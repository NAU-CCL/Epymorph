"""
The basis of the movement model system in epymorph.
This module contains all of the elements needed to define a
movement model, but Rume of it is left to the mm_exec module.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray

from epymorph.compartment_model import CompartmentModel
from epymorph.error import AttributeException, MmValidationException
from epymorph.geo.geo import Geo
from epymorph.movement.parser import Attribute as MmAttribute
from epymorph.params import ContextParams
from epymorph.simulation import (DataSource, SimDimensions, SimDType, Tick,
                                 TickDelta)
from epymorph.util import NumpyTypeError


class MovementContext(Protocol):
    """The subset of the RumeContext that the movement model clauses need."""
    # This machine avoids circular deps.
    dim: SimDimensions
    geo: Geo
    ipm: CompartmentModel
    params: ContextParams
    rng: np.random.Generator
    version: int


PredefParams = ContextParams
PredefClause = Callable[[MovementContext], PredefParams]


class TravelClause(ABC):
    """A clause moving individuals from their home location to another."""

    name: str

    @abstractmethod
    def mask(self, ctx: MovementContext) -> NDArray[np.bool_]:
        """Calculate the movement mask for this clause."""

    @abstractmethod
    def predicate(self, ctx: MovementContext, tick: Tick) -> bool:
        """Should this clause apply this tick?"""

    @abstractmethod
    def requested(self, ctx: MovementContext, predef: PredefParams, tick: Tick) -> NDArray[SimDType]:
        """Evaluate this clause for the given tick, returning a requested movers array (N,N)."""

    @abstractmethod
    def returns(self, ctx: MovementContext, tick: Tick) -> TickDelta:
        """Calculate when this clause's movers should return (which may vary from tick-to-tick)."""


MaskPredicate = Callable[[MovementContext], NDArray[np.bool_]]
"""
A predicate which creates a per-IPM-compartment mask:
should this compartment be subject to movement by this clause?
"""

MovementPredicate = Callable[[MovementContext, Tick], bool]
"""A predicate which decides if a clause should fire this tick."""

MovementFunction = Callable[[MovementContext, PredefParams, Tick], NDArray[SimDType]]
"""
A function which calculates the requested number of individuals to move due to this clause this tick.
Returns an (N,N) array of integers.
"""

ReturnsFunction = Callable[[MovementContext, Tick], TickDelta]
"""A function which decides when this clause's movers should return."""


class DynamicTravelClause(TravelClause):
    """
    A travel clause implementation where each method proxies to a lambda.
    This allows us to build travel clauses dynamically at runtime.
    """

    name: str

    _mask: MaskPredicate
    _move: MovementPredicate
    _requested: MovementFunction
    _returns: ReturnsFunction

    def __init__(self,
                 name: str,
                 mask_predicate: MaskPredicate,
                 move_predicate: MovementPredicate,
                 requested: MovementFunction,
                 returns: ReturnsFunction):
        self.name = name
        self._mask = mask_predicate
        self._move = move_predicate
        self._requested = requested
        self._returns = returns

    def mask(self, ctx: MovementContext) -> NDArray[np.bool_]:
        return self._mask(ctx)

    def predicate(self, ctx: MovementContext, tick: Tick) -> bool:
        return self._move(ctx, tick)

    def requested(self, ctx: MovementContext, predef: PredefParams, tick: Tick) -> NDArray[SimDType]:
        try:
            return self._requested(ctx, predef, tick)
        except KeyError as e:
            # NOTE: catching KeyError here will be necessary (to get nice error messages)
            # until we can properly validate the MM clauses.
            msg = f"Missing attribute {e} required by movement model clause '{self.name}'."
            raise AttributeException(msg) from None

    def returns(self, ctx: MovementContext, tick: Tick) -> TickDelta:
        return self._returns(ctx, tick)


@dataclass(frozen=True)
class MovementModel:
    """
    The movement model divides a day into simulation parts (tau steps) under the assumption
    that each day part will have movement characteristics relevant to the simulation.
    That is: there is no reason to have tau steps smaller than 1 day unless it's relevant
    to movement.
    """

    tau_steps: list[float]
    """The tau steps for the simulation."""

    attributes: list[MmAttribute]

    predef: PredefClause
    """The predef clause for this movement model."""
    predef_context_hash: Callable[[MovementContext], int]
    """
    A hash function which determines if the given MovementContext has changed in a way that would
    necessitate the predef be re-calculated.
    """

    clauses: list[TravelClause]
    """The clauses which express the movement model"""


# Validation


def validate_mm(
    mm_attribs: list[MmAttribute],
    dim: SimDimensions,
    geo_data: DataSource,  # GeoData
    params_data: DataSource,  # ParamsData
) -> None:
    """Validate that the MM has all required attributes."""

    def _validate_attr(attr: MmAttribute) -> Exception | None:
        try:
            name = attr.name
            match attr.source:
                case 'geo':
                    if not name in geo_data:
                        msg = f"Missing geo attribute '{name}'"
                        raise AttributeException(msg)
                    value = geo_data[name]
                case 'params':
                    if not name in params_data:
                        msg = f"Missing params attribute '{name}'"
                        raise AttributeException(msg)
                    value = params_data[name]

            # NOTE: Movement Model currently DOES NOT allow shape polymorphism.
            # It doesn't have an infrastructure to adapt input arrays like the IPM does.
            # The goal is to support this, but I want to delay to 0.5 so as to contain
            # the scope of changes in 0.4. Until then, you have to provide exact shape matches,
            # both in the attributes declaration and in the submitted parameters (which has been the case).

            if not attr.shape.matches(dim, value, False):
                msg = f"Attribute '{attr.name}' was expected to be an array of shape {attr.shape} " + \
                    f"-- got {value.shape}."
                raise AttributeException(msg)

            # if attr.dtype == int:
            #     dtype = np.dtype(np.int64)
            # elif attr.dtype == float:
            #     dtype = np.dtype(np.float64)
            # elif attr.dtype == str:
            #     dtype = np.dtype(np.str_)
            # else:
            #     raise ValueError(f"Unsupported dtype: {attr.dtype}")

            if not np.can_cast(value.dtype, attr.dtype):
                msg = f"Attribute '{attr.name}' was expected to be an array of type {attr.dtype} " + \
                    f"-- got {value.dtype}."
                raise AttributeException(msg)

            return None
        except NumpyTypeError as e:
            msg = f"Attribute '{attr.name}' is not properly specified. {e}"
            return AttributeException(msg)
        except AttributeException as e:
            return e

    # Collect all attribute errors to raise as a group.
    errors = [x for x in (_validate_attr(attr) for attr in mm_attribs)
              if x is not None]

    if len(errors) > 0:
        msg = "MM attribute requirements were not met. See errors:" + \
            "".join(f"\n- {e}" for e in errors)
        raise MmValidationException(msg)
