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

from epymorph.data_type import AttributeArray, SimDType
from epymorph.error import AttributeException, MmSimException
from epymorph.simulation import (AttributeDef, NamespacedAttributeResolver,
                                 SimDimensions, Tick, TickDelta)


class MovementContext(Protocol):
    """The subset of the RumeContext that the movement model clauses need."""

    @property
    @abstractmethod
    def dim(self) -> SimDimensions:
        """The simulation dimensions."""

    @property
    @abstractmethod
    def rng(self) -> np.random.Generator:
        """The simulation's random number generator."""

    @property
    @abstractmethod
    def data(self) -> NamespacedAttributeResolver:
        """The resolver for simulation data."""


PredefData = dict[str, AttributeArray]
PredefClause = Callable[[MovementContext], PredefData]


class TravelClause(ABC):
    """A clause moving individuals from their home location to another."""

    name: str

    @abstractmethod
    def predicate(self, ctx: MovementContext, tick: Tick) -> bool:
        """Should this clause apply this tick?"""

    @abstractmethod
    def requested(self, ctx: MovementContext, tick: Tick) -> NDArray[SimDType]:
        """Evaluate this clause for the given tick, returning a requested movers array (N,N)."""

    @abstractmethod
    def returns(self, ctx: MovementContext, tick: Tick) -> TickDelta:
        """Calculate when this clause's movers should return (which may vary from tick-to-tick)."""


MovementPredicate = Callable[[MovementContext, Tick], bool]
"""A predicate which decides if a clause should fire this tick."""

MovementFunction = Callable[[MovementContext, Tick], NDArray[SimDType]]
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

    _move: MovementPredicate
    _requested: MovementFunction
    _returns: ReturnsFunction

    def __init__(self,
                 name: str,
                 move_predicate: MovementPredicate,
                 requested: MovementFunction,
                 returns: ReturnsFunction):
        self.name = name
        self._move = move_predicate
        self._requested = requested
        self._returns = returns

    def predicate(self, ctx: MovementContext, tick: Tick) -> bool:
        return self._move(ctx, tick)

    def requested(self, ctx: MovementContext, tick: Tick) -> NDArray[SimDType]:
        try:
            return self._requested(ctx, tick)
        except KeyError as e:
            # NOTE: catching KeyError here will be necessary (to get nice error messages)
            # until we can properly validate the MM clauses.
            msg = f"Missing attribute {e} required by movement model clause '{self.name}'."
            raise AttributeException(msg) from None
        except Exception as e:
            # NOTE: catching exceptions here is necessary to get nice error messages
            # for some value error cause by incorrect parameter and/or clause definition
            msg = f"Error from applying clause '{self.name}': see exception trace"
            raise MmSimException(msg) from e

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

    attributes: list[AttributeDef]

    predef: PredefClause
    """The predef clause for this movement model."""

    clauses: list[TravelClause]
    """The clauses which express the movement model"""
