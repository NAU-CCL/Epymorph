from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from epymorph.clock import Tick
from epymorph.context import SimContext
from epymorph.data_shape import (Arbitrary, DataShape, Node, NodeAndArbitrary,
                                 Scalar, Shapes, Time, TimeAndArbitrary,
                                 TimeAndNode, TimeAndNodeAndArbitrary)
from epymorph.ipm.sympy_shim import Symbol, to_symbol
from epymorph.movement.world import Location

AttributeType = type[int] | type[float] | type[str]


class AttributeException(Exception):
    """An error matching an attribute to those provided in a context."""


@dataclass(frozen=True)
class AttributeDef(ABC):
    """Definition of a simulation attribute."""
    symbol: Symbol
    attribute_name: str
    shape: DataShape
    dtype: AttributeType
    allow_broadcast: bool

    @abstractmethod
    def get_value(self, ctx: SimContext) -> NDArray:
        """
        Returns the value drawn from the correct data dict.
        Raises AttributeException if the attribute is not found.
        """

    def verify(self, ctx: SimContext) -> None:
        """
        Check the presence, shape, and datatype of this attribute in the given context.
        Raises AttributeException on mismatch.
        """
        value = self.get_value(ctx)

        if not self.shape.matches(ctx, value, self.allow_broadcast):
            msg = f"Attribute '{self.attribute_name}' was expected to be an array of shape {self.shape} " + \
                f"-- got {value.shape}."
            raise AttributeException(msg)

        if not value.dtype.type == np.dtype(self.dtype).type:
            msg = f"Attribute '{self.attribute_name}' was expected to be an array of type {self.dtype} " +\
                f"-- got {value.dtype}."
            raise AttributeException(msg)

    def get_adapted(self, ctx: SimContext) -> NDArray:
        """
        Adapt the given value to the shape and type expected for this attribute.
        Raises AttributeException if the value cannot be adapted.
        """
        adapted = self.shape.adapt(ctx, self.get_value(ctx), self.allow_broadcast)
        if adapted is None:
            msg = f"Attribute '{self.attribute_name}' could not be adpated to the required shape."
            raise AttributeException(msg)
        return adapted


@dataclass(frozen=True)
class GeoDef(AttributeDef):
    """An attribute drawn from the Geo Model."""

    def get_value(self, ctx: SimContext) -> NDArray:
        if not self.attribute_name in ctx.geo.attributes:
            msg = f"Missing geo attribute '{self.attribute_name}'"
            raise AttributeException(msg)
        return ctx.geo[self.attribute_name]


def geo(symbol_name: str,
        attribute_name: str | None = None,
        shape: DataShape = Shapes.S,
        dtype: AttributeType = float,
        allow_broadcast: bool = True) -> GeoDef:
    """Convenience constructor for GeoDef."""
    if attribute_name is None:
        attribute_name = symbol_name
    return GeoDef(to_symbol(symbol_name), attribute_name, shape, dtype, allow_broadcast)


def quick_geos(symbol_names: str) -> list[AttributeDef]:
    """Convenience constructor: create several geo attributes from a whitespace-delimited string."""
    return [geo(name) for name in symbol_names.split()]


@dataclass(frozen=True)
class ParamDef(AttributeDef):
    """An attribute drawn from the simulation parameters."""

    def get_value(self, ctx: SimContext) -> NDArray:
        if not self.attribute_name in ctx.param:
            msg = f"Missing params attribute '{self.attribute_name}'"
            raise AttributeException(msg)
        return np.asarray(ctx.param[self.attribute_name])


def param(symbol_name: str,
          attribute_name: str | None = None,
          shape: DataShape = Shapes.S,
          dtype: AttributeType = float,
          allow_broadcast: bool = True) -> ParamDef:
    """Convenience constructor for ParamDef."""
    if attribute_name is None:
        attribute_name = symbol_name
    return ParamDef(to_symbol(symbol_name), attribute_name, shape, dtype, allow_broadcast)


def quick_params(symbol_names: str) -> list[AttributeDef]:
    """Convenience constructor: create several param attributes from a whitespace-delimited string."""
    return [param(name) for name in symbol_names.split()]


AttributeGetter = Callable[[Location, Tick], int | float | str]


def compile_getter(ctx: SimContext, attr: AttributeDef) -> AttributeGetter:
    """Create an accessor lambda for the given attribute and context."""

    data = attr.shape.adapt(
        ctx,
        attr.get_value(ctx),
        attr.allow_broadcast
    )

    if data is None:
        # This should be caught during the verification step, but just in case.
        msg = f"Attribute '{attr.attribute_name}' cannot broadcast to the required shape."
        raise AttributeException(msg)

    match attr.shape:
        case Scalar():
            return lambda loc, tick: data  # type: ignore
        case Time():
            return lambda loc, tick: data[tick.day]
        case Node():
            return lambda loc, tick: data[loc.get_index()]
        case TimeAndNode():
            return lambda loc, tick: data[tick.day, loc.get_index()]
        case Arbitrary(a):
            return lambda loc, tick: data[a]
        case TimeAndArbitrary(a):
            return lambda loc, tick: data[tick.day, a]
        case NodeAndArbitrary(a):
            return lambda loc, tick: data[loc.get_index(), a]
        case TimeAndNodeAndArbitrary(a):
            return lambda loc, tick: data[tick.day, loc.get_index(), a]
        case _:
            raise ValueError(f"Unsupported shape: {attr.shape}")
