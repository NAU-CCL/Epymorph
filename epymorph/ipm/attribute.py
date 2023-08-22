from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, Callable, ClassVar

import numpy as np
from numpy.typing import DTypeLike, NDArray

from epymorph.clock import Tick
from epymorph.context import (Arbitrary, DataShape, Node, Scalar, Shapes,
                              SimContext, Time, TimeAndNode)
from epymorph.ipm.sympy_shim import Symbol, to_symbol
from epymorph.movement.world import Location

AttributeType = type[int] | type[float] | type[str]


@dataclass(frozen=True)
class AttributeDef(ABC):
    """Definition of a simulation attribute."""
    symbol: Symbol
    attribute_name: str
    shape: DataShape
    dtype: AttributeType

    data_source: ClassVar[str]
    """Which data source does this attribute draw from?"""


@dataclass(frozen=True)
class GeoDef(AttributeDef):
    """An attribute drawn from the Geo Model."""
    data_source = 'geo'


def geo(symbol_name: str,
        attribute_name: str | None = None,
        shape: DataShape = Shapes.S,
        dtype: AttributeType = float) -> GeoDef:
    """Convenience constructor for GeoDef."""
    if attribute_name is None:
        attribute_name = symbol_name
    return GeoDef(to_symbol(symbol_name), attribute_name, shape, dtype)


def quick_geos(symbol_names: str) -> list[AttributeDef]:
    """Convenience constructor: create several geo attributes from a whitespace-delimited string."""
    return [geo(name) for name in symbol_names.split()]


@dataclass(frozen=True)
class ParamDef(AttributeDef):
    """An attribute drawn from the simulation parameters."""
    data_source = 'param'


def param(symbol_name: str,
          attribute_name: str | None = None,
          shape: DataShape = Shapes.S,
          dtype: AttributeType = float) -> ParamDef:
    """Convenience constructor for ParamDef."""
    if attribute_name is None:
        attribute_name = symbol_name
    return ParamDef(to_symbol(symbol_name), attribute_name, shape, dtype)


def quick_params(symbol_names: str) -> list[AttributeDef]:
    """Convenience constructor: create several param attributes from a whitespace-delimited string."""
    return [param(name) for name in symbol_names.split()]


AttributeGetter = Callable[[Location, Tick], int | float | str]


def compile_getter(ctx: SimContext, attribute: AttributeDef) -> AttributeGetter:
    match attribute:
        case GeoDef(_, name, _, _):
            data = ctx.geo[name]
        case ParamDef(_, name, _, _):
            data = ctx.param[name]
        case _:
            raise ValueError(f"Unsupported attribute type {type(attribute)}")

    # with this knowledge we can now handle every possible case:
    match attribute.shape:
        case Scalar():
            return lambda loc, tick: data
        case Time():
            return lambda loc, tick: data[tick.day]
        case Node():
            return lambda loc, tick: data[loc.get_index()]
        case TimeAndNode():
            return lambda loc, tick: data[tick.day, loc.get_index()]
        case Arbitrary(a, None):
            return lambda loc, tick: data[a]
        case Arbitrary(a, Shapes.T):
            return lambda loc, tick: data[tick.day, a]
        case Arbitrary(a, Shapes.N):
            return lambda loc, tick: data[loc.get_index(), a]
        case Arbitrary(a, Shapes.TxN):
            return lambda loc, tick: data[tick.day, loc.get_index(), a]
        case _:
            raise ValueError(f"Unsupported shape: {attribute.shape}")


def _check_type_scalar(attr: AttributeDef, value: Any) -> bool:
    return isinstance(value, attr.dtype)


def _check_type_numpy(attr: AttributeDef, value: NDArray) -> bool:
    if attr.dtype == str:
        return value.dtype.type == np.dtype(np.str_)
    if attr.dtype == int:
        return value.dtype.type == np.dtype(np.int_)
    if attr.dtype == float:
        return value.dtype.type == np.dtype(np.float_)
    raise ValueError(f"Unsupported dtype: {attr.dtype}")


def verify_attribute(ctx: SimContext, attr: AttributeDef) -> str | None:
    data = getattr(ctx, attr.data_source)
    if not attr.attribute_name in data:
        return f"Attribute {attr.attribute_name} missing from {attr.data_source}."

    value = data[attr.attribute_name]
    match attr.shape:
        case Scalar():
            # check scalar values
            if not attr.shape.matches(ctx, value):
                return f"Attribute {attr.attribute_name} was expected to be a scalar."
            if not _check_type_scalar(attr, value):
                return f"Attribute {attr.attribute_name} was expected to be a scalar {attr.dtype}."
        case shape:
            # check numpy array values
            if not isinstance(value, np.ndarray):
                return f"Attribute {attr.attribute_name} was expected to be an array."
            if not shape.matches(ctx, value):
                return f"Attribute {attr.attribute_name} was expected to be an array of shape {attr.shape}."
            if not _check_type_numpy(attr, value):
                return f"Attribute {attr.attribute_name} was expected to be an array of type {attr.dtype}."

    return None  # no errors!


def process_params(params: dict[str, Any], dtype: dict[str, DTypeLike] | None = None) -> dict[str, Any]:
    """Pre-process parameter dictionaries."""
    # This simplifies attribute verification: checking lists is more tedious/error-prone than checking ndarrays.
    if dtype is None:
        dtype = {}
    ps = params.copy()
    # Replace parameter lists with numpy arrays.
    for key, value in ps.items():
        if isinstance(value, list):
            dt = dtype.get(key, None)
            ps[key] = np.array(value, dtype=dt)
    return ps
