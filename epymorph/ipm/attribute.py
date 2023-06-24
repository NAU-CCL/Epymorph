import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Literal, cast

import numpy as np
import sympy
from numpy.typing import NDArray
from sympy import Symbol

from epymorph.clock import Tick
from epymorph.context import SimContext
from epymorph.world import Location


def _to_symbol(name: str) -> Symbol:
    # sympy doesn't provide explicit typing, so we adapt
    return cast(Symbol, sympy.symbols(name))


AttributeType = Literal['int'] | Literal['float'] | Literal['str']
AttributeShape = str

# IPMs can use parameters of any of these shapes, where:
# - A is an "arbitrary" integer index, 0 or more
# - S is a single scalar value
# - T is the number of ticks
# - N is the number of nodes
# ---
# S
# A
# T
# N
# TxA
# NxA
# TxN
shape_regex = re.compile(r"A|[STN]|[TN]xA|TxN(xA)?"
                         .replace("A", "(0|[1-9][0-9]*)"))
parts_regex = re.compile(r"(.*?)([0-9]*)")


def validate_shape(shape: AttributeShape) -> None:
    """Is this string a valid shape? If not, throw `ValueError`."""
    if not shape_regex.fullmatch(shape):
        raise ValueError(f"{shape} is not a valid shape specification.")


def shape_matches(ctx: SimContext, expected_shape: AttributeShape, value: NDArray) -> bool:
    match = parts_regex.fullmatch(expected_shape)
    if not match:
        # This method should only be called on shape we know to be valid.
        # But if not...
        raise Exception(f"Unsupported shape {expected_shape}.")

    prefix, arbitrary_index = match.groups()
    match prefix:
        case 'S':
            # array is not a scalar
            return False
        case "T":
            return value.shape == (ctx.clock.num_days,)
        case "N":
            return value.shape == (ctx.nodes,)
        case "TxN":
            return value.shape == (ctx.clock.num_days, ctx.nodes)
        case "":
            a = int(arbitrary_index)
            return len(value.shape) == 1 and value.shape[0] > a
        case "Tx":
            a = int(arbitrary_index)
            return len(value.shape) == 2 and value.shape[0] == ctx.clock.num_days and value.shape[1] > a
        case "Nx":
            a = int(arbitrary_index)
            return len(value.shape) == 2 and value.shape[0] == ctx.nodes and value.shape[1] > a
        case "TxNx":
            a = int(arbitrary_index)
            return len(value.shape) == 3 and value.shape[0] == ctx.clock.num_days and value.shape[1] == ctx.nodes and value.shape[2] > a
        case _:
            raise Exception(f"Unsupported shape {expected_shape}.")


@dataclass(frozen=True)
class AttributeDef(ABC):
    symbol: Symbol
    attribute_name: str
    shape: AttributeShape
    dtype: AttributeType

    data_source: ClassVar[str]
    """Which data source does this attribute draw from?"""


@dataclass(frozen=True)
class GeoDef(AttributeDef):
    data_source = 'geo'


def geo(symbol_name: str, attribute_name: str | None = None, shape: str = 'S', dtype: AttributeType = 'float') -> GeoDef:
    validate_shape(shape)
    if attribute_name is None:
        attribute_name = symbol_name
    return GeoDef(_to_symbol(symbol_name), attribute_name, shape, dtype)


def quick_geos(symbol_names: str) -> list[AttributeDef]:
    return [geo(name) for name in symbol_names.split()]


@dataclass(frozen=True)
class ParamDef(AttributeDef):
    data_source = 'param'


def param(symbol_name: str, attribute_name: str | None = None, shape: str = 'S', dtype: AttributeType = 'float') -> ParamDef:
    validate_shape(shape)
    if attribute_name is None:
        attribute_name = symbol_name
    return ParamDef(_to_symbol(symbol_name), attribute_name, shape, dtype)


def quick_params(symbol_names: str) -> list[AttributeDef]:
    return [param(name) for name in symbol_names.split()]


AttributeGetter = Callable[[Location, Tick], int | float | str]


def compile_getter(ctx: SimContext, attribute: AttributeDef) -> AttributeGetter:
    match attribute:
        case GeoDef(_, name, _, _):
            data = ctx.geo[name]
        case ParamDef(_, name, _, _):
            data = ctx.param[name]

    # if we match (this should match all valid shapes):
    # the first group, prefix, will contain everything except the arbitrary index (if present)
    # and the second group will have the index
    # prefix will either be empty or have a trailing 'x' if an arbitrary index is present
    match = parts_regex.fullmatch(attribute.shape)
    if match is None:
        raise Exception(f"Unsupported shape: {attribute.shape}")
    prefix, arbitrary_index = match.groups()

    # with this knowledge we can now handle every possible case:
    match prefix:
        case "S":
            return lambda loc, tick: data
        case "T":
            return lambda loc, tick: data[tick.day]
        case "N":
            return lambda loc, tick: data[loc.index]
        case "TxN":
            return lambda loc, tick: data[tick.day, loc.index]
        case "":
            a = int(arbitrary_index)
            return lambda loc, tick: data[a]
        case "Tx":
            a = int(arbitrary_index)
            return lambda loc, tick: data[tick.day, a]
        case "Nx":
            a = int(arbitrary_index)
            return lambda loc, tick: data[loc.index, a]
        case "TxNx":
            a = int(arbitrary_index)
            return lambda loc, tick: data[tick.day, loc.index, a]
        case _:
            raise Exception(f"Unsupported shape: {attribute.shape}")

    # unfortunately Python's structural pattern matching doesn't allow regex
    # or this would be a lot cleaner


def _check_type_scalar(attr: AttributeDef, value: Any) -> bool:
    match attr.dtype:
        case 'str':
            return isinstance(value, str)
        case 'int':
            return isinstance(value, int)
        case 'float':
            return isinstance(value, float)


def _check_type_numpy(attr: AttributeDef, value: NDArray) -> bool:
    match attr.dtype:
        case 'str':
            return value.dtype.type == np.dtype(np.str_)
        case 'int':
            return value.dtype.type == np.dtype(np.int_)
        case 'float':
            return value.dtype.type == np.dtype(np.float_)


def verify_attribute(ctx: SimContext, attr: AttributeDef) -> str | None:
    data = getattr(ctx, attr.data_source)
    if not attr.attribute_name in data:
        return f"Attribute {attr.attribute_name} missing from {attr.data_source}."

    value = data[attr.attribute_name]
    if attr.shape == 'S':
        # check scalar values
        if not _check_type_scalar(attr, value):
            return f"Attribute {attr.attribute_name} was expected to be a scalar {attr.dtype}."
    else:
        # check numpy array values
        if not isinstance(value, np.ndarray):
            return f"Attribute {attr.attribute_name} was expected to be an array."
        elif not _check_type_numpy(attr, value):
            return f"Attribute {attr.attribute_name} was expected to be an array of type {attr.dtype}."
        elif not shape_matches(ctx, attr.shape, value):
            return f"Attribute {attr.attribute_name} was expected to be an arry of shape {attr.shape}."

    return None  # no errors!
