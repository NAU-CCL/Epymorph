"""General simulation data types, events, and utility functions."""
import logging
import textwrap
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta
from importlib import reload
from typing import (Callable, Iterable, Literal, Mapping, NamedTuple, Protocol,
                    Self)

import numpy as np
import sympy
from numpy.random import SeedSequence

from epymorph.data_shape import (AttributeGetter, DataShape, Shapes,
                                 SimDimensions)
from epymorph.data_type import (AttributeArray, AttributeScalar, DataDType,
                                DataPyScalar, dtype_as_np, dtype_check,
                                dtype_str)
from epymorph.error import AttributeException
from epymorph.sympy_shim import to_symbol
from epymorph.util import MemoDict


class DataSource(Protocol):
    """A generic simulation data source."""

    @abstractmethod
    def __getitem__(self, name: str, /) -> AttributeArray:
        raise NotImplementedError

    @abstractmethod
    def __contains__(self, name: str, /) -> bool:
        raise NotImplementedError


GeoData = DataSource
ParamsData = DataSource


class DataMapping(DataSource):
    """A mapping of a data source, allowing you to override the keys of certain attributes."""

    _source: DataSource
    _overrides: Mapping[str, str]

    def __init__(self, source: DataSource, overrides: Mapping[str, str]):
        self._source = source
        self._overrides = overrides

    def __getitem__(self, name: str, /) -> AttributeArray:
        if name in self._overrides:
            remapped_name = self._overrides[name]
        else:
            remapped_name = name
        return self._source[remapped_name]

    def __contains__(self, name: str, /) -> bool:
        return name in self._overrides or name in self._source


@dataclass(frozen=True, slots=True)
class AttributeDef:
    """Definition of a simulation attribute."""
    name: str
    source: Literal['geo', 'params']
    dtype: DataDType
    shape: DataShape
    symbol: sympy.Symbol = field(default=None, compare=False)
    default_value: DataPyScalar | None = field(default=None, compare=False)
    comment: str | None = field(default=None, compare=False)

    def __post_init__(self):
        if self.default_value is not None and not dtype_check(self.dtype, self.default_value):
            print(dtype_str(self.dtype))
            print(str(self.default_value), type(self.default_value))
            msg = "AttributeDef's default value does not align with its dtype."
            raise ValueError(msg)

        if self.symbol is None:
            object.__setattr__(self, 'symbol', to_symbol(self.name))

    @property
    def dtype_as_np(self) -> np.dtype:
        """Return the dtype of this attribute in a numpy-equivalent type."""
        return dtype_as_np(self.dtype)

    @property
    def description(self) -> str:
        """Returns a textual description of this attribute."""
        properties = [
            f"type: {dtype_str(self.dtype)}",
            f"shape: {self.shape}",
        ]
        if self.default_value is not None:
            properties.append(f"default: {self.default_value}")
        lines = [
            f"- {self.name} ({', '.join(properties)})",
        ]
        if self.comment is not None:
            lines.extend(
                textwrap.wrap(self.comment,
                              initial_indent="    ",
                              subsequent_indent="    ")
            )
        return "\n".join(lines)


def geo_attrib(name: str,
               dtype: DataDType,
               shape: DataShape = Shapes.N,
               symbolic_name: str | None = None,
               default_value: DataPyScalar | None = None,
               comment: str | None = None) -> AttributeDef:
    """
    Convenience constructor for a geo attribute.
    If `symbolic_name` is None, the attribute name will be used.
    """
    symbol = to_symbol(symbolic_name) if symbolic_name is not None else to_symbol(name)
    return AttributeDef(name, 'geo', dtype, shape, symbol, default_value, comment)


def params_attrib(name: str,
                  dtype: DataDType,
                  shape: DataShape = Shapes.S,
                  symbolic_name: str | None = None,
                  default_value: DataPyScalar | None = None,
                  comment: str | None = None) -> AttributeDef:
    """
    Convenience constructor for a params attribute.
    If `symbolic_name` is None, the attribute name will be used.
    """
    symbol = to_symbol(symbolic_name) if symbolic_name is not None else to_symbol(name)
    return AttributeDef(name, 'params', dtype, shape, symbol, default_value, comment)


def default_rng(seed: int | SeedSequence | None = None) -> Callable[[], np.random.Generator]:
    """
    Convenience constructor to create a factory function for a simulation's random number generator,
    optionally with a given seed.
    """
    return lambda: np.random.default_rng(seed)


@dataclass(frozen=True)
class TimeFrame:
    """The time frame of a simulation."""

    @classmethod
    def of(cls, start_date_iso8601: str, duration_days: int) -> Self:
        """Alternate constructor for TimeFrame, parsing start date from an ISO-8601 string."""
        return cls(date.fromisoformat(start_date_iso8601), duration_days)

    start_date: date
    duration_days: int


class Tick(NamedTuple):
    """
    A Tick bundles related time-step information. For instance, each time step corresponds to a calendar day,
    a numeric day (i.e., relative to the start of the simulation), which tau step this corresponds to, and so on.
    """
    index: int  # step increment regardless of tau (0,1,2,3,...)
    day: int  # day increment, same for each tau step (0,0,1,1,2,2,...)
    date: date  # calendar date corresponding to `day`
    step: int  # which tau step? (0,1,0,1,0,1,...)
    tau: float  # the current tau length (0.666,0.333,0.666,0.333,...)


class TickDelta(NamedTuple):
    """
    An offset relative to a Tick expressed as a number of days which should elapse,
    and the step on which to end up. In applying this delta, it does not matter which
    step we start on. We need the Clock configuration to apply a TickDelta, so see
    Clock for the relevant method.
    """
    days: int  # number of whole days
    step: int  # which tau step within that day (zero-indexed)


NEVER = TickDelta(-1, -1)
"""
A special TickDelta value which expresses an event that should never happen.
Any Tick plus Never returns Never.
"""


def resolve_tick(dim: SimDimensions, tick: Tick, delta: TickDelta) -> int:
    """Add a delta to a tick to get the index of the resulting tick."""
    return -1 if delta.days == -1 else \
        tick.index - tick.step + (dim.tau_steps * delta.days) + delta.step


def simulation_clock(dim: SimDimensions) -> Iterable[Tick]:
    """Generator for the sequence of ticks which makes up the simulation clock."""
    one_day = timedelta(days=1)
    tau_steps = list(enumerate(dim.tau_step_lengths))
    curr_index = 0
    curr_date = dim.start_date
    for day in range(dim.days):
        for step, tau in tau_steps:
            yield Tick(curr_index, day, curr_date, step, tau)
            curr_index += 1
        curr_date += one_day


class CachingGetAttributeMixin:
    """
    A mixin for adding cached attribute getter behavior to a context class.
    Implements the `get_attribute()` method of RumeContext.
    Make sure to call this class' constructor.
    """

    _attribute_getters: MemoDict[AttributeDef, AttributeGetter]

    def __init__(self, geo: GeoData, params: ParamsData, dim: SimDimensions):

        def get_attribute_value(attr: AttributeDef) -> AttributeArray:
            """Retrieve the value associated with the given attribute."""
            match attr.source:
                case 'geo':
                    source = geo
                case 'params':
                    source = params

            if not attr.name in source:
                msg = f"Missing {attr.source} attribute '{attr.name}'"
                raise AttributeException(msg)
            return source[attr.name]

        def create_attribute_getter(attr: AttributeDef) -> AttributeGetter:
            """Create a tick-and-node accessor function for the given attribute."""
            data_raw = get_attribute_value(attr)
            data = attr.shape.adapt(dim, data_raw, True)
            if data is None:
                # TODO: should `adapt` raise the exception?
                msg = f"Attribute '{attr.name}' could not be adapted to the required shape."
                raise AttributeException(msg)
            return attr.shape.accessor(data)

        self._attribute_getters = MemoDict(create_attribute_getter)

    def clear_attribute_getter(self, name: str) -> None:
        """Clear the attribute getter for a particular attribute (by name)."""
        for a in (a for a in self._attribute_getters if a.name == name):
            del self._attribute_getters[a]

    def get_attribute(self, attr: AttributeDef, tick: Tick, node: int) -> AttributeScalar:
        """Get an attribute value at a specific tick and node."""
        return self._attribute_getters[attr](tick.day, node)


def enable_logging(filename: str = 'debug.log', movement: bool = True) -> None:
    """Enable simulation logging to file."""
    reload(logging)
    logging.basicConfig(filename=filename, filemode='w')
    if movement:
        logging.getLogger('movement').setLevel(logging.DEBUG)
