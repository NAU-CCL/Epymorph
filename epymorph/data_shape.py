"""
Utility classes to verify the shape of data as numpy arrays
whose dimensions can be relative to the simulation context,
and to adapt equivalent shapes.
"""
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, NamedTuple, Sequence

import numpy as np
from numpy.typing import NDArray

from epymorph.data_type import AttributeArray, AttributeScalar
from epymorph.util import Matcher

AttributeGetter = Callable[[int, int], AttributeScalar]
"""A function for retrieving a scalar value for a given simulation day and a given geo node."""


class SimDimensions(NamedTuple):
    """The dimensionality of a simulation."""

    @classmethod
    def build(cls, tau_step_lengths: Sequence[float], start_date: date, days: int, nodes: int, compartments: int, events: int):
        """Convenience constructor which reduces the overhead of initializing duplicative fields."""
        tau_steps = len(tau_step_lengths)
        ticks = tau_steps * days
        return cls(
            tau_step_lengths, tau_steps, start_date, days, ticks,
            nodes, compartments, events,
            (ticks, nodes, compartments, events))

    tau_step_lengths: Sequence[float]
    """The lengths of each tau step in the MM."""
    tau_steps: int
    """How many tau steps are in the MM?"""
    start_date: date
    """On what calendar date did the simulation start?"""
    days: int
    """How many days are we going to run the simulation for?"""
    ticks: int
    """How many clock ticks are we going to run the simulation for?"""
    nodes: int
    """How many nodes are there in the GEO?"""
    compartments: int
    """How many disease compartments are in the IPM?"""
    events: int
    """How many transition events are in the IPM?"""
    TNCE: tuple[int, int, int, int]
    """
    The critical dimensionalities of the simulation, for ease of unpacking.
    T: number of ticks;
    N: number of geo nodes;
    C: number of IPM compartments;
    E: number of IPM events (transitions)
    """

    @property
    def end_date(self) -> date:
        """The end date (the first day not included in the simulation)."""
        return self.start_date + timedelta(days=self.days)


class DataShape(ABC):
    """Description of a data attribute's shape relative to a simulation context."""

    @abstractmethod
    def matches(self, dim: SimDimensions, value: NDArray, allow_broadcast: bool) -> bool:
        """Does the given value match this shape expression?"""

    @abstractmethod
    def adapt(self, dim: SimDimensions, value: NDArray, allow_broadcast: bool) -> NDArray | None:
        """Adapt the given value to this shape, if possible."""

    @abstractmethod
    def as_tuple(self, nodes: int, days: int) -> tuple[int, ...]:
        """Return a concrete shape as a tuple, using the given number of nodes and days to fill in for N and T."""

    @abstractmethod
    def accessor(self, data: AttributeArray) -> AttributeGetter:
        """
        Returns an accessor function for the given shape on the given data array.
        The accessor function is designed to retrieve a scalar value for a given simulation day
        and a given geo node (by index).
        """


class DataShapeMatcher(Matcher[NDArray]):
    """Matches a DataShape, given the SimDimensions."""
    _shape: DataShape
    _dim: SimDimensions
    _allow_broadcast: bool

    def __init__(self, shape: DataShape, dim: SimDimensions, allow_broadcast: bool):
        self._shape = shape
        self._dim = dim
        self._allow_broadcast = allow_broadcast

    def expected(self) -> str:
        """Describes what the expected value is."""
        return str(self._shape)

    def __call__(self, value: NDArray) -> bool:
        return self._shape.matches(self._dim, value, self._allow_broadcast)


@dataclass(frozen=True)
class Scalar(DataShape):
    """A scalar value."""

    def matches(self, dim: SimDimensions, value: NDArray, allow_broadcast: bool) -> bool:
        return value.shape == tuple()

    def adapt(self, dim: SimDimensions, value: NDArray, allow_broadcast: bool) -> NDArray | None:
        if not self.matches(dim, value, allow_broadcast):
            return None
        return value

    def as_tuple(self, nodes: int, days: int) -> tuple[int, ...]:
        return ()

    def accessor(self, data: AttributeArray) -> AttributeGetter:
        return lambda _day, _node: data  # type: ignore

    def __str__(self):
        return "S"


@dataclass(frozen=True)
class Time(DataShape):
    """An array of at least size T (the number of simulation days)."""

    def matches(self, dim: SimDimensions, value: NDArray, allow_broadcast: bool) -> bool:
        if value.ndim == 1 and value.shape[0] >= dim.days:
            return True
        if allow_broadcast and value.shape == tuple():
            return True
        return False

    def adapt(self, dim: SimDimensions, value: NDArray, allow_broadcast: bool) -> NDArray | None:
        if value.ndim == 1 and value.shape[0] >= dim.days:
            return value[:dim.days]
        if allow_broadcast and value.shape == tuple():
            return np.broadcast_to(value, shape=(dim.days,))
        return None

    def as_tuple(self, nodes: int, days: int) -> tuple[int, ...]:
        return (days,)

    def accessor(self, data: AttributeArray) -> AttributeGetter:
        return lambda day, _node: data[day]

    def __str__(self):
        return "T"


@dataclass(frozen=True)
class Node(DataShape):
    """An array of size N (the number of simulation nodes)."""

    def matches(self, dim: SimDimensions, value: NDArray, allow_broadcast: bool) -> bool:
        if value.ndim == 1 and value.shape[0] == dim.nodes:
            return True
        if allow_broadcast and value.shape == tuple():
            return True
        return False

    def adapt(self, dim: SimDimensions, value: NDArray, allow_broadcast: bool) -> NDArray | None:
        if value.ndim == 1 and value.shape[0] == dim.nodes:
            return value
        if allow_broadcast and value.shape == tuple():
            return np.broadcast_to(value, shape=(dim.nodes,))
        return None

    def as_tuple(self, nodes: int, days: int) -> tuple[int, ...]:
        return (nodes,)

    def accessor(self, data: AttributeArray) -> AttributeGetter:
        return lambda _day, node: data[node]

    def __str__(self):
        return "N"


@dataclass(frozen=True)
class NodeAndNode(DataShape):
    """An array of size NxN."""

    def matches(self, dim: SimDimensions, value: NDArray, allow_broadcast: bool) -> bool:
        if value.shape == (dim.nodes, dim.nodes):
            return True
        if allow_broadcast:
            if value.shape == tuple():
                return True
        return False

    def adapt(self, dim: SimDimensions, value: NDArray, allow_broadcast: bool) -> NDArray | None:
        if value.shape == (dim.nodes, dim.nodes):
            return value
        if allow_broadcast and value.shape == tuple():
            return np.broadcast_to(value, shape=(dim.nodes, dim.nodes))
        return None

    def as_tuple(self, nodes: int, days: int) -> tuple[int, ...]:
        return (nodes, nodes)

    def accessor(self, data: AttributeArray) -> AttributeGetter:
        return lambda _day, node: data[node]

    def __str__(self):
        return "N"


@dataclass(frozen=True)
class TimeAndNode(DataShape):
    """An array of size at-least-T by exactly-N."""

    def matches(self, dim: SimDimensions, value: NDArray, allow_broadcast: bool) -> bool:
        if value.ndim == 2 and value.shape[0] >= dim.days and value.shape[1] == dim.nodes:
            return True
        if allow_broadcast:
            if value.shape == tuple():
                return True
            if value.shape == (dim.nodes,):
                return True
            if value.ndim == 1 and value.shape[0] >= dim.days:
                return True
        return False

    def adapt(self, dim: SimDimensions, value: NDArray, allow_broadcast: bool) -> NDArray | None:
        if value.ndim == 2 and value.shape[0] >= dim.days and value.shape[1] == dim.nodes:
            return value[:dim.days, :]
        if allow_broadcast:
            if value.shape == tuple():
                return np.broadcast_to(value, shape=(dim.days, dim.nodes))
            if value.shape == (dim.nodes,):
                return np.broadcast_to(value, shape=(dim.days, dim.nodes))
            if value.ndim == 1 and value.shape[0] >= dim.days:
                return np.broadcast_to(value[:dim.days, np.newaxis], shape=(dim.days, dim.nodes))
        return None

    def as_tuple(self, nodes: int, days: int) -> tuple[int, ...]:
        return (days, nodes)

    def accessor(self, data: AttributeArray) -> AttributeGetter:
        return lambda day, node: data[day, node]

    def __str__(self):
        return "TxN"


@dataclass(frozen=True)
class Arbitrary(DataShape):
    """An array of arbitrary length greater than `index`."""

    index: int

    def __post_init__(self):
        if self.index < 0:
            raise ValueError("Arbitrary shape cannot specify negative indices.")

    def matches(self, dim: SimDimensions, value: NDArray, allow_broadcast: bool) -> bool:
        if value.ndim == 1 and value.shape[0] > self.index:
            return True
        return False

    def adapt(self, dim: SimDimensions, value: NDArray, allow_broadcast: bool) -> NDArray | None:
        if value.ndim == 1 and value.shape[0] > self.index:
            return value
        return None

    def as_tuple(self, nodes: int, days: int) -> tuple[int, ...]:
        # for arbitrary, assume "index" represents the length
        # this is a bit odd, but arises because we need shapes in two subtly different situations:
        # 1. in the MM/IPM to "pull" specific attributes, and 2. in the GEO to define available attributes
        # The GEO defines what it has; while the MM/IPM must slice the available data for its needs.
        # The MM/IPM usage is more like a selection than a shape... maybe we should be using Slice for that...
        return (self.index,)

    def accessor(self, data: AttributeArray) -> AttributeGetter:
        return lambda _day, _node: data[self.index]

    def __str__(self):
        return f"A({self.index})"


@dataclass(frozen=True)
class TimeAndArbitrary(DataShape):
    """An array of size at-least-T by an arbitrary length greater than `index`."""

    index: int

    def __post_init__(self):
        if self.index < 0:
            raise ValueError("Arbitrary shape cannot specify negative indices.")

    def matches(self, dim: SimDimensions, value: NDArray, allow_broadcast: bool) -> bool:
        if value.ndim == 2 and value.shape[0] >= dim.days and value.shape[1] > self.index:
            return True
        if allow_broadcast:
            if value.ndim == 1 and value.shape[0] > self.index:
                return True
        return False

    def adapt(self, dim: SimDimensions, value: NDArray, allow_broadcast: bool) -> NDArray | None:
        if value.ndim == 2 and value.shape[0] >= dim.days and value.shape[1] > self.index:
            return value[:dim.days, :]
        if allow_broadcast:
            if value.ndim == 1 and value.shape[0] > self.index:
                return np.broadcast_to(value, shape=(dim.days, value.shape[0]))
        return None

    def as_tuple(self, nodes: int, days: int) -> tuple[int, ...]:
        return (days, self.index)

    def accessor(self, data: AttributeArray) -> AttributeGetter:
        return lambda day, _node: data[day, self.index]

    def __str__(self):
        return f"TxA({self.index})"


@dataclass(frozen=True)
class NodeAndArbitrary(DataShape):
    """An array of size exactly-N by an arbitrary length greater than `index`."""

    index: int

    def __post_init__(self):
        if self.index < 0:
            raise ValueError("Arbitrary shape cannot specify negative indices.")

    def matches(self, dim: SimDimensions, value: NDArray, allow_broadcast: bool) -> bool:
        if value.ndim == 2 and value.shape[0] == dim.nodes and value.shape[1] > self.index:
            return True
        if allow_broadcast:
            if value.ndim == 1 and value.shape[0] > self.index:
                return True
        return False

    def adapt(self, dim: SimDimensions, value: NDArray, allow_broadcast: bool) -> NDArray | None:
        if value.ndim == 2 and value.shape[0] == dim.nodes and value.shape[1] > self.index:
            return value
        if allow_broadcast:
            if value.ndim == 1 and value.shape[0] > self.index:
                return np.broadcast_to(value, shape=(dim.nodes, value.shape[0]))
        return None

    def as_tuple(self, nodes: int, days: int) -> tuple[int, ...]:
        return (nodes, self.index)

    def accessor(self, data: AttributeArray) -> AttributeGetter:
        return lambda _day, node: data[node, self.index]

    def __str__(self):
        return f"NxA({self.index})"


@dataclass(frozen=True)
class TimeAndNodeAndArbitrary(DataShape):
    """An array of size at-least-T by exactly-N by an arbitrary length greater than `index`."""

    index: int

    def __post_init__(self):
        if self.index < 0:
            raise ValueError("Arbitrary shape cannot specify negative indices.")

    def matches(self, dim: SimDimensions, value: NDArray, allow_broadcast: bool) -> bool:
        if value.ndim == 3 and value.shape[0] >= dim.days and value.shape[1] == dim.nodes and value.shape[2] > self.index:
            return True
        if allow_broadcast and value.ndim > 0:
            a = value.shape[-1]
            if value.ndim == 1 and a > self.index:
                return True
            if value.ndim == 2 and value.shape[0] == dim.nodes and a > self.index:
                return True
            if value.ndim == 2 and value.shape[0] >= dim.days and a > self.index:
                return True
        return False

    def adapt(self, dim: SimDimensions, value: NDArray, allow_broadcast: bool) -> NDArray | None:
        if value.ndim == 3 and value.shape[0] >= dim.days and value.shape[1] == dim.nodes and value.shape[2] > self.index:
            return value[:dim.days, :, :]
        if allow_broadcast and value.ndim > 0:
            a = value.shape[-1]
            if value.ndim == 1 and a > self.index:
                return np.broadcast_to(value, shape=(dim.days, dim.nodes, a))
            if value.ndim == 2 and value.shape[0] == dim.nodes and a > self.index:
                return np.broadcast_to(value, shape=(dim.days, dim.nodes, a))
            if value.ndim == 2 and value.shape[0] >= dim.days and a > self.index:
                return np.broadcast_to(value[:dim.days, np.newaxis, :], shape=(dim.days, dim.nodes, a))
        return None

    def as_tuple(self, nodes: int, days: int) -> tuple[int, ...]:
        return (days, nodes, self.index)

    def accessor(self, data: AttributeArray) -> AttributeGetter:
        return lambda day, node: data[day, node, self.index]

    def __str__(self):
        return f"TxNxA({self.index})"


@dataclass(frozen=True)
class Shapes:
    """Static instances for all available shapes."""

    S = Scalar()
    T = Time()
    N = Node()
    NxN = NodeAndNode()
    TxN = TimeAndNode()
    A = Arbitrary
    TxA = TimeAndArbitrary
    NxA = NodeAndArbitrary
    TxNxA = TimeAndNodeAndArbitrary

# IPMs can use parameters of any of these shapes, where:
# - A is an "arbitrary" integer index, 0 or more
# - S is a single scalar value
# - T is the number of ticks
# - N is the number of nodes
# ---
# S; A; T; N; NxN; TxA; NxA; TxN; TxNxA


_shape_regex = re.compile(r"A|[STN]|NxN|[TN]xA|TxN(xA)?"
                          .replace("A", "(0|[1-9][0-9]*)"))
_parts_regex = re.compile(r"(.*?)([0-9]*)")


def parse_shape(shape: str) -> DataShape:
    """Attempt to parse an AttributeShape from a string."""

    parts_match = _parts_regex.fullmatch(shape)
    if not _shape_regex.fullmatch(shape) or not parts_match:
        raise ValueError(f"'{shape}' is not a valid shape specification.")
    prefix, index = parts_match.groups()
    match prefix:
        case "S":
            return Shapes.S
        case "T":
            return Shapes.T
        case "N":
            return Shapes.N
        case "NxN":
            return Shapes.NxN
        case "TxN":
            return Shapes.TxN
        # blank or trailing 'x' means there's an index at the end -> Arbitrary
        case "":
            return Shapes.A(int(index))
        case "Tx":
            return Shapes.TxA(int(index))
        case "Nx":
            return Shapes.NxA(int(index))
        case "TxNx":
            return Shapes.TxNxA(int(index))
        case _:
            raise ValueError(f"'{shape}' is not a valid shape specification.")
