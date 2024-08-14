"""
Utility classes to verify the shape of data as numpy arrays
whose dimensions can be relative to the simulation context,
and to adapt equivalent shapes.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta
from typing import NamedTuple, Sequence, TypeVar

import numpy as np
from numpy.typing import NDArray

from epymorph.util import Matcher

T = TypeVar('T', bound=np.generic)


class SimDimensions(NamedTuple):
    """The dimensionality of a simulation."""

    @classmethod
    def build(
        cls,
        tau_step_lengths: Sequence[float],
        start_date: date,
        days: int,
        nodes: int,
        compartments: int,
        events: int,
    ):
        """Convenience constructor which reduces the overhead of initializing duplicative fields."""
        tau_steps = len(tau_step_lengths)
        ticks = tau_steps * days
        return cls(
            tuple(tau_step_lengths), tau_steps, start_date, days, ticks,
            nodes, compartments, events,
            (ticks, nodes, compartments, events))

    tau_step_lengths: tuple[float, ...]
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
    def adapt(self, dim: SimDimensions, value: NDArray[T], allow_broadcast: bool) -> NDArray[T] | None:
        """Adapt the given value to this shape, if possible."""


@dataclass(frozen=True)
class Scalar(DataShape):
    """A scalar value."""

    def matches(self, dim: SimDimensions, value: NDArray, allow_broadcast: bool) -> bool:
        return value.shape == tuple()

    def adapt(self, dim: SimDimensions, value: NDArray[T], allow_broadcast: bool) -> NDArray[T] | None:
        if not self.matches(dim, value, allow_broadcast):
            return None
        return value

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

    def adapt(self, dim: SimDimensions, value: NDArray[T], allow_broadcast: bool) -> NDArray[T] | None:
        if value.ndim == 1 and value.shape[0] >= dim.days:
            return value[: dim.days]
        if allow_broadcast and value.shape == tuple():
            return np.broadcast_to(value, shape=(dim.days,))
        return None

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

    def adapt(self, dim: SimDimensions, value: NDArray[T], allow_broadcast: bool) -> NDArray[T] | None:
        if value.ndim == 1 and value.shape[0] == dim.nodes:
            return value
        if allow_broadcast and value.shape == tuple():
            return np.broadcast_to(value, shape=(dim.nodes,))
        return None

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

    def adapt(self, dim: SimDimensions, value: NDArray[T], allow_broadcast: bool) -> NDArray[T] | None:
        if value.shape == (dim.nodes, dim.nodes):
            return value
        if allow_broadcast and value.shape == tuple():
            return np.broadcast_to(value, shape=(dim.nodes, dim.nodes))
        return None

    def __str__(self):
        return "NxN"


@dataclass(frozen=True)
class NodeAndCompartment(DataShape):
    """An array of size NxC."""

    def matches(self, dim: SimDimensions, value: NDArray, allow_broadcast: bool) -> bool:
        N, C = dim.nodes, dim.compartments
        if value.shape == (N, C):
            return True
        if allow_broadcast:
            if value.shape == tuple():
                return True
            if value.shape == (N,):
                return True
            if value.shape == (C,):
                return True
        return False

    def adapt(self, dim: SimDimensions, value: NDArray[T], allow_broadcast: bool) -> NDArray[T] | None:
        N, C = dim.nodes, dim.compartments
        if value.shape == (N, C):
            return value
        if allow_broadcast:
            if value.shape == tuple():
                return np.broadcast_to(value, shape=(N, C))
            if value.shape == (N,):
                return np.broadcast_to(value[:, np.newaxis], shape=(N, C))
            if value.shape == (C,):
                return np.broadcast_to(value, shape=(N, C))
        return None

    def __str__(self):
        return "NxC"


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

    def adapt(self, dim: SimDimensions, value: NDArray[T], allow_broadcast: bool) -> NDArray[T] | None:
        if value.ndim == 2 and value.shape[0] >= dim.days and value.shape[1] == dim.nodes:
            return value[: dim.days, :]
        if allow_broadcast:
            if value.shape == tuple():
                return np.broadcast_to(value, shape=(dim.days, dim.nodes))
            if value.shape == (dim.nodes,):
                return np.broadcast_to(value, shape=(dim.days, dim.nodes))
            if value.ndim == 1 and value.shape[0] >= dim.days:
                return np.broadcast_to(value[: dim.days, np.newaxis], shape=(dim.days, dim.nodes))
        return None

    def __str__(self):
        return "TxN"


class NodeAndArbitrary(DataShape):
    """An array of size exactly-N by any dimension."""

    def matches(self, dim: SimDimensions, value: NDArray, allow_broadcast: bool) -> bool:
        if value.ndim == 2 and value.shape[0] == dim.nodes:
            return True
        return False

    def adapt(self, dim: SimDimensions, value: NDArray[T], allow_broadcast: bool) -> NDArray[T] | None:
        return value if self.matches(dim, value, allow_broadcast) else None

    def __str__(self):
        return "NxA"


class ArbitraryAndNode(DataShape):
    """An array of size any dimension by exactly-N."""

    def matches(self, dim: SimDimensions, value: NDArray, allow_broadcast: bool) -> bool:
        if value.ndim == 2 and value.shape[1] == dim.nodes:
            return True
        return False

    def adapt(self, dim: SimDimensions, value: NDArray[T], allow_broadcast: bool) -> NDArray[T] | None:
        return value if self.matches(dim, value, allow_broadcast) else None

    def __str__(self):
        return "AxN"


@dataclass(frozen=True)
class Shapes:
    """Static instances for all available shapes."""

    # Data can be in any of these shapes, where:
    # - S is a single scalar value
    # - T is the number of days
    # - N is the number of nodes
    # - C is the number of IPM compartments
    # - A is any length (arbitrary; this dimension is effectively unchecked)

    S = Scalar()
    T = Time()
    N = Node()
    NxC = NodeAndCompartment()
    NxN = NodeAndNode()
    TxN = TimeAndNode()
    NxA = NodeAndArbitrary()
    AxN = ArbitraryAndNode()


def parse_shape(shape: str) -> DataShape:
    """Attempt to parse an AttributeShape from a string."""
    match shape:
        case "S":
            return Shapes.S
        case "T":
            return Shapes.T
        case "N":
            return Shapes.N
        case "NxC":
            return Shapes.NxC
        case "NxN":
            return Shapes.NxN
        case "TxN":
            return Shapes.TxN
        case "NxA":
            return Shapes.NxA
        case "AxN":
            return Shapes.AxN
        case _:
            raise ValueError(f"'{shape}' is not a valid shape specification.")


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
