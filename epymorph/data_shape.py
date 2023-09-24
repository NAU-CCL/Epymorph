"""
Utility classes to verify the shape of data as numpy arrays
whose dimensions can be relative to the simulation context,
and to adapt equivalent shapes.
"""
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class SimDimension:
    """The dimensionality of a simulation."""

    nodes: int
    """How many nodes are there in the GEO?"""
    compartments: int
    """How many disease compartments are in the IPM?"""
    events: int
    """How many transition events are in the IPM?"""
    ticks: int
    """How many ticks are we going to run the simulation for?"""
    days: int
    """How many days are we going to run the simulation for?"""

    TNCE: tuple[int, int, int, int]
    """
    The critical dimensionalities of the simulation, for ease of unpacking.
    T: number of ticks;
    N: number of geo nodes;
    C: number of IPM compartments;
    E: number of IPM events (transitions)
    """


class DataShape(ABC):
    """Description of a data attribute's shape relative to a simulation context."""

    @abstractmethod
    def matches(self, dim: SimDimension, value: NDArray, allow_broadcast: bool) -> bool:
        """Does the given value match this shape expression?"""

    @abstractmethod
    def adapt(self, dim: SimDimension, value: NDArray, allow_broadcast: bool) -> NDArray | None:
        """Adapt the given value to this shape, if possible."""


@dataclass(frozen=True)
class Scalar(DataShape):
    """A scalar value."""

    def matches(self, dim: SimDimension, value: NDArray, allow_broadcast: bool) -> bool:
        return value.shape == tuple()

    def adapt(self, dim: SimDimension, value: NDArray, allow_broadcast: bool) -> NDArray | None:
        if not self.matches(dim, value, allow_broadcast):
            return None
        return value

    def __str__(self):
        return "S"


@dataclass(frozen=True)
class Time(DataShape):
    """An array of at least size T (the number of simulation days)."""

    def matches(self, dim: SimDimension, value: NDArray, allow_broadcast: bool) -> bool:
        if len(value.shape) == 1 and value.shape[0] >= dim.days:
            return True
        if allow_broadcast and value.shape == tuple():
            return True
        return False

    def adapt(self, dim: SimDimension, value: NDArray, allow_broadcast: bool) -> NDArray | None:
        if len(value.shape) == 1 and value.shape[0] >= dim.days:
            return value[:dim.days]
        if allow_broadcast and value.shape == tuple():
            return np.broadcast_to(value, shape=(dim.days,))
        return None

    def __str__(self):
        return "T"


@dataclass(frozen=True)
class Node(DataShape):
    """An array of size N (the number of simulation nodes)."""

    def matches(self, dim: SimDimension, value: NDArray, allow_broadcast: bool) -> bool:
        if len(value.shape) == 1 and value.shape[0] == dim.nodes:
            return True
        if allow_broadcast and value.shape == tuple():
            return True
        return False

    def adapt(self, dim: SimDimension, value: NDArray, allow_broadcast: bool) -> NDArray | None:
        if len(value.shape) == 1 and value.shape[0] == dim.nodes:
            return value
        if allow_broadcast and value.shape == tuple():
            return np.broadcast_to(value, shape=(dim.nodes,))
        return None

    def __str__(self):
        return "N"


@dataclass(frozen=True)
class NodeAndNode(DataShape):
    """An array of size NxN."""

    def matches(self, dim: SimDimension, value: NDArray, allow_broadcast: bool) -> bool:
        if value.shape == (dim.nodes, dim.nodes):
            return True
        if allow_broadcast:
            if value.shape == tuple():
                return True
        return False

    def adapt(self, dim: SimDimension, value: NDArray, allow_broadcast: bool) -> NDArray | None:
        if value.shape == (dim.nodes, dim.nodes):
            return value
        if allow_broadcast and value.shape == tuple():
            return np.broadcast_to(value, shape=(dim.nodes, dim.nodes))
        return None

    def __str__(self):
        return "N"


@dataclass(frozen=True)
class TimeAndNode(DataShape):
    """An array of size at-least-T by exactly-N."""

    def matches(self, dim: SimDimension, value: NDArray, allow_broadcast: bool) -> bool:
        if len(value.shape) == 2 and value.shape[0] >= dim.days and value.shape[1] == dim.nodes:
            return True
        if allow_broadcast:
            if value.shape == tuple():
                return True
            if value.shape == (dim.nodes,):
                return True
            if len(value.shape) == 1 and value.shape[0] >= dim.days:
                return True
        return False

    def adapt(self, dim: SimDimension, value: NDArray, allow_broadcast: bool) -> NDArray | None:
        if len(value.shape) == 2 and value.shape[0] >= dim.days and value.shape[1] == dim.nodes:
            return value[:dim.days, :]
        if allow_broadcast:
            if value.shape == tuple():
                return np.broadcast_to(value, shape=(dim.days, dim.nodes))
            if value.shape == (dim.nodes,):
                return np.broadcast_to(value, shape=(dim.days, dim.nodes))
            if len(value.shape) == 1 and value.shape[0] >= dim.days:
                return np.broadcast_to(value[:dim.days, np.newaxis], shape=(dim.days, dim.nodes))
        return None

    def __str__(self):
        return "TxN"


@dataclass(frozen=True)
class Arbitrary(DataShape):
    """An array of arbitrary length greater than `index`."""

    index: int

    def __post_init__(self):
        if self.index < 0:
            raise ValueError("Arbitrary shape cannot specify negative indices.")

    def matches(self, dim: SimDimension, value: NDArray, allow_broadcast: bool) -> bool:
        if len(value.shape) == 1 and value.shape[0] > self.index:
            return True
        return False

    def adapt(self, dim: SimDimension, value: NDArray, allow_broadcast: bool) -> NDArray | None:
        if len(value.shape) == 1 and value.shape[0] > self.index:
            return value
        return None

    def __str__(self):
        return f"A({self.index})"


@dataclass(frozen=True)
class TimeAndArbitrary(DataShape):
    """An array of size at-least-T by an arbitrary length greater than `index`."""

    index: int

    def __post_init__(self):
        if self.index < 0:
            raise ValueError("Arbitrary shape cannot specify negative indices.")

    def matches(self, dim: SimDimension, value: NDArray, allow_broadcast: bool) -> bool:
        a = value.shape[-1]
        if len(value.shape) == 2 and value.shape[0] >= dim.days and a > self.index:
            return True
        if allow_broadcast:
            if len(value.shape) == 1 and a > self.index:
                return True
        return False

    def adapt(self, dim: SimDimension, value: NDArray, allow_broadcast: bool) -> NDArray | None:
        a = value.shape[-1]
        if len(value.shape) == 2 and value.shape[0] >= dim.days and a > self.index:
            return value[:dim.days, :]
        if allow_broadcast:
            if len(value.shape) == 1 and a > self.index:
                return np.broadcast_to(value, shape=(dim.days, a))
        return None

    def __str__(self):
        return f"TxA({self.index})"


@dataclass(frozen=True)
class NodeAndArbitrary(DataShape):
    """An array of size exactly-N by an arbitrary length greater than `index`."""

    index: int

    def __post_init__(self):
        if self.index < 0:
            raise ValueError("Arbitrary shape cannot specify negative indices.")

    def matches(self, dim: SimDimension, value: NDArray, allow_broadcast: bool) -> bool:
        a = value.shape[-1]
        if len(value.shape) == 2 and value.shape[0] == dim.nodes and a > self.index:
            return True
        if allow_broadcast:
            if len(value.shape) == 1 and a > self.index:
                return True
        return False

    def adapt(self, dim: SimDimension, value: NDArray, allow_broadcast: bool) -> NDArray | None:
        a = value.shape[-1]
        if len(value.shape) == 2 and value.shape[0] == dim.nodes and a > self.index:
            return value
        if allow_broadcast:
            if len(value.shape) == 1 and a > self.index:
                return np.broadcast_to(value, shape=(dim.nodes, a))
        return None

    def __str__(self):
        return f"NxA({self.index})"


@dataclass(frozen=True)
class TimeAndNodeAndArbitrary(DataShape):
    """An array of size at-least-T by exactly-N by an arbitrary length greater than `index`."""

    index: int

    def __post_init__(self):
        if self.index < 0:
            raise ValueError("Arbitrary shape cannot specify negative indices.")

    def matches(self, dim: SimDimension, value: NDArray, allow_broadcast: bool) -> bool:
        a = value.shape[-1]
        if len(value.shape) == 3 and value.shape[0] >= dim.days and value.shape[1] == dim.nodes and a > self.index:
            return True
        if allow_broadcast:
            if len(value.shape) == 1 and a > self.index:
                return True
            if len(value.shape) == 2 and value.shape[0] == dim.nodes and a > self.index:
                return True
            if len(value.shape) == 2 and value.shape[0] >= dim.days and a > self.index:
                return True
        return False

    def adapt(self, dim: SimDimension, value: NDArray, allow_broadcast: bool) -> NDArray | None:
        a = value.shape[-1]
        if len(value.shape) == 3 and value.shape[0] >= dim.days and value.shape[1] == dim.nodes and a > self.index:
            return value[:dim.days, :, :]
        if allow_broadcast:
            if len(value.shape) == 1 and a > self.index:
                return np.broadcast_to(value, shape=(dim.days, dim.nodes, a))
            if len(value.shape) == 2 and value.shape[0] == dim.nodes and a > self.index:
                return np.broadcast_to(value, shape=(dim.days, dim.nodes, a))
            if len(value.shape) == 2 and value.shape[0] >= dim.days and a > self.index:
                return np.broadcast_to(value[:dim.days, np.newaxis, :], shape=(dim.days, dim.nodes, a))
        return None

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
# S; A; T; N; TxA; NxA; TxN; TxNxA


_shape_regex = re.compile(r"A|[STN]|[TN]xA|TxN(xA)?"
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
