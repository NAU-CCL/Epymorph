"""
The need for certain info about a simulation cuts across modules (ipm, movement, geo), so
the SimContext structure is here to contain that info and avoid circular dependencies.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from epymorph.clock import Clock
from epymorph.util import DataDict

SimDType = np.int64
"""
This is the numpy datatype that should be used to represent internal simulation data.
Where segments of the application maintain compartment and/or event counts,
they should take pains to use this type at all times (if possible).
"""

# SimDType being centrally-located means we can change it reliably.

Compartments = NDArray[SimDType]
"""Alias for ndarrays representing compartment counts."""

Events = NDArray[SimDType]
"""Alias for ndarrays representing event counts."""

# Aliases (hopefully) make it a bit easier to keep all these NDArrays sorted out.


class SimDimension:
    """The subset of SimContext that is the dimensionality of a simulation."""

    nodes: int
    compartments: int
    events: int
    ticks: int
    days: int

    TNCE: tuple[int, int, int, int]
    """
    The critical dimensionalities of the simulation, for ease of unpacking.
    T: number of ticks;
    N: number of geo nodes;
    C: number of IPM compartments;
    E: number of IPM events (transitions)
    """


@dataclass(frozen=True)
class SimContext(SimDimension):
    """Metadata about the simulation being run."""

    # geo info
    nodes: int
    labels: list[str]
    geo: DataDict
    # ipm info
    compartments: int
    compartment_tags: list[list[str]]
    events: int
    # run info
    param: DataDict
    clock: Clock
    rng: np.random.Generator
    # denormalized info
    ticks: int = field(init=False)
    days: int = field(init=False)
    TNCE: tuple[int, int, int, int] = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'ticks', self.clock.num_ticks)
        object.__setattr__(self, 'days', self.clock.num_days)
        tnce = (self.clock.num_ticks, self.nodes,
                self.compartments, self.events)
        object.__setattr__(self, 'TNCE', tnce)

    @property
    def population(self) -> NDArray[np.integer]:
        """Get the population of each node."""
        # This is for convenient type-safety.
        # TODO: when we construct the geo we should be verifying this fact.
        return self.geo['population']


# Data shapes


class DataShape(ABC):
    """Description of a data attribute's shape relative to a simulation context."""

    @abstractmethod
    def matches(self, dim: SimDimension, value: Any) -> bool:
        """Does the given value match this shape expression?"""


@dataclass(frozen=True)
class Scalar(DataShape):
    """A scalar value."""

    def matches(self, dim: SimDimension, value: Any) -> bool:
        return np.isscalar(value)

    def __str__(self):
        return "S"


class BaseShape(Protocol):
    """A shape which can be extended with an arbitrary index."""
    base_dimensions: int

    @abstractmethod
    def matches_base(self, dim: SimDimension, value: NDArray) -> bool:
        """Check the base characteristics only."""
        # Arbitrary needs to defer to this method.


@dataclass(frozen=True)
class Time(BaseShape, DataShape):
    """An array of at least size T (the number of simulation days)."""
    base_dimensions = 1

    def matches_base(self, dim: SimDimension, value: NDArray) -> bool:
        return value.shape[0] >= dim.days

    def matches(self, dim: SimDimension, value: Any) -> bool:
        return isinstance(value, np.ndarray) \
            and len(value.shape) == self.base_dimensions \
            and self.matches_base(dim, value)

    def __getitem__(self, index: int) -> Arbitrary:
        return Arbitrary(index, self)

    def __str__(self):
        return "T"


@dataclass(frozen=True)
class Node(BaseShape, DataShape):
    """An array of size N (the number of simulation nodes)."""
    base_dimensions = 1

    def matches_base(self, dim: SimDimension, value: NDArray) -> bool:
        return value.shape[0] == dim.nodes

    def matches(self, dim: SimDimension, value: Any) -> bool:
        return isinstance(value, np.ndarray) \
            and len(value.shape) == self.base_dimensions \
            and self.matches_base(dim, value)

    def __getitem__(self, index: int) -> Arbitrary:
        return Arbitrary(index, self)

    def __str__(self):
        return "N"


@dataclass(frozen=True)
class TimeAndNode(BaseShape, DataShape):
    """An array of size at-least-T by exactly-N."""
    base_dimensions = 2

    def matches_base(self, dim: SimDimension, value: NDArray) -> bool:
        return value.shape[0] >= dim.days and value.shape[1] == dim.nodes

    def matches(self, dim: SimDimension, value: Any) -> bool:
        return isinstance(value, np.ndarray) \
            and len(value.shape) == self.base_dimensions \
            and self.matches_base(dim, value)

    def __getitem__(self, index: int) -> Arbitrary:
        return Arbitrary(index, self)

    def __str__(self):
        return "TxN"

    # = Time | Node | TimeAndNode


@dataclass(frozen=True)
class Arbitrary(DataShape):
    """
    A shape whose final dimension is an arbitrary index and may have
    a base shape which is relative to the simulation context.

    For example: "TxNx3" would describe a three-dimensional array
    whose first dimension is time, second dimension is the number of
    nodes, and third dimension contains at least four values so that
    we can select the fourth (zero-indexed).
    """

    index: int
    base: BaseShape | None = field(default=None)

    def __post_init__(self):
        if self.index < 0:
            raise ValueError("Arbitrary shape cannot specify negative indices.")

    def matches(self, dim: SimDimension, value: Any) -> bool:
        if self.base is not None:
            return isinstance(value, np.ndarray) \
                and len(value.shape) == self.base.base_dimensions + 1 \
                and self.base.matches_base(dim, value) \
                and value.shape[-1] > self.index
        else:
            return isinstance(value, np.ndarray) \
                and len(value.shape) == 1 \
                and value.shape[-1] > self.index

    def __str__(self):
        if self.base is not None:
            return f"{self.base}x{self.index}"
        return str(self.index)


class ArbitraryFactory:
    """Syntactic sugar to create an Arbitrary instance with base=None using index syntax (square brackets)."""

    def __getitem__(self, index: int) -> Arbitrary:
        return Arbitrary(index)


@dataclass(frozen=True)
class Shapes:
    """Static instances for all available shapes."""

    S = Scalar()
    T = Time()
    N = Node()
    TxN = TimeAndNode()
    A = ArbitraryFactory()

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
        case "TxN":
            return Shapes.TxN
        # blank or trailing 'x' means there's an index at the end -> Arbitrary
        case "":
            return Arbitrary(int(index))
        case "Tx":
            return Arbitrary(int(index), Shapes.T)
        case "Nx":
            return Arbitrary(int(index), Shapes.N)
        case "TxNx":
            return Arbitrary(int(index), Shapes.TxN)
        case _:
            raise ValueError(f"'{shape}' is not a valid shape specification.")
