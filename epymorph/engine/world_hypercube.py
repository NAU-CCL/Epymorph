"""World implementation: HypercubeWorld."""
from typing import Literal, Self, overload

import numpy as np
import psutil
from numpy.typing import NDArray
from typing_extensions import deprecated

from epymorph.engine.world import World
from epymorph.error import InitException
from epymorph.simulation import SimDimensions, SimDType, Tick


def to_gib(n_bytes: int) -> float:
    """Convert bytes to GiB."""
    return n_bytes / (1024 * 1024 * 1024)


def _mem_check(dim: SimDimensions) -> None:
    """Will this simulation fit in memory?"""
    T, N, C, _ = dim.TNCE
    required = np.dtype(SimDType).itemsize * ((T * N * N * C) + (2 * N * C))
    available = psutil.virtual_memory().available

    if available < required:
        msg = f"""\
Insufficient memory: the simulation is too large (using HypercubeEngine).
T:{T}, N:{N}, C:{C} requires {to_gib(required):.1f} GiB;
available memory is {to_gib(available):.1f} GiB"""
        raise InitException(msg)


@deprecated("Very slow, at the moment. Do not use in production code.")
class HypercubeWorld(World):
    """
    A world model which tracks movers with a giant hypercube.
    """

    # Developer note:
    # The goal here is to avoid memory allocations (e.g., as in ListWorld's creation of Cohort objects
    # and management of lists). Testing in a previous implementation suggested this could
    # be more efficient for simulations with many nodes. However, in refactoring this implementation
    # turned out many times slower than ListWorld and it's not worth the effort of debugging at the moment.
    # This may be of interest in future, however, so we'll leave the code here.

    dim: SimDimensions
    """The simulation dimensions."""

    ledger: NDArray[SimDType]
    """
    All travelers, shape (T,N,N,C):
    - axis 0: when they return home,
    - axis 1: where they came from,
    - axis 2: where they're visiting,
    - axis 3: number of individuals by compartment.
    """

    time_offset: int
    """The start of the ledger's active chunk: based on what timestep we're on."""

    time_frontier: int
    """The end of the ledger's active chunk: what's the furthest-out group of travelers?"""

    _ident: NDArray[SimDType]
    """
    An (N,N,1) array where the diagonals are all 1 --
    useful for turning an (N,C) array into an (N,N,C) array to apply to the 'home' row.
    """

    @classmethod
    def from_initials(cls, dim: SimDimensions, initial_compartments: NDArray[SimDType]) -> Self:
        """
        Create a world with the given initial compartments:
        assumes everyone starts at home, no travelers initially.
        initial_compartments is an (N,C) array.
        """
        _mem_check(dim)
        home = initial_compartments.copy()
        return cls(dim, home)

    def __init__(self, dim: SimDimensions, home: NDArray[SimDType]):
        self.dim = dim
        T, N, C, _ = dim.TNCE
        self.ledger = np.zeros((T + 1, N, N, C), dtype=SimDType)
        self.time_offset = 0
        self.time_frontier = 1
        self._ident = np.identity(N, dtype=SimDType).reshape((N, N, 1))
        self.ledger[0, :, :, :] = home * self._ident

    def get_cohort_array(self, location_idx: int) -> NDArray[SimDType]:
        _, N, C, _ = self.dim.TNCE
        ti, tf = self.time_offset, self.time_frontier
        return self.ledger[ti:tf, :, location_idx, :].reshape((N * (tf - ti), C)).copy()

    def get_local_array(self) -> NDArray[SimDType]:
        return self.ledger[self.time_offset, :, :, :].sum(axis=1, dtype=SimDType)

    def apply_cohort_delta(self, location_idx: int, delta: NDArray[SimDType]) -> None:
        _, N, C, _ = self.dim.TNCE
        ti, tf = self.time_offset, self.time_frontier
        visitors_delta = delta.reshape((tf - ti, N, C))
        self.ledger[ti:tf, :, location_idx, :] += visitors_delta

    def apply_travel(self, travelers: NDArray[SimDType], return_tick: int) -> None:
        trav_by_source = travelers.sum(axis=1, dtype=SimDType) * self._ident
        self.ledger[self.time_offset, :, :, :] -= trav_by_source
        self.ledger[return_tick + 1, :, :, :] += travelers
        self.time_frontier = max(self.time_frontier, return_tick + 2)

    @overload
    def apply_return(self, tick: Tick, *, return_stats: Literal[False]) -> None:
        ...

    @overload
    def apply_return(self, tick: Tick, *, return_stats: Literal[True]) -> NDArray[SimDType]:
        ...

    def apply_return(self, tick: Tick, *, return_stats: bool) -> NDArray[SimDType] | None:
        # we have to transpose the movers "stats" result since they're being stored here as
        # (home, visiting) and our result needs to be
        # (moving from "visiting", moving to "home")
        movers = self.ledger[self.time_offset + 1, :, :, :].transpose((1, 0, 2)).copy()
        movers_by_home = self.ledger[self.time_offset + 1, :, :, :].sum(
            axis=1, dtype=SimDType) * self._ident
        self.ledger[self.time_offset + 1, :, :, :] = movers_by_home + \
            self.ledger[self.time_offset, :, :, :]
        self.time_offset += 1  # assumes there's only ever one return clause per tick
        return movers if return_stats else None
