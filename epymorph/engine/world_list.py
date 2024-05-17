"""World implementation: ListWorld."""
from operator import attrgetter
from typing import Any, Iterable, Literal, Self, overload

import numpy as np
from numpy.typing import NDArray

from epymorph.data_type import SimDType
from epymorph.engine.world import World
from epymorph.simulation import Tick


class Cohort:
    """
    Represents a group of individuals, divided into IPM compartments as appropriate for the simulation.
    These individuals share the same "home location" and a time at which they should return there.

    These are somewhat abstract concepts, however; a completely nomadic group doesn't really have a home location,
    merely the next location in a chain of movements.
    """

    SORT_KEY = attrgetter('return_tick', 'return_location')
    """The natural sort order of a Cohort."""

    compartments: NDArray[SimDType]
    return_location: int
    return_tick: int

    # Note: when a population is "home",
    # its `return_location` is the same as their home/current location,
    # and its `return_tick` is set to -1 (the "Never" placeholder value).

    def __init__(self, compartments: NDArray[SimDType], return_location: int, return_tick: int):
        self.compartments = compartments
        self.return_location = return_location
        self.return_tick = return_tick

    def can_merge_with(self, other: Self) -> bool:
        """Returns true if two cohorts can be merged."""
        return self.return_tick == other.return_tick \
            and self.return_location == other.return_location

    def merge_from(self, from_cohort: Self) -> None:
        """Merges another cohort into this one, modifying in-place."""
        self.compartments += from_cohort.compartments

    def __eq__(self, other) -> bool:
        if not isinstance(other, Cohort):
            return False
        return np.array_equal(self.compartments, other.compartments) and \
            self.return_location == other.return_location and \
            self.return_tick == other.return_tick

    def __repr__(self) -> str:
        return f"Cohort({str(self.compartments)}, rloc={self.return_location}, rtic={self.return_tick})"


class ListWorld(World):
    """
    A world model which tracks state with lists: the world is a list of locations,
    and each location is a list of cohorts.
    """

    HOME_TICK = -1
    """The value of a population's `return_tick` when the population is home."""

    nodes: int
    compartments: int
    locations: list[list[Cohort]]

    @classmethod
    def from_initials(cls, initial_compartments: NDArray[SimDType]) -> Self:
        """
        Create a world with the given initial compartments:
        assumes everyone starts at home, no travelers initially.
        initial_compartments is an (N,C) array.
        """
        locations = [[Cohort(cs, i, ListWorld.HOME_TICK)]
                     for (i, cs) in enumerate(initial_compartments.copy())]
        return cls(locations)

    def __init__(self, locations: list[list[Cohort]]):
        self.nodes = len(locations)
        self.compartments = len(locations[0][0].compartments)
        self.locations = locations

    def normalize(self) -> None:
        """
        Sorts all cohorts within each location and combines mergeable cohorts (in place).
        The world should be normalized after any modification.
        """
        for cohorts in self.locations:
            cohorts.sort(key=Cohort.SORT_KEY)
            # Iterate over all sequential pairs, starting from (0,1)
            j = 1
            while j < len(cohorts):
                prev = cohorts[j - 1]
                curr = cohorts[j]
                if prev.can_merge_with(curr):
                    prev.merge_from(curr)
                    del cohorts[j]
                else:
                    j += 1

    def get_cohort_array(self, location_idx: int) -> NDArray[SimDType]:
        return np.array([
            c.compartments for c in self.locations[location_idx]
        ], dtype=SimDType)

    def get_local_array(self) -> NDArray[SimDType]:
        # assumes world was normalized after any modification
        return np.array([
            cohorts[0].compartments for cohorts in self.locations
        ], dtype=SimDType)

    def get_local_cohorts(self) -> Iterable[Cohort]:
        """Iterate over all locations returning just the local cohort from each."""
        for loc in self.locations:
            yield loc[0]

    def apply_cohort_delta(self, location_idx: int, delta: NDArray[SimDType]) -> None:
        for i, cohort in enumerate(self.locations[location_idx]):
            cohort.compartments += delta[i]

    def apply_travel(self, travelers: NDArray[SimDType], return_tick: int) -> None:
        travelers_leaving = travelers.sum(axis=1, dtype=SimDType)
        travelers_nxn_sum = travelers.sum(axis=2, dtype=SimDType)

        # Modify world: remove locals, add movement cohorts.
        for src in range(self.nodes):
            self.locations[src][0].compartments -= travelers_leaving[src]
            for dst in range(self.nodes):
                if src != dst and travelers_nxn_sum[src, dst] > 0:
                    p = Cohort(travelers[src, dst, :], src, return_tick)
                    self.locations[dst].append(p)

        self.normalize()

    @overload
    def apply_return(self, tick: Tick, *, return_stats: Literal[False]) -> None:
        ...

    @overload
    def apply_return(self, tick: Tick, *, return_stats: Literal[True]) -> NDArray[SimDType]:
        ...

    def apply_return(self, tick: Tick, *, return_stats: bool) -> NDArray[SimDType] | None:
        movers: Any = None
        if return_stats:
            size = (self.nodes, self.nodes, self.compartments)
            movers = np.zeros(size, dtype=SimDType)

        next_locations = [[locals] for locals in self.get_local_cohorts()]
        for i, location in enumerate(self.locations):
            for cohort in location:
                if cohort.return_tick == ListWorld.HOME_TICK:
                    # locals are already where they need to be
                    continue
                elif cohort.return_tick == tick.index:
                    # cohort ready to go home, merge with locals
                    j = cohort.return_location
                    next_locations[j][0].compartments += cohort.compartments
                    if return_stats:
                        movers[i, j, :] = cohort.compartments
                else:
                    # cohort staying
                    next_locations[i].append(cohort)

        self.locations = next_locations
        self.normalize()
        return movers if return_stats else None
