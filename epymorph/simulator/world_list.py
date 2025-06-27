"""
A `World` implementation -- `ListWorld` -- which tracks the simulation state's
movement cohorts in a Python list-of-lists.
"""

from operator import attrgetter
from typing import Any, ClassVar, Iterable, Literal, Self, Sequence, overload

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.data_type import SimDType
from epymorph.simulation import Tick
from epymorph.simulator.world import World


class Cohort:
    """
    Represents a group of individuals, divided into IPM compartments as appropriate for
    the simulation. These individuals share the same "home location" and a time at which
    they should return there.

    These are somewhat abstract concepts, however; a completely nomadic group doesn't
    really have a home location, merely the next location in a chain of movements.

    Parameters
    ----------
    compartments :
        The group membership by compartment. A (C,)-shaped array.
    return_location :
        The location to which this cohort will return, or if they are home, the location
        where they currently are.
    return_tick :
        The simulation tick on which they will return. -1 implies never.
    """

    SORT_KEY: ClassVar = attrgetter("return_tick", "return_location")
    """The natural sort order of a Cohort."""

    compartments: NDArray[SimDType]
    """The group membership by compartment. A (C,)-shaped array."""
    return_location: int
    """
    The location to which this cohort will return, or if they are home, the location
    where they currently are.
    """
    return_tick: int
    """The simulation tick on which they will return. -1 implies never."""

    # Note: when a population is "home",
    # its `return_location` is the same as their home/current location,
    # and its `return_tick` is set to -1 (the "Never" placeholder value).

    def __init__(
        self,
        compartments: NDArray[SimDType],
        return_location: int,
        return_tick: int,
    ):
        self.compartments = compartments
        self.return_location = return_location
        self.return_tick = return_tick

    def can_merge_with(self, other: Self) -> bool:
        """
        Check if two cohorts can be merged.

        Cohorts at the same location with the same return location and tick can act as
        a single group.

        Parameters
        ----------
        other :
            The other cohort to consider.

        Returns
        -------
        :
            True if these cohorts can be merged.
        """
        return (
            self.return_tick == other.return_tick
            and self.return_location == other.return_location
        )

    def merge_from(self, from_cohort: Self) -> None:
        """
        Merge another cohort into this one, modifying in-place.

        Parameters
        ----------
        from_cohort :
            The other cohort to be "absorbed" into this one.
        """
        self.compartments += from_cohort.compartments

    def __eq__(self, other) -> bool:
        if not isinstance(other, Cohort):
            return False
        return (
            np.array_equal(self.compartments, other.compartments)
            and self.return_location == other.return_location
            and self.return_tick == other.return_tick
        )

    def __repr__(self) -> str:
        return (
            f"Cohort({str(self.compartments)}, "
            f"rloc={self.return_location}, "
            f"rtic={self.return_tick})"
        )


class ListWorld(World):
    """
    A world model which which tracks the simulation state's movement cohorts in a
    Python list-of-lists. That is, the world is a list of locations, and each location
    is a list of cohorts.

    Parameters
    ----------
    locations :
        The initial state of the world. It is assumed there is at least one location
        and at least one cohort at each.
    """

    HOME_TICK: ClassVar = -1
    """The value of a population's `return_tick` when the population is home."""

    nodes: int
    """The number of simulation nodes."""
    compartments: int
    """The number of disease compartments."""
    locations: list[list[Cohort]]
    """All locations and all cohorts at each location."""

    @classmethod
    def from_initials(cls, initial_compartments: NDArray[SimDType]) -> Self:
        """
        Create a world with the given initial state given as a numpy array.

        Everyone is assumed to be in their home location, there are no travelers.

        Parameters
        ----------
        initial_compartments :
            An (N,C)-shaped array of the compartment population at each node.

        Returns
        -------
        :
            The new world instance.
        """
        locations = [
            [Cohort(cs, i, ListWorld.HOME_TICK)]
            for (i, cs) in enumerate(initial_compartments.copy())
        ]
        return cls(locations)

    def __init__(self, locations: list[list[Cohort]]):
        self.nodes = len(locations)
        self.compartments = locations[0][0].compartments.shape[0]
        self.locations = locations

    def normalize(self) -> None:
        """
        Sorts all cohorts within each location and combines mergeable cohorts
        (in place). The world should be normalized after any modification.
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

    @override
    def get_cohorts(self, location_idx: int) -> Sequence[Cohort]:
        return self.locations[location_idx]

    @override
    def get_cohort_array(self, location_idx: int) -> NDArray[SimDType]:
        return np.array(
            [c.compartments for c in self.locations[location_idx]], dtype=SimDType
        )

    @override
    def get_local_array(self) -> NDArray[SimDType]:
        # assumes world was normalized after any modification
        return np.array(
            [cohorts[0].compartments for cohorts in self.locations], dtype=SimDType
        )

    def get_local_cohorts(self) -> Iterable[Cohort]:
        """
        Iterate over all locations returning just the local cohort from each.

        Returns
        -------
        :
            The local cohorts.
        """
        for loc in self.locations:
            yield loc[0]

    @override
    def apply_cohort_delta(self, location_idx: int, delta: NDArray[SimDType]) -> None:
        for i, cohort in enumerate(self.locations[location_idx]):
            cohort.compartments += delta[i]

    @override
    def apply_travel(self, travelers: NDArray[SimDType], return_tick: int) -> None:
        travelers_leaving = travelers.sum(axis=1, dtype=SimDType)
        travelers_nxn_sum = travelers.sum(axis=2, dtype=SimDType)

        # Modify world: remove locals, add movement cohorts.
        for src in range(self.nodes):
            self.locations[src][0].compartments -= travelers_leaving[src]
            for dst in range(self.nodes):
                if src != dst and travelers_nxn_sum[src, dst] > 0:
                    # Normally a Cohort's home is the source index.
                    # But if the return clause is "never"
                    # it needs to be the destination index.
                    # This makes the Cohort consider its new location to be home
                    # so it will merge with the local population.
                    home = dst if return_tick == -1 else src
                    p = Cohort(travelers[src, dst, :], home, return_tick)
                    self.locations[dst].append(p)

        self.normalize()

    @overload
    def apply_return(self, tick: Tick, *, return_stats: Literal[False]) -> None: ...

    @overload
    def apply_return(
        self, tick: Tick, *, return_stats: Literal[True]
    ) -> NDArray[SimDType]: ...

    @override
    def apply_return(
        self, tick: Tick, *, return_stats: bool
    ) -> NDArray[SimDType] | None:
        movers: Any = None
        if return_stats:
            size = (self.nodes, self.nodes, self.compartments)
            movers = np.zeros(size, dtype=SimDType)

        next_locations = [[c] for c in self.get_local_cohorts()]
        for i, location in enumerate(self.locations):
            for cohort in location:
                if cohort.return_tick == ListWorld.HOME_TICK:
                    # locals are already where they need to be
                    continue
                elif cohort.return_tick == tick.sim_index:
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
