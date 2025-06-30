"""
The `World` represents the simulation state at a given point in time.
`World` implementations keep track of how many locations are being simulated,
and the compartmental breakdown of individuals in each of those locations.
They also keep track of groups of individuals which have moved between locations
and will eventually be returning or moving to another location.
"""

from abc import ABC, abstractmethod
from typing import Literal, Protocol, Sequence, overload

from numpy.typing import NDArray

from epymorph.data_type import SimDType
from epymorph.simulation import Tick


class Cohort(Protocol):
    """
    A group of individuals who may be home or traveling.

    If they are traveling, they may have a location to which they'll return and
    knowledge of when to do so.
    """

    compartments: NDArray[SimDType]
    """The group membership by compartment. A (C,)-shaped array."""
    return_location: int
    """
    The location to which this cohort will return, or if they are home, the location
    where they currently are.
    """
    return_tick: int
    """The simulation tick on which they will return. -1 implies never."""


class World(ABC):
    """An abstract world model."""

    @abstractmethod
    def get_cohorts(self, location_idx: int) -> Sequence[Cohort]:
        """
        Iterate over the cohorts present in a single location.

        Parameters
        ----------
        location_idx :
            The index of the location.

        Returns
        -------
        :
            The cohorts present in the location.
        """

    @abstractmethod
    def get_cohort_array(self, location_idx: int) -> NDArray[SimDType]:
        """
        Retrieve the cohorts in a single location as a numpy array.

        Parameters
        ----------
        location_idx :
            The index of the location.

        Returns
        -------
        :
            The cohorts at the location. This is an (X,C)-shaped array where X is the
            number of cohorts, which can be arbitrarily long.
        """

    @abstractmethod
    def get_local_array(self) -> NDArray[SimDType]:
        """
        Get the local populations of each node.

        This is the individuals which are theoretically eligible for movement.

        Returns
        -------
        :
            The local populations as an (N,C)-shaped array.
        """

    @abstractmethod
    def apply_cohort_delta(self, location_idx: int, delta: NDArray[SimDType]) -> None:
        """
        Apply a transition delta for all cohorts at the given location, updating the
        populations in each compartment.

        Parameters
        ----------
        location_idx :
            The index of the location.
        delta :
            The transitions delta, an (X,C)-shape array where X is the number of cohorts
            and each value is the net positive or negative change to the compartment's
            population.
        """

    @abstractmethod
    def apply_travel(self, travelers: NDArray[SimDType], return_tick: int) -> None:
        """
        Apply travel flows to the entire world, updating the cohorts at each location.

        Parameters
        ----------
        travelers :
            An (N,N,C)-shaped array determining who should travel
            -- from-source-to-destination-by-compartment.
        return_tick :
            The tick on which any newly-moved cohort should return home.
        """

    @abstractmethod
    @overload
    def apply_return(self, tick: Tick, *, return_stats: Literal[False]) -> None: ...

    @abstractmethod
    @overload
    def apply_return(
        self, tick: Tick, *, return_stats: Literal[True]
    ) -> NDArray[SimDType]: ...

    @abstractmethod
    def apply_return(
        self, tick: Tick, *, return_stats: bool
    ) -> NDArray[SimDType] | None:
        """
        Apply return-home flows to the entire world, updating the cohorts at each
        location. This finds cohorts which should return home (by looking at their
        return tick) and does so.

        Parameters
        ----------
        tick :
            The current simulation tick.
        return_stats :
            True to collect movement statistics and provide them as the function's
            return value.

        Returns
        -------
        :
            If `return_stats` is True, an (N,N,C)-shaped array containing the
            number of individuals moved during this return phase. Otherwise returns
            `None`.
        """
