"""
The World represents the simulation state at a given point in time.
World implementations keep track of how many locations are being simulated,
and the compartmental breakdown of individuals in each of those locations.
They also keep track of groups of individuals which have moved between locations
and will eventually be returning or moving to another location.
"""
from abc import ABC, abstractmethod

from numpy.typing import NDArray

from epymorph.simulation import SimDType, Tick


class World(ABC):
    """
    An abstract world model.
    """

    @abstractmethod
    def get_cohort_array(self, location_idx: int) -> NDArray[SimDType]:
        """
        Retrieve an (X,C) array containing all cohorts at a single location,
        where X is the number of cohorts.
        """

    @abstractmethod
    def get_local_array(self) -> NDArray[SimDType]:
        """
        Get the local populations of each node as an (N,C) array.
        This is the individuals which are theoretically eligible for movement.
        """

    @abstractmethod
    def apply_cohort_delta(self, location_idx: int, delta: NDArray[SimDType]) -> None:
        """
        Apply the disease delta for all cohorts at the given location.
        `delta` is an (X,C) array where X is the number of cohorts.
        """

    @abstractmethod
    def apply_travel(self, travelers: NDArray[SimDType], return_tick: int) -> None:
        """
        Given an (N,N,C) array determining who should travel -- from-source-to-destination-by-compartment --
        modify the world state as a result of that movement.
        """

    @abstractmethod
    def apply_return(self, tick: Tick) -> int:
        """
        Modify the world state as a result of returning all movers that are ready to be returned home.
        """
