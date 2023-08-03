from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Iterable, TypeVar

from epymorph.context import Compartments

"""
TODO: this text needs to be adapted...
ObjWorld tracks the state of a simulation's populations and subpopulations at a particular timeslice.
Each node in the Geo Model is represented as a Location; each Location has a list of Cohorts
that are considered to be well-mixed at that location; and each Cohort is divided into
Compartments as defined by the Intra-Population Model.
"""


class Location(ABC):

    @abstractmethod
    def get_index(self) -> int:
        pass

    @abstractmethod
    def get_compartments(self) -> Compartments:
        """
        Retrieve compartment totals for this location.
        This is a 1D array containing the compartment-sum of every cohort here.
        """
        pass

    @abstractmethod
    def get_cohorts(self) -> Compartments:
        """
        Retrieve all individual cohorts at this location.
        This is as 2D array where each row is a cohort and columns
        contain the compartment-counts for that cohort.
        """
        pass

    @abstractmethod
    def update_cohorts(self, deltas: Compartments) -> None:
        """
        Update the compartment counts of all cohorts at this location.
        `deltas` should be a 2D array containing one row for each cohort
        (in the order they were given by the `get_cohorts()` method)
        with a delta value for each compartment (by column).
        """
        pass


class World(ABC):

    @abstractmethod
    def get_locations(self) -> Iterable[Location]:
        pass

    @abstractmethod
    def get_locals(self) -> Compartments:
        """Returns a 2D array by location (axis 0) of the locals in each compartment (axis 1)."""
        pass

    @abstractmethod
    def get_travelers(self) -> Compartments:
        """Returns a 2D array by location (axis 0) of the non-locals in each compartment (axis 1)."""
        pass

    @abstractmethod
    def get_travelers_by_home(self) -> Compartments:
        """Returns a 2D array by home-location (axis 0) of all travelers in each compartment (axis 1)."""
        pass
