"""
Classes for representing simulation results.
"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from epymorph.data_shape import SimDimensions
from epymorph.data_type import SimDType


@dataclass(frozen=True)
class Output:
    """
    The output of a simulation run, including prevalence for all populations and all IPM compartments
    and incidence for all populations and all IPM events.
    """

    dim: SimDimensions
    geo_labels: Sequence[str]
    compartment_labels: Sequence[str]
    event_labels: Sequence[str]

    initial: NDArray[SimDType]
    """
    Initial prevalence data by population and compartment.
    Array of shape (N, C) where N is the number of populations, and C is the number of compartments
    """

    prevalence: NDArray[SimDType] = field(init=False)
    """
    Prevalence data by timestep, population, and compartment.
    Array of shape (T,N,C) where T is the number of ticks in the simulation,
    N is the number of populations, and C is the number of compartments.
    """

    incidence: NDArray[SimDType] = field(init=False)
    """
    Incidence data by timestep, population, and event.
    Array of shape (T,N,E) where T is the number of ticks in the simulation,
    N is the number of populations, and E is the number of events.
    """

    def __post_init__(self):
        T, N, C, E = self.dim.TNCE
        object.__setattr__(self, "prevalence", np.zeros((T, N, C), dtype=SimDType))
        object.__setattr__(self, "incidence", np.zeros((T, N, E), dtype=SimDType))

    @property
    def incidence_per_day(self) -> NDArray[SimDType]:
        """
        Returns this output's `incidence` from a per-tick value to a per-day value.
        Returns a shape (D,N,E) array, where D is the number of simulation days.
        """
        T, N, _, E = self.dim.TNCE
        taus = self.dim.tau_steps
        return np.sum(
            self.incidence.reshape((T // taus, taus, N, E)), axis=1, dtype=SimDType
        )

    @property
    def ticks_in_days(self) -> NDArray[np.float64]:
        """
        Create a series with as many values as there are simulation ticks,
        but in the scale of fractional days. That is: the cumulative sum of
        the simulation's tau step lengths across the simulation duration.
        Returns a shape (T,) array, where T is the number of simulation ticks.
        """
        return np.cumsum(
            np.tile(self.dim.tau_step_lengths, self.dim.days), dtype=np.float64
        )
