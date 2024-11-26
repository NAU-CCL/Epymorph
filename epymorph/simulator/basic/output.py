"""
Classes for representing simulation results.
"""

from dataclasses import dataclass, field
from functools import cached_property
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import deprecated, override

from epymorph.compartment_model import BaseCompartmentModel
from epymorph.data_shape import SimDimensions
from epymorph.data_type import SimDType
from epymorph.geography.scope import GeoScope
from epymorph.time import TimeFrame
from epymorph.tools.out_map import MapRendererMixin
from epymorph.tools.out_plot import PlotRendererMixin
from epymorph.tools.out_table import TableRendererMixin


@dataclass(frozen=True)
class Output(TableRendererMixin, PlotRendererMixin, MapRendererMixin):
    """
    The output of a simulation run, including compartment data for all populations and
    all IPM compartments and event data for all populations and all IPM events.
    """

    dim: SimDimensions
    scope: GeoScope
    geo_labels: Sequence[str]
    time_frame: TimeFrame
    ipm: BaseCompartmentModel

    initial: NDArray[SimDType]
    """
    Initial compartments by population and compartment.
    Array of shape (N,C) where N is the number of populations,
    and C is the number of compartments
    """

    compartments: NDArray[SimDType] = field(init=False)
    """
    Compartment data by timestep, population, and compartment.
    Array of shape (S,N,C) where S is the number of ticks in the simulation,
    N is the number of populations, and C is the number of compartments.
    """

    events: NDArray[SimDType] = field(init=False)
    """
    Event data by timestep, population, and event.
    Array of shape (S,N,E) where S is the number of ticks in the simulation,
    N is the number of populations, and E is the number of events.
    """

    def __post_init__(self):
        S = self.dim.ticks
        N = self.dim.nodes
        C = self.dim.compartments
        E = self.dim.events
        object.__setattr__(self, "compartments", np.zeros((S, N, C), dtype=SimDType))
        object.__setattr__(self, "events", np.zeros((S, N, E), dtype=SimDType))

    @cached_property
    def compartment_labels(self) -> Sequence[str]:
        return [c.name.full for c in self.ipm.compartments]

    @cached_property
    def event_labels(self) -> Sequence[str]:
        return [e.name.full for e in self.ipm.events]

    @property
    def events_per_day(self) -> NDArray[SimDType]:
        """
        Returns this output's `incidence` from a per-tick value to a per-day value.
        Returns a shape (T,N,E) array, where T is the number of simulation days.
        """
        S = self.dim.ticks
        N = self.dim.nodes
        E = self.dim.events
        taus = self.dim.tau_steps
        return np.sum(
            self.events.reshape((S // taus, taus, N, E)), axis=1, dtype=SimDType
        )

    @property
    def ticks_in_days(self) -> NDArray[np.float64]:
        """
        Create a series with as many values as there are simulation ticks,
        but in the scale of fractional days. That is: the cumulative sum of
        the simulation's tau step lengths across the simulation duration.
        Returns a shape (S,) array, where S is the number of simulation ticks.
        """
        return np.cumsum(
            np.tile(self.dim.tau_step_lengths, self.dim.days), dtype=np.float64
        )

    @property
    @override
    def dataframe(self) -> pd.DataFrame:
        # NOTE: reshape ordering is critical, because the index column creation
        # must assume ordering happens in a specific way.
        # C ordering causes the later index (node) to change fastest and the
        # earlier index (time) to change slowest. (The quantity index is left as-is.)
        # Thus "tick" goes 0,0,0,...,1,1,1,... (similar situation with "date")
        # and "node" goes 1,2,3,...,1,2,3,...
        data_np = np.concatenate(
            (self.compartments, self.events),
            axis=2,
        ).reshape(
            (-1, self.dim.compartments + self.dim.events),
            order="C",
        )

        # Here I'm concatting two DFs sideways so that the index columns come first.
        # Could use insert, but this is nicer.
        return pd.concat(
            (
                # A dataframe for the various indices
                pd.DataFrame(
                    {
                        "tick": np.arange(self.dim.ticks).repeat(self.dim.nodes),
                        "date": self.time_frame.to_numpy().repeat(
                            self.dim.nodes * self.dim.tau_steps
                        ),
                        "node": np.tile(self.scope.node_ids, self.dim.ticks),
                    }
                ),
                # A dataframe for the data columns
                pd.DataFrame(
                    data=data_np,
                    columns=[*self.compartment_labels, *self.event_labels],
                ),
            ),
            axis=1,  # stick them together side-by-side
        )

    @property
    @deprecated("Use `compartments`", category=None)
    def prevalence(self) -> NDArray[SimDType]:
        """Deprecated alias for compartments."""
        return self.compartments

    @property
    @deprecated("Use `events`", category=None)
    def incidence(self) -> NDArray[SimDType]:
        return self.events

    @property
    @deprecated("Use `events_per_day`", category=None)
    def incidence_per_day(self) -> NDArray[SimDType]:
        return self.events_per_day
