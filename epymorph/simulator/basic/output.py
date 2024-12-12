"""
Classes for representing simulation results.
"""

import dataclasses
from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal, Sequence

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
    """The dimensions of the simulation that generated this output."""
    scope: GeoScope
    """The geo scope of the simulation that generated this output."""
    geo_labels: Sequence[str]
    """Labels for the geo nodes."""
    time_frame: TimeFrame
    """The time frame of the simulation that generated this output."""
    ipm: BaseCompartmentModel
    """The IPM used in the simulation that generated this output."""

    initial: NDArray[SimDType]
    """
    Initial compartments by population and compartment.
    Array of shape (N,C) where N is the number of populations,
    and C is the number of compartments
    """

    visit_compartments: NDArray[SimDType]
    """Compartment data collected in the node where individuals are visiting.
    See `compartments` for more information."""
    visit_events: NDArray[SimDType]
    """Event data collected in the node where individuals are visiting.
    See `events` for more information."""
    home_compartments: NDArray[SimDType]
    """Compartment data collected in the node where individuals reside.
    See `compartments` for more information."""
    home_events: NDArray[SimDType]
    """Event data collected in the node where individuals reside.
    See `events` for more information."""

    data_mode: Literal["visit", "home"] = field(default="visit")
    """Controls which data is returned by the `compartments` and `events` properties.
    Although you can access both data sets, it's helpful to have a default for things
    like our plotting and mapping tools. Defaults to "visit".

    See `data_by_visit` and `data_by_home`
    to obtain an Output object that uses a different data mode.
    """

    def _with_data_mode(self, data_mode: Literal["visit", "home"]) -> "Output":
        return (
            self
            if self.data_mode == data_mode
            else dataclasses.replace(self, data_mode=data_mode)
        )

    @cached_property
    def data_by_visit(self) -> "Output":
        """Returns an Output object that contains the same set of data, but uses
        'visit' as the default data mode."""
        return self._with_data_mode("visit")

    @cached_property
    def data_by_home(self) -> "Output":
        """Returns an Output object that contains the same set of data, but uses
        'home' as the default data mode."""
        return self._with_data_mode("home")

    @property
    def compartments(self) -> NDArray[SimDType]:
        """
        Compartment data by timestep, population, and compartment.
        Array of shape (S,N,C) where S is the number of ticks in the simulation,
        N is the number of populations, and C is the number of compartments.
        """
        if self.data_mode == "visit":
            return self.visit_compartments
        else:
            return self.home_compartments

    @property
    def events(self) -> NDArray[SimDType]:
        """
        Event data by timestep, population, and event.
        Array of shape (S,N,E) where S is the number of ticks in the simulation,
        N is the number of populations, and E is the number of events.
        """
        if self.data_mode == "visit":
            return self.visit_events
        else:
            return self.home_events

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
        """Returns the output data in DataFrame form, using the current data mode."""
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
