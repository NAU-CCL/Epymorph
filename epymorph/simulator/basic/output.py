"""The simulation results for basic simulations."""

import dataclasses
from dataclasses import dataclass, field
from functools import cached_property
from typing import Generic, Literal, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.data_type import SimDType
from epymorph.geography.scope import GeoScope
from epymorph.rume import RUME
from epymorph.tools.out_map import MapRendererMixin
from epymorph.tools.out_plot import PlotRendererMixin
from epymorph.tools.out_table import TableRendererMixin

RUMEType_co = TypeVar("RUMEType_co", covariant=True, bound=RUME[GeoScope])
"""The type of a `RUME` used to produce an `Output`."""


@dataclass(frozen=True)
class Output(
    TableRendererMixin,
    PlotRendererMixin,
    MapRendererMixin,
    Generic[RUMEType_co],
):
    """
    The output of a `BasicSimulation` run, including time series for compartment data
    and transition events for all locations.

    Parameters
    ----------
    rume :
        The `RUME` used in the simulation that generated this output.
    initial :
        The simulation's initial compartments by location and compartment.
    visit_compartments :
        Compartment data collected in the node where individuals are visiting.
    visit_events :
        Event data collected in the node where individuals are visiting.
    home_compartments :
        Compartment data collected in the node where individuals reside.
    home_events :
        Event data collected in the node where individuals reside.

    See Also
    --------
    [epymorph.simulator.basic.basic_simulator.BasicSimulator][] which is the most common
    way to create an output.
    """

    rume: RUMEType_co  # type: ignore (pylance can't tell that rume is immutable)
    """The `RUME` used in the simulation that generated this output."""

    initial: NDArray[SimDType]
    """
    The simulation's initial compartments by location and compartment.
    Array of shape (N,C) where N is the number of locations,
    and C is the number of compartments
    """

    visit_compartments: NDArray[SimDType]
    """
    Compartment data collected in the node where individuals are visiting.
    See `compartments` for more information.
    """
    visit_events: NDArray[SimDType]
    """
    Event data collected in the node where individuals are visiting.
    See `events` for more information.
    """
    home_compartments: NDArray[SimDType]
    """
    Compartment data collected in the node where individuals reside.
    See `compartments` for more information.
    """
    home_events: NDArray[SimDType]
    """
    Event data collected in the node where individuals reside.
    See `events` for more information.
    """

    data_mode: Literal["visit", "home"] = field(default="visit")
    """
    Controls which data is returned by the `compartments` and `events` properties.
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
        """
        Return an `Output` object that contains the same set of data, but uses
        'visit' as the default data mode.
        """
        return self._with_data_mode("visit")

    @cached_property
    def data_by_home(self) -> "Output":
        """
        Return an `Output` object that contains the same set of data, but uses
        'home' as the default data mode.
        """
        return self._with_data_mode("home")

    @property
    def compartments(self) -> NDArray[SimDType]:
        """
        Compartment data by timestep, location, and compartment.
        Array of shape (S,N,C) where S is the number of ticks in the simulation,
        N is the number of locations, and C is the number of compartments.
        """
        if self.data_mode == "visit":
            return self.visit_compartments
        else:
            return self.home_compartments

    @property
    def events(self) -> NDArray[SimDType]:
        """
        Event data by timestep, location, and event.
        Array of shape (S,N,E) where S is the number of ticks in the simulation,
        N is the number of locations, and E is the number of events.
        """
        if self.data_mode == "visit":
            return self.visit_events
        else:
            return self.home_events

    @property
    def events_per_day(self) -> NDArray[SimDType]:
        """
        Returns this output's `incidence` from a per-tick value to a per-day value.
        Returns a shape (T,N,E) array, where T is the number of simulation days.
        """
        S = self.rume.num_ticks
        N = self.rume.scope.nodes
        E = self.rume.ipm.num_events
        taus = self.rume.num_tau_steps
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
            np.tile(self.rume.tau_step_lengths, self.rume.time_frame.days),
            dtype=np.float64,
        )

    @property
    @override
    def dataframe(self) -> pd.DataFrame:
        """Returns the output data as a data frame, using the current data mode."""
        # NOTE: reshape ordering is critical, because the index column creation
        # must assume ordering happens in a specific way.
        # C ordering causes the later index (node) to change fastest and the
        # earlier index (time) to change slowest. (The quantity index is left as-is.)
        # Thus "tick" goes 0,0,0,...,1,1,1,... (similar situation with "date")
        # and "node" goes 1,2,3,...,1,2,3,...
        C = self.rume.ipm.num_compartments
        E = self.rume.ipm.num_events
        N = self.rume.scope.nodes
        S = self.rume.num_ticks
        tau_steps = self.rume.num_tau_steps
        data_np = np.concatenate(
            (self.compartments, self.events),
            axis=2,
        ).reshape((-1, C + E), order="C")

        # Here I'm concatting two DFs sideways so that the index columns come first.
        # Could use insert, but this is nicer.
        return pd.concat(
            (
                # A dataframe for the various indices
                pd.DataFrame(
                    {
                        "tick": np.arange(S).repeat(N),
                        "date": self.rume.time_frame.to_numpy().repeat(N * tau_steps),
                        "node": np.tile(self.rume.scope.node_ids, S),
                    }
                ),
                # A dataframe for the data columns
                pd.DataFrame(
                    data=data_np,
                    columns=[
                        *(c.name.full for c in self.rume.ipm.compartments),
                        *(e.name.full for e in self.rume.ipm.events),
                    ],
                ),
            ),
            axis=1,  # stick them together side-by-side
        )
