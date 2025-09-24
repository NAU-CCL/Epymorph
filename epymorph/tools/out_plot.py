"""Tools for rendering graphs from epymorph simulation output data."""

from datetime import timedelta
from itertools import cycle
from pathlib import Path
from typing import Callable, Literal, OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.dates import AutoDateLocator, DateFormatter
from matplotlib.lines import Line2D
from matplotlib.ticker import EngFormatter

from epymorph.compartment_model import QuantityAggregation, QuantitySelection
from epymorph.geography.scope import GeoAggregation, GeoSelection
from epymorph.time import TimeAggregation, TimeSelection
from epymorph.tools.data import Output, munge
from epymorph.util import identity

OrderingOption = Literal["location", "quantity"]
"""Options for plot line ordering."""

TimeFormatOption = Literal["auto", "date", "day"]
"""Options for the plot's time-axis format."""

LegendOption = Literal["on", "off", "outside", "auto"]
"""Options for the plot's legend display."""


class PlotRenderer:
    """
    Provides methods for rendering an output in plot form.

    Most commonly, you will use `PlotRenderer` starting from a simulation output object
    that supports it:

    ```python
    out = BasicSimulation(rume).run()
    out.plot.line(...)

    Parameters
    ----------
    output :
        The output the renderer will use.
    ```
    """

    output: Output
    """The output the renderer will use."""

    def __init__(self, output: Output):
        self.output = output

    def _sorting(
        self,
        quantities_disambiguated: OrderedDict[str, str],
        ordering: OrderingOption,
    ) -> Callable:
        # Return a sort key function that implements the requested ordering option.
        quantity_order = {s: i for i, s in enumerate(quantities_disambiguated.keys())}

        if ordering == "quantity":

            def qty_sort(group):
                (location, quantity), _ = group
                return (quantity_order[quantity], location)

            return qty_sort

        else:

            def loc_sort(group):
                (location, quantity), _ = group
                return (location, quantity_order[quantity])

            return loc_sort

    def _time_format(
        self,
        time: TimeSelection | TimeAggregation,  # what format did we produce?
        requested_time_format: TimeFormatOption,  # what format do we want?
    ) -> tuple[
        Literal["tick", "date", "day", "other"],  # what format can we actually do?
        Callable[[pd.Series], pd.Series],  # converts time axis into format
    ]:
        """
        Figures out time-axis formatting for plots. This is basically a
        best-effort negotiation depending on the time format we have after
        applying time selection/aggregation (if any) and the time format
        requested.
        """

        tau_step_lengths = self.output.rume.tau_step_lengths
        num_tau_steps = self.output.rume.num_tau_steps
        start_date = self.output.rume.time_frame.start_date
        S = self.output.rume.num_ticks
        T = self.output.rume.time_frame.days
        match (time.group_format, requested_time_format):
            case ("tick", "auto" | "day"):
                # Convert ticks to simulation-day scale:
                # e.g.: [0.333, 1.0, 1.333, ...]
                # NOTE: each tick is represented as the end of its timespan
                def ticks_to_days(time_groups: pd.Series) -> pd.Series:
                    deltas = np.array(tau_step_lengths).cumsum()
                    days = (
                        np.arange(T).repeat(num_tau_steps)  #
                        + np.tile(deltas, T)
                    )
                    ticks = np.arange(S)
                    time_map = dict(zip(ticks, days))
                    return time_groups.apply(lambda x: time_map[x])

                return "day", ticks_to_days

            case ("tick", "date"):
                # Convert ticks to date scale:
                # e.g.: [2020-01-01T08:00, 2020-01-02T00:00, 2020-01-02T08:00, ...]
                # NOTE: each tick is represented as the end of its timespan
                def ticks_to_dates(time_groups: pd.Series) -> pd.Series:
                    deltas = np.array(
                        [timedelta(days=x) for x in tau_step_lengths],
                        dtype=np.timedelta64,
                    ).cumsum()
                    dates = (
                        pd.date_range(start=start_date, periods=T).repeat(
                            num_tau_steps
                        )  #
                        + np.tile(deltas, T)  #
                    )
                    ticks = np.arange(S)
                    time_map = dict(zip(ticks, dates))
                    return time_groups.apply(lambda x: time_map[x])

                return "date", ticks_to_dates

            case ("date", "day"):
                # Convert dates to simulation-day scale:
                # e.g.: [0, 1, 2, 3, 4, ...]
                # Note: this can produce "negative" days;
                # e.g., if you group by week but the first day of the week is Monday
                # and you start the sim on a Tuesday.
                def dates_to_days(time_groups: pd.Series) -> pd.Series:
                    start = pd.Timestamp(start_date)
                    return time_groups.apply(lambda x: (x - start).days)

                return "day", dates_to_days

            case (actual, _):
                # Any other combo doesn't need to be or can't be mapped.
                return actual, identity

    def line(
        self,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantitySelection | QuantityAggregation,
        *,
        label_format: str = "{n}: {q}",
        legend: LegendOption = "auto",
        line_kwargs: list[dict] | None = None,
        ordering: OrderingOption = "location",
        time_format: TimeFormatOption = "auto",
        title: str | None = None,
        to_file: str | Path | None = None,
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> None:
        """
        Render a line plot using matplotlib showing the given selections.

        The plot will either be immediately rendered by this function by calling
        `plt.show()` or, if you specify the `to_file` argument, saved to a file.
        This is intended as a quick plotting method to cover most casual use-cases.
        If you want more control over how the plot is drawn, see method `line_plt()`.

        Parameters
        ----------
        geo :
            The geographic selection to make on the output data.
        time :
            The time selection to make on the output data.
        quantity :
            The quantity selection to make on the output data.
        label_format :
            A format for the items displayed in the legend;
            the string will be used in a call to `format()`
            with the replacement variables `{n}` for the name of the geo node
            and `{q}` for the name of the quantity.
        legend :
            Whether and how to draw the plot legend.

            - "auto" will draw the legend unless it would be too large
            - "on" forces the legend to be drawn
            - "off" forces the legend to not be drawn
            - "outside" forces the legend to be drawn next to the plot area
            (instead of inside it)
        line_kwargs :
            A list of keyword arguments to be passed to the matplotlib function
            that draws each line. If the list contains less items than there are lines,
            we will cycle through the list as many times as needed. Lines are drawn
            in the order defined by the `ordering` parameter.
            See matplotlib documentation for the supported options.
        ordering :
            Controls the order in which lines will be drawn;
            both location and quantity are used to sort the resulting rows,
            this just decides which gets priority.
        time_format :
            Controls the formatting of the time axis (the horizontal axis);
            "auto" will use the format defined by the grouping of the `time` parameter,
            "date" attempts to display calendar dates,
            "day" attempts to display days numerically indexed from the start of the
            simulation with the first day being 0.
            If the system cannot convert to the requested time format, this argument
            may be ignored.
        title :
            A title to draw on the plot.
        to_file :
            Specify a path to save the plot to a file instead of calling `plt.show()`.
        transform :
            Allows you to specify an arbitrary transform function for the source
            dataframe before we plot it, e.g., to rescale the values.
            The function will be called once per geo/quantity group -- once per line,
            essentially -- with a dataframe that contains just the data for that group.
            The dataframe given as the argument is the result of applying
            all selections and the projection if specified.
            You should return a dataframe with the same format, where the
            values of the data column have been modified for your purposes.

            Dataframe columns:

            - "time": the time series column
            - "geo": the node ID (same value per group)
            - "quantity": the label of the quantity (same value per group)
            - "value": the data column
        """
        try:
            _, ax = plt.subplots(layout="constrained")

            lines = self.line_plt(
                ax,
                geo,
                time,
                quantity,
                line_kwargs=line_kwargs,
                label_format=label_format,
                ordering=ordering,
                time_format=time_format,
                transform=transform,
            )

            # Y-axis
            plt.ylabel("count")
            ax.yaxis.set_major_formatter(EngFormatter(sep=""))

            # X-axis
            _time_format, _ = self._time_format(time, time_format)
            if _time_format == "date":
                plt.xlabel("date")
                ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
                ax.xaxis.set_major_locator(
                    AutoDateLocator(minticks=6, maxticks=12, interval_multiples=True)
                )
                plt.xticks(rotation=45)
            elif _time_format == "day":
                plt.xlabel("day")
            elif _time_format == "tick":
                plt.xlabel("tick")
            else:
                plt.xticks(rotation=45)
                plt.xlabel("time")

            # Legend
            if legend == "auto":
                # auto: show a legend if there are at most 12 lines.
                legend = "on" if len(lines) <= 12 else "off"

            if legend == "on":
                plt.legend()
            elif legend == "outside":
                plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

            if title is not None:
                plt.title(title)

            if to_file is None:
                plt.show()
            else:
                path = Path(to_file)
                plt.savefig(path)
        except:
            plt.close()
            raise

    def line_plt(
        self,
        ax: Axes,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantitySelection | QuantityAggregation,
        *,
        label_format: str = "{n}: {q}",
        line_kwargs: list[dict] | None = None,
        ordering: OrderingOption = "location",
        time_format: TimeFormatOption = "auto",
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> list[Line2D]:
        """
        Draws lines onto the given matplotlib `Axes` to show the given selections.
        This is a variant of the method `line()` that gives you more control over
        the rendering of a plot by letting you do most of the work with
        matplotlib's API.

        Parameters
        ----------
        ax :
            The plot axes on which to draw lines.
        geo :
            The geographic selection to make on the output data.
        time :
            The time selection to make on the output data.
        quantity :
            The quantity selection to make on the output data.
        label_format :
            A format for the items displayed in the legend;
            the string will be used in a call to `format()`
            with the replacement variables `{n}` for the name of the geo node
            and `{q}` for the name of the quantity.
        line_kwargs :
            A list of keyword arguments to be passed to the matplotlib function
            that draws each line. If the list contains less items than there are lines,
            we will cycle through the list as many times as needed. Lines are drawn
            in the order defined by the `ordering` parameter.
            See matplotlib documentation for the supported options.
        ordering :
            Controls the order in which lines will be drawn;
            both location and quantity are used to sort the resulting rows,
            this just decides which gets priority.
        time_format :
            Controls the formatting of the time axis (the horizontal axis);
            "auto" will use the format defined by the grouping of the `time` parameter,
            "date" attempts to display calendar dates,
            "day" attempts to display days numerically indexed from the start of the
            simulation with the first day being 0.
            If the system cannot convert to the requested time format, this argument
            may be ignored.
        transform :
            Allows you to specify an arbitrary transform function for the source
            dataframe before we plot it, e.g., to rescale the values.
            The function will be called once per geo/quantity group -- one per line,
            essentially -- with a dataframe that contains just the data for that group.
            The dataframe given as the argument is the result of applying
            all selections and the projection if specified.
            You should return a dataframe with the same format, where the
            values of the data column have been modified for your purposes.

            Dataframe columns:

            - "time": the time series column
            - "geo": the node ID (same value per group)
            - "quantity": the label of the quantity (same value per group)
            - "value": the data column

        Returns
        -------
        :
            The `Line2D` object for each line drawn; you can use this to have finer
            control over the presentation of the lines.
        """
        if line_kwargs is None or len(line_kwargs) == 0:
            line_kwargs = [{}]
        if transform is None:
            transform = identity

        data_df = munge(self.output, geo, time, quantity)

        # Map time labels:
        _, map_time_axis = self._time_format(time, time_format)
        data_df["time"] = map_time_axis(data_df["time"])

        # Map geo labels:
        result_scope = geo.to_scope()
        if (labels := result_scope.labels_option) is not None:
            geo_map = dict(zip(result_scope.node_ids, labels))
            data_df["geo"] = data_df["geo"].apply(lambda x: geo_map[x])

        # Before melting, disambiguate any quantities with the same name.
        q_mapping = quantity.disambiguate_groups()

        groups_df = (
            data_df.set_axis(["time", "geo", *q_mapping.keys()], axis=1)
            .melt(id_vars=["time", "geo"], var_name="quantity")
            .groupby(["geo", "quantity"], sort=False)
        )

        sort_key = self._sorting(q_mapping, ordering)

        lines = list[Line2D]()
        for (group, data), kwargs in zip(
            sorted(groups_df, key=sort_key),
            cycle(line_kwargs),
        ):
            n_label, q_label_dis = group
            q_label = q_mapping[q_label_dis]
            label = label_format.format(n=n_label, q=q_label)
            curr_kwargs = {"label": label, **kwargs}
            data = transform(data.assign(quantity=q_label))
            ls = ax.plot(data["time"], data["value"], **curr_kwargs)
            lines.extend(ls)
        return lines


class PlotRendererMixin(Output):
    """Mixin class that adds a convenient method for rendering plots from an output."""

    @property
    def plot(self) -> PlotRenderer:
        """Render a plot from this output."""
        return PlotRenderer(self)
