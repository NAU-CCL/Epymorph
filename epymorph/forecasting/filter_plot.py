from datetime import timedelta
from itertools import cycle
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.dates import AutoDateLocator, DateFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from scipy.stats import gaussian_kde

from epymorph.compartment_model import QuantityStrategy
from epymorph.forecasting.munge_realizations import (
    ParameterStrategy,
    RealizationAggregation,
    RealizationSelection,
)
from epymorph.forecasting.pipeline import PipelineOutput, munge_pipeline_output
from epymorph.geography.scope import GeoAggregation, GeoSelection
from epymorph.time import TimeAggregation, TimeSelection
from epymorph.tools.out_plot import LegendOption, OrderingOption, TimeFormatOption
from epymorph.util import identity


class PlotRendererPipeline:
    """
    Provides methods for rendering an output in plot form.

    Most commonly, you will use `PlotRendererPipeline` starting from a pipeline output
    object that supports it:

    ```python
    out = EnsembleKalmanFilterSimulator(...).run(...)
    out.plot.line(...)
    ```

    Parameters
    ----------
    output :
        The PipelineOutput the renderer will use.
    """

    output: PipelineOutput
    """The output the renderer will use."""

    def __init__(self, output: PipelineOutput):
        self.output = output

    def _compute_quantile_range(self, credible_interval: float) -> list[float]:
        """Computes the quantiles corresponding the a given credible interval."""
        exterior = (100.0 - credible_interval) / 2
        exterior = round(exterior, 1)
        return [0.0 + exterior, 100.0 - exterior]

    def _sorting(
        self, quantities_disambiguated: Mapping[str, str], ordering: OrderingOption
    ) -> Callable:
        quantity_order = {
            name: index for index, name in enumerate(quantities_disambiguated)
        }

        if ordering == "quantity":

            def quantity_sort(group):
                (location, quantity), _ = group
                return (quantity_order[quantity], location)

            return quantity_sort

        def location_sort(group):
            (location, quantity), _ = group
            return (location, quantity_order[quantity])

        return location_sort

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

    def spaghetti(
        self,
        realization: RealizationSelection,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantityStrategy | ParameterStrategy,
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
        Produces a spaghetti plot of a filter output. This is a plot where
        each realization corresponds to a specific line on a plot.

        The plot will either be immediately rendered by this function by calling
        `plt.show()` or, if you specify the `to_file` argument, saved to a file.
        This is intended as a quick plotting method to cover most casual use-cases.
        If you want more control over how the plot is drawn, see method
        `spaghetti_plt()`.

        Parameters
        ----------
        realization :
            A realization selection to make on the output data,
            you can either select all the realizations or a random subset.
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
            Each location/quantity group has one label.
        legend :
            Whether and how to draw the plot legend.

            - "auto" will draw the legend if there are at most 12 groups
            - "on" forces the legend to be drawn
            - "off" forces the legend to not be drawn
            - "outside" forces the legend to be drawn next to the plot area
            (instead of inside it)
        line_kwargs :
            A list of keyword arguments to be passed to the matplotlib function
            that draws each line. Each dictionary applies to all realizations in one
            location/quantity group; the list cycles in the order defined by the
            `ordering` parameter.
            See matplotlib documentation for the supported options.
        ordering :
            Whether location or quantity takes priority when ordering groups.
            Locations are sorted by label and quantities follow selection order.
        time_format :
            Controls the formatting of the time axis (the horizontal axis);

            - "auto" will use the format defined by the grouping of the
            `time` parameter
            - "date" attempts to display calendar dates
            - "day" attempts to display days numerically indexed from the start of the
            simulation with the first day being 0

            If the system cannot convert to the requested time format, this argument
            may be ignored.
        title :
            A title to draw on the plot.
        to_file :
            Specify a path to save the plot to a file instead of calling `plt.show()`.
        transform :
            Allows you to specify an arbitrary transform function for the source
            dataframe before we plot it, e.g., to rescale the values.
            The function will be called once per realization per geo/quantity group
            -- once per line, essentially -- with a dataframe that contains just the
            data for that realization and group. The dataframe given as the argument
            is the result of applying all selections. You should return a dataframe
            with the same format, where the values of the data column have been modified
            for your purposes.

            Dataframe columns:

            - "realization": the realization index (same value per group)
            - "time": the time series column
            - "geo": the node ID (same value per group)
            - "quantity": the label of the quantity (same value per group)
            - "value": the data column
        """

        if not isinstance(realization, RealizationSelection):
            raise ValueError("Spaghetti plots only support RealizationSelection.")

        fig, ax = plt.subplots(layout="constrained")
        try:
            self.spaghetti_plt(
                ax,
                realization,
                geo,
                time,
                quantity,
                label_format=label_format,
                line_kwargs=line_kwargs,
                ordering=ordering,
                time_format=time_format,
                transform=transform,
            )

            ax.set_ylabel("count")
            actual_time_format, _ = self._time_format(time, time_format)
            if actual_time_format == "date":
                ax.set_xlabel("date")
                ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
                ax.xaxis.set_major_locator(
                    AutoDateLocator(minticks=6, maxticks=12, interval_multiples=True)
                )
                ax.tick_params(axis="x", labelrotation=45)
            elif actual_time_format == "day":
                ax.set_xlabel("day")
            elif actual_time_format == "tick":
                ax.set_xlabel("tick")
            else:
                ax.set_xlabel("time")
                ax.tick_params(axis="x", labelrotation=45)

            if legend == "auto":
                legend = "on" if len(ax.get_legend_handles_labels()[0]) <= 12 else "off"
            if legend == "on":
                ax.legend()
            elif legend == "outside":
                ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

            if title is not None:
                ax.set_title(title)

            if to_file is None:
                plt.show()
            else:
                fig.savefig(Path(to_file))

        except BaseException:
            plt.close(fig)
            raise

    def spaghetti_plt(
        self,
        ax: Axes,
        realization: RealizationSelection,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantityStrategy | ParameterStrategy,
        *,
        label_format: str = "{n}: {q}",
        line_kwargs: list[dict] | None = None,
        ordering: OrderingOption = "location",
        time_format: TimeFormatOption = "auto",
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> list[Line2D]:
        """
        Draw spaghetti lines onto one matplotlib `Axes`. This variant of
        `spaghetti` leaves plot formatting and legend display to the caller.
        To use subplots, call this method for each axes with narrowed selections.

        Parameters
        ----------
        ax :
            The plot axes on which to draw lines.
        realization :
            A realization selection to make on the output data.
        geo :
            The geographic selection to make on the output data.
        time :
            The time selection to make on the output data.
        quantity :
            The quantity selection to make on the output data.
        label_format :
            A format for legend labels, with `{n}` for the geo node name and
            `{q}` for the quantity name. Each location/quantity group has one label.
        line_kwargs :
            A list of keyword arguments to be passed to the matplotlib function that
            draws each line. Each dictionary applies to all realizations in one
            location/quantity group; the list cycles in `ordering` order. If no color
            is specified, realizations in the same group share a default color.
            See matplotlib documentation for the supported options.
        ordering :
            Whether location or quantity takes priority when ordering groups.
            Locations are sorted by label and quantities follow selection order.
        time_format :
            Controls the formatting of the time axis (the horizontal axis);

            - "auto" will use the format defined by the grouping of the
            `time` parameter
            - "date" attempts to display calendar dates
            - "day" attempts to display days numerically indexed from the start of the
            simulation with the first day being 0

            If the system cannot convert to the requested time format, this argument
            may be ignored.
        transform :
            Allows you to specify an arbitrary transform function for the source
            dataframe before we plot it, e.g., to rescale the values.
            The function will be called once per realization per geo/quantity group
            -- once per line, essentially -- with a dataframe that contains just the
            data for that realization and group. The dataframe given as the argument
            is the result of applying all selections. You should return a dataframe
            with the same format, where the values of the data column have been modified
            for your purposes.

            Dataframe columns:

            - "realization": the realization index (same value per group)
            - "time": the time series column
            - "geo": the node ID (same value per group)
            - "quantity": the label of the quantity (same value per group)
            - "value": the data column

        Returns
        -------
        :
            The list of `Line2D` objects for each line drawn.
        """

        if not isinstance(realization, RealizationSelection):
            raise ValueError("Spaghetti plots only support RealizationSelection.")

        if line_kwargs is None or len(line_kwargs) == 0:
            line_kwargs = [{}]

        if transform is None:
            transform = identity

        data_df = munge_pipeline_output(self.output, realization, geo, time, quantity)

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
            data_df.set_axis(["realization", "time", "geo", *q_mapping.keys()], axis=1)
            .melt(id_vars=["realization", "time", "geo"], var_name="quantity")
            .groupby(["geo", "quantity"], sort=False)
        )
        lines = list[Line2D]()
        for (group_key, group), kwargs in zip(
            sorted(groups_df, key=self._sorting(q_mapping, ordering)),
            cycle(line_kwargs),
        ):
            location, quantity_name = cast(tuple[str, str], group_key)
            q_name = q_mapping[quantity_name]
            label = label_format.format(n=location, q=q_name)
            group_kwargs = {"label": label, **kwargs}
            for index, (_, realization_data) in enumerate(group.groupby("realization")):
                plot_kwargs = dict(group_kwargs)
                if index > 0:
                    plot_kwargs["label"] = "_nolegend_"
                data = transform(
                    realization_data.sort_values("time").assign(quantity=q_name)
                )
                line = ax.plot(data["time"], data["value"], **plot_kwargs)[0]
                lines.append(line)
                if (
                    index == 0
                    and kwargs.get("color") is None
                    and kwargs.get("c") is None
                ):
                    color_key = "c" if "c" in kwargs else "color"
                    group_kwargs[color_key] = line.get_color()

        return lines

    def quantiles(
        self,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantityStrategy | ParameterStrategy,
        credible_intervals: Sequence[float] | None = None,
        *,
        label_format: str = "{n}: {q}: {c}",
        legend: LegendOption = "auto",
        median_label_format: str = "{n}: {q}",
        fill_kwargs: list[dict] | None = None,
        line_kwargs: list[dict] | None = None,
        ordering: OrderingOption = "location",
        time_format: TimeFormatOption = "auto",
        title: str | None = None,
        to_file: str | Path | None = None,
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> None:
        """
        Produces a quantile plot of a filter output, typically as a median line
        with one or more shaded credible intervals.

        The plot will either be immediately rendered by this function by calling
        `plt.show()` or, if you specify the `to_file` argument, saved to a file.
        This is intended as a quick plotting method to cover most casual use-cases.
        If you want more control over how the plot is drawn, see method
        `quantiles_plt()`.

        Parameters
        ----------
        geo :
            The geographic selection to make on the output data.
        time :
            The time selection to make on the output data.
        quantity :
            The quantity selection to make on the output data.
        credible_intervals :
            A list of credible intervals you wish to plot, in percentage points.
            The supported interval widths are 0, 5, 10, ..., 100.
            Defaults to [95].
        label_format :
            A format for the items displayed in the legend for shaded intervals;
            the string will be used in a call to `format()` with the replacement
            variables `{n}` for the name of the geo node, `{q}` for the name of the
            quantity, and `{c}` for the list of credible intervals.
        legend :
            Whether and how to draw the plot legend.

            - "auto" will draw the legend unless it would be too large
            - "on" forces the legend to be drawn
            - "off" forces the legend to not be drawn
            - "outside" forces the legend to be drawn next to the plot area
            (instead of inside it)
        median_label_format :
            A format for the items displayed in the legend for median lines;
            the string will be used in a call to `format()` with the replacement
            variables `{n}` for the name of the geo node and `{q}` for the name of
            the quantity. The suffix `: Median` is appended to each label.
        fill_kwargs :
            A list of keyword arguments to be passed to the matplotlib function
            that draws each shaded interval. Each dictionary applies to all intervals
            in one location/quantity group; the list cycles in the order defined by
            the `ordering` parameter.
            See matplotlib documentation for the supported options.
        line_kwargs :
            A list of keyword arguments to be passed to the matplotlib function
            that draws each median line. The list cycles in the order defined by
            the `ordering` parameter.
            See matplotlib documentation for the supported options.
        ordering :
            Whether location or quantity takes priority when ordering groups.
            Locations are sorted by label and quantities follow selection order.
        time_format :
            Controls the formatting of the time axis (the horizontal axis);

            - "auto" will use the format defined by the grouping of the
            `time` parameter,
            - "date" attempts to display calendar dates,
            - "day" attempts to display days numerically indexed from the start of the
            simulation with the first day being 0.

            If the system cannot convert to the requested time format, this argument
            may be ignored.
        title :
            A title to draw on the plot.
        to_file :
            Specify a path to save the plot to a file instead of calling `plt.show()`.
        transform :
            Allows you to specify an arbitrary transform function for the source
            dataframe before we plot it, e.g., to rescale the quantiles.
            The function will be called once per quantile per geo/quantity group
            -- once per line, essentially -- with a dataframe that contains just the
            data for that group. The dataframe given as the argument is the result
            of applying all selections and computing the quantiles. You should return a
            dataframe with the same format, where the values of the data column have
            been modified for your purposes.

            Dataframe columns:

            - "value": the data column
        """

        fig, ax = plt.subplots(layout="constrained")
        try:
            self.quantiles_plt(
                ax,
                geo,
                time,
                quantity,
                credible_intervals=credible_intervals,
                label_format=label_format,
                median_label_format=median_label_format,
                fill_kwargs=fill_kwargs,
                line_kwargs=line_kwargs,
                ordering=ordering,
                time_format=time_format,
                transform=transform,
            )

            ax.set_ylabel("count")
            actual_time_format, _ = self._time_format(time, time_format)
            if actual_time_format == "date":
                ax.set_xlabel("date")
                ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
                ax.xaxis.set_major_locator(
                    AutoDateLocator(minticks=6, maxticks=12, interval_multiples=True)
                )
                ax.tick_params(axis="x", labelrotation=45)
            elif actual_time_format == "day":
                ax.set_xlabel("day")
            elif actual_time_format == "tick":
                ax.set_xlabel("tick")
            else:
                ax.set_xlabel("time")
                ax.tick_params(axis="x", labelrotation=45)

            if legend == "auto":
                legend = "on" if len(ax.get_legend_handles_labels()[0]) <= 12 else "off"
            if legend == "on":
                ax.legend()
            elif legend == "outside":
                ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

            if title is not None:
                ax.set_title(title)
            if to_file is None:
                plt.show()
            else:
                fig.savefig(Path(to_file))

        except BaseException:
            plt.close(fig)
            raise

    def quantiles_plt(
        self,
        ax: Axes,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantityStrategy | ParameterStrategy,
        credible_intervals: Sequence[float] | None = None,
        *,
        label_format: str = "{n}: {q}: {c}",
        median_label_format: str = "{n}: {q}",
        fill_kwargs: list[dict] | None = None,
        line_kwargs: list[dict] | None = None,
        ordering: OrderingOption = "location",
        time_format: TimeFormatOption = "auto",
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> list[Line2D]:
        """
        Draw quantile plots onto one matplotlib `Axes`. This variant of
        `quantiles` leaves plot formatting and legend display to the caller.
        To use subplots, call this method for each axes with narrowed selections.

        Parameters
        ----------
        ax :
            The plot axes on which to draw lines and shaded intervals.
        geo :
            The geographic selection to make on the output data.
        time :
            The time selection to make on the output data.
        quantity :
            The quantity selection to make on the output data.
        credible_intervals :
            A list of credible intervals you wish to plot, in percentage points.
            The supported interval widths are 0, 5, 10, ..., 100.
            Defaults to [95].
        label_format :
            A format for the items displayed in the legend for shaded intervals;
            the string will be used in a call to `format()` with the replacement
            variables `{n}` for the name of the geo node, `{q}` for the name of the
            quantity, and `{c}` for the list of credible intervals.
        median_label_format :
            A format for the items displayed in the legend for median lines;
            the string will be used in a call to `format()` with the replacement
            variables `{n}` for the name of the geo node and `{q}` for the name of
            the quantity. The suffix `: Median` is appended to each label.
        fill_kwargs :
            A list of keyword arguments to be passed to the matplotlib function
            that draws each shaded interval. Each dictionary applies to all intervals
            in one location/quantity group; the list cycles in the order defined by
            the `ordering` parameter.
            See matplotlib documentation for the supported options.
        line_kwargs :
            A list of keyword arguments to be passed to the matplotlib function
            that draws each median line. The list cycles in the order defined by
            the `ordering` parameter.
            See matplotlib documentation for the supported options.
        ordering :
            Whether location or quantity takes priority when ordering groups.
            Locations are sorted by label and quantities follow selection order.
        time_format :
            Controls the formatting of the time axis (the horizontal axis);

            - "auto" will use the format defined by the grouping of the
            `time` parameter,
            - "date" attempts to display calendar dates,
            - "day" attempts to display days numerically indexed from the start of the
            simulation with the first day being 0.

            If the system cannot convert to the requested time format, this argument
            may be ignored.
        transform :
            Allows you to specify an arbitrary transform function for the source
            dataframe before we plot it, e.g., to rescale the quantiles.
            The function will be called once per quantile per geo/quantity group
            -- once per line, essentially -- with a dataframe that contains just the
            data for that group. The dataframe given as the argument is the result
            of applying all selections and computing the quantiles. You should return a
            dataframe with the same format, where the values of the data column have
            been modified for your purposes.

            Dataframe columns:

            - "value": the data column

        Returns
        -------
        :
            The list of `Line2D` objects for each median line drawn. Shaded intervals
            can be accessed through the axes' collections.
        """
        if line_kwargs is None or len(line_kwargs) == 0:
            line_kwargs = [{"color": "black"}]

        if fill_kwargs is None or len(fill_kwargs) == 0:
            fill_kwargs = [{"color": "tab:blue", "alpha": 0.3}]

        if transform is None:
            transform = identity

        if credible_intervals is None or len(credible_intervals) == 0:
            credible_intervals = [95]
        if any(
            not np.isfinite(interval)
            or interval < 0
            or interval > 100
            or interval % 5 != 0
            for interval in credible_intervals
        ):
            raise ValueError(
                "Credible intervals must be between 0 and 100 in increments of 5."
            )

        quantile_list = []
        for interval in sorted(credible_intervals, reverse=True):
            lower, upper = self._compute_quantile_range(interval)
            quantile_list.append((f"quantile_{lower}", f"quantile_{upper}"))

        flat_quantile_list = list(
            dict.fromkeys(
                [quantile for pair in quantile_list for quantile in pair]
                + ["quantile_50.0"]
            )
        )
        realizations_agg = self.output.select.all().agg(flat_quantile_list)

        data_df = munge_pipeline_output(
            self.output, realizations_agg, geo, time, quantity
        )

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

        data_df.columns = pd.MultiIndex.from_tuples(
            [("time", ""), ("geo", "")]
            + [
                (quantity_name, quantile)
                for quantity_name in q_mapping
                for quantile in flat_quantile_list
            ]
        )
        groups_df = data_df.groupby("geo", sort=False)
        groups = (
            ((location, quantity_name), group)
            for location, group in groups_df
            for quantity_name in q_mapping
        )

        lines = list[Line2D]()
        for ((location, quantity_name), group), l_kwargs, f_kwargs in zip(
            sorted(groups, key=self._sorting(q_mapping, ordering)),
            cycle(line_kwargs),
            cycle(fill_kwargs),
        ):
            quantity_label = q_mapping[quantity_name]
            group = group.sort_values(("time", ""))
            for index, (lower, upper) in enumerate(quantile_list):
                data_lower = transform(
                    pd.DataFrame({"value": group[quantity_name][lower]})
                )
                data_upper = transform(
                    pd.DataFrame({"value": group[quantity_name][upper]})
                )
                label = (
                    label_format.format(
                        n=location,
                        q=quantity_label,
                        c=credible_intervals,
                    )
                    if index == len(quantile_list) - 1
                    else "_nolegend_"
                )
                plot_kwargs = {"label": label, **f_kwargs}
                if index != len(quantile_list) - 1:
                    plot_kwargs["label"] = "_nolegend_"
                ax.fill_between(
                    group["time"],
                    data_lower["value"],
                    data_upper["value"],
                    **plot_kwargs,
                )
            data_median = transform(
                pd.DataFrame({"value": group[quantity_name]["quantile_50.0"]})
            )
            label = (
                f"{median_label_format.format(n=location, q=quantity_label)}: Median"
            )
            lines.extend(
                ax.plot(
                    group["time"],
                    data_median["value"],
                    **{"label": label, "zorder": 100, **l_kwargs},
                )
            )

        return lines

    def histogram(
        self,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantityStrategy | ParameterStrategy,
        *,
        label_format: str = "{n}: {q}: {t}",
        legend: LegendOption = "auto",
        hist_kwargs: list[dict] | None = None,
        ordering: OrderingOption = "location",
        title: str | None = None,
        to_file: str | Path | None = None,
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> None:
        """
        Produces a histogram plot of a filter output. This is a plot where
        a specific time instance is taken and plotted as a histogram.

        The plot will either be immediately rendered by this function by calling
        `plt.show()` or, if you specify the `to_file` argument, saved to a file.
        This is intended as a quick plotting method to cover most casual use-cases.
        If you want more control over how the plot is drawn, see method
        `histogram_plt()`.

        Parameters
        ----------
        geo :
            The geographic selection to make on the output data.
        time :
            The time selection to make on the output data. For
            this plot the time selection must be a single time instant.
            For instance you could use
            `rume.time_frame.select.days(100, 100).group("day").agg()`
            to create a histogram corresponding to a single day.
        quantity :
            The quantity selection to make on the output data.
        label_format :
            A format for the items displayed in the legend;
            the string will be used in a call to `format()` with the replacement
            variables `{n}` for the name of the geo node, `{q}` for the name of the
            quantity, and `{t}` for the selected date range.
        legend :
            Whether and how to draw the plot legend.

            - "auto" will draw the legend unless it would be too large
            - "on" forces the legend to be drawn
            - "off" forces the legend to not be drawn
            - "outside" forces the legend to be drawn next to the plot area
            (instead of inside it)
        hist_kwargs :
            A list of keyword arguments to be passed to the matplotlib function
            that draws each histogram. The list cycles in the order defined by
            the `ordering` parameter.
            See matplotlib documentation for the supported options.
        ordering :
            Whether location or quantity takes priority when ordering groups.
            Locations are sorted by label and quantities follow selection order.
        title :
            A title to draw on the plot.
        to_file :
            Specify a path to save the plot to a file instead of calling `plt.show()`.
        transform :
            Allows you to specify an arbitrary transform function for the source
            dataframe before we plot it, e.g., to rescale the values.
            The function will be called once per geo/quantity group with a dataframe
            that contains just the data for that group. The dataframe given as the
            argument is the result of applying all selections.
            You should return a dataframe with the same format, where the
            values of the data column have been modified for your purposes.

            Dataframe columns:

            - "value": the data column
        """

        fig, ax = plt.subplots(layout="constrained")
        try:
            self.histogram_plt(
                ax,
                geo,
                time,
                quantity,
                label_format=label_format,
                hist_kwargs=hist_kwargs,
                ordering=ordering,
                transform=transform,
            )

            ax.set_xlabel("value")
            ax.set_ylabel("count")

            if legend == "auto":
                legend = "on" if len(ax.get_legend_handles_labels()[0]) <= 12 else "off"
            if legend == "on":
                ax.legend()
            elif legend == "outside":
                ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

            if title is not None:
                ax.set_title(title)

            if to_file is None:
                plt.show()
            else:
                fig.savefig(Path(to_file))

        except BaseException:
            plt.close(fig)
            raise

    def histogram_plt(
        self,
        ax: Axes,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantityStrategy | ParameterStrategy,
        *,
        label_format: str = "{n}: {q}: {t}",
        hist_kwargs: list[dict] | None = None,
        ordering: OrderingOption = "location",
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> list[BarContainer | list[Polygon]]:
        """
        Draw histograms onto one matplotlib `Axes`. This variant of
        `histogram` leaves plot formatting and legend display to the caller.
        To use subplots, call this method for each axes with narrowed selections.

        Parameters
        ----------
        ax :
            The plot axes on which to draw histograms.
        geo :
            The geographic selection to make on the output data.
        time :
            The time selection to make on the output data. For
            this plot the time selection must be a single time instant.
            For instance you could use
            `rume.time_frame.select.days(100, 100).group("day").agg()`
            to create a histogram corresponding to a single day.
        quantity :
            The quantity selection to make on the output data.
        label_format :
            A format for the items displayed in the legend;
            the string will be used in a call to `format()` with the replacement
            variables `{n}` for the name of the geo node, `{q}` for the name of the
            quantity, and `{t}` for the selected date range.
        hist_kwargs :
            A list of keyword arguments to be passed to the matplotlib function
            that draws each histogram. The list cycles in the order defined by
            the `ordering` parameter.
            See matplotlib documentation for the supported options.
        ordering :
            Whether location or quantity takes priority when ordering groups.
            Locations are sorted by label and quantities follow selection order.
        transform :
            Allows you to specify an arbitrary transform function for the source
            dataframe before we plot it, e.g., to rescale the values.
            The function will be called once per geo/quantity group with a dataframe
            that contains just the data for that group. The dataframe given as the
            argument is the result of applying all selections.
            You should return a dataframe with the same format, where the
            values of the data column have been modified for your purposes.

            Dataframe columns:

            - "value": the data column

        Returns
        -------
        :
            The matplotlib artists for each histogram drawn, in `ordering` order.
            Bar histograms return a `BarContainer`; step histograms return a list of
            `Polygon` objects.
        """
        if hist_kwargs is None or len(hist_kwargs) == 0:
            hist_kwargs = [{}]

        if transform is None:
            transform = identity

        realizations_agg = self.output.select.all()
        data_df = munge_pipeline_output(
            self.output, realizations_agg, geo, time, quantity
        )

        if len(data_df["time"].unique()) > 1:
            err = (
                "When drawing a histogram plot, please ensure that you choose a "
                "time aggregation strategy that reduces the time series to a "
                "single point (scalar)."
            )
            raise ValueError(err)

        # Map geo labels:
        result_scope = geo.to_scope()
        if (labels := result_scope.labels_option) is not None:
            geo_map = dict(zip(result_scope.node_ids, labels))
            data_df["geo"] = data_df["geo"].apply(lambda x: geo_map[x])

        # Before melting, disambiguate any quantities with the same name.
        q_mapping = quantity.disambiguate_groups()

        groups_df = (
            data_df.set_axis(["realization", "time", "geo", *q_mapping.keys()], axis=1)
            .melt(id_vars=["realization", "time", "geo"], var_name="quantity")
            .groupby(["geo", "quantity"], sort=False)
        )

        plots = list[BarContainer | list[Polygon]]()
        for (group_key, group), kwargs in zip(
            sorted(groups_df, key=self._sorting(q_mapping, ordering)),
            cycle(hist_kwargs),
        ):
            location, quantity_name = cast(tuple[str, str], group_key)
            label = label_format.format(
                n=location,
                q=q_mapping[quantity_name],
                t=time.date_bounds_label,
            )
            data = transform(group[["value"]])
            _, _, artists = ax.hist(data["value"], **{"label": label, **kwargs})
            if isinstance(artists, BarContainer):
                plots.append(artists)
            elif isinstance(artists, list) and all(
                isinstance(artist, Polygon) for artist in artists
            ):
                plots.append(cast(list[Polygon], artists))
            else:
                raise TypeError("Matplotlib returned unexpected histogram artists.")

        return plots

    def line(
        self,
        realization: RealizationAggregation,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantityStrategy | ParameterStrategy,
        *,
        label_format: str = "{n}: {q}: {m}",
        legend: LegendOption = "auto",
        line_kwargs: list[dict] | None = None,
        ordering: OrderingOption = "location",
        time_format: TimeFormatOption = "auto",
        title: str | None = None,
        to_file: str | Path | None = None,
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> None:
        """
        Produces a line plot of a filter output, which may plot the time series of an
        IPM quantity or a filter parameter.

        Filter outputs typically contain many realizations of a simulation. In order to
        condense the realization axis to a single time-series, you are required to
        specify an aggregation method across realizations.

        The plot will either be immediately rendered by this function by calling
        `plt.show()` or, if you specify the `to_file` argument, saved to a file.
        This is intended as a quick plotting method to cover most casual use-cases.
        If you want more control over how the plot is drawn, see method `line_plt()`.

        Parameters
        ----------
        realization :
            A realization aggregation to make on the output data.
        geo :
            The geographic selection to make on the output data.
        time :
            The time selection to make on the output data.
        quantity :
            The quantity selection to make on the output data.
        label_format :
            A format for the items displayed in the legend;
            the string will be used in a call to `format()`
            with the replacement variables `{n}` for the name of the geo node,
            `{q}` for the name of the quantity, and `{m}` for the aggregation name
            corresponding to the realization aggregation.
        legend :
            Whether and how to draw the plot legend.

            - "auto" will draw the legend unless it would be too large
            - "on" forces the legend to be drawn
            - "off" forces the legend to not be drawn
            - "outside" forces the legend to be drawn next to the plot area
            (instead of inside it)
        line_kwargs :
            A list of keyword arguments to be passed to the matplotlib function
            that draws each line. The list cycles in the order defined by
            the `ordering` parameter.
            See matplotlib documentation for the supported options.
        ordering :
            Whether location or quantity takes priority when ordering groups.
            Locations are sorted by label and quantities follow selection order.
            Within each group, lines are sorted by aggregation name.
        time_format :
            Controls the formatting of the time axis (the horizontal axis);

            - "auto" will use the format defined by the grouping of the
            `time` parameter,
            - "date" attempts to display calendar dates,
            - "day" attempts to display days numerically indexed from the start of the
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
            The function will be called once per aggregation per geo/quantity group
            -- once per line, essentially -- with a dataframe that contains just the
            data for that group. The dataframe given as the argument is the result of
            applying all selections and aggregations. You should return a dataframe
            with the same format, where the values of the data column have been modified
            for your purposes.

            Dataframe columns:

            - "metric": the aggregation method for the realizations
            - "value": the data column
        """

        if not isinstance(realization, RealizationAggregation):
            raise ValueError("Line plots only support RealizationAggregation.")

        try:
            _, ax = plt.subplots(layout="constrained")

            lines = self.line_plt(
                ax,
                realization,
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
        realization: RealizationAggregation,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantityStrategy | ParameterStrategy,
        *,
        label_format: str = "{n}: {q}: {m}",
        line_kwargs: list[dict] | None = None,
        ordering: OrderingOption = "location",
        time_format: TimeFormatOption = "auto",
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> list[Line2D]:
        """
        Draw lines onto one matplotlib `Axes`. This variant of
        `line` leaves plot formatting and legend display to the caller.
        To use subplots, call this method for each axes with narrowed selections.

        Parameters
        ----------
        ax :
            The plot axes on which to draw lines.
        realization :
            A realization aggregation to make on the output data.
        geo :
            The geographic selection to make on the output data.
        time :
            The time selection to make on the output data.
        quantity :
            The quantity selection to make on the output data.
        label_format :
            A format for the items displayed in the legend;
            the string will be used in a call to `format()`
            with the replacement variables `{n}` for the name of the geo node,
            `{q}` for the name of the quantity, and `{m}` for the aggregation name
            corresponding to the realization aggregation.
        line_kwargs :
            A list of keyword arguments to be passed to the matplotlib function
            that draws each line. The list cycles in the order defined by
            the `ordering` parameter.
            See matplotlib documentation for the supported options.
        ordering :
            Whether location or quantity takes priority when ordering groups.
            Locations are sorted by label and quantities follow selection order.
            Within each group, lines are sorted by aggregation name.
        time_format :
            Controls the formatting of the time axis (the horizontal axis);

            - "auto" will use the format defined by the grouping of the
            `time` parameter,
            - "date" attempts to display calendar dates,
            - "day" attempts to display days numerically indexed from the start of the
            simulation with the first day being 0.

            If the system cannot convert to the requested time format, this argument
            may be ignored.
        transform :
            Allows you to specify an arbitrary transform function for the source
            dataframe before we plot it, e.g., to rescale the values.
            The function will be called once per aggregation per geo/quantity group
            -- once per line, essentially -- with a dataframe that contains just the
            data for that group. The dataframe given as the argument is the result of
            applying all selections and aggregations. You should return a dataframe
            with the same format, where the values of the data column have been modified
            for your purposes.

            Dataframe columns:

            - "metric": the aggregation method for the realizations
            - "value": the data column

        Returns
        -------
        :
            The `Line2D` object for each line drawn; you can use this to have finer
            control over the presentation of the lines.

        """

        if not isinstance(realization, RealizationAggregation):
            raise ValueError("Line plots only support RealizationAggregation.")

        if line_kwargs is None or len(line_kwargs) == 0:
            line_kwargs = [{}]

        if transform is None:
            transform = identity

        data_df = munge_pipeline_output(self.output, realization, geo, time, quantity)

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

        quantity_names = (name for name in q_mapping for _ in realization.aggregation)
        data_df.columns = pd.MultiIndex.from_tuples(
            [("time", ""), ("geo", "")]
            + [
                (name, metric)
                for name, (_, metric) in zip(
                    quantity_names, data_df.columns[2:], strict=True
                )
            ]
        )

        lines = list[Line2D]()
        geo_groups = data_df.groupby("geo", sort=False)
        groups = (
            ((location, quantity_name), group)
            for location, group in geo_groups
            for quantity_name in q_mapping
        )
        kwargs_cycle = cycle(line_kwargs)
        for (location, quantity_name), group in sorted(
            groups, key=self._sorting(q_mapping, ordering)
        ):
            metric_groups = (
                group[quantity_name].melt(var_name="metric").groupby("metric")
            )
            for metric_name, metric_data in metric_groups:
                label = label_format.format(
                    n=location,
                    q=q_mapping[quantity_name],
                    m=metric_name,
                )
                data = transform(metric_data)
                lines.extend(
                    ax.plot(
                        group["time"],
                        data["value"],
                        **{"label": label, **next(kwargs_cycle)},
                    )
                )

        return lines

    def kde(
        self,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantityStrategy | ParameterStrategy,
        *,
        label_format: str = "{n}: {q}: {t}",
        legend: LegendOption = "auto",
        line_kwargs: list[dict] | None = None,
        ordering: OrderingOption = "location",
        bandwidth: float | str = "scott",
        delta_t: float | None = None,
        title: str | None = None,
        to_file: str | Path | None = None,
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> None:
        """
        Produces a kernel density plot of a filter output. This is a plot where
        a specific time instance is taken and plotted as a KDE.

        The plot will either be immediately rendered by this function by calling
        `plt.show()` or, if you specify the `to_file` argument, saved to a file.
        This is intended as a quick plotting method to cover most casual use-cases.
        If you want more control over how the plot is drawn, see method `kde_plt()`.

        Parameters
        ----------
        geo :
            The geographic selection to make on the output data.
        time :
            The time selection to make on the output data. For
            this plot the time selection must be a single time instant.
            For instance you could use
            `rume.time_frame.select.days(100, 100).group("day").agg()`
            to create a KDE corresponding to a single day.
        quantity :
            The quantity selection to make on the output data.
        label_format :
            A format for the items displayed in the legend;
            the string will be used in a call to `format()` with the replacement
            variables `{n}` for the name of the geo node, `{q}` for the name of the
            quantity, and `{t}` for the selected date range.
        legend :
            Whether and how to draw the plot legend.

            - "auto" will draw the legend unless it would be too large
            - "on" forces the legend to be drawn
            - "off" forces the legend to not be drawn
            - "outside" forces the legend to be drawn next to the plot area
            (instead of inside it)
        line_kwargs :
            A list of keyword arguments to be passed to the matplotlib function
            that draws each line. The list cycles in the order defined by
            the `ordering` parameter.
            See matplotlib documentation for the supported options.
        ordering :
            Whether location or quantity takes priority when ordering groups.
            Locations are sorted by label and quantities follow selection order.
        bandwidth :
            The bandwidth for the convolution kernel. Specify either a float or
            a string for an adaptive scheme. Options are "scott", "silverman".
            See
            [https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html)
            for details.
        delta_t :
            Specifies the positive, finite interval at which to sample the density.
            If `None`, the interval is `1/100` of the data range for each group.
        title :
            A title to draw on the plot.
        to_file :
            Specify a path to save the plot to a file instead of calling `plt.show()`.
        transform :
            Allows you to specify an arbitrary transform function for the source
            dataframe before we plot it, e.g., to rescale the values.
            The function will be called once per geo/quantity group with a dataframe
            that contains just the data for that group. The dataframe given as the
            argument is the result of applying all selections.
            You should return a dataframe with the same format, where the
            values of the data column have been modified for your purposes.

            Dataframe columns:

            - "value": the data column
        """
        fig, ax = plt.subplots(layout="constrained")
        try:
            self.kde_plt(
                ax,
                geo,
                time,
                quantity,
                label_format=label_format,
                line_kwargs=line_kwargs,
                ordering=ordering,
                delta_t=delta_t,
                bandwidth=bandwidth,
                transform=transform,
            )

            ax.set_xlabel("value")
            ax.set_ylabel("density")

            if legend == "auto":
                legend = "on" if len(ax.get_legend_handles_labels()[0]) <= 12 else "off"
            if legend == "on":
                ax.legend()
            elif legend == "outside":
                ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

            if title is not None:
                ax.set_title(title)

            if to_file is None:
                plt.show()
            else:
                fig.savefig(Path(to_file))

        except BaseException:
            plt.close(fig)
            raise

    def kde_plt(
        self,
        ax: Axes,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantityStrategy | ParameterStrategy,
        *,
        label_format: str = "{n}: {q}: {t}",
        line_kwargs: list[dict] | None = None,
        ordering: OrderingOption = "location",
        bandwidth: float | str = "scott",
        delta_t: float | None = None,
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> list[Line2D]:
        """
        Draw KDE lines onto one matplotlib `Axes`. This variant of
        `kde` leaves plot formatting and legend display to the caller.
        To use subplots, call this method for each axes with narrowed selections.

        Parameters
        ----------
        ax :
            The plot axes on which to draw lines.
        geo :
            The geographic selection to make on the output data.
        time :
            The time selection to make on the output data. For
            this plot the time selection must be a single time instant.
            For instance you could use
            `rume.time_frame.select.days(100, 100).group("day").agg()`
            to create a KDE corresponding to a single day.
        quantity :
            The quantity selection to make on the output data.
        label_format :
            A format for the items displayed in the legend;
            the string will be used in a call to `format()` with the replacement
            variables `{n}` for the name of the geo node, `{q}` for the name of the
            quantity, and `{t}` for the selected date range.
        line_kwargs :
            A list of keyword arguments to be passed to the matplotlib function
            that draws each line. The list cycles in the order defined by
            the `ordering` parameter.
            See matplotlib documentation for the supported options.
        ordering :
            Whether location or quantity takes priority when ordering groups.
            Locations are sorted by label and quantities follow selection order.
        bandwidth :
            The bandwidth for the convolution kernel. Specify either a float or
            a string for an adaptive scheme. Options are "scott", "silverman".
            See
            [https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html)
            for details.
        delta_t :
            Specifies the positive, finite interval at which to sample the density.
            If `None`, the interval is `1/100` of the data range for each group.
        transform :
            Allows you to specify an arbitrary transform function for the source
            dataframe before we plot it, e.g., to rescale the values.
            The function will be called once per geo/quantity group with a dataframe
            that contains just the data for that group. The dataframe given as the
            argument is the result of applying all selections.
            You should return a dataframe with the same format, where the
            values of the data column have been modified for your purposes.

            Dataframe columns:

            - "value": the data column

        Returns
        -------
        :
            The list of `Line2D` objects for each line drawn.
        """

        if delta_t is not None and (not np.isfinite(delta_t) or delta_t <= 0):
            raise ValueError("delta_t must be positive and finite.")

        realizations_agg = self.output.select.all()
        data_df = munge_pipeline_output(
            self.output, realizations_agg, geo, time, quantity
        )

        if len(data_df["time"].unique()) > 1:
            err = (
                "When drawing a KDE plot, please ensure that you choose a "
                "time aggregation strategy that reduces the time series to a "
                "single point (scalar)."
            )
            raise ValueError(err)

        if line_kwargs is None or len(line_kwargs) == 0:
            line_kwargs = [{}]

        if transform is None:
            transform = identity

        # Map geo labels:
        result_scope = geo.to_scope()
        if (labels := result_scope.labels_option) is not None:
            geo_map = dict(zip(result_scope.node_ids, labels))
            data_df["geo"] = data_df["geo"].apply(lambda x: geo_map[x])

        # Before melting, disambiguate any quantities with the same name.
        q_mapping = quantity.disambiguate_groups()

        groups_df = (
            data_df.set_axis(["realization", "time", "geo", *q_mapping.keys()], axis=1)
            .melt(id_vars=["realization", "time", "geo"], var_name="quantity")
            .groupby(["geo", "quantity"], sort=False)
        )

        lines = list[Line2D]()
        for (group_key, group), kwargs in zip(
            sorted(groups_df, key=self._sorting(q_mapping, ordering)),
            cycle(line_kwargs),
        ):
            location, quantity_name = cast(tuple[str, str], group_key)
            label = label_format.format(
                n=location,
                q=q_mapping[quantity_name],
                t=time.date_bounds_label,
            )
            data = transform(group[["value"]])["value"].to_numpy()
            density = gaussian_kde(data, bw_method=bandwidth)
            d_min = data.min()
            d_max = data.max()
            step = delta_t if delta_t is not None else (d_max - d_min) / 100
            eval_range = np.arange(d_min, d_max + step, step)
            lines.extend(
                ax.plot(
                    eval_range,
                    density.evaluate(eval_range),
                    **{"label": label, **kwargs},
                )
            )

        return lines
