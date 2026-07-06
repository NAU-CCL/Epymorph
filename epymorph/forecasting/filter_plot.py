from datetime import timedelta
from itertools import cycle
from math import ceil
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Sequence,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.dates import AutoDateLocator, DateFormatter
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from scipy.stats import gaussian_kde

from epymorph.compartment_model import (
    QuantityStrategy,
)
from epymorph.forecasting.munge_realizations import (
    ParameterStrategy,
    RealizationAggregation,
    RealizationSelection,
)
from epymorph.forecasting.pipeline import PipelineOutput, munge_pipeline_output
from epymorph.geography.scope import GeoAggregation, GeoSelection
from epymorph.time import (
    TimeAggregation,
    TimeSelection,
)
from epymorph.tools.out_plot import LegendOption, TimeFormatOption
from epymorph.util import identity


class PlotRendererPipeline:
    """
    Provides methods for rendering an output in plot form.

    Most commonly, you will use `PlotRendererFilter` starting
    from a filter output object
    that supports it:

    Parameters
    ----------
    output : PipelineOutput
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
        sharex: bool = True,
        ncols: int = 3,
        legend: LegendOption = "auto",
        line_kwargs: list[dict] | None = None,
        time_format: TimeFormatOption = "auto",
        title: str | None = None,
        to_file: str | Path | None = None,
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> None:
        """
        Produces a spaghetti plot of a filter output. This is a plot where
        each realization corresponds to a specific line on a plot.

        Parameters
        ----------
        realization:
            A realization selection to make on the output data,
            you can either select all the realizations or a random subset.
        geo :
            The geographic selection to make on the output data.
        time :
            The time selection to make on the output data.
        quantity :
            The quantity selection to make on the output data.
        sharex :
            Whether or not the subplots should share the x-axis ticks.
        ncols :
            The number of columns in the resulting subplot matrix. The
            number of rows is set dynamically.
        line_kwargs :
            A list of dictionaries of keyword arguments to be passed to the matplotlib
            function that draws each line. Each dictionary corresponds
            to a single quantity.
            See matplotlib documentation for the supported options.
        time_format :
            Controls the formatting of the time axis (the horizontal axis);
            "auto" will use the format defined by the grouping of the `time` parameter,
            "date" attempts to display calendar dates,
            "day" attempts to display days numerically indexed from the start of the
            simulation with the first day being 0.
            If the system cannot convert to the requested time format, this argument
            may be ignored.
        legend :
            Whether and how to draw the plot legend.
            - "auto" will draw the legend unless it would be too large
            - "on" forces the legend to be drawn
            - "off" forces the legend to not be drawn
            - "outside" forces the legend to be drawn next to the plot area
            (instead of inside it)
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

        if not isinstance(realization, RealizationSelection):
            raise ValueError("Spaghetti plots only support RealizationSelection.")

        try:
            # Initialize subplots and info
            num_nodes = self.output.rume.scope.nodes
            nrows = ceil(num_nodes / ncols)
            fig, axs = plt.subplots(
                nrows,
                ncols,
                figsize=(ncols * 5, nrows * 3),
                sharex=sharex,
                layout="constrained",
            )

            # Y-axis
            fig.supylabel("count")

            # Title
            if title is not None:
                fig.suptitle(t=title)

            # Legend
            if legend == "auto":
                # auto: show a legend if there are at most 4 quantities.
                legend = "on" if len(quantity.labels) <= 4 else "off"

            # Call the spaghetti plot function, this returns the lines
            _ = self.spaghetti_plt(
                axs,
                realization,
                geo,
                time,
                quantity,
                legend=legend,
                line_kwargs=line_kwargs,
                time_format=time_format,
                label_format="{q}",
                transform=transform,
            )

            if to_file is None:
                plt.show()
            else:
                path = Path(to_file)
                fig.savefig(path)

        except:
            plt.close()
            raise

    def spaghetti_plt(
        self,
        axs: Axes | Iterable[Axes] | NDArray[Any],
        realization: RealizationSelection,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantityStrategy | ParameterStrategy,
        *,
        legend: LegendOption = "auto",
        kwarg_type: str = "quantity",
        ax_title: str = "{n}",
        line_kwargs: list[dict] | None = None,
        time_format: TimeFormatOption = "auto",
        label_format: str = "{n}: {q}",
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> list[Line2D]:
        """
        Draw spaghetti plots onto the array of matplotlib `Axes`, such as what is
        returned by matplotlib `subplots`. This is a variant of the method
        `spaghetti`.

        Parameters
        ----------
        axs:
            The array of matplotlib `Axes` on which to draw the plots.
        realization:
            A realization selection to make on the output data.
        geo :
            The geographic selection to make on the output data.
        time :
            The time selection to make on the output data.
        quantity :
            The quantity selection to make on the output data.
        legend :
            Whether and how to draw the plot legend.
        kwarg_type :
            Whether to iterate the kwargs over the quantities or the geos.
            Options are "geo", default is quantity iteration.
        ax_title "
            A format string to display as the title for each subplot.
            Defaults to displaying the geo.
        line_kwargs :
            A list of dictionaries of keyword arguments to be passed to the matplotlib
            function that draws each line.
        time_format :
            Controls the formatting of the time axis (the horizontal axis).
        label_format :
            A format for the items displayed in the legend.
        transform :
            Allows you to specify an arbitrary transform function for the source
            dataframe before we plot it.

        Returns
        -------
        :
            The list of `Line2D` objects for each line drawn.
        """

        if isinstance(axs, np.ndarray):
            ax_list = list(axs.flat)
        elif isinstance(axs, Axes):
            ax_list = [axs]
        else:
            ax_list = list(axs)

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

        # Group by geo location
        groups_df = data_df.set_axis(
            ["realization", "time", "geo", *q_mapping.keys()], axis=1
        ).groupby("geo")

        _time_format, _ = self._time_format(time, time_format)

        lines = list[Line2D]()
        plot_index = 0
        for (geo_group_name, gdf), gkwargs in zip(groups_df, cycle(line_kwargs)):
            ax = ax_list[plot_index]

            ax_title_str = ax_title.format(n=geo_group_name)

            ax.set_title(ax_title_str)

            ax.tick_params(axis="x", labelrotation=45)

            quantity_groups = gdf.melt(
                id_vars=["realization", "time", "geo"], var_name="quantity"
            ).groupby("quantity")

            # Line kwargs cycle over the quantity axis,
            # not setting colors for individual lines!
            for (quantity_group_name, qdf), kwargs in zip(
                quantity_groups,
                cycle(line_kwargs),
            ):
                if kwarg_type == "geo":
                    kwargs = gkwargs

                q_name = q_mapping[str(quantity_group_name)]
                label = label_format.format(n=geo_group_name, q=q_name)
                curr_kwargs = {"label": label, **kwargs}

                realization_groups = qdf.groupby("realization")

                # Iterate over each realization and apply args
                for realization_index, (realization_group_name, rdf) in enumerate(
                    realization_groups
                ):
                    plot_kwargs = curr_kwargs.copy()
                    plot_kwargs["label"] = (
                        label if realization_index == 1 else "_nolegend_"
                    )
                    rdf = rdf.sort_values("time")
                    data = transform(rdf.assign(quantity=q_name))
                    ls = ax.plot(rdf["time"], data["value"], **plot_kwargs)
                    lines.extend(ls)

            ##Labels and Legend
            if legend == "on":
                ax.legend()
            elif legend == "outside":
                ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

            subplotspec = ax.get_subplotspec()
            if subplotspec is not None and subplotspec.is_last_row():
                if _time_format == "date":
                    ax.set_xlabel("date")
                    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
                    ax.xaxis.set_major_locator(
                        AutoDateLocator(
                            minticks=6, maxticks=12, interval_multiples=True
                        )
                    )

                elif _time_format == "day":
                    ax.set_xlabel("day")
                elif _time_format == "tick":
                    ax.set_xlabel("tick")
                else:
                    ax.set_xlabel("time")

            plot_index += 1
            plot_index = plot_index % len(ax_list)

        return lines

    def quantiles(
        self,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantityStrategy | ParameterStrategy,
        credible_intervals: Sequence[float] | None = None,
        *,
        sharex: bool = True,
        ncols: int = 3,
        legend: LegendOption = "auto",
        fill_kwargs: list[dict] | None = None,
        line_kwargs: list[dict] | None = None,
        time_format: TimeFormatOption = "auto",
        title: str | None = None,
        to_file: str | Path | None = None,
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ):
        """
        Produces a quantile plot of a filter output. This is a plot where
        each realization corresponds to a specific line on a plot.

        Parameters
        ----------
        geo :
            The geographic selection to make on the output data.
        time :
            The time selection to make on the output data.
        quantity :
            The quantity selection to make on the output data.
        credible_intervals :
            A list of credible intervals you wish to plot.
            This argument only accepts CI's in 2.5% increments,
            i.e. 2.5,5.0,7.5,...,97.5,100.0.
        sharex :
            Whether or not the subplots should share the x-axis ticks.
        ncols :
            The number of columns in the resulting subplot matrix. The
            number of rows is set dynamically.
        legend :
            Whether and how to draw the plot legend.

            - "auto" will draw the legend unless it would be too large
            - "on" forces the legend to be drawn
            - "off" forces the legend to not be drawn
            - "outside" forces the legend to be drawn next to the plot area
            (instead of inside it)
        fill_kwargs :
            A list of dictionaries corresponding to each CI.
            This tells the plotting function how to fill the interior of the CI.
            See matplotlib documentation for the supported options.
        line_kwargs :
            A list of dictionaries correspondng to each CI's median.
            See matplotlib documentation for the supported options.
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
            num_nodes = self.output.rume.scope.nodes
            nrows = ceil(num_nodes / ncols)
            fig, axs = plt.subplots(
                nrows,
                ncols,
                figsize=(ncols * 5, nrows * 3),
                sharex=sharex,
                layout="constrained",
            )

            # Y-axis
            fig.supylabel("count")

            # Title
            if title is not None:
                fig.suptitle(t=title)

            # Legend
            if legend == "auto":
                # auto: show a legend if there are at most 5 realizations.
                legend = "on" if len(quantity.labels) <= 4 else "off"

            self.quantiles_plt(
                axs,
                geo,
                time,
                quantity,
                legend=legend,
                credible_intervals=credible_intervals,
                fill_kwargs=fill_kwargs,
                line_kwargs=line_kwargs,
                time_format=time_format,
                transform=transform,
            )

            if to_file is None:
                plt.show()
            else:
                path = Path(to_file)
                fig.savefig(path)

        except:
            plt.close()
            raise

    def quantiles_plt(
        self,
        axs: Axes | Iterable[Axes] | NDArray[Any],
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantityStrategy | ParameterStrategy,
        *,
        credible_intervals: Sequence[float] | None = None,
        legend: LegendOption = "auto",
        kwarg_type: str = "quantity",
        fill_kwargs: list[dict] | None = None,
        line_kwargs: list[dict] | None = None,
        time_format: TimeFormatOption = "auto",
        label_format: str = "{n}: {q}: {c}",
        ax_title: str = "{n}",
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ):
        """
        Draw quantile plots onto the array of matplotlib `Axes`, such as what is
        returned by matplotlib `subplots`. This is a variant of the method
        `quantile`.

        Parameters
        ----------
        axs:
            The array of matplotlib `Axes` on which to draw the plots.
        geo :
            The geographic selection to make on the output data.
        time :
            The time selection to make on the output data.
        quantity :
            The quantity selection to make on the output data.
        credible_intervals :
            A list of credible intervals you wish to plot.
        legend :
            Whether and how to draw the plot legend.
        kwarg_type :
            Whether to iterate the kwargs over the quantities or the geos.
            Options are "geo", default is quantity iteration.
        fill_kwargs :
            A list of dictionaries corresponding to each credible interval.
        line_kwargs :
            A list of dictionaries correspondng to each credible interval's median.
        time_format :
            Controls the formatting of the time axis (the horizontal axis).
        label_format :
            A format string describing the labels in the legend. Defaults to
            {n} : {q} : {c}, with n the geo, q the quantity,
            and c the list of credible intervals.
        ax_title :
            A format string for the title of each subplot. Defaults to
            {n}, with n the geo.
        transform :
            Allows you to specify an arbitrary transform function for the source
            dataframe before we plot it.
        """
        if isinstance(axs, np.ndarray):
            ax_list = list(axs.flat)
        elif isinstance(axs, Axes):
            ax_list = [axs]
        else:
            ax_list = list(axs)

        if line_kwargs is None or len(line_kwargs) == 0:
            line_kwargs = [{"color": "black"}]

        if fill_kwargs is None or len(fill_kwargs) == 0:
            fill_kwargs = [{"color": "tab:blue", "alpha": 0.3}]

        if transform is None:
            transform = identity

        if credible_intervals is None or len(credible_intervals) == 0:
            credible_intervals = [95]

        quantile_list = list(list())
        for interval in sorted(credible_intervals, reverse=True):
            lower, upper = self._compute_quantile_range(interval)
            quantile_list.append([f"quantile_{lower}", f"quantile_{upper}"])

        flat_quantile_list = [
            quantile for sublist in quantile_list for quantile in sublist
        ]
        flat_quantile_list.append("quantile_50.0")
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

        data_df = data_df.rename(
            columns={
                "time": "time",
                "geo": "geo",
                **{v: k for k, v in q_mapping.items()},
            }
        )

        groups_df = data_df.groupby("geo")

        _time_format, _ = self._time_format(time, time_format)

        plot_index = 0
        for (geo_group_name, gdf), gl_kwargs, gf_kwargs in zip(
            groups_df, cycle(line_kwargs), cycle(fill_kwargs)
        ):
            ax = ax_list[plot_index]

            for (quantity_dis_label, quantity_label), l_kwargs, f_kwargs in zip(
                q_mapping.items(), cycle(line_kwargs), cycle(fill_kwargs)
            ):
                if kwarg_type == "geo":
                    l_kwargs = gl_kwargs
                    f_kwargs = gf_kwargs

                for ci_index, (upper, lower) in enumerate(quantile_list):
                    data_lower = transform(
                        pd.DataFrame({"value": gdf[quantity_dis_label][lower]})
                    )
                    data_upper = transform(
                        pd.DataFrame({"value": gdf[quantity_dis_label][upper]})
                    )

                    label = ""
                    if ci_index == (len(quantile_list) - 1):
                        label = label_format.format(
                            n=geo_group_name, q=quantity_label, c=credible_intervals
                        )

                    ax.fill_between(
                        gdf["time"],
                        data_lower["value"],
                        data_upper["value"],
                        label=label,
                        **f_kwargs,
                    )
                data_median = transform(
                    pd.DataFrame({"value": gdf[quantity_dis_label]["quantile_50.0"]})
                )

                median_label = f"{geo_group_name}: {quantity_label}: Median"
                ax.plot(
                    gdf["time"],
                    data_median["value"],
                    label=median_label,
                    zorder=100,
                    **l_kwargs,
                )

            ax_title_str = ax_title.format(n=geo_group_name, t=time.date_bounds)
            ax.set_title(ax_title_str)
            ax.tick_params(axis="x", labelrotation=45)

            ##Labels and Legend
            if legend == "on":
                leg = ax.legend()
                leg.set_zorder(2e10)
            elif legend == "outside":
                leg = ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
                leg.set_zorder(2e10)

            if ax.get_subplotspec().is_last_row():
                if _time_format == "date":
                    ax.set_xlabel("date")
                    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
                    ax.xaxis.set_major_locator(
                        AutoDateLocator(
                            minticks=6, maxticks=12, interval_multiples=True
                        )
                    )

                elif _time_format == "day":
                    ax.set_xlabel("day")
                elif _time_format == "tick":
                    ax.set_xlabel("tick")
                else:
                    ax.set_xlabel("time")

            plot_index += 1
            plot_index = plot_index % len(ax_list)

    def histogram(
        self,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantityStrategy | ParameterStrategy,
        *,
        hist_kwargs: list[dict] | None = None,
        ncols: int = 3,
        legend: LegendOption = "auto",
        time_format: TimeFormatOption = "auto",
        title: str | None = None,
        to_file: str | Path | None = None,
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ):
        """
        Produces a histogram plot of a filter output. This is a plot where
        a specific time instance is taken and plotted as a histogram.

        Parameters
        ----------
        geo :
            The geographic selection to make on the output data.
        time :
            The time selection to make on the output data. For
            this plot the time selection must be a single time instant.
            for instance you could use
            'rume.time_frame.select.days(100, 100).group("day").agg()'
            to create a histogram corresponding to a single day.
        quantity :
            The quantity selection to make on the output data.
        ncols :
            The number of columns in the resulting subplot matrix. The
            number of rows is set dynamically.
        hist_kwargs :
            A list of keyword arguments to be passed to the matplotlib function
            that draws the bin plot.
            See matplotlib documentation for the supported options.
        legend :
            Whether and how to draw the plot legend.

            - "auto" will draw the legend unless it would be too large
            - "on" forces the legend to be drawn
            - "off" forces the legend to not be drawn
            - "outside" forces the legend to be drawn next to the plot area
            (instead of inside it)
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
            num_nodes = self.output.rume.scope.nodes
            nrows = ceil(num_nodes / ncols)
            fig, axs = plt.subplots(
                nrows,
                ncols,
                figsize=(ncols * 5, nrows * 3),
                layout="constrained",
            )

            # Y-axis
            fig.supylabel("Density")

            # X-axis
            fig.supxlabel("Count")

            # Title
            if title is not None:
                fig.suptitle(t=title)

            # Legend
            if legend == "auto":
                # auto: show a legend if there are at most 5 realizations.
                legend = "on" if len(quantity.labels) <= 4 else "off"

            self.histogram_plt(
                axs,
                geo,
                time,
                quantity,
                legend=legend,
                hist_kwargs=hist_kwargs,
                label_format="{q}",
                time_format=time_format,
                transform=transform,
            )

            if to_file is None:
                plt.show()
            else:
                path = Path(to_file)
                fig.savefig(path)

        except:
            plt.close()
            raise

    def histogram_plt(
        self,
        axs: Axes | Iterable[Axes] | NDArray[Any],
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantityStrategy | ParameterStrategy,
        *,
        legend: LegendOption = "auto",
        hist_kwargs: list[dict] | None = None,
        kwarg_type="quantity",
        ax_title: str = "{n}: {t}",
        label_format="{n}: {q}: {t}",
        time_format: TimeFormatOption = "auto",
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ):
        """
        Draw histogram plots onto the array of matplotlib `Axes`, such as what is
        returned by matplotlib `subplots`. This is a variant of the method
        `histogram`.

        Parameters
        ----------
        axs:
            The array of matplotlib `Axes` on which to draw the plots.
        geo :
            The geographic selection to make on the output data.
        time :
            The time selection to make on the output data.
        quantity :
            The quantity selection to make on the output data.
        hist_kwargs :
            A list of keyword arguments to be passed to the matplotlib function
            that draws the bin plot.
        kwarg_type :
            A string describing whether hist_kwargs should iterate over
            the geo or quantity axis. Default is "quantity", specify "geo"
            for geo iteration.
        ax_title :
            Specifies the format of the title for the subplots.
            Defaults to {n}: {t} where n is the geo and t is the time.
        label_format :
            Specifies the label format for the legend.
            Defaults to {n}: {q}: {t} where n is the geo, q is the
            quantity, and t is the time.
        legend :
            Whether and how to draw the plot legend.
        time_format :
            Controls the formatting of the time axis (the horizontal axis).
        transform :
            Allows you to specify an arbitrary transform function for the source
            dataframe before we plot it.
        """
        if isinstance(axs, np.ndarray):
            ax_list = list(axs.flat)
        elif isinstance(axs, Axes):
            ax_list = [axs]
        else:
            ax_list = list(axs)
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

        data_df = data_df.rename(
            columns={
                "time": "time",
                "geo": "geo",
                **{v: k for k, v in q_mapping.items()},
            }
        )

        groups_df = data_df.groupby("geo")

        plot_index = 0
        ax_time_str = (
            f"{time.date_bounds[0].isoformat()}/{time.date_bounds[1].isoformat()}"
        )
        for (geo_group_name, gdf), gwargs in zip(groups_df, cycle(hist_kwargs)):
            ax = ax_list[plot_index]
            ax.tick_params(axis="x", labelrotation=45)
            ax_title_str = ax_title.format(
                n=geo_group_name,
                t=ax_time_str,
            )
            ax.set_title(ax_title_str)

            for (quantity_dis_label, quantity_label), kwargs in zip(
                q_mapping.items(), cycle(hist_kwargs)
            ):
                if kwarg_type == "geo":
                    kwargs = gwargs

                label = label_format.format(
                    n=geo_group_name,
                    q=quantity_label,
                    t=ax_time_str,
                )
                curr_kwargs = {"label": label, **kwargs}

                ax.hist(
                    transform(pd.DataFrame({"value": gdf[quantity_dis_label]})),
                    **curr_kwargs,
                )
            # Labels and Legend
            if legend == "on":
                ax.legend()
            elif legend == "outside":
                ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

            plot_index += 1
            plot_index = plot_index % len(ax_list)

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
        time_format: TimeFormatOption = "auto",
        title: str | None = None,
        to_file: str | Path | None = None,
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> None:
        """
        Produces a line plot of a filter output. This is a plot where
        a single line is plotted for each location in the GeoStrategy.

        Parameters
        ----------
        realization :
            A realization aggregation which returns some number of lines per
            location in the GeoStrategy.
        geo :
            The geographic selection to make on the output data.
        time :
            The time selection to make on the output data.
        quantity :
            The quantity selection to make on the output data.
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
            we will cycle through the list as many times as needed.
            See matplotlib documentation for the supported options.
        time_format :
            Controls the formatting of the time axis (the horizontal axis);
            "auto" will use the format defined by the grouping of the `time` parameter,
            "date" attempts to display calendar dates,
            "day" attempts to display days numerically indexed from the start of the
            simulation with the first day being 0.
            If the system cannot convert to the requested time format, this argument
            may be ignored.
        label_format :
            A format for the items displayed in the legend;
            the string will be used in a call to `format()`
            with the replacement variables `{n}` for the name of the geo node,
            `{q}` for the name of the quantity, and '{m}' for the aggregation name
            corresponding to the realization aggregation.
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
        realization: RealizationSelection | RealizationAggregation,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantityStrategy | ParameterStrategy,
        *,
        label_format: str = "{n}: {q}: {m}",
        line_kwargs: list[dict] | None = None,
        time_format: TimeFormatOption = "auto",
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> list[Line2D]:
        """
        Draw lines onto the matplotlib `Axes`. This is a variant of the method
        `line`.

        Parameters
        ----------
        ax:
            The `Axes` on which to draw.
        realization :
            A realization aggregation which returns some number of lines per
            location in the GeoStrategy.
        geo :
            The geographic selection to make on the output data.
        time :
            The time selection to make on the output data.
        quantity :
            The quantity selection to make on the output data.
        line_kwargs :
            A list of keyword arguments to be passed to the matplotlib function
            that draws each line.
        time_format :
            Controls the formatting of the time axis (the horizontal axis).
        label_format :
            A format for the items displayed in the legend.
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

        data_df = data_df.rename(
            columns={
                "time": "time",
                "geo": "geo",
                **{v: k for k, v in q_mapping.items()},
            }
        )

        lines = list[Line2D]()

        geo_groups = data_df.groupby("geo")

        line_index = 0
        for group_name, gdf in geo_groups:
            for quantity_dis_label, quantity_label in q_mapping.items():
                qdf = gdf[quantity_dis_label]
                qdf = qdf.melt(var_name="metric")

                metric_groups = qdf.groupby("metric")

                for metric_name, mdf in metric_groups:
                    kwargs = line_kwargs[line_index % len(line_kwargs)]
                    label = label_format.format(
                        n=group_name, q=quantity_label, m=metric_name
                    )
                    curr_kwargs = {"label": label, **kwargs}
                    data = transform(mdf)
                    ls = ax.plot(gdf["time"], data["value"], **curr_kwargs)
                    lines.extend(ls)
                    line_index += 1

        return lines

    def kde(
        self,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantityStrategy | ParameterStrategy,
        *,
        line_kwargs: list[dict] | None = None,
        ncols: int = 3,
        legend: LegendOption = "auto",
        delta_t: float | None = None,
        bandwidth: float | str = "scott",
        time_format: TimeFormatOption = "auto",
        title: str | None = None,
        to_file: str | Path | None = None,
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ):
        """
        Produces a kernel density plot of a filter output. This is a plot where
        a specific time instance is taken and plotted as a KDE.

        Parameters
        ----------
        geo :
            The geographic selection to make on the output data.
        time :
            The time selection to make on the output data. For
            this plot the time selection must be a single time instant.
            for instance you could use
            'rume.time_frame.select.days(100, 100).group("day").agg()'
            to create a histogram corresponding to a single day.
        quantity :
            The quantity selection to make on the output data.
        ncols :
            The number of columns in the resulting subplot matrix. The
            number of rows is set dynamically.
        line_kwargs :
            A list of keyword arguments to be passed to the matplotlib function
            that draws the kde plot.
        legend :
            Whether and how to draw the plot legend.
        delta_t :
            Specifies the interval at which to sample the kernelized density.
            Defaults to 1/100.
        bandwidth :
            The bandwidth for the convolution kernel. Specify either a float or
            a string for an adaptive scheme. Options are "scott", "silverman".
            Defaults to "scott". See scipy.stats.gaussian_kde for details.
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
            num_nodes = self.output.rume.scope.nodes
            nrows = ceil(num_nodes / ncols)
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(ncols * 5, nrows * 3),
                layout="constrained",
            )

            # Y-axis
            fig.supylabel("Density")

            # X-axis
            fig.supxlabel("Value")

            # Title
            fig.suptitle(t=title)  # type: ignore

            # Legend
            if legend == "auto":
                # auto: show a legend if there are at most 5 realizations.
                legend = "on" if len(quantity.labels) <= 4 else "off"

            self.kde_plt(
                axes,
                geo,
                time,
                quantity,
                line_kwargs=line_kwargs,
                delta_t=delta_t,
                legend=legend,
                bandwidth=bandwidth,
                label_format="{q}",
                time_format=time_format,
                transform=transform,
            )

            if to_file is None:
                plt.show()
            else:
                path = Path(to_file)
                fig.savefig(path)

        except:
            plt.close()
            raise

    def kde_plt(
        self,
        axs: Axes | Iterable[Axes] | NDArray[Any],
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantityStrategy | ParameterStrategy,
        line_kwargs: list[dict] | None = None,
        ax_title: str = "{n}: {t}",
        kwarg_type: str = "quantity",
        bandwidth: float | str = "scott",
        delta_t: float | None = None,
        label_format: str = "{n}: {q}: {t}",
        legend: LegendOption = "auto",
        time_format: TimeFormatOption = "auto",
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ):
        """
        Draw kde plots onto the array of matplotlib `Axes`, such as what is
        returned by matplotlib `subplots`. This is a variant of the method
        `kde`.

        Parameters
        ----------
        axs:
            The array of matplotlib `Axes` on which to draw the plots.
        geo :
            The geographic selection to make on the output data.
        time :
            The time selection to make on the output data.
        quantity :
            The quantity selection to make on the output data.
        line_kwargs :
            A list of keyword arguments to be passed to the matplotlib function
            that draws the lines.
        ax_title :
            A format string specifying the format of the title.
            Defaults to "{n}: {t}", where n is the geo and t is the time.
        kwarg_type :
            A string describing whether hist_kwargs should iterate over
            the geo or quantity axis. Default is "quantity", specify "geo"
            for geo iteration.
        bandwidth :
            The bandwidth for the convolution kernel. Specify either a float or
            a string for an adaptive scheme. Options are "scott", "silverman".
            Defaults to "scott". See scipy.stats.gaussian_kde for details.
        delta_t :
            Specifies the interval at which to sample the kernelized density.
            Defaults to 1/100.
        ax_title :
            Specifies the format of the title for the subplots.
            Defaults to {n}: {t} where n is the geo and t is the time.
        label_format :
            Specifies the label format for the legend.
            Defaults to {n}: {q}: {t} where n is the geo, q is the
            quantity, and t is the time.
        legend :
            Whether and how to draw the plot legend.
        time_format :
            Controls the formatting of the time axis (the horizontal axis).
        transform :
            Allows you to specify an arbitrary transform function for the source
            dataframe before we plot it.
        """

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

        if isinstance(axs, np.ndarray):
            ax_list = list(axs.flat)
        elif isinstance(axs, Axes):
            ax_list = [axs]
        else:
            ax_list = list(axs)

        if line_kwargs is None or len(line_kwargs) == 0:
            line_kwargs = [{}]

        if transform is None:
            transform = identity

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

        data_df = data_df.rename(
            columns={
                "time": "time",
                "geo": "geo",
                **{v: k for k, v in q_mapping.items()},
            }
        )

        groups_df = data_df.groupby("geo")

        # Plotting
        ax_time_str = (
            f"{time.date_bounds[0].isoformat()}/{time.date_bounds[1].isoformat()}"
        )

        plot_index = 0
        for (geo_group_name, gdf), gwargs in zip(groups_df, cycle(line_kwargs)):
            ax = ax_list[plot_index]
            ax.tick_params(axis="x", labelrotation=45)
            ax_title_str = ax_title.format(
                n=geo_group_name,
                t=ax_time_str,
            )
            ax.set_title(ax_title_str)

            for (quantity_dis_label, quantity_label), kwargs in zip(
                q_mapping.items(), cycle(line_kwargs)
            ):
                if kwarg_type == "geo":
                    kwargs = gwargs

                label = label_format.format(
                    n=geo_group_name,
                    q=quantity_label,
                    t=f"{time.date_bounds[0].isoformat()}:{time.date_bounds[1].isoformat()}",
                )
                curr_kwargs = {"label": label, **kwargs}

                data = (
                    transform(pd.DataFrame({"value": gdf[quantity_dis_label]}))
                    .to_numpy()
                    .squeeze()
                )

                d_max = data.max()
                d_min = data.min()
                if delta_t is None:
                    delta_t = (d_max - d_min) / 100

                eval_range = np.arange(d_min, d_max + delta_t, delta_t)
                kde = gaussian_kde(data, bw_method=bandwidth)
                ax.plot(
                    eval_range,
                    kde.evaluate(eval_range),
                    **curr_kwargs,
                )

            ##Labels and Legend
            if legend == "on":
                ax.legend()
            elif legend == "outside":
                ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

            plot_index += 1
            plot_index = plot_index % len(ax_list)
