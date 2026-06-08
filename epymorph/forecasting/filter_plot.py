from datetime import timedelta
from itertools import chain, cycle
from math import ceil
from pathlib import Path
from typing import (
    Callable,
    Literal,
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
    ```
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
        line_kwargs: list[dict] = [{}],
        time_format: TimeFormatOption = "auto",
        label_format: str = "{q}",
        title: str | None = None,
        to_file: str | Path | None = None,
        transform: Callable[[pd.DataFrame], pd.DataFrame] = identity,
    ) -> None:
        """
        Produces a spaghetti plot of a filter output. This is a plot where
        each realization corresponds to a specific line on a plot.

        Parameters
        ----------
        realization:
            A realization selection to make on the output data,
            you can either select all vthe realizations or a random subset.

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

        label_format :
            A format for the items displayed in the legend;
            the string will be used in a call to `format()`
            with the replacement variable {q}` for the name of the quantity.
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
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(ncols * 5, nrows * 3),
                sharex=sharex,
                layout="constrained",
            )

            # Y-axis
            fig.supylabel("count")

            # Title
            fig.suptitle(t=title)  # type: ignore

            # Legend
            if legend == "auto":
                # auto: show a legend if there are at most 4 quantities.
                legend = "on" if len(quantity.labels) <= 4 else "off"

            # Call the spaghetti plot function, this returns the lines
            lines = self.spaghetti_plt(  # noqa: F841
                axes,
                realization,
                geo,
                time,
                quantity,
                "quantity",
                legend,
                line_kwargs,
                time_format,
                label_format,
                transform,
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
        axes: NDArray[Axes],  # type: ignore
        realizations: RealizationSelection,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantityStrategy | ParameterStrategy,
        kwarg_type: str = "quantity",
        legend: LegendOption = "auto",
        line_kwargs: list[dict] = [{}],
        time_format: TimeFormatOption = "auto",
        label_format: str = "{n}: {q}",
        transform: Callable[[pd.DataFrame], pd.DataFrame] = identity,
    ) -> list[Line2D]:
        data_df = munge_pipeline_output(self.output, realizations, geo, time, quantity)

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

        # Plotting
        axes = axes.flatten() if len(axes.shape) > 1 else axes
        _time_format, _ = self._time_format(time, time_format)

        lines = list[Line2D]()
        plot_index = 0
        for (geo_group_name, gdf), gkwargs in zip(groups_df, cycle(line_kwargs)):
            quantity_groups = gdf.melt(
                id_vars=["realization", "time", "geo"], var_name="quantity"
            ).groupby("quantity")

            # Line kwargs cycle over the quantity axis,
            # not setting colors for individual lines!
            for (quantity_group_name, qdf), kwargs in zip(
                quantity_groups,
                cycle(line_kwargs),  # type: ignore
            ):
                if kwarg_type == "geo":
                    kwargs = gkwargs

                q_name = q_mapping[quantity_group_name]  # type: ignore
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
                    data = transform(rdf.assign(quantity=q_name))  # type: ignore
                    ls = axes[plot_index].plot(
                        rdf["time"], data["value"], **plot_kwargs
                    )
                    lines.extend(ls)

            ##Labels and Legend
            axes[plot_index].tick_params(axis="x", labelrotation=45)
            if legend == "on":
                axes[plot_index].legend()
            elif legend == "outside":
                axes[plot_index].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

            if axes[plot_index].get_subplotspec().is_last_row():  # type: ignore
                if _time_format == "date":
                    axes[plot_index].set_xlabel("date")
                    axes[plot_index].xaxis.set_major_formatter(
                        DateFormatter("%Y-%m-%d")
                    )
                    axes[plot_index].xaxis.set_major_locator(
                        AutoDateLocator(
                            minticks=6, maxticks=12, interval_multiples=True
                        )
                    )

                elif _time_format == "day":
                    axes[plot_index].set_xlabel("day")
                elif _time_format == "tick":
                    axes[plot_index].set_xlabel("tick")
                else:
                    axes[plot_index].set_xlabel("time")

            plot_index += 1
            plot_index = plot_index % len(axes)

        return lines

    def quantiles(
        self,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantityStrategy | ParameterStrategy,
        credible_intervals: list[float] = [95.0],
        *,
        sharex: bool = True,
        ncols: int = 3,
        legend: LegendOption = "auto",
        fill_kwargs: list[dict] = [{}],
        line_kwargs: list[dict] = [{}],
        time_format: TimeFormatOption = "auto",
        title: str | None = None,
        to_file: str | Path | None = None,
        transform: Callable[[pd.DataFrame], pd.DataFrame] = identity,
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
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(ncols * 5, nrows * 3),
                sharex=sharex,
                layout="constrained",
            )

            # Convert to array to keep the type consistent
            axes = np.array([axes]) if ncols == 1 else axes

            # Y-axis
            fig.supylabel("count")

            # Title
            fig.suptitle(t=title)  # type: ignore

            # Legend
            if legend == "auto":
                # auto: show a legend if there are at most 5 realizations.
                legend = "on" if len(quantity.labels) <= 4 else "off"

            self.quantiles_plt(
                axes,
                geo,
                time,
                quantity,
                credible_intervals,
                legend,
                "quantity",
                fill_kwargs,
                line_kwargs,
                time_format,
                transform,
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
        axes: NDArray[Axes],  # type: ignore
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantityStrategy | ParameterStrategy,
        credible_intervals: list[float] = [95],
        legend: LegendOption = "on",
        kwarg_type: str = "quantity",
        fill_kwargs: list[dict] = [{}],
        line_kwargs: list[dict] = [{}],
        time_format: TimeFormatOption = "auto",
        transform: Callable[[pd.DataFrame], pd.DataFrame] = identity,
    ):
        # Generate list of quantiles from the CIs
        quantile_list = list(list())
        for interval in sorted(credible_intervals, reverse=True):
            lower, upper = self._compute_quantile_range(interval)
            quantile_list.append([f"quantile_{lower}", f"quantile_{upper}"])

        # Flatten and append the median
        flat_quantile_list = list(chain.from_iterable(quantile_list))
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

        # Group the data
        groups_df = data_df.groupby("geo")

        # Plotting
        axes = axes.flatten() if len(axes.shape) > 1 else axes
        _time_format, _ = self._time_format(time, time_format)

        plot_index = 0
        for (geo_group_name, gdf), gl_kwargs, gf_kwargs in zip(
            groups_df, cycle(line_kwargs), cycle(fill_kwargs)
        ):
            for quantity_name, l_kwargs, f_kwargs in zip(
                quantity.labels, cycle(line_kwargs), cycle(fill_kwargs)
            ):
                if kwarg_type == "geo":
                    l_kwargs = gl_kwargs
                    f_kwargs = gf_kwargs

                for ci_index, (upper, lower) in enumerate(quantile_list):
                    data_lower = transform(
                        pd.DataFrame({"value": gdf[quantity_name][lower]})
                    )
                    data_upper = transform(
                        pd.DataFrame({"value": gdf[quantity_name][upper]})
                    )

                    label = ""
                    if ci_index == (len(quantile_list) - 1):
                        label = (
                            f"{geo_group_name}: {quantity_name}: {credible_intervals}"
                        )

                    axes[plot_index].fill_between(
                        gdf["time"],
                        data_lower["value"],
                        data_upper["value"],
                        label=label,
                        **f_kwargs,
                    )
                data_median = transform(
                    pd.DataFrame({"value": gdf[quantity_name]["quantile_50.0"]})
                )

                axes[plot_index].plot(
                    gdf["time"],
                    data_median["value"],
                    label=f"Median of {quantity_name}",
                    zorder=100,
                    **l_kwargs,
                )

            axes[plot_index].tick_params(axis="x", labelrotation=45)

            ##Labels and Legend
            if legend == "on":
                axes[plot_index].legend()
            elif legend == "outside":
                axes[plot_index].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

            if axes[plot_index].get_subplotspec().is_last_row():
                if _time_format == "date":
                    axes[plot_index].set_xlabel("date")
                    axes[plot_index].xaxis.set_major_formatter(
                        DateFormatter("%Y-%m-%d")
                    )
                    axes[plot_index].xaxis.set_major_locator(
                        AutoDateLocator(
                            minticks=6, maxticks=12, interval_multiples=True
                        )
                    )

                elif _time_format == "day":
                    axes[plot_index].set_xlabel("day")
                elif _time_format == "tick":
                    axes[plot_index].set_xlabel("tick")
                else:
                    axes[plot_index].set_xlabel("time")

            plot_index += 1
            plot_index = plot_index % len(axes)

    def histogram(
        self,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantityStrategy | ParameterStrategy,
        *,
        hist_kwargs: list[dict] = [{}],
        ncols: int = 3,
        legend: LegendOption = "auto",
        time_format: TimeFormatOption = "auto",
        title: str | None = None,
        label_format: str = "{q}: {t}",
        to_file: str | Path | None = None,
        transform: Callable[[pd.DataFrame], pd.DataFrame] = identity,
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
        sharex :
            Whether or not the subplots should share the x-axis ticks.
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
        ```
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
            fig.supxlabel("Count")

            # Title
            fig.suptitle(t=title)  # type: ignore

            # Legend
            if legend == "auto":
                # auto: show a legend if there are at most 5 realizations.
                legend = "on" if len(quantity.labels) <= 4 else "off"

            self.histogram_plt(
                axes,
                geo,
                time,
                quantity,
                hist_kwargs,
                "quantity",
                label_format,
                legend,
                time_format,
                transform,  # type: ignore
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
        axes: NDArray[Axes],  # type: ignore
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantityStrategy | ParameterStrategy,
        hist_kwargs: list[dict] = [{}],
        kwarg_type="quantity",
        label_format="{n}: {q}: {t}",
        legend: LegendOption = "auto",
        time_format: TimeFormatOption = "auto",
        transform: Callable[[pd.DataFrame], pd.DataFrame] = identity,
    ):
        realizations_agg = self.output.select.all()
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

        groups_df = data_df.groupby("geo")

        # Plotting
        axes = axes.flatten() if len(axes.shape) > 1 else axes

        plot_index = 0
        for (geo_group_name, gdf), gwargs in zip(groups_df, cycle(hist_kwargs)):
            axes[plot_index].tick_params(axis="x", labelrotation=45)

            for quantity_name, kwargs in zip(quantity.labels, cycle(hist_kwargs)):
                if kwarg_type == "geo":
                    kwargs = gwargs

                label = label_format.format(
                    n=geo_group_name, q=quantity_name, t=time.date_bounds
                )
                curr_kwargs = {"label": label, **kwargs}

                axes[plot_index].hist(
                    transform(gdf[quantity_name].to_frame()),
                    **curr_kwargs,
                )

            ##Labels and Legend
            if legend == "on":
                axes[plot_index].legend()
            elif legend == "outside":
                axes[plot_index].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

            plot_index += 1
            plot_index = plot_index % len(axes)

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
        line_kwargs :
            A list of keyword arguments to be passed to the matplotlib function
            that draws each line. If the list contains less items than there are lines,
            we will cycle through the list as many times as needed. Lines are drawn
            in the order defined by the `ordering` parameter.
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
            essentially -- with a dataframe that contain                print(gdf[quantity_name].to_frame().shape)s just the data for that group.
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
        line_kwargs: list[dict] = [{}],
        ncols: int = 3,
        legend: LegendOption = "auto",
        time_format: TimeFormatOption = "auto",
        title: str | None = None,
        label_format: str = "{q}: {t}",
        to_file: str | Path | None = None,
        transform: Callable[[pd.DataFrame], pd.DataFrame] = identity,
    ):
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
            fig.supxlabel("Count")

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
                line_kwargs,
                "quantity",
                label_format,
                legend,
                time_format,
                transform,  # type: ignore
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
        axes: NDArray[Axes],  # type: ignore
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantityStrategy | ParameterStrategy,
        line_kwargs: list[dict] = [{}],
        kwarg_type: str = "quantity",
        bandwidth: float|str = "scott",
        label_format="{n}: {q}: {t}",
        legend: LegendOption = "auto",
        time_format: TimeFormatOption = "auto",
        transform: Callable[[pd.DataFrame], pd.DataFrame] = identity,
    ):
        realizations_agg = self.output.select.all()
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

        groups_df = data_df.groupby("geo")

        # Plotting
        axes = axes.flatten() if len(axes.shape) > 1 else axes

        plot_index = 0
        for (geo_group_name, gdf), gwargs in zip(groups_df, cycle(line_kwargs)):
            axes[plot_index].tick_params(axis="x", labelrotation=45)

            for quantity_name, kwargs in zip(quantity.labels, cycle(line_kwargs)):
                if kwarg_type == "geo":
                    kwargs = gwargs

                label = label_format.format(
                    n=geo_group_name, q=quantity_name, t=time.date_bounds
                )
                curr_kwargs = {"label": label, **kwargs}

                data = transform(gdf[quantity_name].to_frame()).to_numpy().squeeze()

                delta_t = 0.01
                eval_range = np.arange(data.min(),data.max() + delta_t ,delta_t)
                kde = gaussian_kde(data,bw_method = bandwidth)
                axes[plot_index].plot(
                    eval_range,
                    kde.evaluate(eval_range),
                    **curr_kwargs,
                )

            ##Labels and Legend
            if legend == "on":
                axes[plot_index].legend()
            elif legend == "outside":
                axes[plot_index].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

            plot_index += 1
            plot_index = plot_index % len(axes)
