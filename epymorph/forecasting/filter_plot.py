from datetime import timedelta
from itertools import cycle
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

from epymorph.compartment_model import (
    QuantityAggregation,
    QuantitySelection,
)
from epymorph.forecasting.munge_realizations import (
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


class PlotRendererFilter:
    """
    Provides methods for rendering an output in plot form.

    Most commonly, you will use `PlotRendererFilter` starting from a filter output object
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
        realizations: RealizationSelection,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantitySelection | QuantityAggregation,
        *,
        sharex: bool = True,
        ncols: int = 3,
        legend: LegendOption = "auto",
        line_kwargs: list[dict] | None = None,
        time_format: TimeFormatOption = "auto",
        label_format: str = "{q}",
        title: str | None = None,
        to_file: str | Path | None = None,
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> None:
        if not isinstance(realizations, RealizationSelection):
            raise ValueError("Spaghetti plots require a RealizationSelection.")

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

            if line_kwargs is None:
                line_kwargs = [{}]

            if transform is None:
                transform = identity

            # Y-axis
            fig.supylabel("count")

            # Title
            fig.suptitle(t=title)  # type: ignore

            # Legend
            if legend == "auto":
                # auto: show a legend if there are at most 4 quantities.
                legend = "on" if len(quantity.labels) <= 4 else "off"

            lines = self.spaghetti_plt(
                axes,
                realizations,
                geo,
                time,
                quantity,
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
        axes: NDArray[Axes], # type: ignore
        realizations: RealizationSelection,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantitySelection | QuantityAggregation,
        legend: LegendOption = "auto",
        line_kwargs: list[dict] | None = None,
        time_format: TimeFormatOption = "auto",
        label_format: str = "{q}",
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
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

        groups_df = data_df.set_axis(
            ["realization", "time", "geo", *q_mapping.keys()], axis=1
        ).groupby("geo")

        # Plotting
        axes = axes.flatten()

        _time_format, _ = self._time_format(time, time_format)

        lines = list[Line2D]()
        plot_index = 0
        for geo_group_name, gdf in groups_df:
            axes[plot_index].set_title(f"{geo_group_name}")
            axes[plot_index].tick_params(axis="x", labelrotation=45)

            quantity_groups = gdf.melt(
                id_vars=["realization", "time", "geo"], var_name="quantity"
            ).groupby("quantity")

            for (quantity_group_name, qdf), kwargs in zip(
                quantity_groups, cycle(line_kwargs)
            ):
                q_name = q_mapping[quantity_group_name]  # type: ignore
                label = label_format.format(q=q_name)
                curr_kwargs = {"label": label, **kwargs}

                realization_groups = qdf.groupby("realization")

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

        return lines

    def quantiles(
        self,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantitySelection | QuantityAggregation,
        credible_intervals: list[float] = [95],
        *,
        sharex: bool = True,
        ncols: int = 3,
        legend: LegendOption = "auto",
        fill_kwargs: list[dict] | None = None,
        line_kwargs: list[dict] | None = None,
        time_format: TimeFormatOption = "auto",
        title: str | None = None,
        to_file: str | Path | None = None,
    ):
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

            if fill_kwargs is None:
                fill_kwargs = [{}]

            if line_kwargs is None:
                line_kwargs = [{}]

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
                fill_kwargs,
                line_kwargs,
                time_format,
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
        axes: NDArray[Axes], # type: ignore
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantitySelection | QuantityAggregation,
        credible_intervals: list[float],
        legend: LegendOption,
        fill_kwargs: list[dict],
        line_kwargs: list[dict],
        time_format: TimeFormatOption,
    ):
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

        groups_df = data_df.groupby("geo")

        # Plotting
        axes = axes.flatten()

        _time_format, _ = self._time_format(time, time_format)

        plot_index = 0
        for geo_group_name, gdf in groups_df:
            axes[plot_index].set_title(f"{geo_group_name}")
            axes[plot_index].tick_params(axis="x", labelrotation=45)

            for quantity_name, l_kwargs in zip(quantity.labels, cycle(line_kwargs)):
                for (ci_index, (upper, lower)), f_kwargs in zip(
                    enumerate(quantile_list), cycle(fill_kwargs)
                ):
                    ci_label = (
                        ""
                        if len(fill_kwargs) == 1
                        else f"{credible_intervals[ci_index]}% CI of {quantity_name}"
                    )

                    axes[plot_index].fill_between(
                        gdf["time"],
                        gdf[quantity_name][lower],
                        gdf[quantity_name][upper],
                        label=ci_label,
                        **f_kwargs,
                    )
                axes[plot_index].plot(
                    gdf["time"],
                    gdf[quantity_name]["quantile_50.0"],
                    label=f"Median of {quantity_name}",
                    zorder=100,
                    **l_kwargs,
                )

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

    def histogram(
        self,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantitySelection | QuantityAggregation,
        *,
        hist_kwargs: list[dict] | None = None,
        ncols: int = 3,
        legend: LegendOption = "auto",
        time_format: TimeFormatOption = "auto",
        title: str | None = None,
        to_file: str | Path | None = None,
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

            if hist_kwargs is None:
                hist_kwargs = [{}]

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
                axes, geo, time, quantity, legend, hist_kwargs, time_format
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
        axes: NDArray[Axes], # type: ignore
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection,
        quantity: QuantitySelection | QuantityAggregation,
        legend: LegendOption,
        hist_kwargs,
        time_format: TimeFormatOption,
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
        axes = axes.flatten()

        plot_index = 0
        for geo_group_name, gdf in groups_df:
            axes[plot_index].set_title(f"{geo_group_name}")
            axes[plot_index].tick_params(axis="x", labelrotation=45)

            for quantity_name, kwargs in zip(quantity.labels, cycle(hist_kwargs)):
                axes[plot_index].hist(
                    gdf[quantity_name],
                    label=f"Histogram of {quantity_name} at Time: {gdf['time'].iloc[0]}",
                    **kwargs,
                )

            ##Labels and Legend
            if legend == "on":
                axes[plot_index].legend()
            elif legend == "outside":
                axes[plot_index].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

            plot_index += 1
