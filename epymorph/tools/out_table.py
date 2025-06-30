"""Tools for rendering tables from epymorph simulation output data."""

import dataclasses
from typing import Callable, Literal, Sequence, overload
from warnings import warn

import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
from sparklines import sparklines
from sympy import Symbol

from epymorph.compartment_model import (
    CompartmentDef,
    EdgeDef,
    QuantityAggregation,
    QuantitySelection,
)
from epymorph.geography.scope import GeoAggregation, GeoSelection
from epymorph.time import NBins, TimeAggregation, TimeSelection
from epymorph.tools.data import Output, munge
from epymorph.util import filter_unique

OrderingOption = Literal["location", "quantity"]
"""Options for table row ordering."""

FormatOption = Literal["dataframe", "string", "print"]
"""Options for table render output format."""


def _to_sparklines(values):
    """Produce a sparkline result, as a Pandas aggregation function."""
    # NOTE: I tried using the same max value for each quantity;
    # in a way this improves comparison between locations, but means that
    # large locations "drown out" smaller locations, hiding the dynamics.
    # I made the judgement call that the dynamics were more important here.
    # If you want to understand relative scale you should use a line graph.
    # NOTE: I'm checking for max == 0 here because if min and max are both zero,
    # sparkline draws a mid-height box instead of a min-height box;
    # with this adjustment, all-zero data looks like zeros.
    maximum = values.max()
    if maximum == 0:
        maximum = 1.0
    (spark,) = sparklines(values, minimum=0, maximum=maximum, num_lines=1)
    return spark


def _process_output(
    out: Output,
    geo: GeoSelection | GeoAggregation,
    time: TimeSelection | TimeAggregation,
    quantity: QuantitySelection | QuantityAggregation,
    ordering: OrderingOption,
    column_names: Sequence[str],
    process_groups: Callable[[DataFrameGroupBy], pd.DataFrame],
) -> pd.DataFrame:
    # The different tables share a lot of data processing logic.
    # The only significant difference is how to turn
    # the grouped (geo,quantity) time series into values.
    # This difference is factored out by taking a lambda for processing groups.

    # Prepare a consistent sort order.
    # By default: we sort by location (alphanumeric by node_id),
    # then by quantity in IPM declaration order (compartments then events).
    # Passing ordering "quantity" just flips the ordering of two columns.

    # Compute the grouped time series.
    data_df = munge(out, geo, time, quantity)

    # Before melting, disambiguate any quantities with the same name.
    q_mapping = quantity.disambiguate_groups()

    groups_df = (
        data_df.set_axis(["time", "geo", *q_mapping.keys()], axis=1)
        .melt(id_vars=["time", "geo"], var_name="quantity")
        .drop(columns="time")
        .groupby(["geo", "quantity"], sort=False)
    )

    # Process the groups into values.
    result_df = process_groups(groups_df)

    # Set column names.
    result_df = result_df.set_axis(["geo", "quantity", *column_names], axis=1)

    # If this is a multistrata model add a column for "other strata";
    # for any quantities that are meta edge events, the value of this column
    # are which other strata are involved (by examining the rate expression).
    # For compartments and events that are purely intrastrata, this should be blank.
    if quantity.ipm.is_multistrata:
        strata = quantity.ipm.strata

        def which_strata(s: Symbol) -> str | None:
            for x in strata:
                if str(s).endswith(f"_{x}"):
                    return x
            return None

        def calc_other_strata(e: EdgeDef) -> str:
            # If we can't tell which other strata are involved,
            # just return empty string.
            own_strata = which_strata(e.compartment_from)
            if own_strata is None:
                return ""
            other_strata = filter_unique(
                strata
                for strata in (
                    which_strata(x)
                    for x in e.rate.free_symbols
                    if isinstance(x, Symbol)
                )
                if strata is not None and strata != own_strata
            )
            return ",".join(other_strata)

        map_other_strata = {
            name: calc_other_strata(q) if isinstance(q, EdgeDef) else ""
            for name, q in zip(q_mapping.keys(), quantity.selected)
        }
        other_strata = result_df["quantity"].apply(lambda x: map_other_strata[x])
        result_df.insert(2, column="other strata", value=other_strata)

    # Sort rows.
    def sort_key(series):
        if series.name == "quantity":
            quantity_order = {s: i for i, s in enumerate(q_mapping.keys())}
            return series.apply(lambda x: quantity_order[x])
        else:
            return series

    sort_cols = ["quantity", "geo"] if ordering == "quantity" else ["geo", "quantity"]
    result_df = result_df.sort_values(sort_cols, key=sort_key, ignore_index=True)

    # Restore quantity names (now in rows).
    qs = result_df["quantity"].apply(lambda x: q_mapping[x])
    result_df = result_df.assign(quantity=qs)

    # If we have a way to do this, replace node IDs with friendly labels.
    result_scope = geo.to_scope()
    if (labels := result_scope.labels_option) is not None:
        geo_map = dict(zip(result_scope.node_ids, labels))
        result_df["geo"] = result_df["geo"].apply(lambda x: geo_map[x])

    return result_df


class TableRenderer:
    """
    Provides a number of methods for rendering an output in tabular form.

    Most commonly, you will use TableRenderer starting from a simulation output object
    that supports it:

    ```python
    out = BasicSimulation(rume).run()
    out.table.quantiles(...)
    ```

    Parameters
    ----------
    output :
        The output the renderer will use.
    """

    output: Output
    """The output the renderer will use."""

    def __init__(self, output: Output):
        self.output = output

    def _format_output(self, result_df: pd.DataFrame, result_format: FormatOption):
        """Produce output according to the given format setting."""
        match result_format:
            case "dataframe":
                return result_df
            case "string":
                return result_df.to_string()
            case "print":
                print(result_df.to_string(index=False))  # noqa: T201
                return None
            case x:
                msg = f"Invalid result_format: {x}"
                raise ValueError(msg)

    @overload
    def quantiles(
        self,
        quantiles: Sequence[float],
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantitySelection | QuantityAggregation,
        *,
        ordering: OrderingOption = "location",
        result_format: Literal["dataframe"] = "dataframe",
        column_names: Sequence[str] | None = None,
    ) -> pd.DataFrame: ...

    @overload
    def quantiles(
        self,
        quantiles: Sequence[float],
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantitySelection | QuantityAggregation,
        *,
        ordering: OrderingOption = "location",
        result_format: Literal["string"],
        column_names: Sequence[str] | None = None,
    ) -> str: ...

    @overload
    def quantiles(
        self,
        quantiles: Sequence[float],
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantitySelection | QuantityAggregation,
        *,
        ordering: OrderingOption = "location",
        result_format: Literal["print"],
        column_names: Sequence[str] | None = None,
    ) -> None: ...

    def quantiles(
        self,
        quantiles: Sequence[float],
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantitySelection | QuantityAggregation,
        *,
        ordering: OrderingOption = "location",
        result_format: FormatOption = "dataframe",
        column_names: Sequence[str] | None = None,
    ) -> pd.DataFrame | str | None:
        """
        Render a table showing time-series quantiles for the given selections.

        Parameters
        ----------
        quantiles :
            The list of quantiles to calculate and display, in the range [0,1].
        geo :
            The geographic selection to make on the output data.
        time :
            The time selection to make on the output data.
        quantity :
            The quantity selection to make on the output data.
        ordering :
            Controls the ordering of rows in the result;
            both location and quantity are used to sort the resulting rows,
            this just decides which gets priority.
        result_format :
            Controls the type of the result of this method;
            "dataframe" returns a Pandas dataframe,
            "string" returns the stringified table, and
            "print" just prints the stringified table directly and returns `None`.
        column_names :
            Overrides the default names of the quantiles columns;
            by default, this is just the quantile value itself.

        Returns
        -------
        :
            Output according to the value of the `result_format` parameter.
        """
        _quantiles = np.array(quantiles)
        if len(quantiles) == 0:
            err = "Please provide a list of quantiles to calculate."
            raise ValueError(err)
        if np.any(_quantiles < 0) or np.any(_quantiles > 1):
            err = "Invalid quantiles: must be in the range [0.0, 1.0]"
            raise ValueError(err)
        if column_names is None:
            column_names = [str(x) for x in quantiles]

        result_df = _process_output(
            self.output,
            geo,
            time,
            quantity,
            ordering,
            column_names,
            lambda df: (
                # `pivot` is more appropriate than `pivot_table` because we don't want
                # aggregation to happen accidentally
                df.quantile(_quantiles)  # noqa: PD010
                .reset_index()
                .pivot(index=["geo", "quantity"], columns=["level_2"])
                .reset_index()
            ),
        )
        return self._format_output(result_df, result_format)

    @overload
    def range(
        self,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantitySelection | QuantityAggregation,
        *,
        ordering: OrderingOption = "location",
        result_format: Literal["dataframe"] = "dataframe",
    ) -> pd.DataFrame: ...

    @overload
    def range(
        self,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantitySelection | QuantityAggregation,
        *,
        ordering: OrderingOption = "location",
        result_format: Literal["string"],
    ) -> str: ...

    @overload
    def range(
        self,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantitySelection | QuantityAggregation,
        *,
        ordering: OrderingOption = "location",
        result_format: Literal["print"],
    ) -> None: ...

    def range(
        self,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantitySelection | QuantityAggregation,
        *,
        ordering: OrderingOption = "location",
        result_format: FormatOption = "dataframe",
    ) -> pd.DataFrame | str | None:
        """
        Render a table showing minimum and maximum values over time for the given
        selections. This is equivalent to calling `quantiles()` with 0 and 1.

        Parameters
        ----------
        geo :
            The geographic selection to make on the output data.
        time :
            The time selection to make on the output data.
        quantity :
            The quantity selection to make on the output data.
        ordering :
            Controls the ordering of rows in the result;
            both location and quantity are used to sort the resulting rows,
            this just decides which gets priority.
        result_format :
            Controls the type of the result of this method;
            "dataframe" returns a Pandas dataframe,
            "string" returns the stringified table, and
            "print" just prints the stringified table directly and returns `None`.

        Returns
        -------
        :
            Output according to the value of the `result_format` parameter.
        """
        return self.quantiles(
            quantiles=(0.0, 1.0),
            geo=geo,
            time=time,
            quantity=quantity,
            ordering=ordering,
            result_format=result_format,
            column_names=("min", "max"),
        )

    @overload
    def sum(
        self,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantitySelection | QuantityAggregation,
        *,
        ordering: OrderingOption = "location",
        result_format: Literal["dataframe"] = "dataframe",
    ) -> pd.DataFrame: ...

    @overload
    def sum(
        self,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantitySelection | QuantityAggregation,
        *,
        ordering: OrderingOption = "location",
        result_format: Literal["string"],
    ) -> str: ...

    @overload
    def sum(
        self,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantitySelection | QuantityAggregation,
        *,
        ordering: OrderingOption = "location",
        result_format: Literal["print"],
    ) -> None: ...

    def sum(
        self,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantitySelection | QuantityAggregation,
        *,
        ordering: OrderingOption = "location",
        result_format: FormatOption = "dataframe",
    ) -> pd.DataFrame | str | None:
        """
        Render a table showing summed values over time for the given selections.

        Because it is not valid to sum compartment values over time -- this would be
        double-counting individuals in a way that has no physical meaning --
        compartment quantities are automatically omitted even if they are part of the
        selection, so only events are reflected in the result.

        Parameters
        ----------
        geo :
            The geographic selection to make on the output data.
        time :
            The time selection to make on the output data.
        quantity :
            The quantity selection to make on the output data.
        ordering :
            Controls the ordering of rows in the result;
            both location and quantity are used to sort the resulting rows,
            this just decides which gets priority.
        result_format :
            Controls the type of the result of this method;
            "dataframe" returns a Pandas dataframe,
            "string" returns the stringified table, and
            "print" just prints the stringified table directly and returns `None`.

        Returns
        -------
        :
            Output according to the value of the `result_format` parameter.
        """
        if any(isinstance(q, CompartmentDef) for q in quantity.selected):
            warn(
                "Although your selection includes IPM compartments, "
                "`sum()` has removed these from the output.\nThis is because "
                "adding a time-series of compartment values is likely to "
                "double-count individuals and produce a misleading result.\n"
                "Adding a time-series of event occurrences, on the other hand, is fine."
            )

        # Since it doesn't make real sense to sum compartment values,
        # exclude compartments from selection.
        C, E = quantity.ipm.num_compartments, quantity.ipm.num_events
        m = np.zeros(shape=C + E, dtype=np.bool_)
        m[C:] = True
        quantity = dataclasses.replace(quantity, selection=quantity.selection & m)

        result_df = _process_output(
            self.output,
            geo,
            time,
            quantity,
            ordering,
            ["sum"],
            lambda df: df.sum().reset_index(),
        )
        return self._format_output(result_df, result_format)

    @overload
    def chart(
        self,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantitySelection | QuantityAggregation,
        *,
        chart_length: int = 20,
        ordering: OrderingOption = "location",
        result_format: Literal["dataframe"] = "dataframe",
    ) -> pd.DataFrame: ...

    @overload
    def chart(
        self,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantitySelection | QuantityAggregation,
        *,
        chart_length: int = 20,
        ordering: OrderingOption = "location",
        result_format: Literal["string"],
    ) -> str: ...

    @overload
    def chart(
        self,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantitySelection | QuantityAggregation,
        *,
        chart_length: int = 20,
        ordering: OrderingOption = "location",
        result_format: Literal["print"],
    ) -> None: ...

    def chart(
        self,
        geo: GeoSelection | GeoAggregation,
        time: TimeSelection | TimeAggregation,
        quantity: QuantitySelection | QuantityAggregation,
        *,
        chart_length: int = 20,
        ordering: OrderingOption = "location",
        result_format: FormatOption = "dataframe",
    ) -> pd.DataFrame | str | None:
        """
        Render a table showing a rough time series bar chart for the given selections
        using ASCII characters.

        It is of course limited by the fact that this is a
        relatively coarse display method. The y-axis of each chart is on its own scale,
        and thus is not comparable to others. However the x-axis is on a shared scale,
        so this can give you an idea of the time-series behavior of your simulation
        and relative timing between the selected quantities and locations.

        Parameters
        ----------
        geo :
            The geographic selection to make on the output data.
        time :
            The time selection to make on the output data.
        quantity :
            The quantity selection to make on the output data.
        chart_length :
            Approximately how many characters should we use to render the charts?
            This is simply a ballpark, similar to automatically selecting the number
            of bins in a histogram, so you may get more or less than you ask for.
            Multiple days may be compressed into one bin, but one day will never be
            split between bins. The last bin may contain less days of data than the
            rest of the bins.
        ordering :
            Controls the ordering of rows in the result;
            both location and quantity are used to sort the resulting rows,
            this just decides which gets priority.
        result_format :
            Controls the type of the result of this method;
            "dataframe" returns a Pandas dataframe,
            "string" returns the stringified table, and
            "print" just prints the stringified table directly and returns `None`.

        Returns
        -------
        :
            Output according to the value of the `result_format` parameter.
        """
        if isinstance(time, TimeSelection):
            # Unless the user supplies their own TimeAggregation,
            # use NBins to condense the series.
            time = time.group(NBins(chart_length)).agg()

        result_df = _process_output(
            self.output,
            geo,
            time,
            quantity,
            ordering,
            ["chart"],
            lambda df: df.agg(_to_sparklines).reset_index(),
        )
        return self._format_output(result_df, result_format)


class TableRendererMixin(Output):
    """Mixin class that adds a convenient method for rendering tables from an output."""

    @property
    def table(self) -> TableRenderer:
        """Render a table from this output."""
        return TableRenderer(self)
