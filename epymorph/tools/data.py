"""General tools for processing epymorph data."""

import dataclasses
from abc import abstractmethod
from pathlib import Path
from typing import Protocol, TypeVar

import numpy as np
import pandas as pd

from epymorph.attribute import NamePattern
from epymorph.compartment_model import (
    CompartmentDef,
    QuantityAggregation,
    QuantitySelection,
)
from epymorph.geography.scope import GeoAggregation, GeoSelection
from epymorph.rume import RUME
from epymorph.time import Dim, TimeAggregation, TimeSelection
from epymorph.util import mask


class Output(Protocol):
    """A generic simulation result."""

    # NOTE: this decouples us from concrete Output implementations.

    rume: RUME
    """The RUME used in the simulation that generated this output."""

    @property
    @abstractmethod
    def dataframe(self) -> pd.DataFrame:
        """The simulation results as a DataFrame."""


def _validate(
    output: Output,
    geo: GeoSelection | GeoAggregation,
    time: TimeSelection | TimeAggregation,
    quantity: QuantitySelection | QuantityAggregation,
) -> None:
    """
    Check that the given axis strategies use the same objects that produced the output.

    For example, it's an error to run a RUME with one scope, then try to render a table
    using a selection on a completely different scope.
    """
    if geo.scope is not output.rume.scope:
        err = (
            "When applying a geo selection to an output, both selection and "
            "output must reference the same GeoScope instance.\n"
            "In this case:\n"
            f"  selection references {object.__repr__(geo.scope)}\n"
            f"     output references {object.__repr__(output.rume.scope)}\n"
            "You might fix this by selecting from the output's RUME's scope, "
            "e.g.: `out.rume.scope.select.all()`"
        )
        raise ValueError(err)
    if time.time_frame is not output.rume.time_frame:
        err = (
            "When applying a time frame selection to an output, both selection and "
            "output must reference the same TimeFrame instance.\n"
            "In this case:\n"
            f"  selection references {object.__repr__(time.time_frame)}\n"
            f"     output references {object.__repr__(output.rume.time_frame)}\n"
            "You might fix this by selecting from the output's RUME's time frame, "
            "e.g.: `out.rume.time_frame.select.all()`"
        )
        raise ValueError(err)
    if quantity.ipm is not output.rume.ipm:
        err = (
            "When applying an IPM quantity selection to an output, both selection and "
            "output must reference the same CompartmentModel instance.\n"
            "In this case:\n"
            f"  selection references {object.__repr__(quantity.ipm)}\n"
            f"     output references {object.__repr__(output.rume.ipm)}\n"
            "You might fix this by selecting from the output's RUME's IPM, "
            "e.g.: `out.rume.ipm.select.events()`"
        )
        raise ValueError(err)


def munge(
    output: Output,
    geo: GeoSelection | GeoAggregation,
    time: TimeSelection | TimeAggregation,
    quantity: QuantitySelection | QuantityAggregation,
) -> pd.DataFrame:
    """
    Apply select/group/aggregate operations to an output dataframe.

    This function powers many of our more-specialized output processing tools, but we
    expose this general utility to enable re-use of this logic in more use-cases.

    Parameters
    ----------
    output :
        The result data to process.
    geo :
        The geo-axis strategy.
    time :
        The time-axis strategy.
    quantity :
        The quantity-axis strategy.

    Returns
    -------
    :
        The munged result.

        It is a dataframe with columns "time", "geo", and a column per selected
        quantity. The values in "time" and "geo" come from the chosen aggregation
        for those axes. Without any group or aggregation specified, "time" is the
        simulation ticks and "geo" is node IDs.

    Raises
    ------
    ValueError
        If the axis strategies don't refer to the same `RUME` used to produce the
        `Output`. Generally it's safest to create the axis strategies using the methods
        from an output's `RUME` object.
    """
    _validate(output, geo, time, quantity)

    N = output.rume.scope.nodes
    S = output.rume.num_ticks
    taus = output.rume.num_tau_steps

    # Apply selections first so that aggregations operate on less data.
    time_mask = np.repeat(mask(S, time.selection_ticks(taus)), N)
    geo_mask = np.tile(geo.selection, S)

    # columns are: ["tick", "date", "node", *quantities]
    columns = np.concatenate(([True, True, True], quantity.selection))
    data_df = output.dataframe.loc[time_mask & geo_mask].loc[:, columns]
    data_df = data_df.set_axis(["tick", "date", "geo", *data_df.columns[3:]], axis=1)

    # I think it makes the most intuitive sense to apply time aggregation last.
    # For example, if the user simulates at tract and wants the peak infection
    # by county -- they probably want each county's peak to represent a single date,
    # rather than summing tract peaks from different days.

    if geo.aggregation is None:
        # Without agg: use node IDs as the geo dimension.
        pass
    else:
        # With agg:
        agg_df = data_df
        if geo.grouping is None:
            # With no grouping: the geo dimension collapses.
            geo_groups = "*"
        else:
            # With group: geo dimension comes from group.
            geo_groups = geo.grouping.map(agg_df["geo"].to_numpy())

        data_df = (
            data_df.assign(geo=geo_groups)
            .groupby(["tick", "date", "geo"], sort=False)
            .agg(geo.aggregation)
            .reset_index()
        )

    if quantity.aggregation is None:
        # For the sake of aggregating and sorting, we need to ensure column names
        # are not ambiguous. But we don't want to alter the names arbitrarily;
        # so we'll rename the columns, do our munging, then restore the original names.
        q_mapping = quantity.disambiguate()
        data_df = data_df.set_axis([*data_df.columns[0:3], *q_mapping.keys()], axis=1)
    else:
        # currently only supported agg is "sum"
        def agg(qty_indices: tuple[int, ...]) -> pd.Series:
            offset = 3  # we have three leading columns before quantities start
            col_indices = [i + offset for i in qty_indices]
            return data_df.iloc[:, col_indices].sum(axis=1)

        group_defs, group_indices = quantity.grouping.map(quantity.selected)
        group_names = [g.name.full for g in group_defs]
        data_df = pd.DataFrame(
            {
                "tick": data_df["tick"],
                "date": data_df["date"],
                "geo": data_df["geo"],
                **{g: agg(xs) for g, xs in zip(group_names, group_indices)},
            }
        )
        # When there is any grouping applied, it should not be possible to
        # produce ambiguous quantity names; but to keep things simple we'll
        # just provide a no-op map for this case.
        q_mapping = dict(zip(group_names, group_names))

    if time.aggregation is None:
        # Without agg: drop date and use ticks as the time dimension.
        data_df = data_df.drop(columns=["date"]).rename(columns={"tick": "time"})
    else:
        # With agg:
        if time.grouping is None:
            # Without group: time dimension collapses.
            time_axis = "*"
        else:
            # With group: time dimension comes from grouping.
            nodes = data_df["geo"].unique().shape[0]
            days = data_df.shape[0] // (nodes * taus)
            time_axis = time.grouping.map(
                Dim(nodes=nodes, days=days, tau_steps=taus),
                data_df["tick"].to_numpy(),
                data_df["date"].to_numpy(),
            )
            if time_axis.shape[0] != data_df.shape[0]:
                err = "Chosen time-axis grouping did not return a group for every row."
                raise ValueError(err)
        data_df = (
            data_df.drop(columns=["tick", "date"])
            .assign(time=time_axis)
            .groupby(["time", "geo"], sort=False)
            .agg(
                {
                    col: (
                        time.aggregation.compartments
                        if isinstance(q, CompartmentDef)
                        else time.aggregation.events
                    )
                    for col, q in zip(q_mapping.keys(), quantity.selected)
                }
            )
            .reset_index()
        )

    # Reset column names
    return data_df.rename(columns=q_mapping)


RumeT = TypeVar("RumeT", bound=RUME)
"""A type of RUME."""


def memoize_rume(
    path: str | Path,
    rume: RumeT,
    *,
    refresh: bool = False,
    rng: np.random.Generator | None = None,
) -> RumeT:
    """
    Cache a RUME's parameter data using a local file.

    If the file doesn't exist, the RUME's parameters are evaluated and saved.
    If the file does exist (and we're not forcing a refresh), values are loaded from
    the file and the RUME's parameters are overridden to use those values.

    This is intended as a utility for working with RUMEs that use data from ADRIOs
    that may be expensive to fetch repeatedly.

    For example, working interactively in a Notebook you may have to reprocess code
    cells many times, which could mean spending a long time re-fetching data and/or
    incurring API usage costs. This function helps you avoid those costs.

    Parameters
    ----------
    path :
        The path to the cache file.
    rume :
        The RUME to use.
    refresh :
        True to force this logic to ignore any existing cache file. Parameters will be
        evaluated and the cache file will be overwritten.
    rng :
        A random number generator instance to use. Otherwise we'll use numpy's default
        RNG.

    Returns
    -------
    :
        A clone of the `RUME` with all parameters replaced by the fully-evaluated numpy
        data arrays.

    Warning
    -------
    This is not a full serialization of the RUME, so if you change the RUME config
    you should _not_ re-use the cache file; passing `refresh=True` is an easy way to
    make this function ignore any previously saved file.
    """
    path = Path(path)
    if not refresh and path.exists():
        # Load from cache
        param_values = dict(np.load(path))
    else:
        # Save to cache
        # Evaluate parameters and store the resulting arrays
        resolver = rume.evaluate_params(rng=rng or np.random.default_rng())
        param_values = resolver.to_dict(simplify_names=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **param_values)

    # In either case: clone RUME and replace all parameters from saved arrays
    params = {NamePattern.parse(key): value for key, value in param_values.items()}
    return dataclasses.replace(rume, params=params)
