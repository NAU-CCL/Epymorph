"""
Tools for processing epymorph data.
"""

import dataclasses
from abc import abstractmethod
from pathlib import Path
from typing import Protocol, TypeVar

import numpy as np
import pandas as pd

from epymorph.compartment_model import (
    BaseCompartmentModel,
    CompartmentDef,
    QuantityAggregation,
    QuantitySelection,
)
from epymorph.data_shape import SimDimensions
from epymorph.database import NamePattern
from epymorph.geography.scope import GeoAggregation, GeoScope, GeoSelection
from epymorph.log.messaging import sim_messaging
from epymorph.rume import Rume
from epymorph.time import Dim, TimeAggregation, TimeFrame, TimeSelection
from epymorph.util import mask


class Output(Protocol):
    """A generic output interface."""

    dim: SimDimensions
    """The simulation dimensions."""
    scope: GeoScope
    """The simulated scope."""
    time_frame: TimeFrame
    """The simulated time frame."""
    ipm: BaseCompartmentModel
    """The IPM used."""

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
    # Check that the given axis strategies are in-fact based on the same elements that
    # produced the output.
    # For example, it's an error to run a RUME with one scope, then
    # try to render a table using a selection on a completely different scope.
    if geo.scope is not output.scope:
        err = (
            "When applying a geo selection, please create that selection "
            "from the same scope you are applying it to."
        )
        raise ValueError(err)
    if time.time_frame is not output.time_frame:
        err = (
            "When applying a time selection, please create that selection "
            "from the same time frame you are applying it to."
        )
        raise ValueError(err)
    if quantity.ipm is not output.ipm:
        err = (
            "When applying an IPM quantity selection, please create that selection "
            "from the same IPM you are applying it to."
        )
        raise ValueError(err)


def munge(
    output: Output,
    geo: GeoSelection | GeoAggregation,
    time: TimeSelection | TimeAggregation,
    quantity: QuantitySelection | QuantityAggregation,
) -> pd.DataFrame:
    """Applies all select/group/aggregate operations to an output dataframe.
    Returns a dataframe with columns "time", "geo", and a column per selected quantity.
    The values in "time" and "geo" come from the chosen aggregation for those axes.
    Without any group or aggregation specified, "time" is the simulation ticks
    and "geo" is node IDs.
    """
    _validate(output, geo, time, quantity)

    T, N, _, _ = output.dim.TNCE
    taus = output.dim.tau_steps

    # Apply selections first so that aggregations operate on less data.
    time_mask = np.repeat(mask(T, time.selection_ticks(taus)), N)
    geo_mask = np.tile(geo.selection, T)

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


RumeT = TypeVar("RumeT", bound=Rume)


def memoize_rume(path: str | Path, rume: RumeT, *, refresh: bool = False) -> RumeT:
    """Stores/loads a RUME's ADRIO data using a local file.

    This is intended as a utility for working with RUMEs that use data from ADRIOs.
    For example, working interactively in a Notebook you may have to reload the Notebook
    many times, which could take seconds or minutes each time just to fetch data.
    It's convenient to avoid that delay so that's where this function comes in.
    This is not a full serialization of the RUME, so if you change the RUME config
    you should not re-use the cache file; passing `refresh=True` is an easy way to make
    this function ignore any previously stored file."""
    path = Path(path)
    if not refresh and path.exists():
        # Load from cache; clone RUME and replace ADRIOs with ndarrays
        cached = dict(np.load(path))
        return dataclasses.replace(
            rume,
            params={
                **rume.params,
                **{NamePattern.parse(key): value for key, value in cached.items()},
            },
        )
    else:
        # Save to cache; evaluate parameters and store the resulting ndarrays
        with sim_messaging():
            values = rume.evaluate_params(rng=np.random.default_rng()).to_dict(
                simplify_names=True
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **values)
        return rume
