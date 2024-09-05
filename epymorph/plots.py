"""Our built-in plotting functions and helpful utilities."""

from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from pandas import DataFrame
from pandas import merge as pd_merge

from epymorph.geography import us_tiger
from epymorph.geography.us_census import STATE, CensusScope, CountyScope, StateScope
from epymorph.simulator.basic.output import Output


def plot_event(out: Output, event_idx: int) -> None:
    """Plot the event with the given index for all locations."""
    fig, ax = plt.subplots()
    ax.set_title(f"Event occurrences by location: {out.event_labels[event_idx]}")
    ax.set_xlabel("days")
    ax.set_ylabel("occurrences")

    x_axis = np.arange(out.dim.days)
    y_axis = out.incidence_per_day[:, :, event_idx]
    ax.plot(x_axis, y_axis, label=out.geo_labels)

    if out.dim.nodes <= 12:
        ax.legend()

    fig.tight_layout()
    plt.show()


def plot_pop(out: Output, pop_idx: int, log_scale: bool = False) -> None:
    """Plot all compartments for the population at the given index."""
    fig, ax = plt.subplots()
    ax.set_title(f"Prevalence by compartment: {out.geo_labels[pop_idx]}")
    ax.set_xlabel("days")
    if not log_scale:
        ax.set_ylabel("individuals")
    else:
        ax.set_ylabel("log(individuals)")
        ax.set_yscale("log")

    x_axis = out.ticks_in_days
    y_axis = out.prevalence[:, pop_idx, :]
    ax.plot(x_axis, y_axis, label=out.compartment_labels)

    if out.dim.compartments <= 12:
        ax.legend()

    fig.tight_layout()
    plt.show()


def _subset_states(gdf: GeoDataFrame, state_fips: tuple[str, ...]) -> GeoDataFrame:
    return cast(GeoDataFrame, gdf[gdf["GEOID"].str.startswith(state_fips)])


def map_data_by_county(
    scope: CountyScope,
    data: NDArray,
    *,
    title: str,
    cmap: Any | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    outline: Literal["states", "counties"] = "counties",
    year: us_tiger.TigerYear = 2020,
) -> None:
    """
    Draw a county-level choropleth map using the given `data`. This must be a numpy array whose
    ordering is the same as the nodes in the geo scope.
    """
    state_fips = tuple(STATE.truncate_list(scope.get_node_ids()))
    gdf_counties = us_tiger.get_counties_geo(year)
    gdf_counties = _subset_states(gdf_counties, state_fips)
    gdf_borders = gdf_counties
    if outline == "states":
        gdf_states = us_tiger.get_states_geo(2020)
        gdf_borders = _subset_states(gdf_states, state_fips)
    return _map_data_by_geo(
        scope,
        data,
        gdf_counties,
        gdf_borders=gdf_borders,
        title=title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )


def map_data_by_state(
    scope: StateScope,
    data: NDArray,
    *,
    title: str,
    cmap: Any | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    year: us_tiger.TigerYear = 2020,
) -> None:
    """
    Draw a state-level choropleth map using the given `data`. This must be a numpy array whose
    ordering is the same as the nodes in the geo scope.
    """
    state_fips = tuple(STATE.truncate_list(scope.get_node_ids()))
    gdf_states = us_tiger.get_states_geo(year)
    gdf_states = _subset_states(gdf_states, state_fips)
    return _map_data_by_geo(
        scope, data, gdf_states, title=title, cmap=cmap, vmin=vmin, vmax=vmax
    )


def _map_data_by_geo(
    scope: CensusScope,
    data: NDArray,
    gdf_nodes: GeoDataFrame,
    *,
    gdf_borders: GeoDataFrame | None = None,
    title: str,
    cmap: Any | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """
    Draw a choropleth map for the geo information given.
    This is an internal function to abstract commonalities -- see `map_data_by_county()` and `map_data_by_state()`.
    """
    df_merged = pd_merge(
        on="GEOID",
        left=gdf_nodes,
        right=DataFrame({"GEOID": scope.get_node_ids(), "data": data}),
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    ax.set_title(title)
    df_merged.plot(ax=ax, column="data", legend=True, cmap=cmap, vmin=vmin, vmax=vmax)
    if gdf_borders is None:
        gdf_borders = gdf_nodes
    gdf_borders.plot(ax=ax, linewidth=1, edgecolor="black", color="none", alpha=0.8)
    fig.tight_layout()
    plt.show()
