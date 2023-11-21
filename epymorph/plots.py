"""Our built-in plotting functions and helpful utilities."""
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygris
from numpy.typing import NDArray

from epymorph.engine.standard_sim import Output
from epymorph.geo.geo import Geo


def plot_event(out: Output, event_idx: int) -> None:
    """Plot the event with the given index for all locations."""
    fig, ax = plt.subplots()
    ax.set_title(f"Event occurrences by location: {out.event_labels[event_idx]}")
    ax.set_xlabel('days')
    ax.set_ylabel('occurrences')

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
    ax.set_xlabel('days')
    if not log_scale:
        ax.set_ylabel('individuals')
    else:
        ax.set_ylabel('log(individuals)')
        ax.set_yscale('log')

    x_axis = out.ticks_in_days
    y_axis = out.prevalence[:, pop_idx, :]
    ax.plot(x_axis, y_axis, label=out.compartment_labels)

    if out.dim.compartments <= 12:
        ax.legend()

    fig.tight_layout()
    plt.show()


def map_data_by_county(
    geo: Geo,
    data: NDArray,
    *,
    df_borders: pd.DataFrame | None = None,
    title: str,
    cmap: Any | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    outline: Literal['states', 'counties'] = 'counties',
) -> None:
    """
    Draw a county-level choropleth map using the given `data`. This must be a numpy array whose
    ordering is the same as the nodes in the geo.
    Assumes that the geo contains an attribute (`geoid`) containing the geoids of its nodes.
    (This information is needed to fetch the map shapes.)
    """
    state_fips = {s[:2] for s in geo['geoid']}
    df_states = pygris.states(cb=True, resolution='5m', cache=True, year=2020)
    df_states = df_states.loc[df_states['GEOID'].isin(state_fips)]
    df_counties = pd.concat([
        pygris.counties(state=s, cb=True, resolution='5m', cache=True, year=2020)
        for s in state_fips
    ])
    if outline == 'counties':
        df_borders = df_counties
    else:
        df_borders = df_states
    return _map_data_by_geo(geo, data, df_counties, df_borders=df_borders, title=title, cmap=cmap, vmin=vmin, vmax=vmax)


def map_data_by_state(
    geo: Geo,
    data: NDArray,
    *,
    title: str,
    cmap: Any | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """
    Draw a state-level choropleth map using the given `data`. This must be a numpy array whose
    ordering is the same as the nodes in the geo.
    Assumes that the geo contains an attribute (`geoid`) containing the geoids of its nodes.
    (This information is needed to fetch the map shapes.)
    """
    state_fips = {s[:2] for s in geo['geoid']}
    df_states = pygris.states(cb=True, resolution='5m', cache=True, year=2020)
    df_states = df_states.loc[df_states['GEOID'].isin(state_fips)]
    return _map_data_by_geo(geo, data, df_states, title=title, cmap=cmap, vmin=vmin, vmax=vmax)


def _map_data_by_geo(
    geo: Geo,
    data: NDArray,
    df_nodes: pd.DataFrame,
    *,
    df_borders: pd.DataFrame | None = None,
    title: str,
    cmap: Any | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """
    Draw a choropleth map for the geo information given.
    This is an internal function to abstract commonalities -- see `map_data_by_county()` and `map_data_by_state()`.
    """
    df_merged = pd.merge(
        on="GEOID",
        left=df_nodes,
        right=pd.DataFrame({'GEOID': geo['geoid'], 'data': data}),
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    ax.set_title(title)
    df_merged.plot(ax=ax, column='data', legend=True, cmap=cmap, vmin=vmin, vmax=vmax)
    if df_borders is None:
        df_borders = df_nodes
    df_borders.plot(ax=ax, linewidth=1, edgecolor='black', color='none', alpha=0.8)
    fig.tight_layout()
    plt.show()
