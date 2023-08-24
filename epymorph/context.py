"""
The need for certain info about a simulation cuts across modules (ipm, movement, geo), so
the SimContext structure is here to contain that info and avoid circular dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import DTypeLike, NDArray

from epymorph.clock import Clock

SimDType = np.int64
"""
This is the numpy datatype that should be used to represent internal simulation data.
Where segments of the application maintain compartment and/or event counts,
they should take pains to use this type at all times (if possible).
"""

# SimDType being centrally-located means we can change it reliably.

Compartments = NDArray[SimDType]
"""Alias for ndarrays representing compartment counts."""

Events = NDArray[SimDType]
"""Alias for ndarrays representing event counts."""

# Aliases (hopefully) make it a bit easier to keep all these NDArrays sorted out.


# DataDict


def normalize_lists(data: dict[str, Any], dtypes: dict[str, DTypeLike] | None = None) -> dict[str, Any]:
    """
    Normalize a dictionary of values so that all lists are replaced with numpy arrays.
    If you would like to force certain values to take certain dtypes, provide the `dtypes` argument 
    with a mapping from key to dtype (types will not affect non-list values).
    """
    if dtypes is None:
        dtypes = {}
    ps = dict[str, Any]()
    # Replace list values with numpy arrays.
    for key, value in data.items():
        if isinstance(value, list):
            dt = dtypes.get(key, None)
            ps[key] = np.asarray(value, dtype=dt)
        else:
            ps[key] = value
    return ps


# SimContext


class SimDimension:
    """The subset of SimContext that is the dimensionality of a simulation."""

    nodes: int
    compartments: int
    events: int
    ticks: int
    days: int

    TNCE: tuple[int, int, int, int]
    """
    The critical dimensionalities of the simulation, for ease of unpacking.
    T: number of ticks;
    N: number of geo nodes;
    C: number of IPM compartments;
    E: number of IPM events (transitions)
    """


@dataclass(frozen=True)
class SimContext(SimDimension):
    """Metadata about the simulation being run."""

    # geo info
    nodes: int
    labels: list[str]
    geo: dict[str, NDArray]
    # ipm info
    compartments: int
    compartment_tags: list[list[str]]
    events: int
    # run info
    param: dict[str, Any]
    clock: Clock
    rng: np.random.Generator
    # denormalized info
    ticks: int = field(init=False)
    days: int = field(init=False)
    TNCE: tuple[int, int, int, int] = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'ticks', self.clock.num_ticks)
        object.__setattr__(self, 'days', self.clock.num_days)
        tnce = (self.clock.num_ticks, self.nodes,
                self.compartments, self.events)
        object.__setattr__(self, 'TNCE', tnce)

    @property
    def population(self) -> NDArray[np.integer]:
        """Get the population of each node."""
        # This is for convenient type-safety.
        # TODO: when we construct the geo we should be verifying this fact.
        return self.geo['population']
