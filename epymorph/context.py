"""
The need for certain info about a simulation cuts across modules (ipm, movement, geo), so
the SimContext structure is here to contain that info and avoid circular dependencies.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from epymorph.clock import Clock
from epymorph.util import DataDict

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


@dataclass(frozen=True)
class SimContext:
    """Metadata about the simulation being run."""

    # geo info
    nodes: int
    labels: list[str]
    geo: DataDict
    # ipm info
    compartments: int
    compartment_tags: list[list[str]]
    events: int
    # run info
    param: DataDict
    clock: Clock
    rng: np.random.Generator

    TNCE: tuple[int, int, int, int] = field(init=False)
    """
    The critical dimensionalities of the simulation, for ease of unpacking.
    T: number of ticks;
    N: number of geo nodes;
    C: number of IPM compartments;
    E: number of IPM events (transitions)
    """

    def __post_init__(self):
        tnce = (self.clock.num_ticks, self.nodes,
                self.compartments, self.events)
        object.__setattr__(self, 'TNCE', tnce)
