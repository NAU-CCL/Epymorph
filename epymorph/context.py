from typing import NamedTuple

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
# Having a centrally-located value for this means we can change it reliably.

Compartments = NDArray[SimDType]
"""Alias for ndarrays representing compartment counts."""

Events = NDArray[SimDType]
"""Alias for ndarrays representing event counts."""
# Aliases (hopefully) make it a bit easier to keep all these NDArrays sorted out.


class SimContext(NamedTuple):
    """Metadata about the simulation being run."""

    # Because the need for this info cuts across modules (ipm, movement, geo),
    # this structure is extracted here to avoid creating circular dependencies.

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

    @property
    def TNCE(self) -> tuple[int, int, int, int]:
        """
        The critical dimensionalities of the simulation, for ease of unpacking.
        T: number of ticks; N: number of geo nodes; C: number of IPM compartments; E: number of IPM events (transitions)
        """
        return (self.clock.num_ticks, self.nodes, self.compartments, self.events)
