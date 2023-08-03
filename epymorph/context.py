from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from epymorph.clock import Clock
from epymorph.util import DataDict

SimDType = np.int_
Compartments = NDArray[SimDType]
Events = NDArray[SimDType]


# Because the need for this info cuts across modules (ipm, movement, geo),
# this structure is extracted here to avoid creating circular dependencies.
class SimContext(NamedTuple):
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

    @property
    def prv_shape(self) -> tuple[int, int, int]:
        """The shape of the prevalence data for this sim."""
        return (self.clock.num_ticks, self.nodes, self.compartments)

    @property
    def inc_shape(self) -> tuple[int, int, int]:
        """The shape of the incidence data for this sim."""
        return (self.clock.num_ticks, self.nodes, self.events)
