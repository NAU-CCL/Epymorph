from typing import NamedTuple

import numpy as np

from epymorph.clock import Clock
from epymorph.util import DataDict


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
    events: int
    # run info
    param: DataDict
    clock: Clock
    rng: np.random.Generator
