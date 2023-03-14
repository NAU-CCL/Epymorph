from typing import NamedTuple

import numpy as np

# The need for this info tends to cut across the various modules (ipm, movement, geo).
# This structure is extracted here to avoid creating circular dependencies.


class SimContext(NamedTuple):
    """Useful information about the context in which a simulation is being run."""
    compartments: int
    events: int
    nodes: int
    labels: list[str]
    rng: np.random.Generator
