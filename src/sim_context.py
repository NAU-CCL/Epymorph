from typing import NamedTuple

import numpy as np


class SimContext(NamedTuple):
    compartments: int
    events: int
    nodes: int
    labels: list[str]
    rng: np.random.Generator
