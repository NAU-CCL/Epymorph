from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class PlotOutput:
    num_particles: int
    parameters_estimated: list
    duration: str
    param_quantiles: Dict[str, List[float]]
    param_values: Dict[str, List[float]]
    true_data: np.ndarray
    model_data: np.ndarray
