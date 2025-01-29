from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class PlotOutput:
    """
    Contains attributes for plotting the output of the particle filter.

    Attributes
    ----------
    num_particles: int
        Number of particles.
    parameters_estimated: list
        A list of parameters which were estimated.
    duration: str
        The duration of the simulation.
    param_quantiles: Dict[str, List[float]]
        The quantiles from the estimated distribution from each parameter at each
        observation time.
    param_values: Dict[str, List[float]]
        The values of the estimated parameters.
    true_data: np.ndarray
        The true data.
    model_data: np.ndarray
        The data predicted by the model.
    """

    num_particles: int
    parameters_estimated: list
    duration: str
    param_quantiles: Dict[str, List[float]]
    param_values: Dict[str, List[float]]
    true_data: np.ndarray
    model_data: np.ndarray
