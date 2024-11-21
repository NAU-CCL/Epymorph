from typing import Dict

import numpy as np


class Particle:
    """
    A class representing a single particle with its state and parameters.

    Attributes:
        state (np.ndarray): The state of the particle, initialized through a
        model-specific method.
        parameters (Dict[str, float]): The dynamic parameters associated
        with the particle (e.g., beta).
    """

    def __init__(
        self,
        state: np.ndarray,
        parameters: Dict[str, np.ndarray],
    ) -> None:
        """
        Initializes a Particle instance.

        Args:
            state (np.ndarray): Initial state of the particle.
            parameters (Dict[str, np.ndarray]): Dictionary of dynamic parameters
            for the particle.
        """
        self.state = state
        self.parameters = parameters
