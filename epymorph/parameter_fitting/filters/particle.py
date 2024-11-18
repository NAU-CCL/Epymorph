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
        events_state (np.ndarray): The current events state of the particle,
        over the given duration
    """

    def __init__(
        self,
        state: np.ndarray,
        parameters: Dict[str, np.ndarray],
        events_state: np.ndarray,
    ) -> None:
        """
        Initializes a Particle instance.
        Args:
            state (np.ndarray): Initial state of the particle.
            parameters (Dict[str, float]): Dictionary of dynamic parameters
            for the particle.
            events_state (np.ndarray): Initial events state of the particle
        """
        self.state = state
        self.parameters = parameters
        self.events_state = events_state

    def update_state(self, new_state: np.ndarray) -> None:
        """
        Updates the state of the particle.
        Args:
            new_state (np.ndarray): The new state to update the particle with.
        """
        self.state = new_state

    def update_parameters(self, new_parameters: Dict[str, np.ndarray]) -> None:
        """
        Updates the parameters of the particle.
        Args:
            new_parameters (Dict[str, float]): Dictionary of new parameters to update the particle with.
        """
        self.parameters.update(new_parameters)
