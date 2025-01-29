from abc import ABC, abstractmethod

import numpy as np


class Distribution(ABC):
    """
    Abstract base class for initial distributions for estimating parameters.
    """

    @abstractmethod
    def rvs(self, size: int, random_state: np.random.Generator):
        """
        Draws random variates from the distribution.

        Parameters
        ----------
        size : int
            Number of random variates to draw.
        random_state : np.random.Generator
            The random number generator to use.
        """
        raise NotImplementedError("Subclasses should implement this method.")
