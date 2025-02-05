from abc import ABC, abstractmethod

import numpy as np
import scipy as sp


class Distribution(ABC):
    """
    Abstract base class for the prior/initial distributions for static/dynamic parameter
    estimation.
    """

    @abstractmethod
    def rvs(self, size: int, random_state: np.random.Generator):
        """
        Draws independent random variates (aka random deviates, realizations,
        or samples) from the distribution.

        Parameters
        ----------
        size : int
            Number of random variates to draw.
        random_state : np.random.Generator
            The random number generator to use.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class Uniform(Distribution):
    """
    Continuous uniform distribution on an interval.

    Attributes
    ----------
    a : float
        The left endpoint of the interval.
    b : float
        The right endpoint of the interval.
    """

    def __init__(self, a: float, b: float):
        """
        Initializes the parameters of the distribution.

        Parameters
        ----------
        a : float
            The left endpoint of the the interval.
        b : float
            The right endpoint of the the interval.
        """
        self.a = a
        self.b = b

    def rvs(self, size=1, random_state: np.random.Generator | None = None):
        """
        Draws from a random continuous uniform distribution on an interval.

        Parameters
        ----------
        size : int
            Number of random variates to draw.
        random_state : np.random.Generator
            The random number generator to use.
        """
        return sp.stats.uniform(loc=self.a, scale=(self.b - self.a)).rvs(
            size=size, random_state=random_state
        )
