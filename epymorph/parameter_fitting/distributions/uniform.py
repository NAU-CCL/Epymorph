import numpy as np
import scipy as sp

from epymorph.parameter_fitting.distributions.base_distribution import Distribution


class Uniform(Distribution):
    """
    Uniform distribution for the initial distribution for estimating parameters.

    Attributes
    ----------
    a : float
        The left endpoint of the distribution.
    b : float
        The right endpoint of the distribution.
    """

    def __init__(self, a: float, b: float):
        """
        Initializes the distribution.

        Parameters
        ----------
        a : float
            The left endpoint of the distribution.
        b : float
            The right endpoint of the distribution.
        """
        self.a = a
        self.b = b

    def rvs(self, size=1, random_state: np.random.Generator | None = None):
        """
        Draws random uniform variates.

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
