from abc import ABC, abstractmethod

from scipy.stats import poisson


class Likelihood(ABC):
    """
    Abstract base class for likelihood functions for observational data.
    """

    @abstractmethod
    def compute(self, observed: int, expected: int):
        """
        Computes the likelihood.

        Parameters
        ----------
        observed : int
            The observational data.
        expected : int
            The data predicted by the model.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class Poisson(Likelihood):
    """
    Encapsulatees the Poisson likelihood function for observational data.
    """

    def __init__(self):
        pass

    def compute(self, observed, expected):
        """
        Computes the Poisson likelihood.

        Parameters
        ----------
        observed : int
            The observational data.
        expected : int
            The data predicted by the model.
        """
        return poisson.pmf(observed, expected)
