from abc import ABC, abstractmethod


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
