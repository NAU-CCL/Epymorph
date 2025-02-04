from scipy.stats import poisson

from .base_likelihood import Likelihood


class PoissonLikelihood(Likelihood):
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
