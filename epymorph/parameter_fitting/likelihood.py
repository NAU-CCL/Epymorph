from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import nbinom, norm, poisson


class Likelihood(ABC):
    """
    Abstract base class for likelihood functions for computing the likelihood of
    observational data predicted by a model.
    """

    @abstractmethod
    def compute(self, observed: int, expected: int):
        """
        Computes the likelihood of the observed data given the data expected by a model.

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
    Encapsulatees the Poisson likelihood function for observational data. The expected
    value of the observation is used as the parameter for the Poisson distribution. The
    observed values must be nonnegative integers.

    Attributes
    ----------
    jitter : float
        A small number added to the expected value to avoid the degenerate case when the
        expected value is zero.
    """

    def __init__(self, jitter: float = 0.0001, shift=0, scale=1):
        self.jitter = jitter
        self.shift = shift
        self.scale = scale

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
        return poisson.pmf(
            np.floor(observed * self.scale + self.shift),
            expected * self.scale + self.jitter + self.shift,
        )

    def compute_log(self, observed, expected):
        """
        Computes the Poisson likelihood.

        Parameters
        ----------
        observed : int
            The observational data.
        expected : int
            The data predicted by the model.
        """
        return poisson.logpmf(
            np.floor(observed * self.scale + self.shift),
            expected + self.jitter + self.shift,
        )


class NegativeBinomial(Likelihood):
    """
    Encapsulatees the Poisson likelihood function for observational data. The expected
    value of the observation is used as the parameter for the Poisson distribution. The
    observed values must be nonnegative integers.

    Attributes
    ----------
    jitter : float
        A small number added to the expected value to avoid the degenerate case when the
        expected value is zero.
    """

    def __init__(self, variance, jitter=0.0001):
        self.variance = variance
        self.jitter = jitter

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
        # return poisson.pmf(observed, expected + self.jitter)
        mean = expected + self.jitter
        return nbinom.pmf(
            observed,
            n=mean**2 / (self.variance - mean),
            p=mean / self.variance,
        )

    def compute_log(self, observed, expected):
        """
        Computes the Poisson likelihood.

        Parameters
        ----------
        observed : int
            The observational data.
        expected : int
            The data predicted by the model.
        """
        # return poisson.logpmf(observed, expected + self.jitter)
        mean = expected + self.jitter
        return nbinom.logpmf(
            observed,
            n=mean**2 / (self.variance - mean),
            p=mean / self.variance,
        )


class Gaussian(Likelihood):
    """
    Encapsulatees the Poisson likelihood function for observational data. The expected
    value of the observation is used as the parameter for the Poisson distribution. The
    observed values must be nonnegative integers.

    Attributes
    ----------
    jitter : float
        A small number added to the expected value to avoid the degenerate case when the
        expected value is zero.
    """

    def __init__(self, variance):
        self.variance = variance

    def compute(self, observed, expected):
        return norm.pdf(observed, loc=expected, scale=np.sqrt(self.variance))

    def compute_log(self, observed, expected):
        return norm.logpdf(observed, loc=expected, scale=np.sqrt(self.variance))
