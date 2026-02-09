from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.stats import nbinom, norm, poisson


@dataclass(frozen=True)
class Likelihood(ABC):
    """
    Abstract base class for likelihood functions for computing the likelihood of
    observational data predicted by a model.
    """

    @abstractmethod
    def compute(self, observed, expected):
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

    @abstractmethod
    def compute_log(self, observed, expected):
        raise NotImplementedError("Subclasses should implement this method.")


@dataclass(frozen=True)
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

    jitter: float = 0.0001
    shift: int = 0
    scale: float = 1.0

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
            observed,
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
            observed,
            expected * self.scale + self.jitter + self.shift,
        )


@dataclass(frozen=True)
class NegativeBinomial(Likelihood):
    """
    Encapsulatees the Negative Binomial likelihood function for observational data. The
    expected value of the observation is used as the parameter for the Negative Binomial
    distribution. The observed values must be nonnegative integers.

    Attributes
    ----------
    r : int
        The overdispersion parameter of the Negative Binomial distribution. As r->inf
        NB -> Poisson
    """

    r: int

    def compute(self, observed, expected):
        return nbinom.pmf(
            observed,
            n=self.r,
            p=1 / (1 + expected / self.r),
        )

    def compute_log(self, observed, expected):
        """
        Computes the Negative Binomial likelihood.

        Parameters
        ----------
        observed : int
            The observational data.
        expected : int
            The data predicted by the model.
        """
        return nbinom.logpmf(
            observed,
            n=self.r,
            p=1 / (1 + expected / self.r),
        )


@dataclass(frozen=True)
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

    variance: float

    def compute(self, observed, expected):
        return norm.pdf(observed, loc=expected, scale=np.sqrt(self.variance))

    def compute_log(self, observed, expected):
        return norm.logpdf(observed, loc=expected, scale=np.sqrt(self.variance))
