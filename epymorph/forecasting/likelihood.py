"""
Components for specifying a likelihood function for comparing observed data to
predicted values for state and parameter fitting.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import nbinom, norm, poisson
from typing_extensions import override


@dataclass(frozen=True)
class Likelihood(ABC):
    """
    Abstract base class for likelihood functions for computing the likelihood of
    observational data predicted by a model.
    """

    @abstractmethod
    def compute(
        self, observed: NDArray[np.float64], expected: NDArray[np.float64]
    ) -> NDArray[np.float64]:
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
    def compute_log(
        self, observed: NDArray[np.float64], expected: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        raise NotImplementedError("Subclasses should implement this method.")


@dataclass(frozen=True)
class PoissonLikelihood(Likelihood):
    """
    Encapsulatees the Poisson likelihood function for observational data. The expected
    value of the observation is used as the parameter for the Poisson distribution. The
    observed values must be nonnegative integers.

    Parameters
    ----------
    jitter :
        A small number added to the expected value to avoid the degenerate case when the
        expected value is zero.
    """

    jitter: float = 0.0001

    @override
    def compute(
        self, observed: NDArray[np.float64], expected: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return poisson.pmf(
            observed,
            expected + self.jitter,
        )

    @override
    def compute_log(
        self, observed: NDArray[np.float64], expected: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return poisson.logpmf(
            observed,
            expected + self.jitter,
        )


@dataclass(frozen=True)
class NegativeBinomialLikelihood(Likelihood):
    """
    Encapsulatees the Negative Binomial likelihood function for observational data.

    Parameters
    ----------
    r :
        The overdispersion parameter of the Negative Binomial distribution. Must be
        positive.
    """

    r: int

    def __post_init__(self):
        if self.r <= 0:
            raise ValueError("The overdispersion parameter, r, must be positive.")

    @override
    def compute(
        self, observed: NDArray[np.float64], expected: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return nbinom.pmf(
            observed,
            n=self.r,
            p=1 / (1 + expected / self.r),
        )

    @override
    def compute_log(
        self, observed: NDArray[np.float64], expected: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return nbinom.logpmf(
            observed,
            n=self.r,
            p=1 / (1 + expected / self.r),
        )


@dataclass(frozen=True)
class GaussianLikelihood(Likelihood):
    """
    A Gaussian likelihood function for observational data.

    Parameters
    ----------
    standard_deviation :
        The standard_deviation of the Gaussian distribution. Must be positive.
    """

    standard_deviation: float

    def __post_init__(self):
        if self.standard_deviation <= 0:
            raise ValueError("The standard_deviation must be positive.")

    @override
    def compute(
        self, observed: NDArray[np.float64], expected: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return norm.pdf(observed, loc=expected, scale=self.standard_deviation)

    @override
    def compute_log(
        self, observed: NDArray[np.float64], expected: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return norm.logpdf(observed, loc=expected, scale=self.standard_deviation)
