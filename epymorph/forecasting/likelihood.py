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
    
    @abstractmethod
    def sample(self,rng,mean,**kwargs):
        """
        Samples the likelihood.

        Parameters
        ----------
        rng : np.random.generator
            The numpy random number generator used to sample from the distribution 
        mean : np.array  
            Mean of the distribution
        """
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

    def sample(self,rng,mean,**kwargs):
        """
        Samples the likelihood.

        Parameters
        ----------
        rng : np.random.generator
            The numpy random number generator used to sample from the distribution 
        mean : np.array
            mean of the distribution
        """
        return rng.poisson(mean)


@dataclass(frozen=True)
class NegativeBinomial(Likelihood):
    """
    Encapsulatees the Negative Binomial likelihood function for observational data. The expected
    value of the observation is used as the parameter for the Negative Binomial distribution. The
    observed values must be nonnegative integers.

    Attributes
    ----------
    r : int
        The overdispersion parameter of the Negative Binomial distribution. As r->inf 
        NB -> Poisson
    """

    r : int

    def compute(self, observed, expected):
        pass

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
            p=self.r/(0.005+expected + self.r),
        )
    
    def sample(self,rng,mean,**kwargs):
        """
        Samples the likelihood.

        Parameters
        ----------
        rng : np.random.generator
            The numpy random number generator used to sample from the distribution 
        mean : np.array
            Mean of the distribution
        """

        return rng.negative_binomial(n = self.r,p = self.r/(self.r + mean + 0.005))


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

    def sample(self,rng,mean,**kwargs):
        """
        Samples the likelihood.

        Parameters
        ----------
        rng : np.random.generator
            The numpy random number generator used to sample from the distribution 
        mean : np.array
            Mean of the distribution
        """

        sample = rng.normal(0.,scale = np.sqrt(self.variance),size = mean.shape)

        return mean + sample

@dataclass(frozen=True)
class AdaptiveGaussian(Likelihood):

    jitter: float = 0.0001

    def compute(self, observed, expected):
        return norm.pdf(observed, loc=expected, scale=np.sqrt(expected + self.jitter))

    def compute_log(self, observed, expected):
        return norm.logpdf(observed, loc=expected, scale=np.sqrt(expected + self.jitter))

    def sample(self,rng,mean,**kwargs):
        """
        Samples the likelihood.

        Parameters
        ----------
        rng : np.random.generator
            The numpy random number generator used to sample from the distribution 
        mean : np.array
            Mean of the distribution
        """

        sample = rng.normal(0.,scale = np.sqrt(mean + self.jitter),size = mean.shape)

        return mean + sample
