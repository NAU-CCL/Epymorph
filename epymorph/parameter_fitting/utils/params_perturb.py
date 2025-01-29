import numpy as np


class Perturb:
    """
    The `Perturb` class provides methods to apply perturbation to model parameters
    using different stochastic processes. In this case, a geometric Brownian motion
    (GBM) is used to perturb a parameter based on volatility and time duration.

    Attributes
    ----------
    duration : int
        The time duration over which the perturbation is applied.
    """

    def __init__(self, duration: int) -> None:
        """
        Initializes the Perturb class with a specified duration.

        Parameters
        ----------
        duration : int
            The time duration for which the perturbation will be applied.
        """
        self.duration = duration

    def gbm(self, param, volatility: float, rng: np.random.Generator) -> float:
        """
        Applies geometric Brownian motion (GBM) to perturb the given parameter.

        Parameters
        ----------
        param : float
            The initial parameter value to be perturbed.
        volatility : float
            The volatility factor that affects the degree of perturbation.

        Returns
        -------
        float
            The perturbed parameter after applying GBM.
        """
        return np.exp(rng.normal(np.log(param), volatility * np.sqrt(self.duration)))
