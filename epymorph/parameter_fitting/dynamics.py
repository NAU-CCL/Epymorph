from abc import ABC


class Dynamics(ABC): ...


class GeometricBrownianMotion(Dynamics):
    """
    Encapsulates the hyperparameters for geometric Brownian motion, where the logarithm
    of geometric Brownian motion is standard Brownian motion.

    Attributes
    ----------
    voliatility : float, optional
        The voliatility of geometric brownian motion.
    """

    def __init__(self, volatility=0.1) -> None:
        self.volatility = volatility
