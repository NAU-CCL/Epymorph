from abc import ABC


class Dynamics(ABC): ...


class GeometricBrownianMotion(Dynamics):
    """
    Encapsulates the hyperparameters for geometric Brownian motion.

    Attributes
    ----------
    voliatility : float, optional
        The voliatility of geometric brownian motion.
    """

    def __init__(self, volatility=0.1) -> None:
        self.volatility = volatility


class Calvetti(Dynamics):
    """
    Encapsulates the hyperparameters for the Calvetti static parameter estimation
    method.

    Attributes
    ----------
    a : float, optional
        The weight on the prior particle cloud.
    """

    def __init__(self, a=0.9):
        self.a = a
