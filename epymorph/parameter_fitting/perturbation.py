from abc import ABC


class Perturbation(ABC): ...


class Calvetti(Perturbation):
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
