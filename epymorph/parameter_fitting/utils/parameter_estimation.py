from typing import Literal

from epymorph.parameter_fitting.distributions.base_distribution import Distribution
from epymorph.parameter_fitting.dynamics.dynamics import Dynamics
from epymorph.parameter_fitting.perturbation.perturbation import Perturbation


class EstimateParameters:
    """
    Contains the information needed to estimate a parameter.

    Attributes
    ----------
    distribution : Distribution
        The prior (initial) distribution for a static (time varying) parameter.
    dynamics : Dynamics
        The dynamics of the parameter.
    """

    def __init__(
        self,
        distribution: Distribution,
        dynamics: Dynamics | None,
        perturbation: Perturbation | None,
    ):
        self.distribution = distribution
        self.dynamics = dynamics
        self.perturbation = perturbation

    @classmethod
    def TimeVarying(
        cls,
        distribution: Distribution,
        dynamics: Dynamics,
        perturbation: Perturbation | None = None,
    ):
        return cls(distribution, dynamics=dynamics, perturbation=perturbation)

    @classmethod
    def Static(
        cls, distribution: Distribution, perturbation: Perturbation | None = None
    ):
        return cls(distribution, dynamics=None, perturbation=perturbation)


class PropagateParams:
    @classmethod
    def propagate_param(cls, approach: Literal["GBM"]):
        return None
