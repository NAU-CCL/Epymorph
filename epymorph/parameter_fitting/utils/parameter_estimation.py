from typing import Literal

from epymorph.parameter_fitting.distribution import Distribution
from epymorph.parameter_fitting.dynamics import Dynamics
from epymorph.parameter_fitting.perturbation import Perturbation


class EstimateParameters:
    """
    Contains the information needed to estimate a single parameter.

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
    def TimeVarying(  # noqa: N802
        cls,
        distribution: Distribution,
        dynamics: Dynamics,
        perturbation: Perturbation | None = None,
    ):
        return cls(distribution, dynamics=dynamics, perturbation=perturbation)

    @classmethod
    def Static(  # noqa: N802
        cls, distribution: Distribution, perturbation: Perturbation | None = None
    ):
        return cls(distribution, dynamics=None, perturbation=perturbation)


class PropagateParams:
    @classmethod
    def propagate_param(cls, approach: Literal["GBM"]):
        return None
