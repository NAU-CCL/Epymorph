from typing import Literal

from epymorph.parameter_fitting.distributions.base_distribution import Distribution
from epymorph.parameter_fitting.dynamics.dynamics import Dynamics


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

    def __init__(self, distribution: Distribution, dynamics: Dynamics):
        self.distribution = distribution
        self.dynamics = dynamics

    @classmethod
    def TimeVarying(cls, distribution: Distribution, dynamics: Dynamics):
        return cls(distribution, dynamics=dynamics)

    @classmethod
    def Static(cls, distribution: Distribution, dynamics: Dynamics):
        return cls(distribution, dynamics=dynamics)


class PropagateParams:
    @classmethod
    def propagate_param(cls, approach: Literal["GBM"]):
        return None
