from typing import Literal

from epymorph.parameter_fitting.distributions.base_distribution import Distribution
from epymorph.parameter_fitting.dynamics.dynamics import Dynamics

# class Distribution:
#     def __init__(self, name: str, *ranges):
#         self.name = name.lower()
#         self.ranges = ranges
#         self.validate()

#     def validate(self):
#         if not hasattr(np.random, self.name):
#             print("Invalid distribution name")


class EstimateParameters:
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
