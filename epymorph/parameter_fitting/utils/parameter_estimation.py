from typing import Literal

import numpy as np


class Distribution:
    def __init__(self, name: str, *ranges):
        self.name = name.lower()
        self.ranges = ranges
        self.validate()

    def validate(self):
        if not hasattr(np.random, self.name):
            print("Invalid distribution name")


class EstimateParameters:
    def __init__(self, distribution: object, dynamics: object):
        self.distribution = distribution
        self.dynamics = dynamics

    @classmethod
    def TimeVarying(cls, distribution: object, dynamics: object):
        return cls(distribution, dynamics=dynamics)

    @classmethod
    def Static(cls, distribution: object, dynamics: object):
        return cls(distribution, dynamics=dynamics)


class PropagateParams:
    @classmethod
    def propagate_param(cls, approach: Literal["GBM"]):
        return None
