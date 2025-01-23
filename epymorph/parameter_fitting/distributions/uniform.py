import numpy as np
import scipy as sp

from epymorph.parameter_fitting.distributions.base_distribution import Distribution


class Uniform(Distribution):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def rvs(self, size=1, random_state: np.random.Generator | None = None):
        return sp.stats.uniform(loc=self.a, scale=(self.b - self.a)).rvs(
            size=size, random_state=random_state
        )
