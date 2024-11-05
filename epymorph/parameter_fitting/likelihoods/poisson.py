from scipy.stats import poisson

from .base_likelihood import Likelihood


class PoissonLikelihood(Likelihood):
    def __init__(self):
        pass

    def compute(self, observed, expected):
        return poisson.pmf(observed, expected)
