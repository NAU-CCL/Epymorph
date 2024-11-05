import numpy as np

rng = np.random.default_rng()


class Perturb:
    def __init__(self, duration: int) -> None:
        self.duration = duration

    def gbm(self, param, volatility):
        return np.exp(rng.normal(np.log(param), volatility * np.sqrt(self.duration)))
