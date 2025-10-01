from abc import ABC, abstractmethod
from copy import deepcopy
from typing import final

import numpy as np
import scipy as sp
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.data_shape import Shapes
from epymorph.params import ParamFunction, ResultDType


class Prior:
    @abstractmethod
    def sample(self, size, rng): ...


class UniformPrior(Prior):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def sample(self, size, rng):
        return sp.stats.uniform.rvs(
            loc=self.lower, scale=(self.upper - self.lower), size=size, random_state=rng
        )


class GaussianPrior(Prior):
    def __init__(self, mean, standard_deviation):
        self.mean = mean
        self.standard_deviation = standard_deviation

    def sample(self, size, rng):
        return sp.stats.norm.rvs(
            loc=self.mean, scale=self.standard_deviation, size=size, random_state=rng
        )


class ParamFunctionDynamics(ParamFunction[ResultDType], ABC):
    dtype: type[ResultDType] | None = None
    initial = None

    def with_initial(self, initial):
        clone = deepcopy(self)
        setattr(clone, "initial", initial)
        return clone

    @final
    @override
    def evaluate(self) -> NDArray[ResultDType]:
        return np.array(self.evaluate_from_initial(self.initial), self.dtype)

    @abstractmethod
    def evaluate_from_initial(self, initial) -> NDArray[ResultDType]: ...


class OrnsteinUhlenbeck(ParamFunctionDynamics[np.float64]):
    requirements = ()

    def __init__(self, initial=None, damping=None, mean=None, standard_deviation=None):
        self.initial = initial
        self.damping = damping
        self.mean = mean
        self.standard_deviation = standard_deviation

    def evaluate_from_initial(self, initial):
        result = np.zeros((self.time_frame.days, self.scope.nodes), np.float64)
        previous = self.initial

        mean = self.mean
        mean = Shapes.TxN.adapt(self.dim, np.array(mean))
        damping = Shapes.TxN.adapt(self.dim, np.array(self.damping))
        standard_deviation = Shapes.TxN.adapt(
            self.dim, np.array(self.standard_deviation)
        )
        delta_t = 1
        A = np.exp(-damping * delta_t)
        M = mean * (np.exp(-damping * delta_t) - 1)
        C = standard_deviation * np.sqrt(1 - np.exp(-2 * damping * delta_t))

        for i_day in range(self.time_frame.days):
            current = (
                A[i_day, ...] * previous
                - M[i_day, ...]
                + C[i_day, ...] * self.rng.standard_normal(size=self.scope.nodes)
            )
            result[i_day, ...] = current
            previous = current
        return result
