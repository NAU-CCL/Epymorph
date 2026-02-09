from abc import ABC, abstractmethod
from copy import deepcopy
from typing import final

import numpy as np
import scipy as sp
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.attribute import AttributeDef
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
        previous = initial

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


class OrnsteinUhlenbeckWithMean(ParamFunctionDynamics[np.float64]):
    # requirements = ()
    @property
    def requirements(self):
        return (self._mean_req,)

    @property
    def mean(self):
        return self.data(self._mean_req.name)

    def __init__(self, initial=None, damping=None, mean=None, standard_deviation=None):
        self.initial = initial
        self.damping = damping
        if isinstance(mean, str):
            self._mean_req = AttributeDef(mean, float, Shapes.TxN)
        self.standard_deviation = standard_deviation

    def evaluate_from_initial(self, initial):
        result = np.zeros((self.time_frame.days, self.scope.nodes), np.float64)
        previous = initial

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


class Static(ParamFunctionDynamics[np.float64]):
    def __init__(self, initial=None):
        self.initial = initial

    def evaluate_from_initial(self, initial):
        result = np.zeros((self.time_frame.days, self.scope.nodes), np.float64)
        result[...] = initial
        return result


class BrownianMotion(ParamFunctionDynamics[np.float64]):
    requirements = ()

    def __init__(self, initial=None, voliatility=0.1):
        self.initial = initial
        self.voliatility = voliatility

    def evaluate_from_initial(self, initial):
        result = np.zeros((self.time_frame.days, self.scope.nodes), np.float64)
        previous = self.initial
        voliatility = self.voliatility
        voliatility = Shapes.TxN.adapt(self.dim, np.array(voliatility))
        for i_day in range(self.time_frame.days):
            current = previous + voliatility[i_day, ...] * self.rng.normal(
                size=self.scope.nodes
            )
            result[i_day, ...] = current
            previous = current
        return result


class ExponentialTransform(ParamFunction[np.float64]):
    @property
    def requirements(self):
        if self._value_req is not None:
            return (self._value_req,)
        else:
            return ()

    def __init__(self, other: str):
        self._value_req = AttributeDef(other, float, Shapes.TxN)

    def evaluate(self):
        return np.exp(self.data(self._value_req.name))


class ShiftTransform(ParamFunction[np.float64]):
    @property
    def requirements(self):
        return (self._first_req, self._second_req)

    def __init__(self, first: str, second: str):
        self._first_req = AttributeDef(first, float, Shapes.TxN)
        self._second_req = AttributeDef(second, float, Shapes.TxN)

    def evaluate(self):
        return self.data(self._first_req.name) + self.data(self._second_req.name)
