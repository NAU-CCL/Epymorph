from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Self, final

import numpy as np
import scipy as sp
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.attribute import AttributeDef
from epymorph.data_shape import Shapes
from epymorph.params import ParamFunction, ResultDType


class Prior(ABC):
    """
    Abstract class representing the prior distribution of an unknown parameter.
    """

    @abstractmethod
    def sample(self, size: tuple[int], rng: np.random.Generator) -> NDArray[np.float64]:
        """
        Sample values from the prior distribution.
        """


class UniformPrior(Prior):
    """
    A uniform prior distribution.
    """

    lower: float
    """
    The lower bound of the uniform distribution.
    """

    upper: float
    """
    The upper bound of the uniform distribution.
    """

    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.upper = upper

    def sample(self, size: tuple[int], rng: np.random.Generator) -> NDArray[np.float64]:
        """
        Sample an array of uniform random variates.
        """
        return sp.stats.uniform.rvs(
            loc=self.lower, scale=(self.upper - self.lower), size=size, random_state=rng
        )


class GaussianPrior(Prior):
    """
    A Gaussian prior distribution.
    """

    mean: float
    """
    The mean of the Gaussian distribution.
    """

    standard_deviation: float
    """
    The standard deviation of the Gaussian distribution.
    """

    def __init__(self, mean: float, standard_deviation: float):
        self.mean = mean
        self.standard_deviation = standard_deviation

    def sample(self, size: tuple[int], rng: np.random.Generator) -> NDArray[np.float64]:
        """
        Sample an array of Gaussian random variates.
        """
        return sp.stats.norm.rvs(
            loc=self.mean, scale=self.standard_deviation, size=size, random_state=rng
        )


class ParamFunctionDynamics(ParamFunction[ResultDType], ABC):
    """
    Base class for the dynamics, e.g. the time-dependence, of an unknown parameter.
    Once given an initial condition, it produces a time-by-node matrix of data.
    """

    _initial: NDArray[ResultDType] | None = None

    def with_initial(self, initial: NDArray[ResultDType]) -> Self:
        """
        Add an initial value so that this parameter is suitable for evaluation from
        within a RUME.
        """
        clone = deepcopy(self)
        setattr(clone, "_initial", initial)
        return clone

    @final
    @override
    def evaluate(self) -> NDArray[ResultDType]:
        if self._initial is None:
            err = (
                f"Tried to evaluate {self.__class__.__name__} without an initial value."
            )
            raise ValueError(err)
        return np.array(self._evaluate_from_initial(self._initial))

    @abstractmethod
    def _evaluate_from_initial(
        self, initial: NDArray[ResultDType]
    ) -> NDArray[ResultDType]:
        """
        Produce a trajectory matching the attribute requirements from the initial value.
        """


class OrnsteinUhlenbeck(ParamFunctionDynamics[np.float64]):
    """
    Model the time dependence of an unknown parameter as an Ornstein-Uhlenbeck process.
    The process is independent for each node.
    """

    requirements = ()

    def __init__(self, damping: float, mean: float, standard_deviation: float):
        self.damping = damping
        self.mean = mean
        self.standard_deviation = standard_deviation

    def _evaluate_from_initial(
        self, initial: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        result = np.zeros((self.time_frame.days, self.scope.nodes), np.float64)
        previous = initial

        mean = self.mean
        mean = Shapes.TxN.adapt(self.dim, np.array(mean))
        damping = Shapes.TxN.adapt(self.dim, np.array(self.damping))
        standard_deviation = Shapes.TxN.adapt(
            self.dim, np.array(self.standard_deviation)
        )
        delta_t = 1

        # These coefficients come from exactly solving the OU process. This is not an
        # approximation, e.g. this is not Euler-Maruyama.
        A = np.exp(-damping * delta_t)  # noqa: N806
        M = mean * (np.exp(-damping * delta_t) - 1)  # noqa: N806
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
    """
    Model an unknown parameter as a static parameter e.g. no time-dependence.
    """

    def __init__(self):
        pass

    def _evaluate_from_initial(
        self, initial: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        result = np.zeros((self.time_frame.days, self.scope.nodes), np.float64)
        result[...] = initial.copy()
        return result


class BrownianMotion(ParamFunctionDynamics[np.float64]):
    """
    Model the time dependence of an unknown parameter as Brownian motion. The Brownian
    motion for each node is independent.
    """

    requirements = ()

    def __init__(self, volatility: float):
        self.volatility = volatility

    def _evaluate_from_initial(
        self, initial: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        result = np.zeros((self.time_frame.days, self.scope.nodes), np.float64)
        previous = initial
        volatility = self.volatility
        volatility = Shapes.TxN.adapt(self.dim, np.array(volatility))
        for i_day in range(self.time_frame.days):
            current = previous + volatility[i_day, ...] * self.rng.normal(
                size=self.scope.nodes
            )
            result[i_day, ...] = current
            previous = current
        return result


class ExponentialTransform(ParamFunction[np.float64]):
    """
    Produces a time-by-node matrix of data by taking the exponential of another
    parameter.

    Parameters
    ----------
    other : str
        The name of the attribute to take the exponential of. This will be added as a
        requirement.
    """

    _value_req: AttributeDef

    @property
    def requirements(self) -> tuple[AttributeDef]:
        return (self._value_req,)

    def __init__(self, other: str):
        self._value_req = AttributeDef(other, float, Shapes.TxN)

    def evaluate(self) -> NDArray[np.float64]:
        return np.exp(self.data(self._value_req))


class ShiftTransform(ParamFunction[np.float64]):
    """
    Produces a time-by-node matrix of data by taking the sum of two other parameters.

    Parameters
    ----------
    first : str
        The name of the first attribute to take the sum of. This will be added as a
        requirement.
    second : str
        The name of the second attribute to take the sum of. This will be added as a
        requirement.
    """

    _first_req: AttributeDef
    _second_req: AttributeDef

    @property
    def requirements(self) -> tuple[AttributeDef, AttributeDef]:
        return (self._first_req, self._second_req)

    def __init__(self, first: str, second: str):
        self._first_req = AttributeDef(first, float, Shapes.TxN)
        self._second_req = AttributeDef(second, float, Shapes.TxN)

    def evaluate(self) -> NDArray[np.float64]:
        return np.add(self.data(self._first_req), self.data(self._second_req))
