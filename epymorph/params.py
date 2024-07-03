"""
Users may wish to express simulation parameters as functions of simulation dimensions
and/or of other simulation attributes. These classes provide a way to do that.
This module also defines the types of acceptable input values for simulation parameters.
"""
from abc import ABC, abstractmethod
from typing import Literal, Sequence, TypeVar, Union, final

import numpy as np
from numpy.typing import NDArray
from sympy import Expr, Symbol

from epymorph.data_type import (AttributeValue, ScalarDType, ScalarValue,
                                StructDType, StructValue)
from epymorph.simulation import SimulationFunction
from epymorph.sympy_shim import lambdify, to_symbol

T_co = TypeVar('T_co', bound=np.generic, covariant=True)
"""The result type of a ParamFunction."""


class ParamFunction(SimulationFunction[T_co], ABC):
    """Parameter functions can be specified in a variety of forms; this class describe the common elements."""


class ParamFunctionNumpy(ParamFunction[T_co]):
    """A param function which produces a numpy array for the full data series."""

    @abstractmethod
    def evaluate(self) -> NDArray[T_co]:
        """
        Produce a numpy array containing all of this parameter's values.
        This method must assure the values are the appropriate shape and data type.
        """


class _ParamFunction1(ParamFunction[T_co], ABC):
    """Base class for parameter functions which calculate results one at a time."""

    dtype: type[T_co] | None = None
    """The result type of this function. If specified, results will be coerced accordingly."""


class ParamFunctionScalar(_ParamFunction1[T_co]):
    """A param function which produces a scalar value (which is the full data series)."""

    @final
    def evaluate(self) -> NDArray[T_co]:
        return np.array(self.evaluate1(), dtype=self.dtype)

    @abstractmethod
    def evaluate1(self) -> AttributeValue:
        """Produce a scalar value for this parameter in the given simulation context."""


class ParamFunctionTime(_ParamFunction1[T_co]):
    """A param function which produces a time-series of data, one value at a time."""

    @final
    def evaluate(self) -> NDArray[T_co]:
        result = [self.evaluate1(day) for day in range(self.dim.days)]
        return np.array(result, dtype=self.dtype)

    @abstractmethod
    def evaluate1(self, day: int) -> AttributeValue:
        """Produce a scalar value for this parameter by day in the given simulation context."""


class ParamFunctionNode(_ParamFunction1[T_co]):
    """A param function which produces a node-series of data, one value at a time."""

    @final
    def evaluate(self) -> NDArray[T_co]:
        result = [self.evaluate1(n) for n in range(self.dim.nodes)]
        return np.array(result, dtype=self.dtype)

    @abstractmethod
    def evaluate1(self, node_index: int) -> AttributeValue:
        """Produce a scalar value for this parameter by node in the given simulation context."""


class ParamFunctionNodeAndNode(_ParamFunction1[T_co]):
    """A param function which produces a node-by-node matrix of data, one value at a time."""

    @final
    def evaluate(self) -> NDArray[T_co]:
        result = [[self.evaluate1(n1, n2) for n2 in range(self.dim.nodes)]
                  for n1 in range(self.dim.nodes)]
        return np.array(result, dtype=self.dtype)

    @abstractmethod
    def evaluate1(self, node_from: int, node_to: int) -> AttributeValue:
        """Produce a scalar value for this parameter by node-pair in the given simulation context."""


class ParamFunctionNodeAndCompartment(_ParamFunction1[T_co]):
    """A param function which produces a node-by-disease-compartment matrix of data, one value at a time."""

    @final
    def evaluate(self) -> NDArray[T_co]:
        result = [[self.evaluate1(n, c) for c in range(self.dim.compartments)]
                  for n in range(self.dim.nodes)]
        return np.array(result, dtype=self.dtype)

    @abstractmethod
    def evaluate1(self, node_index: int, compartment_index: int) -> AttributeValue:
        """Produce a scalar value for this parameter by node and disease compartment in the given simulation context."""


class ParamFunctionTimeAndNode(_ParamFunction1[T_co]):
    """A param function which produces a time-by-node matrix of data, one value at a time."""

    @final
    def evaluate(self) -> NDArray[T_co]:
        result = [[self.evaluate1(day, n) for n in range(self.dim.nodes)]
                  for day in range(self.dim.days)]
        return np.array(result, dtype=self.dtype)

    @abstractmethod
    def evaluate1(self, day: int, node_index: int) -> AttributeValue:
        """Produce a scalar value for this parameter by day and node in the given simulation context."""

# NOTE: data_shape does not yet support TxNxN attributes, so this type of function is moot for now
#       (commented out to prevent confusion)
# class ParamFunctionTimeAndNodeAndNode(_ParamFunction1[T_co]):
#     """A param function which produces a time-by-node-by-node 3-dimensional array of data, one value at a time."""
#
#     @final
#     def evaluate(self) -> NDArray[T_co]:
#         result = [[[self.evaluate1(day, n1, n2) for n2 in range(self.dim.nodes)]
#                    for n1 in range(self.dim.nodes)]
#                   for day in range(self.dim.days)]
#         return np.array(result, dtype=self.dtype)
#
#     @abstractmethod
#     def evaluate1(self, day: int, node_from: int, node_to: int) -> AttributeValue:
#         """Produce a scalar value for this parameter by day and node-pair in the given simulation context."""


_ALL_PARAMS = ('day', 'node_index', 'duration_days', 'nodes')
_ALL_PARAM_SYMBOLS = [to_symbol(x) for x in _ALL_PARAMS]
_PARAMS_MAP = dict(zip(_ALL_PARAMS, _ALL_PARAM_SYMBOLS))

ParamSymbol = Literal['day', 'node_index', 'duration_days', 'nodes']


def simulation_symbols(*symbols: ParamSymbol) -> tuple[Symbol, ...]:
    """Convenient function to retrieve the symbols used to represent simulation quantities."""
    return tuple(_PARAMS_MAP[x] for x in symbols if x in _PARAMS_MAP)


class ParamExpressionTimeAndNode(ParamFunction[np.float64]):
    """A param function based on a sympy expression for a time-by-node matrix of data."""

    attributes = ()

    _expr: Expr

    def __init__(self, expression: Expr):
        bad_symbols = [
            x for x in expression.free_symbols
            if x not in _ALL_PARAM_SYMBOLS
        ]
        if len(bad_symbols) > 0:
            bs = ", ".join(map(str, bad_symbols))
            raise ValueError(f"expression uses unsupported symbols: {bs}")
        self._expr = expression

    # TODO: support data attributes
    # 1. user needs a way to acquire valid symbols, maybe: self.symbol(attribute)
    # 2. evaluate() would need to include attr symbols in lambdify(), values in fn()

    @final
    def evaluate(self) -> NDArray[np.float64]:
        D = self.dim.days
        N = self.dim.nodes
        ds = np.broadcast_to(np.arange(D).reshape((D, 1)), (D, N))
        ns = np.broadcast_to(np.arange(N).reshape((1, N)), (D, N))
        fn = lambdify(_ALL_PARAM_SYMBOLS, self._expr)
        result = fn([ds, ns, D, N])
        if isinstance(result, np.ndarray):
            return result.astype(dtype=np.float64, copy=False)
        return np.broadcast_to(np.array(result, dtype=np.float64), (D, N))


ListValue = Sequence[Union[ScalarValue, StructValue, 'ListValue']]
ParamValue = ScalarValue | StructValue | ListValue | ParamFunction | Expr \
    | NDArray[ScalarDType | StructDType]
"""All acceptable input forms for parameter values."""
