"""
Users may wish to express simulation parameters as functions of simulation dimensions
and/or of other simulation attributes. These classes provide a way to do that.
This module also defines the types of acceptable input values for simulation parameters.
"""

from abc import ABC, abstractmethod
from typing import Literal, TypeVar, final

import numpy as np
from numpy.typing import NDArray
from sympy import Expr, Symbol
from typing_extensions import override

from epymorph.attribute import AbsoluteName
from epymorph.compartment_model import BaseCompartmentModel
from epymorph.data_type import (
    AttributeArray,
    AttributeValue,
)
from epymorph.database import DataResolver, evaluate_param
from epymorph.error import DataAttributeError
from epymorph.geography.scope import GeoScope
from epymorph.simulation import Context, SimulationFunction
from epymorph.sympy_shim import lambdify, to_symbol
from epymorph.time import TimeFrame

ResultDType = TypeVar("ResultDType", bound=np.generic)
"""The result type of a `ParamFunction`."""


class ParamFunction(SimulationFunction[NDArray[ResultDType]], ABC):
    """
    Base class for parameter functions.

    `ParamFunction`s are generic on the dtype of the numpy array
    that they produce (`ResultDType`).
    """


@evaluate_param.register
def _(
    value: ParamFunction,
    name: AbsoluteName,
    data: DataResolver,
    scope: GeoScope | None,
    time_frame: TimeFrame | None,
    ipm: BaseCompartmentModel | None,
    rng: np.random.Generator | None,
) -> AttributeArray:
    # depth-first evaluation guarantees `data` has our dependencies.
    ctx = Context.of(name, data, scope, time_frame, ipm, rng)
    sim_func = value.with_context_internal(ctx)
    return sim_func.evaluate()


class ParamFunctionNumpy(ParamFunction[ResultDType]):
    """
    Base class for param functions whose evaluate method produces
    the whole data series as a numpy array.
    """

    @override
    @abstractmethod
    def evaluate(self) -> NDArray[ResultDType]:
        """
        Produce a numpy array containing all of this parameter's values.
        This method must assure the values are the appropriate shape and data type.

        Returns
        -------
        :
            The data array.
        """


class ParamFunction1(ParamFunction[ResultDType], ABC):
    """Base class for parameter functions which calculate results one at a time."""

    dtype: type[ResultDType] | None = None
    """
    The result type of this function. If specified, results will be coerced accordingly.
    """


class ParamFunctionScalar(ParamFunction1[ResultDType]):
    """
    Base class for param functions whose evaluate method produces
    a scalar value (which is the full data series).

    Implement by overriding `evaluate1`.
    """

    @final
    @override
    def evaluate(self) -> NDArray[ResultDType]:
        """
        Fully evaluate the parameter function in the current simulation context.

        Returns
        -------
        :
            The data array.
        """
        return np.array(self.evaluate1(), dtype=self.dtype)

    @abstractmethod
    def evaluate1(self) -> AttributeValue:
        """
        Produce a scalar value for this parameter in the given simulation context.

        Returns
        -------
        :
            The data value.
        """


class ParamFunctionTime(ParamFunction1[ResultDType]):
    """
    Base class for param functions whose evaluate method produces
    a time-series of data, one value at a time.

    Implement by overriding `evaluate1`.
    """

    @final
    @override
    def evaluate(self) -> NDArray[ResultDType]:
        """
        Fully evaluate the parameter function in the current simulation context.

        Returns
        -------
        :
            The data array.
        """
        result = [self.evaluate1(day) for day in range(self.time_frame.days)]
        return np.array(result, dtype=self.dtype)

    @abstractmethod
    def evaluate1(self, day: int) -> AttributeValue:
        """
        Produce the daily value for this parameter.

        Parameters
        ----------
        day :
            The simulation day (0-indexed from the start of the simulation).

        Returns
        -------
        :
            The data value.
        """


class ParamFunctionNode(ParamFunction1[ResultDType]):
    """
    Base class for param functions whose evaluate method produces
    a node-series of data, one value at a time.

    Implement by overriding `evaluate1`.
    """

    @final
    @override
    def evaluate(self) -> NDArray[ResultDType]:
        """
        Fully evaluate the parameter function in the current simulation context.

        Returns
        -------
        :
            The data array.
        """
        result = [self.evaluate1(n) for n in range(self.scope.nodes)]
        return np.array(result, dtype=self.dtype)

    @abstractmethod
    def evaluate1(self, node_index: int) -> AttributeValue:
        """
        Produce the per-node value for this parameter.

        Parameters
        ----------
        node_index :
            The node index for which to compute the value.

        Returns
        -------
        :
            The data value.
        """


class ParamFunctionNodeAndNode(ParamFunction1[ResultDType]):
    """
    Base class for param functions whose evaluate method produces
    a node-by-node matrix of data, one value at a time.

    Often, this kind of data is posed as having an originating node
    and a destination node, for example: data describing the normal
    number of commuters between locations. For some data this relationship
    may be more figurative than literal.

    Implement by overriding `evaluate1`.
    """

    @final
    @override
    def evaluate(self) -> NDArray[ResultDType]:
        """
        Fully evaluate the parameter function in the current simulation context.

        Returns
        -------
        :
            The data array.
        """
        result = [
            [self.evaluate1(n1, n2) for n2 in range(self.scope.nodes)]
            for n1 in range(self.scope.nodes)
        ]
        return np.array(result, dtype=self.dtype)

    @abstractmethod
    def evaluate1(self, node_from: int, node_to: int) -> AttributeValue:
        """
        Produce the per-node-pair value for this parameter.

        Parameters
        ----------
        node_from :
            The origin node.
        node_to :
            The destination node.

        Returns
        -------
        :
            The data value.
        """


class ParamFunctionNodeAndCompartment(ParamFunction1[ResultDType]):
    """
    Base class for param functions whose evaluate method produces
    a node-by-disease-compartment matrix of data, one value at a time.

    Implement by overriding 'evaluate1`.
    """

    @final
    @override
    def evaluate(self) -> NDArray[ResultDType]:
        """
        Fully evaluate the parameter function in the current simulation context.

        Returns
        -------
        :
            The data array.
        """
        result = [
            [self.evaluate1(n, c) for c in range(self.ipm.num_compartments)]
            for n in range(self.scope.nodes)
        ]
        return np.array(result, dtype=self.dtype)

    @abstractmethod
    def evaluate1(self, node_index: int, compartment_index: int) -> AttributeValue:
        """
        Produce the per-node and per-disease-compartment value for this parameter.

        Parameters
        ----------
        node_index :
            The node index for which to compute the value.
        compartment_index :
            The disease compartment index for which to compute the value.
            This is taken from the IPM compartment definition order, indexed from 0.

        Returns
        -------
        :
            The data value.
        """


class ParamFunctionTimeAndNode(ParamFunction1[ResultDType]):
    """
    Base class for param functions whose evaluate method produces
    a time-by-node matrix of data, one value at a time.

    Implement by overriding `evaluate1`.
    """

    @final
    @override
    def evaluate(self) -> NDArray[ResultDType]:
        """
        Fully evaluate the parameter function in the current simulation context.

        Returns
        -------
        :
            The data array.
        """
        result = [
            [self.evaluate1(day, n) for n in range(self.scope.nodes)]
            for day in range(self.time_frame.days)
        ]
        return np.array(result, dtype=self.dtype)

    @abstractmethod
    def evaluate1(self, day: int, node_index: int) -> AttributeValue:
        """
        Produce the per-day and per-node value for this parameter.

        Parameters
        ----------
        day :
            The simulation day (0-indexed from the start of the simulation).
        node_index :
            The node index for which to compute the value.

        Returns
        -------
        :
            The data value.
        """


_ALL_PARAMS = ("day", "node_index", "duration_days", "nodes")
_ALL_PARAM_SYMBOLS = [to_symbol(x) for x in _ALL_PARAMS]
_PARAMS_MAP = dict(zip(_ALL_PARAMS, _ALL_PARAM_SYMBOLS))

ParamSymbol = Literal["day", "node_index", "duration_days", "nodes"]
"""
The names of common symbols used in epymorph simulation expressions.
For example, building a `ParamExpressionTimeAndNode` may make use of these symbols.

See Also
--------
[epymorph.rume.RUME.symbols][] and [epymorph.params.simulation_symbols][]
which are methods for using these names to obtain symbol references.
"""


def simulation_symbols(*symbols: ParamSymbol) -> tuple[Symbol, ...]:
    """
    Retrieve the symbols used to represent simulation quantities.

    Parameters
    ----------
    *symbols :
        The names of symbols to retrieve, as var-args.

    Returns
    -------
    :
        A tuple containing the symbols named, in the order requested.
    """
    return tuple(_PARAMS_MAP[x] for x in symbols if x in _PARAMS_MAP)


class ParamExpressionTimeAndNode(ParamFunction[np.float64]):
    """
    A param function based on a sympy expression for a time-by-node matrix of data.

    Parameters
    ----------
    expression :
        The sympy expression to use.

    See Also
    --------
    [`epymorph.params.simulation_symbols`][] which enables you to obtain
    symbols standing for simulation properties like "day" and "node_index";
    you will likely make use of the symbols in writing the expression.
    """

    requirements = ()
    """
    Param expressions do not support data requirements, so this list must be empty.
    """
    # TODO (ISSUE #250): support data requirements

    _expr: Expr

    def __init__(self, expression: Expr):
        bad_symbols = [
            x for x in expression.free_symbols if x not in _ALL_PARAM_SYMBOLS
        ]
        if len(bad_symbols) > 0:
            bs = ", ".join(map(str, bad_symbols))
            raise ValueError(f"expression uses unsupported symbols: {bs}")
        self._expr = expression

    @final
    @override
    def evaluate(self) -> NDArray[np.float64]:
        """
        Fully evaluate the parameter function in the current simulation context.

        Returns
        -------
        :
            The data array.
        """
        T = self.time_frame.days
        N = self.scope.nodes
        ds = np.broadcast_to(np.arange(T).reshape((T, 1)), (T, N))
        ns = np.broadcast_to(np.arange(N).reshape((1, N)), (T, N))
        fn = lambdify(_ALL_PARAM_SYMBOLS, self._expr)
        result = fn([ds, ns, T, N])
        if isinstance(result, np.ndarray):
            return result.astype(dtype=np.float64, copy=False)
        return np.broadcast_to(np.array(result, dtype=np.float64), (T, N))


@evaluate_param.register
def _(
    value: Expr,
    name: AbsoluteName,
    data: DataResolver,
    scope: GeoScope | None,
    time_frame: TimeFrame | None,
    ipm: BaseCompartmentModel | None,
    rng: np.random.Generator | None,
) -> AttributeArray:
    # Automatically convert sympy expressions into a ParamFunction instance.
    try:
        expr_func = ParamExpressionTimeAndNode(value)
    except ValueError as e:
        raise DataAttributeError(str(e)) from None
    return evaluate_param(expr_func, name, data, scope, time_frame, ipm, rng)
