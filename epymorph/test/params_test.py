# pylint: disable=missing-docstring
import unittest
from datetime import date
from math import cos

import numpy as np
import numpy.testing as npt
import sympy
from numpy.typing import NDArray

from epymorph.data_shape import SimDimensions
from epymorph.data_type import AttributeArray, AttributeValue
from epymorph.database import Database, ModuleNamespace
from epymorph.geography.custom import CustomScope
from epymorph.geography.scope import GeoScope
from epymorph.params import (
    ParamExpressionTimeAndNode,
    ParamFunctionNode,
    ParamFunctionNodeAndCompartment,
    ParamFunctionNodeAndNode,
    ParamFunctionNumpy,
    ParamFunctionScalar,
    ParamFunctionTime,
    ParamFunctionTimeAndNode,
    simulation_symbols,
)
from epymorph.simulation import NamespacedAttributeResolver


class ParamFunctionsTest(unittest.TestCase):
    def _dim_data_scope(
        self,
    ) -> tuple[SimDimensions, NamespacedAttributeResolver, GeoScope]:
        dim = SimDimensions.build(
            tau_step_lengths=[1 / 3, 2 / 3],
            start_date=date(2021, 1, 1),
            days=100,
            nodes=4,
            compartments=3,
            events=2,
        )
        data = NamespacedAttributeResolver(
            data=Database[AttributeArray]({}),
            dim=dim,
            namespace=ModuleNamespace("gpm:all", "ipm"),
        )
        scope = CustomScope(["a", "b", "c", "d"])
        return dim, data, scope

    def test_numpy_1(self):
        dim, data, scope = self._dim_data_scope()

        class TestFunc(ParamFunctionNumpy):
            def evaluate(self) -> NDArray[np.int64]:
                return np.arange(400).reshape((4, 100))

        f = TestFunc()

        npt.assert_array_equal(
            f.evaluate_in_context(data, dim, scope, np.random.default_rng(1)),
            np.arange(400).reshape((4, 100)),
        )

    def test_scalar_1(self):
        dim, data, scope = self._dim_data_scope()

        class TestFunc(ParamFunctionScalar):
            dtype = np.float64

            def evaluate1(self) -> AttributeValue:
                return 42.0

        f = TestFunc()

        npt.assert_array_equal(
            f.evaluate_in_context(data, dim, scope, np.random.default_rng(1)),
            np.array(42.0, dtype=np.float64),
        )

    def test_time_1(self):
        dim, data, scope = self._dim_data_scope()

        class TestFunc(ParamFunctionTime):
            dtype = np.float64

            def evaluate1(self, day: int) -> AttributeValue:
                return 2 * day

        f = TestFunc()

        npt.assert_array_equal(
            f.evaluate_in_context(data, dim, scope, np.random.default_rng(1)),
            2 * np.arange(100, dtype=np.float64),
        )

    def test_node_1(self):
        dim, data, scope = self._dim_data_scope()

        class TestFunc(ParamFunctionNode):
            dtype = np.float64

            def evaluate1(self, node_index: int) -> AttributeValue:
                return 3 * node_index

        f = TestFunc()

        npt.assert_array_equal(
            f.evaluate_in_context(data, dim, scope, np.random.default_rng(1)),
            3 * np.arange(4, dtype=np.float64),
        )

    def test_node_and_node_1(self):
        dim, data, scope = self._dim_data_scope()

        class TestFunc(ParamFunctionNodeAndNode):
            dtype = np.int64

            def evaluate1(self, node_from: int, node_to: int) -> AttributeValue:
                return node_from * 10 + node_to

        f = TestFunc()

        npt.assert_array_equal(
            f.evaluate_in_context(data, dim, scope, np.random.default_rng(1)),
            np.array(
                [
                    [0, 1, 2, 3],
                    [10, 11, 12, 13],
                    [20, 21, 22, 23],
                    [30, 31, 32, 33],
                ],
                dtype=np.int64,
            ),
        )

    def test_node_and_compartment_1(self):
        dim, data, scope = self._dim_data_scope()

        class TestFunc(ParamFunctionNodeAndCompartment):
            dtype = np.int64

            def evaluate1(
                self, node_index: int, compartment_index: int
            ) -> AttributeValue:
                return node_index * 10 + compartment_index

        f = TestFunc()

        npt.assert_array_equal(
            f.evaluate_in_context(data, dim, scope, np.random.default_rng(1)),
            np.array(
                [
                    [0, 1, 2],
                    [10, 11, 12],
                    [20, 21, 22],
                    [30, 31, 32],
                ],
                dtype=np.int64,
            ),
        )

    def test_time_and_node_1(self):
        dim, data, scope = self._dim_data_scope()

        class TestFunc(ParamFunctionTimeAndNode):
            dtype = np.float64

            def evaluate1(self, day: int, node_index: int) -> AttributeValue:
                return (1.0 + node_index) * cos(day / self.dim.days)

        f = TestFunc()

        cosine = np.cos(np.arange(100) / 100, dtype=np.float64)
        npt.assert_array_equal(
            f.evaluate_in_context(data, dim, scope, np.random.default_rng(1)),
            np.stack(
                [
                    1.0 * cosine,
                    2.0 * cosine,
                    3.0 * cosine,
                    4.0 * cosine,
                ],
                axis=1,
                dtype=np.float64,
            ),
        )

    def test_expr_time_and_node_1(self):
        dim, data, scope = self._dim_data_scope()

        d, n, days = simulation_symbols("day", "node_index", "duration_days")
        f = ParamExpressionTimeAndNode((1.0 + n) * sympy.cos(d / days))

        cosine = np.cos(np.arange(100) / 100, dtype=np.float64)
        npt.assert_array_equal(
            f.evaluate_in_context(data, dim, scope, np.random.default_rng(1)),
            np.stack(
                [
                    1.0 * cosine,
                    2.0 * cosine,
                    3.0 * cosine,
                    4.0 * cosine,
                ],
                axis=1,
                dtype=np.float64,
            ),
        )
