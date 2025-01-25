import unittest
from math import cos
from unittest.mock import MagicMock

import numpy as np
import numpy.testing as npt
import sympy
from numpy.typing import NDArray

from epymorph.attribute import AbsoluteName
from epymorph.compartment_model import BaseCompartmentModel
from epymorph.data_type import AttributeValue
from epymorph.geography.custom import CustomScope
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
from epymorph.time import TimeFrame


class ParamFunctionsTest(unittest.TestCase):
    def _eval_context(self):
        ipm = MagicMock(spec=BaseCompartmentModel)
        ipm.num_compartments = 3
        ipm.num_events = 2
        scope = CustomScope(["a", "b", "c", "d"])
        return (
            AbsoluteName("gpm:all", "ipm", "test"),
            {},
            scope,
            TimeFrame.of("2021-01-01", 100),
            ipm,
            np.random.default_rng(1),
        )

    def test_numpy_1(self):
        class TestFunc(ParamFunctionNumpy):
            def evaluate(self) -> NDArray[np.int64]:
                return np.arange(400).reshape((4, 100))

        f = TestFunc()

        npt.assert_array_equal(
            f.with_context(*self._eval_context()).evaluate(),
            np.arange(400).reshape((4, 100)),
        )

    def test_scalar_1(self):
        class TestFunc(ParamFunctionScalar):
            dtype = np.float64

            def evaluate1(self) -> AttributeValue:
                return 42.0

        f = TestFunc()

        npt.assert_array_equal(
            f.with_context(*self._eval_context()).evaluate(),
            np.array(42.0, dtype=np.float64),
        )

    def test_time_1(self):
        class TestFunc(ParamFunctionTime):
            dtype = np.float64

            def evaluate1(self, day: int) -> AttributeValue:
                return 2 * day

        f = TestFunc()

        npt.assert_array_equal(
            f.with_context(*self._eval_context()).evaluate(),
            2 * np.arange(100, dtype=np.float64),
        )

    def test_node_1(self):
        class TestFunc(ParamFunctionNode):
            dtype = np.float64

            def evaluate1(self, node_index: int) -> AttributeValue:
                return 3 * node_index

        f = TestFunc()

        npt.assert_array_equal(
            f.with_context(*self._eval_context()).evaluate(),
            3 * np.arange(4, dtype=np.float64),
        )

    def test_node_and_node_1(self):
        class TestFunc(ParamFunctionNodeAndNode):
            dtype = np.int64

            def evaluate1(self, node_from: int, node_to: int) -> AttributeValue:
                return node_from * 10 + node_to

        f = TestFunc()

        npt.assert_array_equal(
            f.with_context(*self._eval_context()).evaluate(),
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
        class TestFunc(ParamFunctionNodeAndCompartment):
            dtype = np.int64

            def evaluate1(
                self, node_index: int, compartment_index: int
            ) -> AttributeValue:
                return node_index * 10 + compartment_index

        f = TestFunc()

        npt.assert_array_equal(
            f.with_context(*self._eval_context()).evaluate(),
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
        class TestFunc(ParamFunctionTimeAndNode):
            dtype = np.float64

            def evaluate1(self, day: int, node_index: int) -> AttributeValue:
                return (1.0 + node_index) * cos(day / self.time_frame.days)

        f = TestFunc()

        cosine = np.cos(np.arange(100) / 100, dtype=np.float64)
        npt.assert_array_equal(
            f.with_context(*self._eval_context()).evaluate(),
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
        d, n, days = simulation_symbols("day", "node_index", "duration_days")
        f = ParamExpressionTimeAndNode((1.0 + n) * sympy.cos(d / days))

        cosine = np.cos(np.arange(100) / 100, dtype=np.float64)
        npt.assert_array_equal(
            f.with_context(*self._eval_context()).evaluate(),
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
