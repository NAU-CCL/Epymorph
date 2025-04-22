# ruff: noqa: PT009,PT027
import unittest
from datetime import date
from functools import cached_property
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from epymorph.attribute import AbsoluteName, AttributeDef
from epymorph.data_shape import Shapes
from epymorph.error import MissingContextError
from epymorph.simulation import (
    SimulationFunction,
    Tick,
    simulation_clock,
)
from epymorph.time import TimeFrame


class TestClock(unittest.TestCase):
    def test_clock(self):
        tau_step_lengths = [2 / 3, 1 / 3]
        time_frame = TimeFrame.of("2023-01-01", 6)
        clock = simulation_clock(time_frame, tau_step_lengths)
        act = list(clock)
        exp = [
            Tick(0, 0, date(2023, 1, 1), 0, 2 / 3),
            Tick(1, 0, date(2023, 1, 1), 1, 1 / 3),
            Tick(2, 1, date(2023, 1, 2), 0, 2 / 3),
            Tick(3, 1, date(2023, 1, 2), 1, 1 / 3),
            Tick(4, 2, date(2023, 1, 3), 0, 2 / 3),
            Tick(5, 2, date(2023, 1, 3), 1, 1 / 3),
            Tick(6, 3, date(2023, 1, 4), 0, 2 / 3),
            Tick(7, 3, date(2023, 1, 4), 1, 1 / 3),
            Tick(8, 4, date(2023, 1, 5), 0, 2 / 3),
            Tick(9, 4, date(2023, 1, 5), 1, 1 / 3),
            Tick(10, 5, date(2023, 1, 6), 0, 2 / 3),
            Tick(11, 5, date(2023, 1, 6), 1, 1 / 3),
        ]
        self.assertEqual(act, exp)


class TestSimulationFunction(unittest.TestCase):
    def _eval_context(self, bar: int):
        name = AbsoluteName("gpm:all", "foo", "foo")
        data = {"bar": np.asarray(bar)}
        scope = None
        time_frame = None
        ipm = None
        rng = None
        return (name, data, scope, time_frame, ipm, rng)

    def test_basic_usage(self):
        class Foo(SimulationFunction[NDArray[np.int64]]):
            requirements = [AttributeDef("bar", int, Shapes.Scalar)]

            baz: int

            def __init__(self, baz: int):
                self.baz = baz

            def evaluate(self):
                return 7 * self.baz * self.data("bar").astype(np.int64)

        f = Foo(3)

        self.assertIsInstance(Foo.requirements, tuple)

        self.assertEqual(42, f.with_context(*self._eval_context(bar=2)).evaluate())

        with self.assertRaises(MissingContextError) as e:
            f.evaluate()
        self.assertIn(
            "missing function context 'data' during evaluation",
            str(e.exception).lower(),
        )

    def test_immutable_requirements(self):
        class Foo(SimulationFunction[NDArray[np.int64]]):
            requirements = [AttributeDef("bar", int, Shapes.Scalar)]

            def evaluate(self):
                return 7 * self.data("bar")

        f = Foo()
        self.assertEqual(Foo.requirements, f.requirements)
        self.assertIsInstance(Foo.requirements, tuple)
        self.assertIsInstance(f.requirements, tuple)

    def test_dynamic_requirements(self):
        class Foo(SimulationFunction[NDArray[np.int64]]):
            @property
            def requirements(self) -> Sequence[AttributeDef]:
                return (AttributeDef("bar", int, Shapes.Scalar),)

            def evaluate(self):
                return 7 * self.data("bar")

        f = Foo()
        self.assertEqual(f.requirements, (AttributeDef("bar", int, Shapes.Scalar),))

    def test_undefined_requirement(self):
        class Foo(SimulationFunction[NDArray[np.int64]]):
            requirements = [AttributeDef("bar", int, Shapes.Scalar)]

            def evaluate(self):
                return 7 * self.data("quux")

        with self.assertRaises(ValueError) as e:
            Foo().with_context(*self._eval_context(bar=2)).evaluate()
        self.assertIn("did not declare as a requirement", str(e.exception).lower())

    def test_bad_definition(self):
        with self.assertRaises(TypeError) as e:

            class Foo1(SimulationFunction[NDArray[np.int64]]):
                requirements = "hey"  # type: ignore

                def evaluate(self):
                    return np.asarray(42)

        self.assertIn("invalid requirements", str(e.exception).lower())

        with self.assertRaises(TypeError) as e:

            class Foo2(SimulationFunction[NDArray[np.int64]]):
                requirements = ["hey"]  # type: ignore

                def evaluate(self):
                    return np.asarray(42)

        self.assertIn("invalid requirements", str(e.exception).lower())

        with self.assertRaises(TypeError) as e:

            class Foo3(SimulationFunction[NDArray[np.int64]]):
                requirements = [
                    AttributeDef("foo", int, Shapes.Scalar),
                    AttributeDef("foo", int, Shapes.Scalar),
                ]

                def evaluate(self):
                    return np.asarray(42)

        self.assertIn("invalid requirements", str(e.exception).lower())

    def test_cached_properties(self):
        class Foo(SimulationFunction[NDArray[np.int64]]):
            requirements = [AttributeDef("bar", int, Shapes.Scalar)]

            @cached_property
            def baz(self):
                return self.data("bar") * 2

            def evaluate(self):
                return 7 * self.baz

        f = Foo()

        self.assertEqual(42, f.with_context(*self._eval_context(bar=3)).evaluate())
        self.assertEqual(84, f.with_context(*self._eval_context(bar=6)).evaluate())
