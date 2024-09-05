# pylint: disable=missing-docstring,unused-variable
import unittest
from datetime import date
from functools import cached_property
from unittest.mock import MagicMock

import numpy as np

from epymorph.data_shape import Shapes
from epymorph.geography.scope import GeoScope
from epymorph.simulation import (
    AttributeDef,
    NamespacedAttributeResolver,
    SimDimensions,
    SimulationFunction,
    Tick,
    simulation_clock,
)


class TestClock(unittest.TestCase):
    def test_clock(self):
        dim = SimDimensions.build(
            tau_step_lengths=[2 / 3, 1 / 3],
            start_date=date(2023, 1, 1),
            days=6,
            # sim clock doesn't depend on GEO/IPM dimensions
            nodes=99,
            compartments=99,
            events=99,
        )
        clock = simulation_clock(dim)
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
    def context(self, bar: int):
        data = MagicMock(spec=NamespacedAttributeResolver)
        data.resolve.return_value = np.array([bar])
        dim = MagicMock(spec=SimDimensions)
        scope = MagicMock(spec=GeoScope)
        rng = MagicMock(spec=np.random.Generator)
        return (data, dim, scope, rng)

    def test_basic_usage(self):
        class Foo(SimulationFunction[int]):
            requirements = [AttributeDef("bar", int, Shapes.S)]

            baz: int

            def __init__(self, baz: int):
                self.baz = baz

            def evaluate(self) -> int:
                return 7 * self.baz * self.data("bar")[0]

        f = Foo(3)

        self.assertIsInstance(Foo.requirements, tuple)

        self.assertEqual(42, f.evaluate_in_context(*self.context(bar=2)))

        with self.assertRaises(TypeError) as e:
            f.evaluate()
        self.assertIn("invalid access of function context", str(e.exception).lower())

    def test_immutable_requirements(self):
        class Foo(SimulationFunction[int]):
            requirements = [AttributeDef("bar", int, Shapes.S)]

            def evaluate(self) -> int:
                return 7 * self.data("bar")[0]

        f = Foo()
        self.assertEqual(Foo.requirements, f.requirements)
        self.assertIsInstance(Foo.requirements, tuple)
        self.assertIsInstance(f.requirements, tuple)

    def test_undefined_requirement(self):
        class Foo(SimulationFunction[int]):
            requirements = [AttributeDef("bar", int, Shapes.S)]

            def evaluate(self) -> int:
                return 7 * self.data("quux")[0]

        with self.assertRaises(ValueError) as e:
            Foo().evaluate_in_context(*self.context(bar=2))
        self.assertIn("did not declare as a requirement", str(e.exception).lower())

    def test_bad_definition(self):
        with self.assertRaises(TypeError) as e:

            class Foo1(SimulationFunction[int]):
                requirements = "hey"  # type: ignore

                def evaluate(self) -> int:
                    return 42

        self.assertIn("invalid requirements", str(e.exception).lower())

        with self.assertRaises(TypeError) as e:

            class Foo2(SimulationFunction[int]):
                requirements = ["hey"]  # type: ignore

                def evaluate(self) -> int:
                    return 42

        self.assertIn("invalid requirements", str(e.exception).lower())

        with self.assertRaises(TypeError) as e:

            class Foo3(SimulationFunction[int]):
                requirements = [
                    AttributeDef("foo", int, Shapes.S),
                    AttributeDef("foo", int, Shapes.S),
                ]

                def evaluate(self) -> int:
                    return 42

        self.assertIn("invalid requirements", str(e.exception).lower())

    def test_cached_properties(self):
        class Foo(SimulationFunction[int]):
            requirements = [AttributeDef("bar", int, Shapes.S)]

            @cached_property
            def baz(self):
                return self.data("bar")[0] * 2

            def evaluate(self) -> int:
                return 7 * self.baz

        f = Foo()

        self.assertEqual(42, f.evaluate_in_context(*self.context(bar=3)))
        self.assertEqual(84, f.evaluate_in_context(*self.context(bar=6)))
