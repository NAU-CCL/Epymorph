# ruff: noqa: PT009,PT027
import unittest
from unittest.mock import MagicMock

import numpy as np
from numpy.typing import NDArray

import epymorph.initializer as init
from epymorph.attribute import AbsoluteName, AttributeDef
from epymorph.compartment_model import (
    BaseCompartmentModel,
    QuantitySelector,
    quick_compartments,
)
from epymorph.data_shape import Shapes
from epymorph.data_type import SimDType
from epymorph.error import DataAttributeError, InitError
from epymorph.geography.custom import CustomScope
from epymorph.time import TimeFrame

_FOOSBALL_CHAMPIONSHIPS = AttributeDef("foosball_championships", int, Shapes.N)

_POP = np.array([100, 200, 300, 400, 500], dtype=SimDType)
_POP_5x3 = np.broadcast_to(_POP[:, np.newaxis], (5, 3))


def _eval_context(additional_data: dict[str, NDArray] | None = None):
    # Creates a basic evaluation context for splatting into .with_context();
    # this should cover the needs of most test cases.
    # - Scope with 5 locations
    # - Time frame of 100 days
    # - IPM with 3 compartments: S,I,R
    # - population and foosball_championships as data attributes
    # - optional additional data attributes
    scope = CustomScope(list("ABCDE"))
    name = AbsoluteName("gpm:all", "init", "init")
    params = {
        "label": scope.node_ids,
        "population": _POP,
        "foosball_championships": np.array([2, 4, 1, 9, 6]),
        **(additional_data or {}),
    }
    time_frame = TimeFrame.of("2020-01-01", 100)
    # mock just enough of the IPM for compartment selections to work
    ipm = MagicMock(spec=BaseCompartmentModel)
    ipm.num_compartments = 3
    ipm.num_events = 2
    ipm.compartments = quick_compartments("S I R")
    ipm.select = QuantitySelector(ipm)
    return (name, params, scope, time_frame, ipm, np.random.default_rng(1))


class TestNoInfection(unittest.TestCase):
    def test_no_01(self):
        actual = init.NoInfection().with_context(*_eval_context()).evaluate()
        expected = _POP_5x3 * np.array([1, 0, 0])
        np.testing.assert_array_equal(expected, actual)

    def test_no_02(self):
        actual = init.NoInfection(1).with_context(*_eval_context()).evaluate()
        expected = _POP_5x3 * np.array([0, 1, 0])
        np.testing.assert_array_equal(expected, actual)

    def test_no_03(self):
        actual = init.NoInfection("S").with_context(*_eval_context()).evaluate()
        expected = _POP_5x3 * np.array([1, 0, 0])
        np.testing.assert_array_equal(expected, actual)

    def test_no_04(self):
        actual = init.NoInfection("R").with_context(*_eval_context()).evaluate()
        expected = _POP_5x3 * np.array([0, 0, 1])
        np.testing.assert_array_equal(expected, actual)

    def test_no_05(self):
        with self.assertRaises(InitError):
            init.NoInfection("BAD_CMPARTMNT").with_context(*_eval_context()).evaluate()


class TestExplicitInitializer(unittest.TestCase):
    def test_explicit_01(self):
        initials = np.array(
            [
                [50, 20, 30],
                [50, 120, 30],
                [100, 100, 100],
                [300, 100, 0],
                [0, 0, 500],
            ]
        )
        actual = init.Explicit(initials).with_context(*_eval_context()).evaluate()
        np.testing.assert_array_equal(actual, initials)
        self.assertIsNot(actual, initials)  # returns a copy

    def test_explicit_01b(self):
        initials = np.array(
            [
                [50, 20, 30],
                [50, 120, 30],
                [100, 100, 100],
                [300, 100, 0],
                [0, 0, 500],
            ],
            dtype=np.int32,  # test with wrong but compatible data type
        )
        actual = init.Explicit(initials).with_context(*_eval_context()).evaluate()
        np.testing.assert_array_equal(actual, initials)

    def test_explicit_02(self):
        initials = [
            [50, 20, 30],
            [50, 120, 30],
            [100, 100, 100],
            [300, 100, 0],
            [0, 0, 500],
        ]
        actual = init.Explicit(initials).with_context(*_eval_context()).evaluate()
        np.testing.assert_array_equal(actual, np.array(initials, SimDType))

    def test_explicit_03(self):
        # test wrong shape
        initials = [
            [50, 20],
            [50, 120],
            [100, 100],
            [300, 100],
            [0, 0],
        ]
        ini = init.Explicit(initials).with_context(*_eval_context())
        with self.assertRaises(InitError):
            ini.evaluate()

    def test_explicit_04(self):
        # test wrong type
        initials = [
            [50, 20, 99.99],
            [50, 120, 30],
            [100, 100, 100],
            [300, 100, 0],
            [0, 0, 500],
        ]
        ini = init.Explicit(initials).with_context(*_eval_context())
        with self.assertRaises(InitError):
            ini.evaluate()


class TestProportionalInitializer(unittest.TestCase):
    def test_proportional(self):
        # All three of these cases should be equivalent.
        # Should work if the ratios are the same as the explicit numbers.
        ratios1 = np.array(
            [
                [50, 20, 30],
                [50, 120, 30],
                [100, 100, 100],
                [300, 100, 0],
                [0, 0, 500],
            ]
        )

        ratios2 = np.array(
            [
                [5, 2, 3],
                [5, 12, 3],
                [1, 1, 1],
                [3, 1, 0],
                [0, 0, 5],
            ]
        )

        ratios3 = np.array(
            [
                [0.5, 0.2, 0.3],
                [0.25, 0.6, 0.15],
                [1 / 3, 1 / 3, 1 / 3],
                [0.75, 0.25, 0],
                [0, 0, 1],
            ]
        )

        expected = ratios1.copy()
        for ratios in [ratios1, ratios2, ratios3]:
            actual = init.Proportional(ratios).with_context(*_eval_context()).evaluate()
            np.testing.assert_array_equal(actual, expected)

    def test_shape_adapt(self):
        ratios = [1, 2, 3]
        pop = np.array([100, 200, 300, 400, 500])
        expected = (
            (pop[:, np.newaxis] * np.array([1 / 6, 1 / 3, 1 / 2]))
            .round()
            .astype(SimDType)
        )
        actual = init.Proportional(ratios).with_context(*_eval_context()).evaluate()
        np.testing.assert_array_equal(actual, expected)

    def test_bad_args(self):
        with self.assertRaises(InitError):
            # row sums to zero!
            ratios = np.array(
                [
                    [50, 20, 30],
                    [50, 120, 30],
                    [0, 0, 0],
                    [300, 100, 0],
                    [0, 0, 500],
                ]
            )
            init.Proportional(ratios).with_context(*_eval_context()).evaluate()

        with self.assertRaises(InitError):
            # row sums to negative!
            ratios = np.array(
                [
                    [50, 20, 30],
                    [50, 120, 30],
                    [0, -50, -50],
                    [300, 100, 0],
                    [0, 0, 500],
                ]
            )
            init.Proportional(ratios).with_context(*_eval_context()).evaluate()

        with self.assertRaises(InitError):
            # bad type
            ratios = np.array(
                [
                    [50, 20, True],
                    [50, 120, 30],
                    [0, 0, 0],
                    [300, 100, 0],
                    [0, 0, 500],
                ]
            )
            init.Proportional(ratios).with_context(*_eval_context()).evaluate()


class TestIndexedInitializer(unittest.TestCase):
    def test_indexed_locations(self):
        out = (
            init.IndexedLocations(
                selection=np.array(
                    [1, -2], dtype=np.intp
                ),  # Negative indices work, too.
                seed_size=100,
            )
            .with_context(*_eval_context())
            .evaluate()
        )
        # Make sure only the selected locations get infected.
        actual = out[:, 1] > 0
        expected = np.array([False, True, False, True, False])
        np.testing.assert_array_equal(actual, expected)
        # And check for 100 infected in total.
        self.assertEqual(out[:, 1].sum(), 100)

        # Repeat test with list of ints
        out = (
            init.IndexedLocations(selection=[1, -2], seed_size=100)
            .with_context(*_eval_context())
            .evaluate()
        )
        # Make sure only the selected locations get infected.
        actual = out[:, 1] > 0
        expected = np.array([False, True, False, True, False])
        np.testing.assert_array_equal(actual, expected)
        # And check for 100 infected in total.
        self.assertEqual(out[:, 1].sum(), 100)

    def test_indexed_locations_02(self):
        # test with non-default compartment configuration
        actual = (
            init.IndexedLocations(
                selection=[2],
                seed_size=100,
                initial_compartment=2,  # R
                infection_compartment="S",
            )
            .with_context(*_eval_context())
            .evaluate()
        )
        expected = np.array(
            [
                [0, 0, 100],
                [0, 0, 200],
                [100, 0, 200],
                [0, 0, 400],
                [0, 0, 500],
            ]
        )
        np.testing.assert_array_equal(expected, actual)

    def test_indexed_locations_bad(self):
        with self.assertRaises(InitError):
            # indices must be one dimension
            init.IndexedLocations(
                selection=np.array([[1], [3]], dtype=np.intp),
                seed_size=100,
            ).with_context(*_eval_context()).evaluate()
        with self.assertRaises(InitError):
            # indices must be in range (positive)
            init.IndexedLocations(
                selection=np.array([1, 8], dtype=np.intp),
                seed_size=100,
            ).with_context(*_eval_context()).evaluate()
        with self.assertRaises(InitError):
            # indices must be in range (negative)
            init.IndexedLocations(
                selection=np.array([1, -8], dtype=np.intp),
                seed_size=100,
            ).with_context(*_eval_context()).evaluate()


class TestLabeledInitializer(unittest.TestCase):
    def test_labeled_locations(self):
        out = (
            init.LabeledLocations(
                labels=np.array(["B", "D"]),
                seed_size=100,
            )
            .with_context(*_eval_context())
            .evaluate()
        )
        # Make sure only the selected locations get infected.
        actual = out[:, 1] > 0
        expected = np.array([False, True, False, True, False])
        np.testing.assert_array_equal(actual, expected)
        # And check for 100 infected in total.
        self.assertEqual(out[:, 1].sum(), 100)

        # repeat test with list
        out = (
            init.LabeledLocations(["B", "D"], seed_size=100)
            .with_context(*_eval_context())
            .evaluate()
        )
        # Make sure only the selected locations get infected.
        actual = out[:, 1] > 0
        expected = np.array([False, True, False, True, False])
        np.testing.assert_array_equal(actual, expected)
        # And check for 100 infected in total.
        self.assertEqual(out[:, 1].sum(), 100)

    def test_labeled_locations_02(self):
        # test with non-default compartment configuration
        actual = (
            init.LabeledLocations(
                labels=["C"],
                seed_size=100,
                initial_compartment=2,  # R
                infection_compartment="S",
            )
            .with_context(*_eval_context())
            .evaluate()
        )
        expected = np.array(
            [
                [0, 0, 100],
                [0, 0, 200],
                [100, 0, 200],
                [0, 0, 400],
                [0, 0, 500],
            ]
        )
        np.testing.assert_array_equal(expected, actual)

    def test_labeled_locations_bad(self):
        with self.assertRaises(InitError):
            init.LabeledLocations(
                labels=np.array(["A", "B", "Z"]),
                seed_size=100,
            ).with_context(*_eval_context()).evaluate()


class TestSingleLocationInitializer(unittest.TestCase):
    def test_single_loc_01(self):
        exp = np.array(
            [
                [100, 0, 0],
                [200, 0, 0],
                [201, 99, 0],
                [400, 0, 0],
                [500, 0, 0],
            ]
        )
        act = (
            init.SingleLocation(
                location=2,
                seed_size=99,
            )
            .with_context(*_eval_context())
            .evaluate()
        )
        np.testing.assert_array_equal(act, exp)

    def test_single_loc_02(self):
        expected = np.array(
            [
                [0, 100, 0],
                [0, 200, 0],
                [0, 201, 99],
                [0, 400, 0],
                [0, 500, 0],
            ]
        )
        actual = (
            init.SingleLocation(
                location=2,
                seed_size=99,
                initial_compartment="I",
                infection_compartment="R",
            )
            .with_context(*_eval_context())
            .evaluate()
        )
        np.testing.assert_array_equal(actual, expected)


class TestTopInitializer(unittest.TestCase):
    def test_top(self):
        out = (
            init.TopLocations(
                top_attribute=_FOOSBALL_CHAMPIONSHIPS,
                num_locations=3,
                seed_size=100,
            )
            .with_context(*_eval_context())
            .evaluate()
        )
        act = out[:, 1] > 0
        exp = np.array([False, True, False, True, True])
        np.testing.assert_array_equal(act, exp)

    def test_top_02(self):
        actual = (
            init.TopLocations(
                top_attribute=_FOOSBALL_CHAMPIONSHIPS,
                num_locations=1,
                seed_size=99,
                initial_compartment="I",
                infection_compartment="R",
            )
            .with_context(*_eval_context())
            .evaluate()
        )
        expected = np.array(
            [
                [0, 100, 0],
                [0, 200, 0],
                [0, 300, 0],
                [0, 301, 99],
                [0, 500, 0],
            ]
        )
        np.testing.assert_array_equal(expected, actual)

    def test_missing_attribute(self):
        with self.assertRaises(DataAttributeError):
            # we didn't provide quidditch_championships data!
            i = init.TopLocations(
                top_attribute=AttributeDef("quidditch_championships", int, Shapes.N),
                num_locations=3,
                seed_size=100,
            )
            i.with_context(*_eval_context()).evaluate()

    def test_wrong_type_attribute(self):
        with self.assertRaises(DataAttributeError):
            # we asked for an int array, but the data is a float array
            i = init.TopLocations(
                top_attribute=AttributeDef("quidditch_championships", int, Shapes.N),
                num_locations=3,
                seed_size=100,
            )
            i.with_context(
                *_eval_context(
                    {
                        "quidditch_championships": np.array(
                            [1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64
                        ),
                    }
                )
            ).evaluate()

    def test_invalid_size_attribute(self):
        with self.assertRaises(InitError):
            # what does "top" mean in an NxN array? too ambiguous
            i = init.TopLocations(
                top_attribute=AttributeDef("quidditch_relative_rank", int, Shapes.NxN),
                num_locations=3,
                seed_size=100,
            )
            i.with_context(
                *_eval_context(
                    {
                        "quidditch_relative_rank": np.arange(
                            25, dtype=np.float64
                        ).reshape((5, 5)),
                    }
                )
            ).evaluate()


class TestBottomInitializer(unittest.TestCase):
    def test_bottom(self):
        out = (
            init.BottomLocations(
                bottom_attribute=_FOOSBALL_CHAMPIONSHIPS,
                num_locations=3,
                seed_size=100,
            )
            .with_context(*_eval_context())
            .evaluate()
        )
        act = out[:, 1] > 0
        exp = np.array([True, True, True, False, False])
        np.testing.assert_array_equal(act, exp)

    def test_bottom_02(self):
        actual = (
            init.BottomLocations(
                bottom_attribute=_FOOSBALL_CHAMPIONSHIPS,
                num_locations=1,
                seed_size=99,
                initial_compartment="I",
                infection_compartment="R",
            )
            .with_context(*_eval_context())
            .evaluate()
        )
        expected = np.array(
            [
                [0, 100, 0],
                [0, 200, 0],
                [0, 201, 99],
                [0, 400, 0],
                [0, 500, 0],
            ]
        )
        np.testing.assert_array_equal(expected, actual)

    def test_missing_attribute(self):
        with self.assertRaises(DataAttributeError):
            # we didn't provide quidditch_championships data!
            i = init.BottomLocations(
                bottom_attribute=AttributeDef("quidditch_championships", int, Shapes.N),
                num_locations=3,
                seed_size=100,
            )
            i.with_context(*_eval_context()).evaluate()

    def test_invalid_size_attribute(self):
        with self.assertRaises(InitError):
            # what does "bottom" mean in an NxN array? too ambiguous
            i = init.BottomLocations(
                bottom_attribute=AttributeDef(
                    "quidditch_relative_rank", int, Shapes.NxN
                ),
                num_locations=3,
                seed_size=100,
            )
            i.with_context(
                *_eval_context(
                    {
                        "quidditch_relative_rank": np.arange(
                            25, dtype=np.float64
                        ).reshape((5, 5)),
                    }
                )
            ).evaluate()


class TestCustomInitializer(unittest.TestCase):
    def test_validation(self):
        # Built-in initializer validation should catch that we're
        # returning negative values.
        class MyInit(init.Initializer):
            def evaluate(self):
                return np.array([-1])

        with self.assertRaises(InitError):
            MyInit().with_context(*_eval_context()).evaluate()
