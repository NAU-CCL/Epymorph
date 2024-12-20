# pylint: disable=missing-docstring
import unittest
from datetime import date

import numpy as np
from numpy.typing import NDArray

import epymorph.initializer as init
from epymorph.data_shape import Shapes, SimDimensions
from epymorph.data_type import SimDType
from epymorph.database import ModuleNamespace
from epymorph.error import AttributeException, InitException
from epymorph.geography.custom import CustomScope
from epymorph.simulation import AttributeDef


def _eval_context(additional_data: dict[str, NDArray] | None = None):
    scope = CustomScope(list("ABCDE"))
    dim = SimDimensions.build(
        tau_step_lengths=[1 / 3, 2 / 3],
        start_date=date(2020, 1, 1),
        days=100,
        nodes=scope.nodes,
        compartments=3,
        events=3,
    )
    params = {
        "label": scope.node_ids,
        "population": np.array([100, 200, 300, 400, 500], dtype=SimDType),
        "foosball_championships": np.array([2, 4, 1, 9, 6]),
        **(additional_data or {}),
    }
    namespace = ModuleNamespace("gpm:all", "init")
    return (namespace, params, dim, scope, np.random.default_rng(1))


_FOOSBALL_CHAMPIONSHIPS = AttributeDef("foosball_championships", int, Shapes.N)


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
        with self.assertRaises(InitException):
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
        with self.assertRaises(InitException):
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
        with self.assertRaises(InitException):
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

        with self.assertRaises(InitException):
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

        with self.assertRaises(InitException):
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

    def test_indexed_locations_bad(self):
        with self.assertRaises(InitException):
            # indices must be one dimension
            init.IndexedLocations(
                selection=np.array([[1], [3]], dtype=np.intp),
                seed_size=100,
            ).with_context(*_eval_context()).evaluate()
        with self.assertRaises(InitException):
            # indices must be in range (positive)
            init.IndexedLocations(
                selection=np.array([1, 8], dtype=np.intp),
                seed_size=100,
            ).with_context(*_eval_context()).evaluate()
        with self.assertRaises(InitException):
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

    def test_labeled_locations_bad(self):
        with self.assertRaises(InitException):
            init.LabeledLocations(
                labels=np.array(["A", "B", "Z"]),
                seed_size=100,
            ).with_context(*_eval_context()).evaluate()


class TestSingleInitializer(unittest.TestCase):
    def test_single_loc(self):
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

    def test_missing_attribute(self):
        with self.assertRaises(AttributeException):
            # we didn't provide quidditch_championships data!
            i = init.TopLocations(
                top_attribute=AttributeDef("quidditch_championships", int, Shapes.N),
                num_locations=3,
                seed_size=100,
            )
            i.with_context(*_eval_context()).evaluate()

    def test_wrong_type_attribute(self):
        with self.assertRaises(AttributeException):
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
        with self.assertRaises(InitException):
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

    def test_missing_attribute(self):
        with self.assertRaises(AttributeException):
            # we didn't provide quidditch_championships data!
            i = init.BottomLocations(
                bottom_attribute=AttributeDef("quidditch_championships", int, Shapes.N),
                num_locations=3,
                seed_size=100,
            )
            i.with_context(*_eval_context()).evaluate()

    def test_invalid_size_attribute(self):
        with self.assertRaises(InitException):
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

        with self.assertRaises(InitException):
            MyInit().with_context(*_eval_context()).evaluate()
