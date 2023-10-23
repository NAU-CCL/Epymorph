import dataclasses
import unittest
from datetime import date
from functools import partial

import numpy as np

from epymorph.clock import Clock
from epymorph.context import SimContext, SimDType
from epymorph.data_shape import Shapes
from epymorph.geo.spec import LABEL, NO_DURATION, AttribDef
from epymorph.geo.static import StaticGeo, StaticGeoSpec
from epymorph.initializer import (InitializerException, bottom_locations,
                                  explicit, indexed_locations, initialize,
                                  labeled_locations, proportional,
                                  single_location, top_locations)


def test_context():
    geo = StaticGeo(
        spec=StaticGeoSpec(
            attributes=[
                LABEL,
                AttribDef('population', np.int64, Shapes.N),
                AttribDef('foosball_championships', np.int64, Shapes.N),
            ],
            time_period=NO_DURATION,
        ),
        values={
            'population': np.array([100, 200, 300, 400, 500], dtype=SimDType),
            'label': np.array(list('ABCDE'), dtype=np.str_),
            'foosball_championships': np.array([2, 4, 1, 9, 6]),
        })
    return SimContext(
        geo=geo,
        compartments=3,
        events=3,
        param={},
        compartment_tags=[[], [], []],
        clock=Clock(date(2019, 1, 1), 20, [.5, .5]),
        rng=np.random.default_rng(1)
    )


def assert_array_equal(testcase, test, expected):
    """Tests if two numpy arrays are equal."""
    testcase.assertTrue(
        np.array_equal(test, expected),
        f"""arrays not equal:
expected: {expected}
received: {test}"""
    )


class TestExplicitInitializer(unittest.TestCase):

    def test_explicit(self):
        initials = np.array([
            [50, 20, 30],
            [50, 120, 30],
            [100, 100, 100],
            [300, 100, 0],
            [0, 0, 500],
        ])
        out = explicit(test_context(), initials)
        assert_array_equal(self, out, initials)


class TestProportionalInitializer(unittest.TestCase):

    def test_proportional(self):
        # All three of these cases should be equivalent.
        # Should work if the ratios are the same as the explicit numbers.
        ratios1 = np.array([
            [50, 20, 30],
            [50, 120, 30],
            [100, 100, 100],
            [300, 100, 0],
            [0, 0, 500],
        ])

        ratios2 = np.array([
            [5, 2, 3],
            [5, 12, 3],
            [1, 1, 1],
            [3, 1, 0],
            [0, 0, 5],
        ])

        ratios3 = np.array([
            [.5, .2, .3],
            [.25, .6, .15],
            [1/3, 1/3, 1/3],
            [0.75, 0.25, 0],
            [0, 0, 1],
        ])

        expected = ratios1
        for rs in [ratios1, ratios2, ratios3]:
            out = proportional(test_context(), ratios=rs)
            assert_array_equal(self, out, expected)

    def test_bad_args(self):
        with self.assertRaises(InitializerException):
            # row sums to zero!
            ratios = np.array([
                [50, 20, 30],
                [50, 120, 30],
                [0, 0, 0],
                [300, 100, 0],
                [0, 0, 500],
            ])
            proportional(test_context(), ratios=ratios)

        with self.assertRaises(InitializerException):
            # row sums to negative!
            ratios = np.array([
                [50, 20, 30],
                [50, 120, 30],
                [0, -50, -50],
                [300, 100, 0],
                [0, 0, 500],
            ])
            proportional(test_context(), ratios=ratios)


class TestIndexedInitializer(unittest.TestCase):

    def test_indexed_locations(self):
        out = indexed_locations(
            test_context(),
            selection=np.array([1, -2], dtype=np.intp),  # Negative indices work, too.
            seed_size=100
        )
        # Make sure only the selected locations get infected.
        act = out[:, 1] > 0
        exp = np.array([False, True, False, True, False])
        assert_array_equal(self, act, exp)
        # And check for 100 infected in total.
        self.assertEqual(out[:, 1].sum(), 100)

    def test_indexed_locations_bad(self):
        with self.assertRaises(InitializerException):
            # indices must be one dimension
            indexed_locations(
                test_context(),
                selection=np.array([[1], [3]], dtype=np.intp),
                seed_size=100
            )
        with self.assertRaises(InitializerException):
            # indices must be in range (positive)
            indexed_locations(
                test_context(),
                selection=np.array([1, 8], dtype=np.intp),
                seed_size=100
            )
        with self.assertRaises(InitializerException):
            # indices must be in range (negative)
            indexed_locations(
                test_context(),
                selection=np.array([1, -8], dtype=np.intp),
                seed_size=100
            )


class TestLabeledInitializer(unittest.TestCase):

    def test_labeled_locations(self):
        out = labeled_locations(
            test_context(),
            labels=np.array(["B", "D"]),
            seed_size=100
        )
        # Make sure only the selected locations get infected.
        act = out[:, 1] > 0
        exp = np.array([False, True, False, True, False])
        assert_array_equal(self, act, exp)
        # And check for 100 infected in total.
        self.assertEqual(out[:, 1].sum(), 100)

    def test_labeled_locations_bad(self):
        with self.assertRaises(InitializerException):
            labeled_locations(
                test_context(),
                labels=np.array(["A", "B", "Z"]),
                seed_size=100
            )


class TestSingleInitializer(unittest.TestCase):

    def test_single_loc(self):
        exp = np.array([
            [100, 0, 0],
            [200, 0, 0],
            [201, 99, 0],
            [400, 0, 0],
            [500, 0, 0],
        ])
        act = single_location(
            test_context(),
            location=2,
            seed_size=99
        )
        assert_array_equal(self, act, exp)


class TestTopInitializer(unittest.TestCase):

    def test_top(self):
        out = top_locations(
            test_context(),
            attribute='foosball_championships',
            num_locations=3,
            seed_size=100
        )
        act = out[:, 1] > 0
        exp = np.array([False, True, False, True, True])
        assert_array_equal(self, act, exp)


class TestBottomInitializer(unittest.TestCase):

    def test_bottom(self):
        out = bottom_locations(
            test_context(),
            attribute='foosball_championships',
            num_locations=3,
            seed_size=100
        )
        act = out[:, 1] > 0
        exp = np.array([True, True, True, False, False])
        assert_array_equal(self, act, exp)


class TestInitialize(unittest.TestCase):

    def test_initialize_01(self):
        # Mostly to determine if auto-wiring from params works as expected.
        ctx = dataclasses.replace(
            test_context(),
            param={
                'selection': np.array([1, 3], dtype=np.intp),
                'seed_size': 100
            }
        )
        out = initialize(indexed_locations, ctx)
        act = out[:, 1] > 0
        exp = np.array([False, True, False, True, False])
        assert_array_equal(self, act, exp)

    def test_initialize_02(self):
        # Mixing partial with some params.
        ctx = dataclasses.replace(
            test_context(),
            param={
                'selection': np.array([1, 3], dtype=np.intp),
            }
        )
        ini = partial(indexed_locations, seed_size=100)
        out = initialize(ini, ctx)
        # Make sure only the selected locations get infected.
        act = out[:, 1] > 0
        exp = np.array([False, True, False, True, False])
        assert_array_equal(self, act, exp)
        # And check for 100 infected in total.
        self.assertEqual(out[:, 1].sum(), 100)

    def test_initialize_03(self):
        # Partial-provided args should take precedence over params.
        ctx = dataclasses.replace(
            test_context(),
            param={
                'selection': np.array([2, 4], dtype=np.intp),
                'seed_size': 100,
            }
        )
        ini = partial(indexed_locations, selection=np.array([1, 3], dtype=np.intp))
        out = initialize(ini, ctx)
        # Make sure only the selected locations get infected.
        act = out[:, 1] > 0
        exp = np.array([False, True, False, True, False])
        assert_array_equal(self, act, exp)

    def test_initialize_bad(self):
        with self.assertRaises(InitializerException):
            # Missing param
            ctx = test_context()
            initialize(single_location, ctx)

        with self.assertRaises(InitializerException):
            # Bad param type
            ctx = dataclasses.replace(test_context(), param={'location': 13})
            initialize(single_location, ctx)

        with self.assertRaises(InitializerException):
            # Bad param type
            ctx = dataclasses.replace(test_context(), param={'location': [1, 2]})
            initialize(single_location, ctx)

        with self.assertRaises(InitializerException):
            # Bad param type
            ctx = dataclasses.replace(test_context(), param={'location': 'abc'})
            initialize(single_location, ctx)

        with self.assertRaises(InitializerException):
            # Bad param type
            ctx = dataclasses.replace(test_context(),
                                      param={'location': np.arange(20).reshape((4, 5))})
            initialize(single_location, ctx)
