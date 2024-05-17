# pylint: disable=missing-docstring
import unittest
from datetime import date
from functools import partial
from unittest.mock import MagicMock

import numpy as np

from epymorph.data_shape import SimDimensions
from epymorph.data_type import SimDType
from epymorph.error import InitException
from epymorph.geo.spec import LABEL, NO_DURATION
from epymorph.geo.static import StaticGeo, StaticGeoSpec
from epymorph.initializer import (InitContext, bottom_locations, explicit,
                                  indexed_locations, initialize,
                                  labeled_locations, proportional,
                                  single_location, top_locations)
from epymorph.params import GeoData
from epymorph.simulation import geo_attrib


def test_context() -> InitContext:
    geo = StaticGeo(
        spec=StaticGeoSpec(
            attributes=[
                LABEL,
                geo_attrib('population', dtype=int),
                geo_attrib('foosball_championships', dtype=int),
            ],
            time_period=NO_DURATION,
        ),
        values={
            'label': np.array(list('ABCDE'), dtype=np.str_),
            'population': np.array([100, 200, 300, 400, 500], dtype=SimDType),
            'foosball_championships': np.array([2, 4, 1, 9, 6]),
        })

    init_context = MagicMock(spec=InitContext)
    init_context.dim = SimDimensions.build(
        tau_step_lengths=[1 / 3, 2 / 3], start_date=date(2020, 1, 1), days=100,
        nodes=geo.nodes, compartments=3, events=3
    )
    init_context.rng = np.random.default_rng(1)
    init_context.geo = geo
    init_context.params = {}

    return init_context


class TestExplicitInitializer(unittest.TestCase):

    def test_explicit(self):
        initials = np.array([
            [50, 20, 30],
            [50, 120, 30],
            [100, 100, 100],
            [300, 100, 0],
            [0, 0, 500],
        ])
        exp = initials.copy()
        act = explicit(test_context(), initials)
        np.testing.assert_array_equal(act, exp)


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
            [1 / 3, 1 / 3, 1 / 3],
            [0.75, 0.25, 0],
            [0, 0, 1],
        ])

        exp = ratios1.copy()
        for ratios in [ratios1, ratios2, ratios3]:
            act = proportional(test_context(), ratios)
            np.testing.assert_array_equal(act, exp)

    def test_bad_args(self):
        with self.assertRaises(InitException):
            # row sums to zero!
            ratios = np.array([
                [50, 20, 30],
                [50, 120, 30],
                [0, 0, 0],
                [300, 100, 0],
                [0, 0, 500],
            ])
            proportional(test_context(), ratios)

        with self.assertRaises(InitException):
            # row sums to negative!
            ratios = np.array([
                [50, 20, 30],
                [50, 120, 30],
                [0, -50, -50],
                [300, 100, 0],
                [0, 0, 500],
            ])
            proportional(test_context(), ratios)


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
        np.testing.assert_array_equal(act, exp)
        # And check for 100 infected in total.
        self.assertEqual(out[:, 1].sum(), 100)

    def test_indexed_locations_bad(self):
        with self.assertRaises(InitException):
            # indices must be one dimension
            indexed_locations(
                test_context(),
                selection=np.array([[1], [3]], dtype=np.intp),
                seed_size=100
            )
        with self.assertRaises(InitException):
            # indices must be in range (positive)
            indexed_locations(
                test_context(),
                selection=np.array([1, 8], dtype=np.intp),
                seed_size=100
            )
        with self.assertRaises(InitException):
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
        np.testing.assert_array_equal(act, exp)
        # And check for 100 infected in total.
        self.assertEqual(out[:, 1].sum(), 100)

    def test_labeled_locations_bad(self):
        with self.assertRaises(InitException):
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
        np.testing.assert_array_equal(act, exp)


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
        np.testing.assert_array_equal(act, exp)


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
        np.testing.assert_array_equal(act, exp)


class TestInitialize(unittest.TestCase):

    def make_geo(self) -> GeoData:
        return {
            'label': np.array(list('ABCDE'), dtype=np.str_),
            'population': np.array([100, 200, 300, 400, 500], dtype=SimDType),
            'foosball_championships': np.array([2, 4, 1, 9, 6]),
        }

    def make_dim(self) -> SimDimensions:
        return SimDimensions.build(
            tau_step_lengths=[1 / 3, 2 / 3], start_date=date(2020, 1, 1), days=100, nodes=5,
            compartments=3, events=3
        )

    def test_initialize_01(self):
        # Mostly to determine if auto-wiring from params works as expected.
        out = initialize(
            init=indexed_locations,
            dim=self.make_dim(),
            geo=self.make_geo(),
            raw_params={
                'selection': np.array([1, 3], dtype=np.intp),
                'seed_size': 100,
            },
            rng=np.random.default_rng(1),
        )
        act = out[:, 1] > 0
        exp = np.array([False, True, False, True, False])
        np.testing.assert_array_equal(act, exp)

    def test_initialize_02(self):
        # Mixing partial with some params.
        out = initialize(
            init=partial(indexed_locations, seed_size=100),
            dim=self.make_dim(),
            geo=self.make_geo(),
            raw_params={
                'selection': np.array([1, 3], dtype=np.intp),
            },
            rng=np.random.default_rng(1),
        )
        # Make sure only the selected locations get infected.
        act = out[:, 1] > 0
        exp = np.array([False, True, False, True, False])
        np.testing.assert_array_equal(act, exp)
        # And check for 100 infected in total.
        self.assertEqual(out[:, 1].sum(), 100)

    def test_initialize_03(self):
        # Partial-provided args should take precedence over params.
        out = initialize(
            init=partial(indexed_locations, selection=np.array([1, 3], dtype=np.intp)),
            dim=self.make_dim(),
            geo=self.make_geo(),
            raw_params={
                'selection': np.array([2, 4], dtype=np.intp),
                'seed_size': 100,
            },
            rng=np.random.default_rng(1),
        )
        # Make sure only the selected locations get infected.
        act = out[:, 1] > 0
        exp = np.array([False, True, False, True, False])
        np.testing.assert_array_equal(act, exp)

    def test_initialize_bad(self):
        with self.assertRaises(InitException):
            # Missing param
            initialize(
                init=single_location,
                dim=self.make_dim(),
                geo=self.make_geo(),
                raw_params={},
                rng=np.random.default_rng(1),
            )

        with self.assertRaises(InitException):
            # Bad param type
            initialize(
                init=single_location,
                dim=self.make_dim(),
                geo=self.make_geo(),
                raw_params={'location': 13},
                rng=np.random.default_rng(1),
            )

        with self.assertRaises(InitException):
            # Bad param type
            initialize(
                init=single_location,
                dim=self.make_dim(),
                geo=self.make_geo(),
                raw_params={'location': [1, 2]},
                rng=np.random.default_rng(1),
            )

        with self.assertRaises(InitException):
            # Bad param type
            initialize(
                init=single_location,
                dim=self.make_dim(),
                geo=self.make_geo(),
                raw_params={'location': 'abc'},
                rng=np.random.default_rng(1),
            )

        with self.assertRaises(InitException):
            # Bad param type
            initialize(
                init=single_location,
                dim=self.make_dim(),
                geo=self.make_geo(),
                raw_params={'location': np.arange(20).reshape((4, 5))},
                rng=np.random.default_rng(1),
            )
