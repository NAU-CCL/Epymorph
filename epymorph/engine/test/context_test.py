# pylint: disable=missing-docstring
import unittest
from datetime import date
from functools import partial
from unittest.mock import MagicMock

import numpy as np

from epymorph.data_shape import Shapes
from epymorph.engine.context import RumeContext, _initialize, _simulation_clock
from epymorph.error import InitException
from epymorph.geo.spec import LABEL, NO_DURATION, AttribDef, StaticGeoSpec
from epymorph.geo.static import StaticGeo
from epymorph.initializer import (Initializer, indexed_locations,
                                  single_location)
from epymorph.params import Params
from epymorph.simulation import SimDimensions, SimDType, Tick


class TestClock(unittest.TestCase):
    def test_clock(self):
        clock = _simulation_clock(
            dim=SimDimensions.build(
                tau_step_lengths=[2 / 3, 1 / 3], days=6,
                # sim clock doesn't depend on GEO/IPM dimensions
                nodes=99, compartments=99, events=99),
            start_date=date(2023, 1, 1)
        )
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


class TestInitialize(unittest.TestCase):

    def make_ctx(self, init: Initializer, params: Params) -> RumeContext:
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
                'label': np.array(list('ABCDE'), dtype=np.str_),
                'population': np.array([100, 200, 300, 400, 500], dtype=SimDType),
                'foosball_championships': np.array([2, 4, 1, 9, 6]),
            })

        ctx = MagicMock(spec=RumeContext)
        ctx.dim = SimDimensions.build(
            tau_step_lengths=[1 / 3, 2 / 3], days=100, nodes=geo.nodes,
            compartments=3, events=3
        )
        ctx.geo = geo
        ctx.raw_params = params
        ctx.initializer = init
        ctx.rng = np.random.default_rng(1)
        return ctx

    def test_initialize_01(self):
        # Mostly to determine if auto-wiring from params works as expected.
        ctx = self.make_ctx(
            init=indexed_locations,
            params={
                'selection': np.array([1, 3], dtype=np.intp),
                'seed_size': 100,
            }
        )
        out = _initialize(ctx)
        act = out[:, 1] > 0
        exp = np.array([False, True, False, True, False])
        np.testing.assert_array_equal(act, exp)

    def test_initialize_02(self):
        # Mixing partial with some params.
        ctx = self.make_ctx(
            init=partial(indexed_locations, seed_size=100),
            params={
                'selection': np.array([1, 3], dtype=np.intp),
            }
        )
        out = _initialize(ctx)
        # Make sure only the selected locations get infected.
        act = out[:, 1] > 0
        exp = np.array([False, True, False, True, False])
        np.testing.assert_array_equal(act, exp)
        # And check for 100 infected in total.
        self.assertEqual(out[:, 1].sum(), 100)

    def test_initialize_03(self):
        # Partial-provided args should take precedence over params.
        ctx = self.make_ctx(
            init=partial(indexed_locations, selection=np.array([1, 3], dtype=np.intp)),
            params={
                'selection': np.array([2, 4], dtype=np.intp),
                'seed_size': 100,
            }
        )
        out = _initialize(ctx)
        # Make sure only the selected locations get infected.
        act = out[:, 1] > 0
        exp = np.array([False, True, False, True, False])
        np.testing.assert_array_equal(act, exp)

    def test_initialize_bad(self):
        with self.assertRaises(InitException):
            # Missing param
            ctx = self.make_ctx(init=single_location, params={})
            _initialize(ctx)

        with self.assertRaises(InitException):
            # Bad param type
            ctx = self.make_ctx(init=single_location, params={'location': 13})
            _initialize(ctx)

        with self.assertRaises(InitException):
            # Bad param type
            ctx = self.make_ctx(init=single_location, params={'location': [1, 2]})
            _initialize(ctx)

        with self.assertRaises(InitException):
            # Bad param type
            ctx = self.make_ctx(init=single_location, params={'location': 'abc'})
            _initialize(ctx)

        with self.assertRaises(InitException):
            # Bad param type
            ctx = self.make_ctx(
                init=single_location,
                params={
                    'location': np.arange(20).reshape((4, 5))
                }
            )
            _initialize(ctx)
