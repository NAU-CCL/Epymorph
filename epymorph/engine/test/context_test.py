# pylint: disable=missing-docstring
import unittest
from datetime import date

from epymorph.data_shape import SimDimensions
from epymorph.engine.context import _simulation_clock
from epymorph.simulation import Tick


class TestClock(unittest.TestCase):
    def test_clock(self):
        dim = SimDimensions.build(
            tau_step_lengths=[2 / 3, 1 / 3],
            start_date=date(2023, 1, 1),
            days=6,
            # sim clock doesn't depend on GEO/IPM dimensions
            nodes=99, compartments=99, events=99,
        )
        clock = _simulation_clock(dim)
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
