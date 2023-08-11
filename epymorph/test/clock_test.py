import unittest
from datetime import date

from epymorph.clock import Clock, Tick


class TestClock(unittest.TestCase):
    def test_clock(self):
        clock = Clock(
            start_date=date(2023, 1, 1),
            num_days=6,
            taus=[2/3, 1/3]
        )
        act = clock.ticks
        exp = [
            Tick(0, 0, date(2023, 1, 1), 0, 2/3, 2/3),
            Tick(1, 0, date(2023, 1, 1), 1, 1/3, 1),
            Tick(2, 1, date(2023, 1, 2), 0, 2/3, 1 + 2/3),
            Tick(3, 1, date(2023, 1, 2), 1, 1/3, 2),
            Tick(4, 2, date(2023, 1, 3), 0, 2/3, 2 + 2/3),
            Tick(5, 2, date(2023, 1, 3), 1, 1/3,  3),
            Tick(6, 3, date(2023, 1, 4), 0, 2/3, 3 + 2/3),
            Tick(7, 3, date(2023, 1, 4), 1, 1/3, 4),
            Tick(8, 4, date(2023, 1, 5), 0, 2/3, 4 + 2/3),
            Tick(9, 4, date(2023, 1, 5), 1, 1/3, 5),
            Tick(10, 5, date(2023, 1, 6), 0, 2/3, 5 + 2/3),
            Tick(11, 5, date(2023, 1, 6), 1, 1/3, 6),
        ]
        self.assertEqual(act, exp)
