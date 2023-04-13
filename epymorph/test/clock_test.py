import unittest
from datetime import date

import numpy as np

from epymorph.clock import Clock, Tick


class TestClock(unittest.TestCase):
    def test_clock(self):
        clock = Clock.init(
            start_date=date(2023, 1, 1),
            num_days=6,
            tau_steps=[np.double(2/3), np.double(1/3)]
        )
        act = clock.ticks
        exp = [
            Tick(0, 0, date(2023, 1, 1), 0, np.double(2/3), np.double(2/3)),
            Tick(1, 0, date(2023, 1, 1), 1, np.double(1/3), np.double(1)),
            Tick(2, 1, date(2023, 1, 2), 0, np.double(2/3), np.double(1 + 2/3)),
            Tick(3, 1, date(2023, 1, 2), 1, np.double(1/3), np.double(2)),
            Tick(4, 2, date(2023, 1, 3), 0, np.double(2/3), np.double(2 + 2/3)),
            Tick(5, 2, date(2023, 1, 3), 1, np.double(1/3),  np.double(3)),
            Tick(6, 3, date(2023, 1, 4), 0, np.double(2/3), np.double(3 + 2/3)),
            Tick(7, 3, date(2023, 1, 4), 1, np.double(1/3), np.double(4)),
            Tick(8, 4, date(2023, 1, 5), 0, np.double(2/3), np.double(4 + 2/3)),
            Tick(9, 4, date(2023, 1, 5), 1, np.double(1/3), np.double(5)),
            Tick(10, 5, date(2023, 1, 6), 0, np.double(2/3), np.double(5 + 2/3)),
            Tick(11, 5, date(2023, 1, 6), 1, np.double(1/3), np.double(6)),
        ]
        self.assertEqual(act, exp)
