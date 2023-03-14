from datetime import date, timedelta
from itertools import accumulate, cycle
from typing import NamedTuple

import numpy as np

from util import stutter


class Tick(NamedTuple):
    time: int  # step increment regardless of tau (0,1,2,3,...)
    day: int  # day increment, same for each tau step (0,0,1,1,2,2,...)
    date: date  # calendar date corresponding to `day`
    step: int  # which tau step? (0,1,0,1,0,1,...)
    tausum: np.double  # the running sum of tau (0,0.666,1.0,1.666,...)


class Clock:
    # `duration` is in days
    def __init__(self, start_date: date, duration: int, taus: list[np.double]):
        num_steps = len(taus)
        self.num_ticks = duration * num_steps
        times = range(self.num_ticks)
        days = [d for d in stutter(range(duration), num_steps)]
        # [tau - taus[0] for tau in taus]
        tauaddend = list(accumulate(taus, lambda a, b: a + b,
                                    initial=np.double(0)))[0:-1]
        tausums = [d + t for d, t in zip(days, cycle(tauaddend))]
        dates = [start_date + timedelta(days=d) for d in days]
        steps = cycle(range(num_steps))
        self.ticks = [Tick(*xs)
                      for xs in zip(times, days, dates, steps, tausums)]
