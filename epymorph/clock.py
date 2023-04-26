from __future__ import annotations

from datetime import date, timedelta
from itertools import accumulate, cycle, islice
from typing import NamedTuple

import numpy as np

from epymorph.util import stutter


class Tick(NamedTuple):
    index: int  # step increment regardless of tau (0,1,2,3,...)
    day: int  # day increment, same for each tau step (0,0,1,1,2,2,...)
    date: date  # calendar date corresponding to `day`
    step: int  # which tau step? (0,1,0,1,0,1,...)
    tau: np.double  # the current tau length (0.666,0.333,0.666,0.333,...)
    # the running sum of tau after this tick is complete (0.666,1.0,1.666,...)
    tausum: np.double


class TickDelta(NamedTuple):
    days: int  # number of whole days
    step: int  # which tau step within that day


Never = TickDelta(-1, -1)


class Clock:
    num_days: int
    num_steps: int
    num_ticks: int
    ticks: list[Tick]

    @classmethod
    def init(cls, start_date: date, num_days: int, tau_steps: list[np.double]) -> Clock:
        # `duration` is in days
        num_steps = len(tau_steps)
        num_ticks = num_days * num_steps

        indices = range(num_ticks)
        days = [d for d in stutter(range(num_days), num_steps)]
        dates = [start_date + timedelta(days=d) for d in days]
        steps = cycle(range(num_steps))
        taus = list(islice(cycle(tau_steps), num_ticks))

        tau_addend = list(accumulate(tau_steps, lambda a, b: a + b))
        tau_addend[-1] = 1
        tau_sums = [d + t for d, t in zip(days, cycle(tau_addend))]

        ticks = [Tick(*xs)
                 for xs in zip(indices, days, dates, steps, taus, tau_sums)]
        return cls(num_days, num_steps, num_ticks, ticks)

    def __init__(self, num_days: int, num_steps: int, num_ticks: int, ticks: list[Tick]):
        self.num_days = num_days
        self.num_steps = num_steps
        self.num_ticks = num_ticks
        self.ticks = ticks

    def tick_plus(self, tick: Tick, delta: TickDelta) -> int:
        return -1 if delta.days == -1 else \
            tick.index - tick.step + (self.num_steps * delta.days) + delta.step
