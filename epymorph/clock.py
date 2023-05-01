from __future__ import annotations

from datetime import date, timedelta
from itertools import accumulate, cycle, islice
from typing import NamedTuple

import numpy as np

from epymorph.util import stutter


class Tick(NamedTuple):
    """
    A Tick bundles related time-step information. For instance, each time step corresponds to a calendar day,
    a numeric day (i.e., relative to the start of the simulation), which tau step this corresponds to, and so on.
    """
    index: int  # step increment regardless of tau (0,1,2,3,...)
    day: int  # day increment, same for each tau step (0,0,1,1,2,2,...)
    date: date  # calendar date corresponding to `day`
    step: int  # which tau step? (0,1,0,1,0,1,...)
    tau: np.double  # the current tau length (0.666,0.333,0.666,0.333,...)
    # the running sum of tau after this tick is complete (0.666,1.0,1.666,...)
    tausum: np.double


class TickDelta(NamedTuple):
    """
    An offset relative to a Tick expressed as a number of days which should elapse,
    and the step on which to end up. In applying this delta, it does not matter which
    step we start on. We need the Clock configuration to apply a TickDelta, so see
    Clock for the relevant method.
    """
    days: int  # number of whole days
    step: int  # which tau step within that day


NEVER = TickDelta(-1, -1)
"""
A special TickDelta value which expresses an event that should never happen.
Any Tick plus Never returns Never.
"""


class Clock:
    """The temporal configuration of a simulation run. Basically: what are the simulation time steps?"""
    num_days: int
    num_steps: int
    num_ticks: int
    ticks: list[Tick]

    @classmethod
    def init(cls, start_date: date, num_days: int, tau_steps: list[np.double]) -> Clock:
        """
        Construct a Clock for a simulation by specifying a start date, the number of days
        to run the simulation, and the tau steps for the simulation (which must sum to 1).
        """
        assert np.sum(tau_steps) == np.double(1), "Tau steps must sum to 1."
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
        """Add a delta to a tick to get the index of the resulting tick."""
        return -1 if delta.days == -1 else \
            tick.index - tick.step + (self.num_steps * delta.days) + delta.step
