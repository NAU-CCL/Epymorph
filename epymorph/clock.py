from __future__ import annotations

from datetime import date, timedelta
from itertools import accumulate
from typing import NamedTuple


class Tick(NamedTuple):
    """
    A Tick bundles related time-step information. For instance, each time step corresponds to a calendar day,
    a numeric day (i.e., relative to the start of the simulation), which tau step this corresponds to, and so on.
    """
    index: int  # step increment regardless of tau (0,1,2,3,...)
    day: int  # day increment, same for each tau step (0,0,1,1,2,2,...)
    date: date  # calendar date corresponding to `day`
    step: int  # which tau step? (0,1,0,1,0,1,...)
    tau: float  # the current tau length (0.666,0.333,0.666,0.333,...)
    # the running sum of tau after this tick is complete (0.666,1.0,1.666,...)
    tausum: float


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
    start_date: date
    num_days: int
    num_ticks: int
    taus: list[float]
    ticks: list[Tick]

    def __init__(self, start_date: date, num_days: int, taus: list[float]):
        """
        Construct a Clock for a simulation by specifying a start date, the number of days
        to run the simulation, and the tau steps for the simulation (which must sum to 1).
        """
        assert sum(taus) == 1, "Tau steps must sum to 1."
        num_steps = len(taus)
        num_ticks = num_days * num_steps
        tau_sum = list(accumulate(taus))

        ticks = []
        for index in range(num_ticks):
            day, step = divmod(index, num_steps)
            date = start_date + timedelta(days=day)
            tau = taus[step]
            tausum = day + tau_sum[step]
            ticks.append(Tick(index, day, date, step, tau, tausum))

        self.start_date = start_date
        self.num_days = num_days
        self.num_ticks = num_ticks
        self.taus = taus
        self.ticks = ticks

    def tick_plus(self, tick: Tick, delta: TickDelta) -> int:
        """Add a delta to a tick to get the index of the resulting tick."""
        return -1 if delta.days == -1 else \
            tick.index - tick.step + (len(self.taus) * delta.days) + delta.step
