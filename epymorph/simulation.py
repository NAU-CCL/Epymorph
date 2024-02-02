"""General simulation data types, events, and utility functions."""
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from functools import partial
from importlib import reload
from time import perf_counter
from typing import (Any, Callable, Generator, NamedTuple, Protocol, Self,
                    Sequence, runtime_checkable)

import numpy as np
from numpy.random import SeedSequence

from epymorph.code import ImmutableNamespace, base_namespace
from epymorph.util import (Event, long_range_von_mises_distribution,
                           mosquito_movement_probability, pairwise_haversine,
                           powerlaw_distribution_probability, progress,
                           row_normalize, subscriptions,
                           weibull_distribution_prob)

SimDType = np.int64
"""
This is the numpy datatype that should be used to represent internal simulation data.
Where segments of the application maintain compartment and/or event counts,
they should take pains to use this type at all times (if possible).
"""

# SimDType being centrally-located means we can change it reliably.


def default_rng(seed: int | SeedSequence | None = None) -> Callable[[], np.random.Generator]:
    """
    Convenience constructor to create a factory function for a simulation's random number generator,
    optionally with a given seed.
    """
    return lambda: np.random.default_rng(seed)


@dataclass(frozen=True)
class TimeFrame:
    """The time frame of a simulation."""

    @classmethod
    def of(cls, start_date_iso8601: str, duration_days: int) -> Self:
        """Alternate constructor for TimeFrame, parsing start date from an ISO-8601 string."""
        return cls(date.fromisoformat(start_date_iso8601), duration_days)

    start_date: date
    duration_days: int

    @property
    def end_date(self) -> date:
        """The end date (the first day not included in the simulation)."""
        return self.start_date + timedelta(days=self.duration_days)


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


class TickDelta(NamedTuple):
    """
    An offset relative to a Tick expressed as a number of days which should elapse,
    and the step on which to end up. In applying this delta, it does not matter which
    step we start on. We need the Clock configuration to apply a TickDelta, so see
    Clock for the relevant method.
    """
    days: int  # number of whole days
    step: int  # which tau step within that day (zero-indexed)


NEVER = TickDelta(-1, -1)
"""
A special TickDelta value which expresses an event that should never happen.
Any Tick plus Never returns Never.
"""


class SimDimensions(NamedTuple):
    """The dimensionality of a simulation."""

    @classmethod
    def build(cls, tau_step_lengths: Sequence[float], days: int, nodes: int, compartments: int, events: int):
        """Convenience constructor which reduces the overhead of initializing duplicative fields."""
        tau_steps = len(tau_step_lengths)
        ticks = tau_steps * days
        return cls(
            tau_step_lengths, tau_steps, days, ticks,
            nodes, compartments, events,
            (ticks, nodes, compartments, events))

    tau_step_lengths: Sequence[float]
    """The lengths of each tau step in the MM."""
    tau_steps: int
    """How many tau steps are in the MM?"""
    days: int
    """How many days are we going to run the simulation for?"""
    ticks: int
    """How many clock ticks are we going to run the simulation for?"""
    nodes: int
    """How many nodes are there in the GEO?"""
    compartments: int
    """How many disease compartments are in the IPM?"""
    events: int
    """How many transition events are in the IPM?"""
    TNCE: tuple[int, int, int, int]
    """
    The critical dimensionalities of the simulation, for ease of unpacking.
    T: number of ticks;
    N: number of geo nodes;
    C: number of IPM compartments;
    E: number of IPM events (transitions)
    """


class OnStart(NamedTuple):
    """The payload of a Simulation on_start event."""
    dim: SimDimensions
    time_frame: TimeFrame


class SimTick(NamedTuple):
    """The payload of a Simulation tick event."""
    tick_index: int
    percent_complete: float


@runtime_checkable
class SimulationEvents(Protocol):
    """Protocol for Simulations that support lifecycle events."""

    on_start: Event[OnStart]
    """
    Event fires at the start of a simulation run. Payload is a subset of the context for this run.
    """

    on_tick: Event[SimTick] | None
    """
    Optional event which fires after each tick has been processed.
    Event payload is a tuple containing the tick index just completed (an integer),
    and the percentage complete (a float).
    """

    on_end: Event[None]
    """
    Event fires after a simulation run is complete.
    """


@contextmanager
def sim_messaging(sim: SimulationEvents) -> Generator[None, None, None]:
    """
    Attach fancy console messaging to a Simulation, e.g., a progress bar.
    This creates subscriptions on `sim`'s events, so you only need to do it once
    per sim. Returns `sim` as a convenience.
    """

    start_time = 0.0
    use_progress_bar = sim.on_tick is not None

    def on_start(ctx: OnStart) -> None:
        start_date = ctx.time_frame.start_date
        duration_days = ctx.time_frame.duration_days
        end_date = ctx.time_frame.end_date

        print(f"Running simulation ({sim.__class__.__name__}):")
        print(f"• {start_date} to {end_date} ({duration_days} days)")
        print(f"• {ctx.dim.nodes} geo nodes")
        if use_progress_bar:
            print(progress(0.0), end='\r')
        else:
            print('Running...')

        nonlocal start_time
        start_time = perf_counter()

    def on_tick(tick: SimTick) -> None:
        print(progress(tick.percent_complete), end='\r')

    def on_end(_: None) -> None:
        end_time = perf_counter()
        if use_progress_bar:
            print(progress(1.0))
        else:
            print('Complete.')
        print(f"Runtime: {(end_time - start_time):.3f}s")

    # Set up a subscriptions context, subscribe our handlers,
    # then yield to the outer context (ostensibly where the sim will be run).
    with subscriptions() as subs:
        subs.subscribe(sim.on_start, on_start)
        if sim.on_tick is not None:
            subs.subscribe(sim.on_tick, on_tick)
        subs.subscribe(sim.on_end, on_end)
        yield


def enable_logging(filename: str = 'debug.log', movement: bool = True) -> None:
    """Enable simulation logging to file."""
    reload(logging)
    logging.basicConfig(filename=filename, filemode='w')
    if movement:
        logging.getLogger('movement').setLevel(logging.DEBUG)


def epymorph_namespace() -> dict[str, Any]:
    """
    Make a safe namespace for user-defined functions,
    including utilities that functions might need in epymorph.
    """
    return {
        'SimDType': SimDType,
        # our utility functions
        'pairwise_haversine': pairwise_haversine,
        'long_range_von_mises_distribution': long_range_von_mises_distribution,
        'powerlaw_distribution_probability': powerlaw_distribution_probability,
        'weibull_distribution_prob': weibull_distribution_prob,
        'mosquito_movement_probability': mosquito_movement_probability,
        'row_normalize': row_normalize,
        # numpy namespace
        'np': ImmutableNamespace({
            # numpy utility functions
            'array': partial(np.array, dtype=SimDType),
            'zeros': partial(np.zeros, dtype=SimDType),
            'zeros_like': partial(np.zeros_like, dtype=SimDType),
            'full': partial(np.full, dtype=SimDType),
            'sum': partial(np.sum, dtype=SimDType),
            'newaxis': np.newaxis,
            # numpy math functions
            'radians': np.radians,
            'degrees': np.degrees,
            'exp': np.exp,
            'log': np.log,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'arcsin': np.arcsin,
            'arccos': np.arccos,
            'arctan': np.arctan,
            'arctan2': np.arctan2,
            'sqrt': np.sqrt,
            'add': np.add,
            'subtract': np.subtract,
            'multiply': np.multiply,
            'divide': np.divide,
            'maximum': np.maximum,
            'minimum': np.minimum,
            'absolute': np.absolute,
            'floor': np.floor,
            'ceil': np.ceil,
            'pi': np.pi,
        }),
        **base_namespace(),
    }
