from __future__ import annotations

import logging
from datetime import date
from time import perf_counter

import numpy as np
from dateutil.relativedelta import relativedelta
from numpy.typing import NDArray

from epymorph.clock import Clock
from epymorph.context import SimContext, SimDType, normalize_params
from epymorph.geo.geo import Geo
from epymorph.initializer import DEFAULT_INITIALIZER, Initializer, initialize
from epymorph.ipm.ipm import IpmBuilder
from epymorph.movement.basic import BasicEngine
from epymorph.movement.engine import MovementBuilder, MovementEngine
from epymorph.util import Event, progress


def configure_sim_logging(enabled: bool) -> None:
    """
    Configure standard logging for simulation runs.
    `True` for verbose logging, `False` to minimize logging.
    """

    if enabled:
        # Verbose output to file.
        logging.basicConfig(filename='debug.log', filemode='w')
        logging.getLogger('movement').setLevel(logging.DEBUG)
    else:
        # Only critical output to console.
        logging.basicConfig(level=logging.CRITICAL)


class Output:
    """
    The output of a simulation run, including prevalence for all populations and all IPM compartments
    and incidence for all populations and all IPM events.
    """

    ctx: SimContext  # TODO: this should be a static subset of SimContext, excluding things like Geo and rng
    """The context under which this output was generated."""

    prevalence: NDArray[SimDType]
    """
    Prevalence data by timestep, population, and compartment.
    Array of shape (T,N,C) where T is the number of ticks in the simulation, N is the number of populations, and C is the number of compartments.
    """

    incidence: NDArray[SimDType]
    """
    Incidence data by timestep, population, and event.
    Array of shape (T,N,E) where T is the number of ticks in the simulation, N is the number of populations, and E is the number of events.
    """

    initial: NDArray[SimDType]
    """
    Initial prevalence data by population and compartment.
    Array of shape (N, C) where N is the number of populations, and C is the number of compartments
    """

    def __init__(self, ctx: SimContext):
        self.ctx = ctx
        T, N, C, E = ctx.TNCE
        self.prevalence = np.zeros((T, N, C), dtype=SimDType)
        self.incidence = np.zeros((T, N, E), dtype=SimDType)
        self.initial = np.zeros((N, C), dtype=SimDType)


class OutputAggregate:
    """
    The output of many simulation runs, reporting the min and max values for:
    - prevalence for all populations and all IPM compartments, and
    - incidence for all populations and all IPM events.
    """

    ctx: SimContext  # TODO: this should be a static subset of SimContext, excluding things like Geo and rng
    min_prevalence: NDArray[SimDType]
    max_prevalence: NDArray[SimDType]
    min_incidence: NDArray[SimDType]
    max_incidence: NDArray[SimDType]

    def __init__(self, ctx: SimContext):
        self.ctx = ctx
        T, N, C, E = ctx.TNCE
        min_int = np.iinfo(SimDType).min
        max_int = np.iinfo(SimDType).max
        self.min_prevalence = np.full((T, N, C), max_int, dtype=SimDType)
        self.max_prevalence = np.full((T, N, C), min_int, dtype=SimDType)
        self.min_incidence = np.full((T, N, E), max_int, dtype=SimDType)
        self.max_incidence = np.full((T, N, E), min_int, dtype=SimDType)

    def __add__(self, that: Output) -> OutputAggregate:
        """Update this aggregate by including the results from the given Output object."""
        np.minimum(self.min_prevalence, that.prevalence,
                   out=self.min_prevalence)
        np.maximum(self.max_prevalence, that.prevalence,
                   out=self.max_prevalence)
        np.minimum(self.min_incidence, that.incidence,
                   out=self.min_incidence)
        np.maximum(self.max_incidence, that.incidence,
                   out=self.max_incidence)
        return self

    def merge(self, that: OutputAggregate) -> OutputAggregate:
        """Merge two OutputAggregates."""
        np.minimum(self.min_prevalence, that.min_prevalence,
                   out=self.min_prevalence)
        np.maximum(self.max_prevalence, that.max_prevalence,
                   out=self.max_prevalence)
        np.minimum(self.min_incidence, that.min_incidence,
                   out=self.min_incidence)
        np.maximum(self.max_incidence, that.max_incidence,
                   out=self.max_incidence)
        return self


class Simulation:
    """
    The combination of a Geo, IPM, and MM which can be executed at a calendar date and
    for a specified duration to produce time-series output.
    """

    geo: Geo
    ipm_builder: IpmBuilder
    mvm_builder: MovementBuilder
    mvm_engine: type[MovementEngine]

    # Progress events
    on_start: Event[SimContext]
    """
    Event fires at the start of every simulation run. Payload is the SimContext for this run.
    """

    on_tick: Event[tuple[int, float]]
    """
    Event fires after each tick has been processed.
    Event payload is a tuple containing the tick index just completed (an integer),
    and the percentage complete (a float).
    """

    on_end: Event[None]
    """
    Event fires after a simulation run is complete.
    """

    def __init__(self, geo: Geo, ipm_builder: IpmBuilder, mvm_builder: MovementBuilder, mvm_engine: type[MovementEngine] | None = None):
        self.geo = geo
        self.ipm_builder = ipm_builder
        self.mvm_builder = mvm_builder
        self.mvm_engine = BasicEngine if mvm_engine is None else mvm_engine
        self.on_start = Event()
        self.on_tick = Event()
        self.on_end = Event()

    def _make_context(self, geo: Geo, param: dict[str, Any], start_date: date, duration_days: int, rng: np.random.Generator | None) -> SimContext:
        return SimContext(
            geo=geo,
            compartments=self.ipm_builder.compartments,
            compartment_tags=self.ipm_builder.compartment_tags(),
            events=self.ipm_builder.events,
            param=normalize_params(param, self.geo, duration_days),
            clock=Clock(start_date, duration_days, self.mvm_builder.taus),
            rng=np.random.default_rng() if rng is None else rng
        )

    def run(self,
            param: dict[str, Any],
            start_date: date,
            duration_days: int,
            initializer: Initializer | None = None,
            rng: np.random.Generator | None = None) -> Output:
        """
        Execute the simulation with the given parameters:

        - param: a dictionary of named simulation parameters available for use by the IPM and MM
        - start_date: the calendar date on which to start the simulation
        - duration: the number of days to run the simulation
        - initializer: a function that initializes the compartments for each geo node; if None is provided, a default initializer will be used
        - rng: (optional) a psuedo-random number generator used in all stochastic calculations
        """

        ctx = self._make_context(self.geo, param, start_date, duration_days, rng)

        # Verification checks:
        self.ipm_builder.verify(ctx)
        self.mvm_builder.verify(ctx)
        # initializer.verify(ctx)

        if initializer is None:
            initializer = DEFAULT_INITIALIZER
        inits = initialize(initializer, ctx)

        mvm = self.mvm_engine(ctx, self.mvm_builder.build(ctx), inits)

        ipm = self.ipm_builder.build(ctx)

        self.on_start.publish(ctx)

        out = Output(ctx)

        out.initial = inits

        for tick in ctx.clock.ticks:
            t = tick.index
            # First do movement
            mvm.apply(tick)
            # Then for each location:
            for p, loc in enumerate(mvm.get_locations()):
                # Calc events by compartment
                es = ipm.events(loc, tick)
                # Store incidence
                out.incidence[t, p] = es
                # Distribute events
                ipm.apply_events(loc, es)
                # Store prevalence
                # TODO: maybe better to do this all at once rather than per loc
                out.prevalence[t, p] = loc.get_compartments()
            self.on_tick.publish((t, (t + 1) / ctx.clock.num_ticks))

        mvm.shutdown()
        self.on_end.publish(None)
        return out


def with_fancy_messaging(sim: Simulation) -> Simulation:
    """
    Attach fancy console messaging to a Simulation, e.g., a progress bar.
    This creates subscriptions on `sim`'s events, so you only need to do it once
    per sim. Returns `sim` as a convenience.
    """

    # Keep track of simulation runtime.
    start_time = 0.0

    def on_start(ctx: SimContext) -> None:
        duration_days = ctx.clock.num_days
        start_date = ctx.clock.start_date
        end_date = start_date + relativedelta(days=duration_days)

        print(f"Running simulation ({sim.mvm_engine.__name__}):")
        print(f"• {start_date} to {end_date} ({duration_days} days)")
        print(f"• {ctx.nodes} geo nodes")
        print(progress(0.0), end='\r')

        nonlocal start_time
        start_time = perf_counter()

    def on_tick(tick: tuple[int, float]) -> None:
        print(progress(tick[1]), end='\r')

    def on_end(_: None) -> None:
        print(progress(1.0))
        end_time = perf_counter()
        print(f"Runtime: {(end_time - start_time):.3f}s")

    sim.on_start.subscribe(on_start)
    sim.on_tick.subscribe(on_tick)
    sim.on_end.subscribe(on_end)

    return sim
