from __future__ import annotations

import logging
from datetime import date

import numpy as np
from numpy.typing import NDArray

from epymorph.clock import Clock
from epymorph.context import SimContext
from epymorph.geo import Geo
from epymorph.ipm.ipm import IpmBuilder
from epymorph.movement.basic import BasicEngine
from epymorph.movement.engine import MovementBuilder, MovementEngine
from epymorph.util import DataDict, Event


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

    ctx: SimContext
    """The context under which this output was generated."""

    prevalence: NDArray[np.int_]
    """
    Prevalence data by timestep, population, and compartment.
    Array of shape (T,P,C) where T is the number of ticks in the simulation, P is the number of populations, and C is the number of compartments.
    """

    incidence: NDArray[np.int_]
    """
    Incidence data by timestep, population, and event.
    Array of shape (T,P,E) where T is the number of ticks in the simulation, P is the number of populations, and E is the number of events.
    """

    def __init__(self, ctx: SimContext):
        self.ctx = ctx
        self.prevalence = np.zeros(ctx.prv_shape, dtype=np.int_)
        self.incidence = np.zeros(ctx.inc_shape, dtype=np.int_)


class OutputAggregate:
    """
    The output of many simulation runs, reporting the min and max values for:
    - prevalence for all populations and all IPM compartments, and
    - incidence for all populations and all IPM events.
    """

    ctx: SimContext
    min_prevalence: NDArray[np.int_]
    max_prevalence: NDArray[np.int_]
    min_incidence: NDArray[np.int_]
    max_incidence: NDArray[np.int_]

    def __init__(self, ctx: SimContext):
        self.ctx = ctx
        prv_shape = ctx.prv_shape
        inc_shape = ctx.inc_shape
        min_int = np.iinfo(np.int_).min
        max_int = np.iinfo(np.int_).max
        self.min_prevalence = np.full(prv_shape, max_int, dtype=np.int_)
        self.max_prevalence = np.full(prv_shape, min_int, dtype=np.int_)
        self.min_incidence = np.full(inc_shape, max_int, dtype=np.int_)
        self.max_incidence = np.full(inc_shape, min_int, dtype=np.int_)

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
    on_start: Event[None]
    on_tick: Event[tuple[int, float]]
    on_end: Event[None]

    def __init__(self, geo: Geo, ipm_builder: IpmBuilder, mvm_builder: MovementBuilder, mvm_engine: type[MovementEngine] | None = None):
        self.geo = geo
        self.ipm_builder = ipm_builder
        self.mvm_builder = mvm_builder
        self.mvm_engine = BasicEngine if mvm_engine is None else mvm_engine
        self.on_start = Event()
        self.on_tick = Event()
        self.on_end = Event()

    def _make_context(self, param: DataDict, start_date: date, duration_days: int, rng: np.random.Generator | None) -> SimContext:
        return SimContext(
            nodes=self.geo.nodes,
            labels=self.geo.labels,
            geo=self.geo.data,
            compartments=self.ipm_builder.compartments,
            compartment_tags=self.ipm_builder.compartment_tags(),
            events=self.ipm_builder.events,
            param=param,
            clock=Clock(start_date, duration_days, self.mvm_builder.taus),
            rng=np.random.default_rng() if rng is None else rng
        )

    def run(self, param: DataDict, start_date: date, duration_days: int, rng: np.random.Generator | None = None) -> Output:
        """
        Execute the simulation with the given parameters:

        - param: a dictionary of named simulation parameters available for use by the IPM and MM
        - start_date: the calendar date on which to start the simulation
        - duration: the number of days to run the simulation
        - rng: (optional) a psuedo-random number generator used in all stochastic calculations
        """

        ctx = self._make_context(param, start_date, duration_days, rng)

        # Verification checks:
        self.ipm_builder.verify(ctx)
        self.mvm_builder.verify(ctx)
        self.mvm_builder.verify(ctx)

        inits = self.ipm_builder.initialize_compartments(ctx)

        mvm = self.mvm_engine(ctx, self.mvm_builder.build(ctx), inits)

        ipm = self.ipm_builder.build(ctx)

        self.on_start.publish(None)

        out = Output(ctx)
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
                out.prevalence[t, p] = loc.get_compartments()
            self.on_tick.publish((t, t / ctx.clock.num_ticks))
        self.on_end.publish(None)
        return out
