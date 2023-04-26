from datetime import date

import numpy as np
from numpy.typing import NDArray

from epymorph.clock import Clock
from epymorph.context import SimContext
from epymorph.epi import IpmBuilder
from epymorph.geo import Geo
from epymorph.movement import MovementBuilder
from epymorph.util import DataDict
from epymorph.world import World


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
        t = ctx.clock.num_ticks
        p, c, e = ctx.nodes, ctx.compartments, ctx.events
        self.prevalence = np.zeros((t, p, c), dtype=np.int_)
        self.incidence = np.zeros((t, p, e), dtype=np.int_)


class Simulation:
    """
    The combination of a Geo, IPM, and MM which can be executed at a calendar date and
    for a specified duration to produce time-series output.
    """

    geo: Geo
    ipm_builder: IpmBuilder
    mvm_builder: MovementBuilder

    def __init__(self, geo: Geo, ipm_builder: IpmBuilder, mvm_builder: MovementBuilder):
        self.geo = geo
        self.ipm_builder = ipm_builder
        self.mvm_builder = mvm_builder

    def run(self, param: DataDict, start_date: date, duration_days: int, rng: np.random.Generator | None = None) -> Output:
        """
        Execute the simulation with the given parameters:

        - param: a dictionary of named simulation parameters available for use by the IPM and MM
        - start_date: the calendar date on which to start the simulation
        - duration: the number of days to run the simulation
        - rng: (optional) a psuedo-random number generator used in all stochastic calculations
        """
        ctx = SimContext(
            nodes=self.geo.nodes,
            labels=self.geo.labels,
            geo=self.geo.data,
            compartments=self.ipm_builder.compartments,
            events=self.ipm_builder.events,
            param=param,
            clock=Clock.init(start_date, duration_days, self.mvm_builder.taus),
            rng=np.random.default_rng() if rng is None else rng
        )

        # Verification checks:
        self.ipm_builder.verify(ctx)
        self.mvm_builder.verify(ctx)
        ipm = self.ipm_builder.build(ctx)
        mvm = self.mvm_builder.build(ctx)

        world = World.initialize(
            self.ipm_builder.initialize_compartments(ctx)
        )
        out = Output(ctx)

        for tick in ctx.clock.ticks:
            t = tick.index

            # First do movement
            mvm.clause.apply(world, tick)

            # Then for each location:
            for p, loc in enumerate(world.locations):
                # Calc events by compartment
                es = ipm.events(loc, tick)
                # Store incidence
                out.incidence[t, p] = es
                # Distribute events
                ipm.apply_events(loc, es)
                # Store prevalence
                out.prevalence[t, p] = loc.compartment_totals

        return out
