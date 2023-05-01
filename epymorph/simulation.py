from datetime import date
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from epymorph.clock import Clock, Tick
from epymorph.context import SimContext
from epymorph.epi import Ipm, IpmBuilder
from epymorph.geo import Geo
from epymorph.movement import Movement, MovementBuilder
from epymorph.util import Compartments, DataDict, Events
from epymorph.world import World


# The output from a single simulation step.
class _TickOutput(NamedTuple):
    prevalence: list[Compartments]
    incidence: list[Events]


class Output:
    """
    The output of a simulation run, including prevalence for all populations and all IPM compartments
    and incidence for all populations and all IPM events.
    """
    clock: Clock
    # TODO: these two should probably just be 2D np.arrays
    prevalence: list[NDArray[np.int_]]
    incidence: list[NDArray[np.int_]]
    pop_labels: list[str]

    def __init__(self, ctx: SimContext):
        self.clock = ctx.clock
        def arr(n): return np.zeros((ctx.clock.num_ticks, n), dtype=np.int_)
        m = ctx.nodes
        self.prevalence = [arr(ctx.compartments) for _ in range(m)]
        self.incidence = [arr(ctx.events) for _ in range(m)]
        self.pop_labels = ctx.labels

    def set_prv(self, loc_idx: int, tick: Tick, prv: Compartments) -> None:
        self.prevalence[loc_idx][tick.index] = prv

    def set_inc(self, loc_idx: int, tick: Tick, inc: Compartments) -> None:
        self.incidence[loc_idx][tick.index] = inc


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
        Run the simulation!
        - `param`: simulation parameters
        - `start_date`: the first calendar day of the simulation
        - `duration_days`: how many days to run
        - `rng`: (optional) a random number generator to use.
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
        for t in ctx.clock.ticks:
            tickout = self._tick(ipm, mvm, world, t)
            for i, inc in enumerate(tickout.incidence):
                out.set_inc(i, t, inc)
            for i, prv in enumerate(tickout.prevalence):
                out.set_prv(i, t, prv)
        return out

    def _tick(self, ipm: Ipm, mvm: Movement, world: World, tick: Tick) -> _TickOutput:
        # First do movement
        mvm.clause.apply(world, tick)

        # Calc events by compartment
        incidence = [ipm.events(loc, tick)
                     for loc in world.locations]

        # Distribute events
        for loc, es in zip(world.locations, incidence):
            ipm.apply_events(loc, es)

        # Calc new effective population by compartment
        prevalence = [loc.compartment_totals for loc in world.locations]

        return _TickOutput(prevalence=prevalence, incidence=incidence)
