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


class TickOutput(NamedTuple):
    prevalence: list[Compartments]
    incidence: list[Events]


class Output:
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
    geo: Geo
    ipmBuilder: IpmBuilder
    mvmBuilder: MovementBuilder

    def __init__(self, geo: Geo, ipmBuilder: IpmBuilder, mvmBuilder: MovementBuilder):
        self.geo = geo
        self.ipmBuilder = ipmBuilder
        self.mvmBuilder = mvmBuilder

    def run(self, param: DataDict, start_date: date, duration: int, rng: np.random.Generator | None = None) -> Output:
        ctx = SimContext(
            nodes=self.geo.nodes,
            labels=self.geo.labels,
            geo=self.geo.data,
            compartments=self.ipmBuilder.compartments,
            events=self.ipmBuilder.events,
            param=param,
            clock=Clock.init(start_date, duration, self.mvmBuilder.taus),
            rng=np.random.default_rng() if rng is None else rng
        )

        # Verification checks:
        self.ipmBuilder.verify(ctx)
        self.mvmBuilder.verify(ctx)
        ipm = self.ipmBuilder.build(ctx)
        mvm = self.mvmBuilder.build(ctx)

        world = World.initialize(
            self.ipmBuilder.initialize_compartments(ctx)
        )

        out = Output(ctx)
        for t in ctx.clock.ticks:
            tickout = self.tick(ipm, mvm, world, t)
            for i, inc in enumerate(tickout.incidence):
                out.set_inc(i, t, inc)
            for i, prv in enumerate(tickout.prevalence):
                out.set_prv(i, t, prv)
        return out

    def tick(self, ipm: Ipm, mvm: Movement, world: World, tick: Tick) -> TickOutput:
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

        return TickOutput(prevalence=prevalence, incidence=incidence)
