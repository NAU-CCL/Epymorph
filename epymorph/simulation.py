from datetime import date
from typing import NamedTuple

import numpy as np

from epymorph.clock import Clock, Tick
from epymorph.epi import Ipm
from epymorph.geo import Geo
from epymorph.movement import Movement
from epymorph.sim_context import SimContext
from epymorph.util import Compartments, Events
from epymorph.world import World


class TickOutput(NamedTuple):
    prevalence: list[Compartments]
    incidence: list[Events]


class Output:
    def __init__(self, ctx: SimContext):
        self.clock = ctx.clock
        def arr(n): return np.zeros((ctx.clock.num_ticks, n), dtype=np.int_)
        m = ctx.nodes
        self.prevalence = [arr(ctx.compartments) for _ in range(m)]
        self.incidence = [arr(ctx.events) for _ in range(m)]
        self.pop_labels = ctx.labels

    def set_prv(self, loc_idx: int, tick: int, prv: Compartments) -> None:
        self.prevalence[loc_idx][tick] = prv

    def set_inc(self, loc_idx: int, tick: int, inc: Compartments) -> None:
        self.incidence[loc_idx][tick] = inc


class Simulation:
    def __init__(self, ipm: Ipm, mvm: Movement, geo: Geo):
        self.ipm = ipm
        self.mvm = mvm
        self.geo = geo

    def run(self, start_date: date, duration: int, rng: np.random.Generator | None = None) -> Output:
        pops = self.ipm.initialize(self.geo.num_nodes)
        world = World.initialize(pops)
        clock = Clock.init(start_date, duration, self.mvm.taus)
        if rng == None:
            rng = np.random.default_rng()
        ctx = SimContext(
            compartments=self.ipm.num_compartments,
            events=self.ipm.num_events,
            nodes=self.geo.num_nodes,
            labels=self.geo.node_labels,
            clock=clock,
            rng=rng
        )
        out = Output(ctx)

        for t in clock.ticks:
            tickout = self.tick(ctx, world, t)
            for i, inc in enumerate(tickout.incidence):
                out.set_inc(i, t.index, inc)
            for i, prv in enumerate(tickout.prevalence):
                out.set_prv(i, t.index, prv)

        return out

    def tick(self, ctx: SimContext, world: World, tick: Tick) -> TickOutput:
        # First do movement
        self.mvm.clause.apply(ctx, world, tick)

        # Calc events by compartment
        incidence = [self.ipm.events(ctx, loc, tick)
                     for loc in world.locations]

        # Distribute events
        for loc, es in zip(world.locations, incidence):
            self.ipm.apply_events(ctx, loc, es)

        # Calc new effective population by compartment
        prevalence = [loc.compartment_totals for loc in world.locations]

        return TickOutput(prevalence=prevalence, incidence=incidence)
