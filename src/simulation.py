from datetime import date
from typing import NamedTuple

import numpy as np

from clock import Clock, Tick
from epi import Ipm
from geo import Geo
from movement import Movement
from sim_context import SimContext
from util import Compartments, Events
from world import World


class TickOutput(NamedTuple):
    prevalence: list[Compartments]
    incidence: list[Events]


class Output:
    def __init__(self, ctx: SimContext, clock: Clock):
        self.clock = clock
        def arr(n): return np.zeros((clock.num_ticks, n), dtype=np.int_)
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
        self.ctx = SimContext(
            compartments=ipm.num_compartments,
            events=ipm.num_events,
            nodes=geo.num_nodes,
            labels=geo.node_labels,
            rng=np.random.default_rng()
        )

    def run(self, start_date: date, duration: int) -> Output:
        pops = self.ipm.initialize(self.geo.num_nodes)
        world = World.initialize(pops)
        clock = Clock(start_date, duration, self.mvm.taus)
        out = Output(self.ctx, clock)

        for t in clock.ticks:
            tickout = self.tick(world, t)
            for i, inc in enumerate(tickout.incidence):
                out.set_inc(i, t.time, inc)
            for i, prv in enumerate(tickout.prevalence):
                out.set_prv(i, t.time, prv)

        return out

    def tick(self, world: World, tick: Tick) -> TickOutput:
        # First do movement
        step = self.mvm.steps[tick.step]
        step.clause.apply(self.ctx, world, tick)

        # Calc events by compartment
        incidence = [self.ipm.events(loc, step.tau, tick)
                     for loc in world.locations]

        # Distribute events
        for loc, es in zip(world.locations, incidence):
            self.ipm.apply_events(loc, es)

        # Calc new effective population by compartment
        prevalence = [loc.compartment_totals for loc in world.locations]

        return TickOutput(prevalence=prevalence, incidence=incidence)
