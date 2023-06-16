from __future__ import annotations

import numpy as np

from epymorph.clock import Tick
from epymorph.context import SimContext
from epymorph.epi import (CompartmentalIpm, CompartmentModel, CompartmentSum,
                          EventId, IndependentEvent, Ipm, IpmBuilder, StateId)
from epymorph.util import Compartments
from epymorph.world import Location


def load() -> IpmBuilder:
    return Builder()


class Builder(IpmBuilder):
    def __init__(self):
        super().__init__(num_compartments=3, num_events=3)

    def build(self, ctx: SimContext) -> Ipm:
        # States
        SUSCEPTIBLE = StateId("Susceptible")
        INFECTED = StateId("Infected")
        RECOVERED = StateId("Recovered")

        # Events
        INFECTION = EventId("Infection")
        RECOVERY = EventId("Recovery")
        RELAPSE = EventId("Relapse")

        # Rate expressions
        def infect(ctx: SimContext, tick: Tick, loc: Location, cs: CompartmentSum) -> int:
            humidity: float = ctx.geo["humidity"][tick.day, loc.index]
            r0_min = 1.3
            r0_max = 2
            a = -180.0
            b = np.log(r0_max - r0_min)
            beta = (np.exp(a * humidity + b) + r0_min) / \
                ctx.param["infection_duration"]
            return beta * cs[0] * cs[1] / cs.total

        def recover(ctx: SimContext, tick: Tick, loc: Location, cs: CompartmentSum) -> int:
            return cs[1] / ctx.param['infection_duration']

        def relapse(ctx: SimContext, tick: Tick, loc: Location, cs: CompartmentSum) -> int:
            return cs[2] / ctx.param['immunity_duration']

        # Model
        model = CompartmentModel(
            states=[SUSCEPTIBLE, INFECTED, RECOVERED],
            events=[INFECTION, RECOVERY, RELAPSE],
            transitions=[
                IndependentEvent(INFECTION, SUSCEPTIBLE, INFECTED, infect),
                IndependentEvent(RECOVERY, INFECTED, RECOVERED, recover),
                IndependentEvent(RELAPSE, RECOVERED, SUSCEPTIBLE, relapse)
            ])

        return CompartmentalIpm(ctx, model)

    def verify(self, ctx: SimContext) -> None:
        if 'population' not in ctx.geo:
            raise Exception("geo missing population")
        if 'humidity' not in ctx.geo:
            raise Exception("geo missing humidity")
        if 'infection_duration' not in ctx.param:
            raise Exception("params missing infection_duration")
        if 'immunity_duration' not in ctx.param:
            raise Exception("params missing immunity_duration")
        if 'infection_seed_loc' not in ctx.param:
            raise Exception("params missing infection_seed_loc")
        if 'infection_seed_size' not in ctx.param:
            raise Exception("params missing infection_seed_size")

    def initialize_compartments(self, ctx: SimContext) -> list[Compartments]:
        # Initial compartments based on population (C0)
        population = ctx.geo['population']
        # With a seeded infection (C1) in one location
        si = ctx.param['infection_seed_loc']
        sn = ctx.param['infection_seed_size']
        cs = np.zeros((ctx.nodes, ctx.compartments), dtype=np.int_)
        cs[:, 0] = population
        cs[si, 0] -= sn
        cs[si, 1] += sn
        return list(cs)
