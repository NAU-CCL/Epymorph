from __future__ import annotations

import numpy as np

from epymorph.clock import Tick
from epymorph.context import SimContext
from epymorph.epi import (EventId, IndependentEvent, Ipm, IpmBuilder,
                          IpmBuilderFromModel, IpmForModel, IpmModel, StateId)
from epymorph.world import Location


def load() -> IpmBuilder:
    return PeiModelBuilder()


class PeiModelBuilder(IpmBuilderFromModel):
    def __init__(self):
        # States
        SUSCEPTIBLE = StateId("Susceptible")
        INFECTED = StateId("Infected")
        RECOVERED = StateId("Recovered")

        # Events
        INFECTION = EventId("Infection")
        RECOVERY = EventId("Recovery")
        RELAPSE = EventId("Relapse")

        # Rate expressions
        def infect(ctx: SimContext, tick: Tick, loc: Location) -> int:
            humidity: float = ctx.geo["humidity"][tick.day, loc.index]
            r0_min = 1.3
            r0_max = 2
            a = -180.0
            b = np.log(r0_max - r0_min)
            beta = np.exp(a * humidity + b) + r0_min / \
                ctx.param["infection_duration"]

            cs = loc.compartment_totals
            total = np.sum(cs)
            return beta * cs[0] * cs[1] / total

        def recover(ctx: SimContext, tick: Tick, loc: Location) -> int:
            cs = loc.compartment_totals
            return cs[1] / ctx.param['infection_duration']

        def relapse(ctx: SimContext, tick: Tick, loc: Location) -> int:
            cs = loc.compartment_totals
            return cs[2] / ctx.param['immunity_duration']

        # Model
        pei_model = IpmModel(
            states=[SUSCEPTIBLE, INFECTED, RECOVERED],
            events=[INFECTION, RECOVERY, RELAPSE],
            transitions=[
                IndependentEvent(INFECTION, SUSCEPTIBLE, INFECTED, infect),
                IndependentEvent(RECOVERY, INFECTED, RECOVERED, recover),
                IndependentEvent(RELAPSE, RECOVERED, SUSCEPTIBLE, relapse)
            ])

        super().__init__(pei_model)

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

    def build(self, ctx: SimContext) -> Ipm:
        return IpmForModel(ctx, self.model)
