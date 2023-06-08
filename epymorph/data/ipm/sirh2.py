from __future__ import annotations

import numpy as np

from epymorph.clock import Tick
from epymorph.context import SimContext
from epymorph.epi import (CompartmentalIpm, CompartmentModel, CompartmentSum,
                          EventId, IndependentEvent, Ipm, IpmBuilder,
                          SplitEvent, StateId, SubEvent)
from epymorph.util import Compartments
from epymorph.world import Location


def load() -> IpmBuilder:
    return Builder()


class Builder(IpmBuilder):
    def __init__(self):
        super().__init__(num_compartments=4, num_events=4)

    def build(self, ctx: SimContext) -> Ipm:
        # States
        SUSCEPTIBLE = StateId("Susceptible")
        INFECTED = StateId("Infected")
        RECOVERED = StateId("Recovered")
        HOSPITALIZED = StateId("Hospitalized")

        # Events
        INFECTION = EventId("Infection")
        RECOVERY = EventId("Recovery")
        HOSPITALIZATION = EventId("Hospitalization")
        DISCHARGE = EventId("Discharge")

        # Rate expressions
        def infect(ctx: SimContext, tick: Tick, loc: Location, cs: CompartmentSum) -> int:
            return ctx.param['beta'] * cs[0] * cs[1] / cs.total

        def recover_hospitalize(ctx: SimContext, tick: Tick, loc: Location, cs: CompartmentSum) -> int:
            return ctx.param['gamma'] * cs[1]

        def discharge(ctx: SimContext, tick: Tick, loc: Location, cs: CompartmentSum) -> int:
            return ctx.param['c'] * cs[3]

        # Model
        model = CompartmentModel(
            states=[SUSCEPTIBLE, INFECTED, RECOVERED, HOSPITALIZED],
            events=[INFECTION, RECOVERY, HOSPITALIZATION, DISCHARGE],
            transitions=[
                IndependentEvent(INFECTION, SUSCEPTIBLE, INFECTED, infect),
                SplitEvent(INFECTED, [
                    SubEvent(RECOVERY, RECOVERED, 1.0 - ctx.param['f']),
                    SubEvent(HOSPITALIZATION, HOSPITALIZED, ctx.param['f'])
                ], recover_hospitalize),
                IndependentEvent(DISCHARGE, HOSPITALIZED, RECOVERED, discharge)
            ])

        return CompartmentalIpm(ctx, model)

    def verify(self, ctx: SimContext) -> None:
        if 'population' not in ctx.geo:
            raise Exception("geo missing population")
        if 'beta' not in ctx.param:
            raise Exception("geo missing beta")
        if 'gamma' not in ctx.param:
            raise Exception("params missing gamma")
        if 'f' not in ctx.param:
            raise Exception("params missing f")
        if 'c' not in ctx.param:
            raise Exception("params missing c")
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
