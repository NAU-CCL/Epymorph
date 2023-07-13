from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from epymorph.clock import Tick
from epymorph.context import SimContext
from epymorph.epi import Ipm, IpmBuilder
from epymorph.util import Compartments, Events, expand_data
from epymorph.world import Location


def load() -> IpmBuilder:
    return Builder()


class Builder(IpmBuilder):
    def __init__(self):
        super().__init__(3, 3)

    def verify(self, ctx: SimContext) -> None:
        if 'population' not in ctx.geo:
            raise Exception("geo missing population")
        if 'beta' not in ctx.param:
            raise Exception("params missing beta")
        if 'infection_duration' not in ctx.param:
            raise Exception("params missing infection_duration")
        if 'immunity_duration' not in ctx.param:
            raise Exception("params missing immunity_duration")
        if 'infection_seed_loc' not in ctx.param:
            raise Exception("params missing infection_seed_loc")
        if 'infection_seed_size' not in ctx.param:
            raise Exception("params missing infection_seed_size")

    def initialize_compartments(self, ctx: SimContext) -> list[Compartments]:
        population = ctx.geo['population']
        num_nodes = len(population)
        cs = [self.compartment_array() for _ in range(num_nodes)]
        # The populations of all locations start off Susceptible.
        for i in range(num_nodes):
            cs[i][0] = population[i]
        # With a seeded infection in one location.
        si = ctx.param['infection_seed_loc']
        # Don't infect more people than exist!
        sn = min(cs[si][0], ctx.param['infection_seed_size'])
        cs[si][0] -= sn
        cs[si][1] += sn
        return cs

    def build(self, ctx: SimContext) -> Ipm:
        return SimpleSIRS(ctx)


class SimpleSIRS(Ipm):
    """A simple SIRS model using a fixed, time-varying, or time-and-population-varying beta."""
    population: NDArray[np.int_]
    beta: NDArray[np.double]
    D: int
    L: int

    event_apply_matrix = np.array([[-1, +1, +0],
                                   [+0, -1, +1],
                                   [+1, +0, -1]])

    def __init__(self, ctx: SimContext):
        super().__init__(ctx)
        self.population = ctx.geo['population']
        self.beta = expand_data(
            ctx.param['beta'], ctx.clock.num_days, ctx.nodes)
        # duration of infection (days)
        self.D = ctx.param['infection_duration']
        # duration of immunity (days)
        self.L = ctx.param['immunity_duration']

    def events(self, loc: Location, tick: Tick) -> Events:
        # TODO: we need a mechanism to make sure we don't exceed the bounds of reality.
        # For instance, in this model, we should never have more infections than there are susceptible people to infect.
        # I don't think that's much of a concern with *this* model, but it will be in the general case, especially
        # as population sizes shrink when we consider more granular spatial scales.
        cs = loc.compartment_totals
        # if there are 0 people here, avoid div-by-0 error
        total = max(1, sum(cs))
        rates = np.array([
            tick.tau * self.beta[tick.day, loc.index] *
            cs[0] * cs[1] / total,
            tick.tau * cs[1] / self.D,
            tick.tau * cs[2] / self.L,
        ])
        return self.ctx.rng.poisson(rates)

    def _draw(self, loc: Location, events: Events, ev_idx: int) -> NDArray[np.int_]:
        # What if a compartment goes negative?! This is best handled during the `events` stage, though.
        # We're protecting against that with `min` here becuase mvhypergeo crashes when it happens.
        # But this is still a problem: incidence counts are no longer entirely accurate to the degree this happens.
        compart_vec = [loc.compartments[ev_idx] for loc in loc.pops]
        # not generalized; assumes event[0] "sources" from compartment[0], etc.
        max_events = min(events[ev_idx], sum(compart_vec))
        return self.ctx.rng.multivariate_hypergeometric(compart_vec, max_events)

    def apply_events(self, loc: Location, es: Events) -> None:
        # Distribute events to subpopulations present.
        events = [self._draw(loc, es, i) for i in range(len(es))]
        for i, pop in enumerate(loc.pops):
            es_pop = [[events[0][i]], [events[1][i]], [events[2][i]]]
            deltas = np.sum(np.multiply(
                es_pop, self.event_apply_matrix), axis=0)
            pop.compartments += deltas
