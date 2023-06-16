from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from epymorph.clock import Tick
from epymorph.context import SimContext
from epymorph.epi import Ipm, IpmBuilder
from epymorph.util import Compartments, Events
from epymorph.world import Location


def load() -> IpmBuilder:
    return PeiModelBuilder()


class PeiModelBuilder(IpmBuilder):
    def __init__(self):
        super().__init__(3, 3)

    def verify(self, ctx: SimContext) -> None:
        # TODO: custom exception type with much more useful info
        # TODO: check types and shapes of these things as well
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
        population = ctx.geo['population']
        num_nodes = len(population)
        cs = [self.compartment_array() for _ in range(num_nodes)]
        # The populations of all locations start off Susceptible.
        for i in range(num_nodes):
            cs[i][0] = population[i]
        # With a seeded infection in one location.
        si = ctx.param['infection_seed_loc']
        sn = ctx.param['infection_seed_size']
        cs[si][0] -= sn
        cs[si][1] += sn
        return cs

    def build(self, ctx: SimContext) -> Ipm:
        return PeiModel(ctx)


class PeiModel(Ipm):
    """The Pei influenza model: an SIR model incorporating absolute humidity over time."""
    population: NDArray[np.int_]
    humidity: NDArray[np.double]
    D: float
    L: float

    # for (row,col): should event (row) be added to or subtracted from compartment (col)?
    # this array has to be [num_events] by [num_compartments] in dimension
    event_apply_matrix = np.array([[-1, +1, +0],
                                   [+0, -1, +1],
                                   [+1, +0, -1]])

    def __init__(self, ctx: SimContext):
        super().__init__(ctx)
        self.population = ctx.geo['population']
        self.humidity = ctx.geo['humidity']
        # duration of infection (days)
        self.D = ctx.param['infection_duration']
        # duration of immunity (days)
        self.L = ctx.param['immunity_duration']

    def _beta(self, loc_idx: int, tick: Tick) -> np.double:
        humidity: np.double = self.humidity[tick.day, loc_idx]
        r0_min = 1.3
        r0_max = 2.0
        a = -180.0
        b = np.log(r0_max - r0_min)
        return (np.exp(a * humidity + b) + r0_min) / self.D

    def events(self, loc: Location, tick: Tick) -> Events:
        # TODO: we need a mechanism to make sure we don't exceed the bounds of reality.
        # For instance, in this model, we should never have more infections than there are susceptible people to infect.
        # I don't think that's much of a concern with *this* model, but it will be in the general case, especially
        # as population sizes shrink when we consider more granular spatial scales.
        cs = loc.compartment_totals
        total = np.sum(cs)
        rates = np.array([
            tick.tau * self._beta(loc.index, tick) * cs[0] * cs[1] / total,
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
