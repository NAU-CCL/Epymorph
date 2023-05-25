from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from epymorph.clock import Tick
from epymorph.context import SimContext
from epymorph.epi import Ipm, IpmBuilder
from epymorph.util import Compartments, Events
from epymorph.world import Location


def load() -> IpmBuilder:
    return sirhBuilder()


class sirhBuilder(IpmBuilder):
    def __init__(self):
        # Creats compartments for SIRH events
        super().__init__(4, 5)

    def verify(self, ctx: SimContext) -> None:
        if "population" not in ctx.geo:
            raise Exception("geo missing population")
        if "beta" not in ctx.param:
            raise Exception("params missing beta")
        if "infection_duration" not in ctx.param:
            raise Exception("params missing infection_duration")
        if "immunity_duration" not in ctx.param:
            raise Exception("params missing immunity_duration")
        if "infection_seed_loc" not in ctx.param:
            raise Exception("params missing infection_seed_loc")
        if "infection_seed_size" not in ctx.param:
            raise Exception("params missing infection_seed_size")
        if "hospitalization_rate" not in ctx.param:
            raise Exception("params missing hospitalization_rate")

    def initialize_compartments(self, ctx: SimContext) -> list[Compartments]:
        population = ctx.geo["population"]
        humidity: NDArray[np.double]
        num_nodes = len(population)
        cs = [self.compartment_array() for _ in range(num_nodes)]
        # The populations of all locations start off Susceptible.
        for i in range(num_nodes):
            cs[i][0] = population[i]
        # With a seeded infection in one location.
        si = ctx.param["infection_seed_loc"]
        sn = ctx.param["infection_seed_size"]
        cs[si][0] -= sn
        cs[si][1] += sn
        return cs

    def build(self, ctx: SimContext) -> Ipm:
        return sirh(ctx)


class sirh(Ipm):
    """A simple SIRS model using a fixed, time-varying, or time-and-population-varying beta."""

    population: NDArray[np.int_]
    humidity: NDArray[np.double]
    beta: NDArray[np.double]
    D: int
    L: int
    H: float
    event_apply_matrix = np.array(
        [
            # S   I   R   H
            [-1, +1, +0, +0],  # S -> E
            [+0, -1, +0, +1],  # I -> H
            [+0, -1, +1, +0],  # I -> R
            [+0, +0, +1, -1],  # H -> R
            [+1, +0, -1, +0],  # R -> S
        ]
    )

    def __init__(self, ctx: SimContext):
        super().__init__(ctx)
        self.population = ctx.geo["population"]
        # duration of infection (days)
        self.D = ctx.param["infection_duration"]
        # duration of immunity (days)
        self.L = ctx.param["immunity_duration"]
        # hospitalization rate
        self.H = ctx.param["hospitalization_rate"]

    def _beta(self, loc_idx: int, tick: Tick) -> np.double:
        humidity: np.double = self.humidity[tick.day, loc_idx]
        r0_min = np.double(1.3)
        r0_max = np.double(2)
        a = np.double(-180)
        b = np.log(r0_max - r0_min)
        return np.exp(a * humidity + b) + r0_min / self.D

    def events(self, loc: Location, tick: Tick) -> Events:
        # TODO: we need a mechanism to make sure we don't exceed the bounds of reality.
        # For instance, in this model, we should never have more infections than there are susceptible people to infect.
        # I don't think that's much of a concern with *this* model, but it will be in the general case, especially
        # as population sizes shrink when we consider more granular spatial scales.

        cs = loc.compartment_totals
        total = np.sum(cs)
        rates = np.array(
            [
                tick.tau
                * self._beta(loc.index, tick)
                * cs[0]
                * cs[1]
                / total,  # S -> E
                tick.tau * np.random.binomial(cs[1], self.H),  # E -> H
                tick.tau * cs[1] / self.D,  # I -> R
                tick.tau * cs[3] / self.D,  # H -> R
                tick.tau * cs[2] / self.L,  # R -> S
            ]
        )
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
            es_pop = [[events[0][i]], [events[1][i]], [events[2][i]], [events[3][i]]]
            deltas = np.sum(np.multiply(es_pop, self.event_apply_matrix), axis=0)
            pop.compartments += deltas
