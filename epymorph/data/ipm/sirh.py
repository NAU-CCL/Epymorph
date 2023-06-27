from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from epymorph.clock import Tick
from epymorph.context import SimContext
from epymorph.ipm.ipm import Ipm, IpmBuilder
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
        if "hospitalization_duration" not in ctx.param:
            raise Exception("params missing hospitalization_duration")

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
        # Don't infect more people than exist!
        sn = min(cs[si][0], ctx.param['infection_seed_size'])
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
    beta: float
    D: int
    L: int
    H: float
    hosp: int
    event_apply_matrix = np.array(
        [
            # S   I   R   H
            [-1, +1, +0, +0],  # S -> 1
            [+0, -1, +1, +0],  # I -> R
            [+0, -1, +0, +1],  # I -> H
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
        # Hospitalization duration
        self.hosp = ctx.param["hospitalization_duration"]
        # Beta (fixed value)
        self.beta = ctx.param["beta"]

    def events(self, loc: Location, tick: Tick) -> Events:
        # TODO: we need a mechanism to make sure we don't exceed the bounds of reality.
        # For instance, in this model, we should never have more infections than there are susceptible people to infect.
        # I don't think that's much of a concern with *this* model, but it will be in the general case, especially
        # as population sizes shrink when we consider more granular spatial scales.

        cs = loc.compartment_totals
        # if there are 0 people here, avoid div-by-0 error
        total = max(1, sum(cs))
        rates = np.array(
            [
                tick.tau * self.beta * cs[0] * cs[1] / total,  # S -> I
                tick.tau * cs[1] / self.D,  # I -> R
                0,  # I -> H
                tick.tau * cs[3] / self.hosp,  # H -> R
                tick.tau * cs[2] / self.L,  # R -> S
            ]
        )
        evs = self.ctx.rng.poisson(rates)

        # spilt from I to either H or R
        I_to_H = np.random.binomial(evs[1], self.H)
        I_to_R = evs[1] - I_to_H

        # reassigning to compartments
        evs[1] = I_to_R
        evs[2] = I_to_H

        # checks for overdraws in compartments
        if evs[0] > cs[0]:
            evs[0] = cs[0]
        if evs[1] > cs[1]:
            evs[1] = cs[1]
        if evs[2] > cs[2]:
            evs[2] = cs[2]
        if evs[3] > cs[3]:
            evs[3] = cs[3]
        if evs[4] > cs[2]:
            evs[4] = cs[2]
        if (evs[1] + evs[2]) > cs[1]:
            evs[2] = cs[1] - evs[1]

        return evs

    def _draw(self, loc: Location, events: Events) -> list[NDArray[np.int_]]:
        # creats a 1D array of compartments (SIRH) from local population
        cs0 = [pop.compartments[0] for pop in loc.pops]  # S
        cs1 = [pop.compartments[1] for pop in loc.pops]  # I
        cs2 = [pop.compartments[2] for pop in loc.pops]  # R
        cs3 = [pop.compartments[3] for pop in loc.pops]  # H
        evs = events
        # distribute events to local compartments (SIRH)
        hypergeo_s_i = self.ctx.rng.multivariate_hypergeometric(cs0, evs[0])
        hypergeo_i_h = self.ctx.rng.multivariate_hypergeometric(cs1, evs[2])
        hypergeo_i_r = self.ctx.rng.multivariate_hypergeometric(cs1, evs[1])
        hypergeo_h_r = self.ctx.rng.multivariate_hypergeometric(cs3, evs[3])
        hypergeo_r_s = self.ctx.rng.multivariate_hypergeometric(cs2, evs[4])
        # creates an array for to store the hypergeo distribution
        hypergeo_sirh = [
            hypergeo_s_i,
            hypergeo_i_h,
            hypergeo_i_r,
            hypergeo_h_r,
            hypergeo_r_s,
        ]
        return hypergeo_sirh

    def apply_events(self, loc: Location, es: Events) -> None:
        # Distribute events to subpopulations present.
        events = self._draw(loc, es)
        for i, pop in enumerate(loc.pops):
            es_pop = [
                [events[0][i]],  # S -> I
                [events[1][i]],  # I -> R
                [events[2][i]],  # I -> H
                [events[3][i]],  # H -> R
                [events[4][i]],  # R -> S
            ]
            deltas = np.sum(np.multiply(
                es_pop, self.event_apply_matrix), axis=0)
            pop.compartments += deltas
