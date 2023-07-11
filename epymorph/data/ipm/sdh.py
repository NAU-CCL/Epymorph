from __future__ import annotations

import os

import numpy as np
from numpy.typing import NDArray

from epymorph.clock import Tick
from epymorph.context import SimContext
from epymorph.data.ipm.initializer import DefaultInitializer, Initializer
from epymorph.epi import Ipm, IpmBuilder
from epymorph.util import Compartments, Events
from epymorph.world import Location

dir = os.path.expanduser("~/Desktop/Github/Epymorph/scratch/output_files")


def load() -> IpmBuilder:
    return sirhBuilder()


class sirhBuilder(IpmBuilder):
    # An initializer instance which can be overridden.
    initializer: Initializer

    def __init__(self):
        # Creats compartments for SIRH events
        super().__init__(4, 5)
        self.initializer = DefaultInitializer()

    def verify(self, ctx: SimContext) -> None:
        if "population" not in ctx.geo:
            raise Exception("geo missing population")
        if "infection_duration" not in ctx.param:
            raise Exception("params missing infection_duration")
        if "immunity_duration" not in ctx.param:
            raise Exception("params missing immunity_duration")
        if "hospitalization_duration" not in ctx.param:
            raise Exception("params missing hospitalization_duration")

    def initialize_compartments(self, ctx: SimContext) -> list[Compartments]:
        # The populations of all locations start off in the first compartment.
        # Note: four compartments is hard-coded here.
        out = [np.array([p, 0, 0, 0], dtype=int) for p in ctx.geo["population"]]
        # Now delegate to our initializer function, writing the result into `out`.
        self.initializer.apply(ctx, out)
        return out
        

    def build(self, ctx: SimContext) -> Ipm:
        return sdh(ctx)


class sdh(Ipm):
    """A simple SIRS model using a fixed, time-varying, or time-and-population-varying beta."""

    population: NDArray[np.int_]
    alpha: NDArray[np.double]
    gamma: NDArray[np.double]
    hosp: float
    D: int
    L: int
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
        # Hospitalization duration
        self.hosp = ctx.param["hospitalization_duration"]
        # alpha
        self.alpha = ctx.param["alpha"]
        # gamma
        self.gamma = ctx.param["gamma"]

    def exp_beta(self, loc_idx: int) -> np.double:
        a0 = self.alpha[0]

        a1 = self.alpha[1]
        x1 = self.ctx.geo["average_household_size"][loc_idx]
        scale_x1 = (x1 - self.ctx.geo["average_household_size"].mean()) / self.ctx.geo[
            "average_household_size"
        ].std()

        a2 = self.alpha[2]
        x2 = self.ctx.geo["pop_density_km2"][loc_idx]
        scale_x2 = (x2 - self.ctx.geo["pop_density_km2"].mean()) / self.ctx.geo[
            "pop_density_km2"
        ].std()
        # np.exp((a0 + (a1 * scale_x1) * (a2 * scale_x2)))
        # print(beta)
        beta = a0 * np.exp(((a1 * scale_x1) + (a2 * scale_x2)))

        return beta

    def _gamma(self, loc_idx) -> float:
        h1 = self.ctx.geo["median_income"][loc_idx]
        h2 = self.ctx.geo["tract_gini_index"][loc_idx]
        g0 = self.gamma

        scale_h1 = (h1 - self.ctx.geo["median_income"].mean()) / self.ctx.geo[
            "median_income"
        ].std()
        scale_h2 = (h2 - self.ctx.geo["tract_gini_index"].mean()) / self.ctx.geo[
            "tract_gini_index"
        ].std()

        gamma = g0 * (
            np.exp((scale_h1 + scale_h2)) / (1 + np.exp((scale_h1 + scale_h2)))
        )
        return gamma

    def events(self, loc: Location, tick: Tick) -> Events:
        # TODO: we need a mechanism to make sure we don't exce3ed the bounds of reality.
        # For instance, in this model, we should never have more infections than there are susceptible people to infect.
        # I don't think that's much of a conceßrn with *this* model, but it will be in the general case, especially
        # as population sizes shrink when we consider more granular spatial scales.

        cs = np.abs(loc.compartment_totals)
        total = np.sum(cs)
        if total == 0:
            return np.zeros(self.ctx.events, dtype=int)
        rates = np.array(
            [
                tick.tau * self.exp_beta(loc.index) * cs[0] * cs[1] / total,  # S -> I
                tick.tau * cs[1] / self.D,  # leaving I compartment (I -> R)
                0,  # I -> H
                tick.tau * cs[3] / self.hosp,  # H -> R
                tick.tau * cs[2] / self.L,  # R -> S
            ]
        )
        # print(self.ctx.geo["labels"][loc.index])
        # print(rates)

        evs = self.ctx.rng.poisson(np.abs(rates))

        # spilt from I to either H or R
        I_to_H = np.random.binomial(evs[1], self._gamma(loc.index))
        I_to_R = evs[1] - I_to_H

        # reassigning to compartments
        evs[1] = I_to_R
        evs[2] = I_to_H

        # checks for overdraws in compartments
        if evs[0] > cs[0]:
            evs[0] = cs[0]
        if evs[1] > cs[1]:
            evs[1] = cs[1]
        if evs[1] < 0:
            evs[1] = 0
        if evs[2] > cs[2]:
            evs[2] = 0
        if evs[2] < 0:
            evs[2] = 0
        if evs[3] > cs[3]:
            evs[3] = cs[3]
        if evs[4] > cs[2]:
            evs[4] = cs[2]
        if (evs[1] + evs[2]) > cs[1]:
            evs[2] = cs[1] - evs[1]

        return evs

    def _draw(self, loc: Location, events: Events) -> list[NDArray[np.int_]]:
        # creats a 1D array of compartments (SIRH) from local population
        cs0 = np.abs([pop.compartments[0] for pop in loc.pops])  # S
        cs1 = np.abs([pop.compartments[1] for pop in loc.pops])  # I
        cs2 = np.abs([pop.compartments[2] for pop in loc.pops])  # R
        cs3 = np.abs([pop.compartments[3] for pop in loc.pops])  # H

        evs = events
        # distribute events to local compartments (SIRH)

        hypergeo_s_i = self.ctx.rng.multivariate_hypergeometric(cs0, evs[0])
        hypergeo_i_r = self.ctx.rng.multivariate_hypergeometric(cs1, evs[1])
        hypergeo_i_h = self.ctx.rng.multivariate_hypergeometric(cs1, evs[2])
        hypergeo_h_r = self.ctx.rng.multivariate_hypergeometric(cs3, evs[3])
        hypergeo_r_s = self.ctx.rng.multivariate_hypergeometric(cs2, evs[4])
        # creates an array for to store the hypergeo distribution
        hypergeo_sirh = [
            hypergeo_s_i,
            hypergeo_i_r,
            hypergeo_i_h,
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
            deltas = np.sum(np.multiply(es_pop, self.event_apply_matrix), axis=0)
            pop.compartments += deltas
