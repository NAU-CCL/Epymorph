from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from epymorph.clock import Tick
from epymorph.context import SimContext
from epymorph.data.ipm.initializer import DefaultInitializer, Initializer
from epymorph.epi import Ipm, IpmBuilder
from epymorph.util import Compartments, Events
from epymorph.world import Location


def load() -> IpmBuilder:
    return sirhBuilder()


class sirhBuilder(IpmBuilder):
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
        out = [np.array([p, 0, 0, 0], dtype=int)
               for p in ctx.geo["population"]]
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
            [-1, +1, +0, +0],  # S -> I
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
        # I don't think that's much of a concern with *this* model, but it will be in the general case, especially
        # as population sizes shrink when we consider more granular spatial scales.

        cs = loc.compartment_totals
        beta = self.ctx.geo["beta"]
        total = np.sum(cs)
        if total == 0:
            return np.zeros(self.ctx.events, dtype=int)

        rates = np.array(
            [
                tick.tau * beta[(tick.day), loc.index] *
                cs[0] * cs[1] / total,  # S -> I
                tick.tau * cs[1] / self.D,  # leaving I compartment (I -> R)
                0,  # I -> H
                tick.tau * cs[3] / self.hosp,  # H -> R
                tick.tau * cs[2] / self.L,  # R -> S
            ]
        )

        evs = self.ctx.rng.poisson(rates)

        # spilt from I to either H or R
        I_to_H = np.random.binomial(evs[1], self._gamma(loc.index))
        I_to_R = evs[1] - I_to_H
        # reassigning to compartments
        evs[1] = I_to_R
        evs[2] = I_to_H

        # checks for overdraws in compartments
        evs[0] = min(evs[0], cs[0])
        if (evs[1] + evs[2]) > cs[1]:
            evs[1] = self.ctx.rng.hypergeometric(evs[1], evs[2], cs[1])
            evs[2] = cs[1] - evs[1]
        evs[3] = min(evs[3], cs[3])
        evs[4] = min(evs[4], cs[2])

        return evs

    # Compartments Reference:
    # S I R H
    # 0 1 2 3

    # Events Reference:
    # e0: (S->I) c0 -> c1
    # e1: (I->R) c1 -> c2
    # e2: (I->H) c1 -> c3
    # e3: (H->R) c3 -> c2
    # e4: (R->S) c2 -> c0

    # Tuples of (event_idx, compartment_index) describing
    # which compartment each event draws from.
    _events: list[tuple[int, int]] = [(0, 0), (1, 1), (2, 1), (3, 3), (4, 2)]

    def apply_events(self, loc: Location, evs: Events) -> None:
        mvhg = self.ctx.rng.multivariate_hypergeometric  # alias

        # PxC array (compartments per population) decremented as we select individuals
        available = np.array([pop.compartments for pop in loc.pops], dtype=int)
        # PxE array (events per population) assignment of events to each population
        occurrences = np.zeros((len(loc.pops), 5), dtype=int)

        # Select individuals for each event (all populations simultaneously).
        for eidx, cidx in self._events:
            selected = mvhg(available[:, cidx], evs[eidx])
            occurrences[:, eidx] = selected
            available[:, cidx] -= selected

        # Update populations.
        for pidx, pop in enumerate(loc.pops):
            deltas = np.matmul(occurrences[pidx], self.event_apply_matrix)
            pop.compartments += deltas
