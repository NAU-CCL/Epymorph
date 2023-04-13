from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from epymorph.clock import Tick
from epymorph.epi import Ipm
from epymorph.geo import ParamN, ParamNT
from epymorph.sim_context import SimContext
from epymorph.util import Compartments, Events
from epymorph.world import Location


class PeiModel(Ipm):
    """The Pei influenza model: an SIR model incorporating absolute humidity over time."""

    # for (row,col): should event (row) be added to or subtracted from compartment (col)?
    # this array has to be [num_events] by [num_compartments] in dimension
    event_apply_matrix = np.array([[-1, +1, +0],
                                   [+0, -1, +1],
                                   [+1, +0, -1]])

    def __init__(self,
                 population: ParamN[np.int_],
                 humidity: ParamNT[np.double],
                 D: np.double,
                 L: np.double):
        super().__init__(3, 3)
        self.population = population
        self.humidity = humidity
        self.D = D  # duration of infection (days)
        self.L = L  # duration of immunity (days)

    def _beta(self, loc_idx: int, tick: Tick) -> np.double:
        humidity: np.double = self.humidity(loc_idx, tick)
        r0_min = np.double(1.3)
        r0_max = np.double(2)
        a = np.double(-180)
        b = np.log(r0_max - r0_min)
        return np.exp(a * humidity + b) + r0_min / self.D

    def initialize(self, num_nodes: int) -> list[Compartments]:
        # The populations of all locations start off Susceptible.
        pops = [self.c() for _ in range(num_nodes)]
        for i in range(num_nodes):
            pops[i][0] = self.population(i)
        # This is where we seed the infection.
        # TODO: this should really be a sort of "initial condition" defined apart from the model.
        pops[0][0] -= 10_000
        pops[0][1] += 10_000
        return pops

    def events(self, sim: SimContext, loc: Location, tick: Tick) -> Events:
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
        return sim.rng.poisson(rates)

    def _draw(self, sim: SimContext, loc: Location, events: Events, ev_idx: int) -> NDArray[np.int_]:
        # What if a compartment goes negative?! This is best handled during the `events` stage, though.
        # We're protecting against that with `min` here becuase mvhypergeo crashes when it happens.
        # But this is still a problem: incidence counts are no longer entirely accurate to the degree this happens.
        compart_vec = [loc.compartments[ev_idx] for loc in loc.pops]
        # not generalized; assumes event[0] "sources" from compartment[0], etc.
        max_events = min(events[ev_idx], sum(compart_vec))
        return sim.rng.multivariate_hypergeometric(compart_vec, max_events)

    def apply_events(self, sim: SimContext, loc: Location, es: Events) -> None:
        # Distribute events to subpopulations present.
        events = [self._draw(sim, loc, es, i) for i in range(len(es))]
        for i, pop in enumerate(loc.pops):
            es_pop = [[events[0][i]], [events[1][i]], [events[2][i]]]
            deltas = np.sum(np.multiply(
                es_pop, self.event_apply_matrix), axis=0)
            pop.compartments += deltas
