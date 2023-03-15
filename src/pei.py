from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import movement as M
from clock import Tick, TickDelta
from epi import Ipm
from geo import Geo, GeoParamN, GeoParamNN, GeoParamNT, ParamN, ParamNT
from sim_context import SimContext
from util import Compartments, Events, is_square
from world import Location

# Geo Model


def load_geo() -> Geo:
    pop_labels = ["FL", "GA", "MD", "NC", "SC", "VA"]
    humidity = np.loadtxt('./data/pei-humidity.csv',
                          delimiter=',', dtype=np.double)
    population = np.loadtxt('./data/pei-population.csv',
                            delimiter=',', dtype=np.int_)
    commuters = np.loadtxt('./data/pei-commuters.csv',
                           delimiter=',', dtype=np.int_)
    return Geo(pop_labels, [
        GeoParamN("population", population),
        GeoParamNT("humidity", humidity),
        GeoParamNN("commuters", commuters)
    ])

# Movement Model


def build_movement(commuters: NDArray[np.int_], move_control: float, theta: float) -> M.Movement:
    assert 0 <= move_control <= 1.0, "Move Control must be in the range [0,1]."
    assert 0 <= theta, "Theta must be not less than zero."
    assert is_square(commuters), "Commuters matrix must be square."

    def commuter_equation() -> M.RowEquation:
        # Total commuters living in each state.
        commuters_by_state = commuters.sum(axis=1, dtype=np.int_)
        # Commuters as a ratio to the total commuters living in that state.
        commuting_prob = commuters / \
            commuters.sum(axis=1, keepdims=True, dtype=np.int_)

        def equation(sim: SimContext, tick: Tick, src_idx: int) -> NDArray[np.int_]:
            # Binomial draw with probability `move_control` to modulate total number of commuters.
            typical = commuters_by_state[src_idx]
            actual = sim.rng.binomial(typical, move_control)
            # Multinomial draw for destination.
            return sim.rng.multinomial(actual, commuting_prob[src_idx])
        return equation

    def disperser_equation() -> M.RowEquation:
        # Pre-compute the average commuters between node pairs.
        commuters_avg = np.zeros(commuters.shape)
        for i in range(commuters.shape[0]):
            for j in range(i + 1, commuters.shape[1]):
                nbar = (commuters[i, j] + commuters[j, i]) // 2
                commuters_avg[i, j] = nbar
                commuters_avg[j, i] = nbar

        def equation(sim: SimContext, tick: Tick, src_idx: int) -> NDArray[np.int_]:
            return sim.rng.poisson(commuters_avg[src_idx] * theta)
        return equation

    return M.Movement(
        # First step is day: 2/3 tau
        # Second step is night: 1/3 tau
        taus=[np.double(2/3), np.double(1/3)],
        clause=M.Sequence([
            # Main commuters: on step 0
            M.GeneralClause.byRow(
                name="Commuters",
                predicate=M.Predicates.everyday(step=0),
                returns=TickDelta(0, 1),  # returns today on step 1
                equation=commuter_equation()
            ),
            # Random dispersers: also on step 0, cumulative effect.
            M.GeneralClause.byRow(
                name="Dispersers",
                predicate=M.Predicates.everyday(step=0),
                returns=TickDelta(0, 1),  # returns today on step 1
                equation=disperser_equation()
            ),
            # Return: always triggers, but only moves pops whose return time is now.
            M.Return()
        ])
    )


# Intra-Population Model

class PeiModel(Ipm):
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
