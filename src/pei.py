from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from clock import Tick
from epi import Ipm
from geo import Geo, GeoParamN, GeoParamNN, GeoParamNT, ParamN, ParamNT
from util import Compartments, Events
from world import Location


def loadGeo() -> Geo:
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
        self.gen = np.random.default_rng()

    def _beta(self, loc_idx: int, tick: Tick) -> np.double:
        humidity: np.double = self.humidity(loc_idx, tick)
        r0_min = np.double(1.3)
        r0_max = np.double(2)
        a = np.double(-180)
        b = np.log(r0_max - r0_min)
        return np.exp(a * humidity + b) + r0_min / self.D

    def initialize(self, num_nodes: int) -> list[Compartments]:
        pops = [self.c() for _ in range(num_nodes)]
        for i in range(num_nodes):
            pops[i][0] = self.population(i)
        pops[0][0] -= 10_000
        pops[0][1] += 10_000
        return pops

    def events(self, loc: Location, tau: np.double, tick: Tick) -> Events:
        cs = loc.compartment_totals
        total = np.sum(cs)
        rates = np.array([
            tau * self._beta(loc.index, tick) * cs[0] * cs[1] / total,
            tau * cs[1] / self.D,
            tau * cs[2] / self.L,
        ])
        return np.random.poisson(rates)
        # Check if we've exceeded population limits.
        # TODO: a check like this doesn't detect if the distribution to subpops is proper, though
        # deltas = np.sum(np.multiply(columnize(events, self.num_events),
        #                 self.event_apply_matrix), axis=0)
        # new_cs = cs + deltas
        # if any(new_cs < 0):
        #     print("WARNING: events exceeded population counts")

    def _draw(self, loc: Location, events: Events, ev_idx: int) -> NDArray[np.int_]:
        # TODO: what if a compartment goes negative?!
        # Actually we're protecting against that with "min" here becuase mvhypergeo crashes when it happens.
        # But this is still a problem: the incidence counts are no longer entirely accurate to the degree this happens.
        compart_vec = [loc.compartments[ev_idx] for loc in loc.pops]
        # not generalized; assumes event[0] "sources" from compartment[0], etc.
        max_events = min(events[ev_idx], sum(compart_vec))
        return self.gen.multivariate_hypergeometric(compart_vec, max_events)

    def apply_events(self, loc: Location, es: Events) -> None:
        # Distribute events to subpopulations present.
        events = [self._draw(loc, es, i) for i in range(len(es))]
        for i, pop in enumerate(loc.pops):
            es_pop = [[events[0][i]], [events[1][i]], [events[2][i]]]
            deltas = np.sum(np.multiply(
                es_pop, self.event_apply_matrix), axis=0)
            pop.compartments += deltas
