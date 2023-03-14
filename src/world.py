from __future__ import annotations

from operator import attrgetter

import numpy as np

from util import Compartments


class Timer:
    Home = -1


class Population:
    def __init__(self, compartments: Compartments, dest: int, timer: int):
        self.compartments = compartments
        self.dest = dest
        self.timer = timer

    def merge(self, other: Population) -> None:
        self.compartments += other.compartments

    @property
    def total(self) -> np.int_:
        return np.sum(self.compartments)

    def __eq__(self, other):
        return (isinstance(other, Population) and
                self.compartments == other.compartments and
                self.dest == other.dest and
                self.timer == other.timer)

    @classmethod
    def can_merge(cls, a: Population, b: Population) -> bool:
        return a.timer == b.timer and a.dest == b.dest

    @classmethod
    def normalize(cls, ps: list[Population]) -> None:
        ps.sort(key=attrgetter('timer', 'dest'))
        # Iterate over all sequential pairs, starting from (0,1)
        j = 1
        while j < len(ps):
            curr = ps[j - 1]
            next = ps[j]
            if Population.can_merge(curr, next):
                curr.merge(next)
                del ps[j]
            else:
                j += 1

    def __str__(self):
        return f"({self.compartments},{self.dest},{self.timer})"


class Location:
    @classmethod
    def initialize(cls, index: int, initial_pop: Compartments):
        pops = [Population(initial_pop, index, Timer.Home)]
        return Location(index, pops)

    def __init__(self, index: int, pops: list[Population]):
        Population.normalize(pops)
        self.index = index
        self.pops = pops

    def locals(self) -> Population:
        # NOTE: this only works if `pops` is normalized after modification
        return self.pops[0]

    @property
    def compartment_totals(self) -> Compartments:
        return np.sum([p.compartments for p in self.pops], axis=0)

    def __eq__(self, other):
        return (isinstance(other, Location) and
                self.index == other.index and
                self.pops == other.pops)

    def __str__(self):
        return ", ".join(map(str, self.pops))


class World:
    @classmethod
    def initialize(cls, initial_pops: list[Compartments]):
        locations = [Location.initialize(i, cs)
                     for (i, cs) in enumerate(initial_pops)]
        return World(locations)

    def __init__(self, locations: list[Location]):
        self.locations = locations

    def normalize(self) -> None:
        for loc in self.locations:
            Population.normalize(loc.pops)

    def __eq__(self, other):
        return (isinstance(other, World) and
                self.locations == other.locations)

    def __str__(self):
        xs = [f"L{i}: {str(x)}" for i, x in enumerate(self.locations)]
        return "\n".join(xs)
