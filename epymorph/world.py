from __future__ import annotations

from operator import attrgetter

import numpy as np

from epymorph.context import SimContext
from epymorph.util import Compartments


class Timer:
    Home = -1


# Note: when a population is "home", its `dest` is set to the home location index, and its `timer` is set to -1.
class Population:
    @classmethod
    def can_merge(cls, a: Population, b: Population) -> bool:
        return a.timer == b.timer and a.dest == b.dest

    @classmethod
    def normalize(cls, ps: list[Population]) -> None:
        """
        Sorts a list of populations and combines equivalent populations.
        Lists of populations should be normalized after creation and any modification.
        """
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

    def __init__(self, compartments: Compartments, dest: int, timer: int):
        self.compartments = compartments
        self.dest = dest
        self.timer = timer

    @property
    def total(self) -> np.int_:
        return np.sum(self.compartments)

    def merge(self, other: Population) -> None:
        self.compartments += other.compartments

    def split(self,
              sim: SimContext,
              src_idx: int,
              dst_idx: int,
              return_tick: int,
              requested: np.int_) -> tuple[np.int_, Population]:
        actual = np.minimum(self.total, requested)
        cs_probability = self.compartments / self.total
        cs = sim.rng.multinomial(actual, cs_probability)
        self.compartments -= cs
        dest = dst_idx if return_tick == Timer.Home else src_idx
        pop = Population(cs, dest, return_tick)
        return (actual, pop)

    def __eq__(self, other):
        return (isinstance(other, Population) and
                self.compartments == other.compartments and
                self.dest == other.dest and
                self.timer == other.timer)

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

    @property
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
    """
    World tracks the state of a simulation's populations and subpopulations at a particular timeslice.
    Each node in the Geo Model is represented as a Location; each Location has a list of Populations
    that are considered to be well-mixed at that location; and each Population is divided into
    Compartments as defined by the Intra-Population Model.
    """
    @classmethod
    def initialize(cls, initial_pops: list[Compartments]):
        locations = [Location.initialize(i, cs)
                     for (i, cs) in enumerate(initial_pops)]
        return World(locations)

    locations: list[Location]
    size: int

    def __init__(self, locations: list[Location]):
        self.locations = locations
        self.size = len(locations)

    def normalize(self) -> None:
        for loc in self.locations:
            Population.normalize(loc.pops)

    def __eq__(self, other):
        return (isinstance(other, World) and
                self.locations == other.locations)

    def __str__(self):
        xs = [f"L{i}: {str(x)}" for i, x in enumerate(self.locations)]
        return "\n".join(xs)
