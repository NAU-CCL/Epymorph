from __future__ import annotations

from operator import attrgetter

import numpy as np

from epymorph.context import SimContext
from epymorph.util import Compartments

_home_tick = -1
"""The value of a population's `return_tick` when the population is home."""


class Population:
    """
    Represents a group of individuals, divided into IPM compartments as appropriate for the simulation.
    These individuals share the same "home location" and a time at which they should return there.

    These are somewhat abstract concepts, however: a completely nomadic group doesn't really have a home location, merely the next
    location in a chain of movements.
    """
    compartments: Compartments
    return_location: int
    return_tick: int

    # Note: when a population is "home",
    # its `return_location` is the same as their current location,
    # and its `return_tick` is set to -1 (the "Never" placeholder value).

    @staticmethod
    def can_merge(a: Population, b: Population) -> bool:
        return a.return_tick == b.return_tick and a.return_location == b.return_location

    @staticmethod
    def normalize(ps: list[Population]) -> None:
        """
        Sorts a list of populations and combines mergeable populations (in place).
        Lists of populations should be normalized after creation and any modification.
        """
        ps.sort(key=attrgetter('return_tick', 'return_location'))
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

    def __init__(self, compartments: Compartments, return_location: int, return_tick: int):
        self.compartments = compartments
        self.return_location = return_location
        self.return_tick = return_tick

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
        """
        Divide a population by splitting off (up to) the `requested` number of individuals.
        These individuals will be randomly drawn from the available compartments.
        The source population (self) will be modified in place. A new Population object will
        be returned as the second element of a tuple; the first element being the actual number
        of individuals (in case you requested more than the number available).
        """
        actual = np.minimum(self.total, requested)
        cs_probability = self.compartments / self.total
        cs = sim.rng.multinomial(actual, cs_probability)
        self.compartments -= cs
        return_location = dst_idx if return_tick == _home_tick else src_idx
        pop = Population(cs, return_location, return_tick)
        return (actual, pop)

    def __eq__(self, other):
        return (isinstance(other, Population) and
                self.compartments == other.compartments and
                self.return_location == other.return_location and
                self.return_tick == other.return_tick)

    def __str__(self):
        return f"({self.compartments},{self.return_location},{self.return_tick})"


class Location:
    @classmethod
    def initialize(cls, index: int, initial_pop: Compartments):
        pops = [Population(initial_pop, index, _home_tick)]
        return cls(index, pops)

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
        return cls(locations)

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
