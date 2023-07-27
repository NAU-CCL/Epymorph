from __future__ import annotations

from dataclasses import dataclass
from operator import attrgetter
from typing import cast

import numpy as np
from numpy.typing import NDArray

from epymorph.context import SimContext
from epymorph.util import Compartments

HOME_TICK = -1
"""The value of a population's `return_tick` when the population is home."""


@dataclass
class Population:
    """
    Represents a group of individuals, divided into IPM compartments as appropriate for the simulation.
    These individuals share the same "home location" and a time at which they should return there.

    These are somewhat abstract concepts, however: a completely nomadic group doesn't really have a home location, merely the next
    location in a chain of movements.
    """

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

    compartments: Compartments
    return_location: int
    return_tick: int

    # Note: when a population is "home",
    # its `return_location` is the same as their current location,
    # and its `return_tick` is set to -1 (the "Never" placeholder value).

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
              requested: int,
              movement_mask: NDArray[np.bool_]) -> tuple[int, Population]:
        """
        Divide a population by splitting off (up to) the `requested` number of individuals.
        These individuals will be randomly drawn from the available compartments.
        The source population (self) will be modified in place. A new Population object will
        be returned as the second element of a tuple; the first element being the actual number
        of individuals (in case you requested more than the number available).
        """
        # How many people are available to move?
        available = self.compartments * movement_mask
        # How many will actually move?
        actual = min(sum(available), requested)
        # Select movers.
        movers = sim.rng.multivariate_hypergeometric(available, actual)
        self.compartments -= movers
        return_location = dst_idx if return_tick == HOME_TICK else src_idx
        pop = Population(movers, return_location, return_tick)
        return (actual, pop)


@dataclass
class Location:

    @classmethod
    def initialize(cls, index: int, initial_pop: Compartments):
        pops = [Population(initial_pop, index, HOME_TICK)]
        return cls(index, pops)

    index: int
    pops: list[Population]

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

    def __str__(self):
        return ", ".join(map(str, self.pops))


@dataclass
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

    def all_locals(self) -> NDArray[np.int_]:
        return np.array([loc.locals.compartments
                        for loc in self.locations], dtype=int)

    def normalize(self) -> None:
        for loc in self.locations:
            Population.normalize(loc.pops)

    def __str__(self):
        xs = [f"L{i}: {str(x)}" for i, x in enumerate(self.locations)]
        return "\n".join(xs)
