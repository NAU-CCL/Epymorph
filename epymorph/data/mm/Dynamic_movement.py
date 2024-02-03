from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from epymorph.data import registry
from epymorph.data_shape import Shapes
from epymorph.data_type import CentroidType, SimDType
from epymorph.movement_model import EveryDay, MovementClause, MovementModel
from epymorph.simulation import AttributeDef, Tick, TickDelta, TickIndex
from epymorph.util import pairwise_haversine, row_normalize


class DynamicClause(MovementClause):
    """The clause of the centroids model."""
    requirements = (
        AttributeDef('population', int, Shapes.N,
                     comment="The total population at each node."),
        AttributeDef('centroid', CentroidType, Shapes.N,
                     comment="The centroids for each node as (longitude, latitude) tuples."),
        AttributeDef('phi', float, Shapes.S, default_value=40.0,
                     comment="Influences the distance that movers tend to travel."),
        AttributeDef('commuter_proportion', float, Shapes.S, default_value=0.1,
                     comment="Decides what proportion of the total population should be commuting normally.")
    )

    predicate = EveryDay()
    leaves = TickIndex(step=0)
    returns = TickDelta(step=1, days=0)

    @cached_property
    def dispersal_kernel(self) -> NDArray[np.float64]:
        """
        The NxN matrix or dispersal kernel describing the tendency for movers to move to a particular location.
        In this model, the kernel is:
            1 / e ^ (distance / phi)
        which is then row-normalized.
        """
        centroid = self.data('centroid')
        phi = self.data('phi')
        distance = pairwise_haversine(centroid['longitude'], centroid['latitude'])
        return row_normalize(1 / np.exp(distance / phi))

    def evaluate(self, tick: Tick) -> NDArray[np.int64]:
        pop = self.data('population')
        comm_prop = self.data('commuter_proportion')
        n_commuters = np.floor(pop * comm_prop).astype(SimDType)
        return self.rng.multinomial(n_commuters, self.dispersal_kernel)

    
@registry.mm('dynamic')
class Dynamic(MovementModel):
    """
    The dynamic MM descibes the movement patterns of individual using celluar ping location 
    provide by Data for Good who's parent company is META. The goal of this movement model is
    to simulate movement patterns of individual on a day to day basis with provided information. 
    These individual travel to another location for 1/3 of the day, and the return home for the 
    remaining 2/3 of the day.

    """
    steps = (1 / 3, 2 / 3)
    clauses = (DynamicClause(),)
