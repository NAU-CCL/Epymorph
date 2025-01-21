from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from epymorph.attribute import AttributeDef
from epymorph.data_shape import Shapes
from epymorph.data_type import CentroidType, SimDType
from epymorph.movement_model import EveryDay, MovementClause, MovementModel
from epymorph.simulation import Tick, TickDelta, TickIndex
from epymorph.util import pairwise_haversine, row_normalize


class CentroidsClause(MovementClause):
    """The clause of the centroids model."""

    requirements = (
        AttributeDef(
            "population", int, Shapes.N, comment="The total population at each node."
        ),
        AttributeDef(
            "centroid",
            CentroidType,
            Shapes.N,
            comment="The centroids for each node as (longitude, latitude) tuples.",
        ),
        AttributeDef(
            "phi",
            float,
            Shapes.Scalar,
            default_value=40.0,
            comment="Influences the distance that movers tend to travel.",
        ),
        AttributeDef(
            "commuter_proportion",
            float,
            Shapes.Scalar,
            default_value=0.1,
            comment="The proportion of the total population which commutes.",
        ),
    )

    predicate = EveryDay()
    leaves = TickIndex(step=0)
    returns = TickDelta(step=1, days=0)

    @cached_property
    def dispersal_kernel(self) -> NDArray[np.float64]:
        """
        The NxN matrix or dispersal kernel describing the tendency for movers to move
        to a particular location. In this model, the kernel is:
            1 / e ^ (distance / phi)
        which is then row-normalized.
        """
        centroid = self.data("centroid")
        phi = self.data("phi")
        distance = pairwise_haversine(centroid)
        return row_normalize(1 / np.exp(distance / phi))

    def evaluate(self, tick: Tick) -> NDArray[np.int64]:
        pop = self.data("population")
        comm_prop = self.data("commuter_proportion")
        n_commuters = np.floor(pop * comm_prop).astype(SimDType)
        return self.rng.multinomial(n_commuters, self.dispersal_kernel)


class Centroids(MovementModel):
    """
    The centroids MM describes a basic commuter movement where a fixed proportion
    of the population commutes every day, travels to another location for 1/3 of a day
    (with a location likelihood that decreases with distance), and then returns home for
    the remaining 2/3 of the day.
    """

    steps = (1 / 3, 2 / 3)
    clauses = (CentroidsClause(),)
