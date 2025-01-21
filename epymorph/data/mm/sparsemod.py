from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from epymorph.attribute import AttributeDef
from epymorph.data_shape import Shapes
from epymorph.data_type import CentroidType, SimDType
from epymorph.movement_model import EveryDay, MovementClause, MovementModel
from epymorph.simulation import Tick, TickDelta, TickIndex
from epymorph.util import pairwise_haversine, row_normalize


class SparsemodClause(MovementClause):
    """The clause of the sparsemod model."""

    requirements = (
        AttributeDef(
            "commuters", int, Shapes.NxN, comment="A node-to-node commuters marix."
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
    )

    predicate = EveryDay()
    leaves = TickIndex(step=0)
    returns = TickDelta(step=1, days=0)

    @cached_property
    def commuters_by_node(self) -> NDArray[SimDType]:
        """
        The number of commuters that live in any particular node
        (regardless of typical commuting destination).
        """
        return np.sum(self.data("commuters"), axis=1)

    @cached_property
    def dispersal_kernel(self) -> NDArray[np.float64]:
        """
        The NxN matrix or dispersal kernel describing the tendency for movers
        to move to a particular location. In this model, the kernel is:
            1 / e ^ (distance / phi)
        which is then row-normalized.
        """
        centroid = self.data("centroid")
        phi = self.data("phi")
        distance = pairwise_haversine(centroid)
        return row_normalize(1 / np.exp(distance / phi))

    def evaluate(self, tick: Tick) -> NDArray[np.int64]:
        return self.rng.multinomial(self.commuters_by_node, self.dispersal_kernel)


class Sparsemod(MovementModel):
    """
    Modeled after the SPARSEMOD COVID-19 paper, this model simulates
    movement using a distance kernel parameterized by phi, and using a commuters
    matrix to determine the total expected number of commuters.
    """

    steps = (1 / 3, 2 / 3)
    clauses = (SparsemodClause(),)
