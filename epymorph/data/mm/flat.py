from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from epymorph.attribute import AttributeDef
from epymorph.data_shape import Shapes
from epymorph.data_type import SimDType
from epymorph.movement_model import EveryDay, MovementClause, MovementModel
from epymorph.simulation import Tick, TickDelta, TickIndex
from epymorph.util import row_normalize


class FlatClause(MovementClause):
    """The clause of the flat model."""

    requirements = (
        AttributeDef(
            "population", int, Shapes.N, comment="The total population at each node."
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
        to a particular location. In this model, the kernel is full of 1s
        except for 0s on the diagonal, which is then row-normalized.
        Effectively: every destination is equally likely.
        """
        ones = np.ones((self.scope.nodes, self.scope.nodes))
        np.fill_diagonal(ones, 0)
        return row_normalize(ones)

    def evaluate(self, tick: Tick) -> NDArray[SimDType]:
        pop = self.data("population")
        comm_prop = self.data("commuter_proportion")
        n_commuters = np.floor(pop * comm_prop).astype(SimDType)
        return self.rng.multinomial(n_commuters, self.dispersal_kernel)


class Flat(MovementModel):
    """
    This model evenly weights the probability of movement to all other nodes.
    It uses parameter 'commuter_proportion' to determine how many people should
    be moving, based on the total normal population of each node.
    """

    steps = (1 / 3, 2 / 3)
    clauses = (FlatClause(),)
