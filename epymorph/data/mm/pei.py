from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from epymorph.attribute import AttributeDef
from epymorph.data_shape import Shapes
from epymorph.data_type import SimDType
from epymorph.movement_model import EveryDay, MovementClause, MovementModel
from epymorph.simulation import Tick, TickDelta, TickIndex
from epymorph.util import row_normalize

_COMMUTERS_ATTRIB = AttributeDef(
    "commuters", int, Shapes.NxN, comment="A node-to-node commuters marix."
)


class Commuters(MovementClause):
    """The commuter clause of the pei model."""

    requirements = (
        _COMMUTERS_ATTRIB,
        AttributeDef(
            "move_control",
            float,
            Shapes.Scalar,
            default_value=0.9,
            comment=(
                "A factor which modulates the number of commuters "
                "by conducting a binomial draw with this probability "
                "and the expected commuters from the commuters matrix."
            ),
        ),
    )

    predicate = EveryDay()
    leaves = TickIndex(step=0)
    returns = TickDelta(step=1, days=0)

    @cached_property
    def commuters_by_node(self) -> NDArray[SimDType]:
        """Total commuters living in each state."""
        commuters = self.data("commuters")
        return np.sum(commuters, axis=1)

    @cached_property
    def commuting_probability(self) -> NDArray[np.float64]:
        """
        Commuters as a ratio to the total commuters living in that state.
        For cases where there are no commuters, avoid div-by-0 errors
        """
        commuters = self.data("commuters")
        return row_normalize(commuters)

    def evaluate(self, tick: Tick) -> NDArray[np.int64]:
        move_control = self.data("move_control")
        actual = self.rng.binomial(self.commuters_by_node, move_control)
        return self.rng.multinomial(actual, self.commuting_probability)


class Dispersers(MovementClause):
    """The dispersers clause of the pei model."""

    requirements = (
        _COMMUTERS_ATTRIB,
        AttributeDef(
            "theta",
            float,
            Shapes.Scalar,
            default_value=0.1,
            comment=(
                "A factor which allows for randomized movement "
                "by conducting a poisson draw with this factor "
                "times the average number of commuters between two nodes "
                "from the commuters matrix."
            ),
        ),
    )

    predicate = EveryDay()
    leaves = TickIndex(step=0)
    returns = TickDelta(step=1, days=0)

    @cached_property
    def commuters_average(self) -> NDArray[SimDType]:
        """Average commuters between locations."""
        commuters = self.data("commuters")
        return (commuters + commuters.T) // 2

    def evaluate(self, tick: Tick) -> NDArray[SimDType]:
        theta = self.data("theta")
        return self.rng.poisson(theta * self.commuters_average)


class Pei(MovementModel):
    """
    Modeled after the Pei influenza paper, this model simulates
    two types of movers -- regular commuters and more-randomized dispersers.
    Each is somewhat stochastic but adhere to the general shape dictated
    by the commuters array. Both kinds of movement can be "tuned" through
    their respective parameters: move_control and theta.
    """

    steps = (1 / 3, 2 / 3)
    clauses = (Commuters(), Dispersers())
