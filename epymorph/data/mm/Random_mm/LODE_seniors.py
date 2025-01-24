from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from epymorph.data import registry
from epymorph.data_shape import Shapes
from epymorph.data_type import SimDType
from epymorph.movement_model import EveryDay, MovementClause, MovementModel
from epymorph.simulation import AttributeDef, Tick, TickDelta, TickIndex
from epymorph.util import row_normalize

_COMMUTERS_ATTRIB = AttributeDef('LODES_commuters', int, Shapes.NxN,
                                 comment="A node-to-node LODES 8 commuters matrix.")


class Commuters(MovementClause):
    """The commuter clause of the LODES 8 model."""

    requirements = (
        _COMMUTERS_ATTRIB,
        AttributeDef('move_control', float, Shapes.S, default_value=0.9,
                     comment="A factor which modulates the number of commuters by conducting a binomial draw "
                     "with this probability and the expected commuters from the commuters matrix."),
    )

    predicate = EveryDay()
    leaves = TickIndex(step=0)
    returns = TickDelta(step=1, days=0)

    @cached_property
    def commuters_by_node(self) -> NDArray[SimDType]:
        """Total workers living in each node."""
        commuters = self.data(_COMMUTERS_ATTRIB)
        return np.sum(commuters, axis=1)

    @cached_property
    def commuting_probability(self) -> NDArray[np.float64]:
        """
        Commuters as a ratio to the total commuters living in that state.
        For cases where there are no commuters, avoid div-by-0 errors
        """
        commuters = self.data(_COMMUTERS_ATTRIB)
        return row_normalize(commuters)

    def evaluate(self, tick: Tick) -> NDArray[np.int64]:
        move_control = self.data('move_control')
        actual = self.rng.binomial(self.commuters_by_node, move_control)
        return self.rng.multinomial(actual, self.commuting_probability)


class Dispersers(MovementClause):
    """The dispersers clause of the LODES model."""

    requirements = (
        _COMMUTERS_ATTRIB,
        AttributeDef('theta', float, Shapes.S, default_value=0.1,
                     comment="A factor which allows for randomized movement by conducting a poisson draw "
                     "with this factor times the average number of commuters between two nodes "
                     "from the commuters matrix."),
    )

    predicate = EveryDay()
    leaves = TickIndex(step=0)
    returns = TickDelta(step=1, days=0)

    @cached_property
    def commuters_average(self) -> NDArray[SimDType]:
        """Average workers between locations."""
        commuters = self.data(_COMMUTERS_ATTRIB)
        return (commuters + commuters.T) // 2

    def evaluate(self, tick: Tick) -> NDArray[SimDType]:
        theta = self.data('theta')
        return self.rng.poisson(theta * self.commuters_average)


@registry.mm('LODES')
class LODES(MovementModel):
    """
    The LODES MM desribes the movement patterns of workers on the census block level. 
    The data was provide by the ASC5 survey that provide us with information on 
    where people travel to work from their home census block group. This movement model
    was designed to reflect a typical work day where an individual would work for 8 hours.
    In order to accomplish this the individuals travel to another location for 1/3 of the day 
    and return home for the remaining 2/3 of the day. 
    """
    steps = (1/3, 1/6, 1/2) # <- this will change to reflect an 8 hour work day
    # Only designed the movement model because we were on a single model 
    # Now each strada can have their own movement model without having the same tau tick system
    clauses = (Commuters(), Dispersers())
