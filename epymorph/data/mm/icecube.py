import numpy as np
from numpy.typing import NDArray

from epymorph.data import registry
from epymorph.data_shape import Shapes
from epymorph.data_type import SimDType
from epymorph.movement_model import EveryDay, MovementClause, MovementModel
from epymorph.simulation import AttributeDef, Tick, TickDelta, TickIndex


class IcecubeClause(MovementClause):
    """The clause of the icecube model."""
    requirements = (
        AttributeDef('population', int, Shapes.N,
                     comment="The total population at each node."),
        AttributeDef('commuter_proportion', float, Shapes.S, default_value=0.1,
                     comment="Decides what proportion of the total population should be commuting normally.")
    )

    predicate = EveryDay()
    leaves = TickIndex(step=0)
    returns = TickDelta(step=1, days=0)

    def evaluate(self, tick: Tick) -> NDArray[np.int64]:
        N = self.dim.nodes
        pop = self.data('population')
        comm_prop = self.data('commuter_proportion')
        commuters = np.zeros((N, N), dtype=SimDType)
        for src in range(N):
            if (src + 1) < N:
                commuters[src, src + 1] = pop[src] * comm_prop
        return commuters


@registry.mm('icecube')
class Icecube(MovementModel):
    """
    A toy example: ice cube tray movement movement model
    Each state sends a fixed number of commuters to the next
    state in the line (without wraparound).
    """
    steps = (1 / 2, 1 / 2)
    clauses = (IcecubeClause(),)
