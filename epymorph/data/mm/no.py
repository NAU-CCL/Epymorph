import numpy as np
from numpy.typing import NDArray

from epymorph.data_type import SimDType
from epymorph.movement_model import EveryDay, MovementClause, MovementModel
from epymorph.simulation import Tick, TickDelta, TickIndex


class NoClause(MovementClause):
    """The clause of the "no" model."""

    requirements = ()
    predicate = EveryDay()
    leaves = TickIndex(step=0)
    returns = TickDelta(step=0, days=0)

    def evaluate(self, tick: Tick) -> NDArray[np.int64]:
        N = self.scope.nodes
        return np.zeros((N, N), dtype=SimDType)


class No(MovementModel):
    """
    No movement at all. This is handy for cases when you want to disable movement
    in an experiment, or for testing.
    """

    steps = (1.0,)
    clauses = (NoClause(),)
