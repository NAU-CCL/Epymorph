from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np

from epymorph.context import SimContext
from epymorph.movement_clause import Clause


class MovementBuilder:
    def __init__(self, taus: list[np.double], clause_compiler: Callable[[SimContext], Clause]):
        assert len(taus) > 0, "Must supply at least one tau step."
        assert np.sum(taus) == np.double(1), "Tau steps must sum to 1."
        self.taus = taus
        self.clause_compiler = clause_compiler

    def verify(self, ctx: SimContext) -> None:
        # TODO: how do we verify the context for movement?
        pass

    def build(self, ctx: SimContext) -> Movement:
        return Movement(self.taus, self.clause_compiler(ctx))


class Movement(NamedTuple):
    taus: list[np.double]
    clause: Clause
