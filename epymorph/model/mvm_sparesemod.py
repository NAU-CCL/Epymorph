from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import epymorph.movement as M
import epymorph.movement_clause as C
from epymorph.clock import Tick, TickDelta
from epymorph.context import SimContext
from epymorph.util import is_square


def sparse_movement(ctx: SimContext) -> C.RowEquation:
    """Sparsemod movement model"""
    commuters = ctx.geo['commuters']
    dispersal_kernel = ctx.geo['dispersal_kernel']
    assert is_square(dispersal_kernel)

    def equation(tick: Tick, src_idx: int) -> NDArray[np.int_]:
        return ctx.rng.multinomial(commuters[src_idx], dispersal_kernel[src_idx, :])
    return equation


def load_mvm():
    return M.MovementBuilder(
        taus=[np.double(2/3), np.double(1/3)],
        clause_compiler=lambda ctx: C.Sequence([
            C.GeneralClause.by_row(
                ctx=ctx,
                name="Commuters",
                predicate=C.Predicates.everyday(step=0),
                returns=TickDelta(0, 1),
                equation=sparse_movement(ctx)
            ),
            C.Return(ctx)
        ])
    )
