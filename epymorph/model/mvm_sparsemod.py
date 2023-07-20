from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import epymorph.movement as M
import epymorph.movement_clause as C
from epymorph.clock import Tick, TickDelta
from epymorph.context import SimContext


def sparse_movement(ctx: SimContext) -> C.RowEquation:
    """Sparsemod movement model"""
    commuters = ctx.geo['commuters']
    distances = ctx.geo['distances']
    n = ctx.nodes
    dispersal_kernel = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dispersal_kernel[i, j] = 1 / \
                (np.exp(distances[i, j]*1/ctx.param['phi']))
        dispersal_kernel[i, ] = dispersal_kernel[i, ] / \
            sum(dispersal_kernel[i, ])

    def equation(tick: Tick, src_idx: int) -> NDArray[np.int_]:
        return ctx.rng.multinomial(commuters[src_idx], dispersal_kernel[src_idx, :])
    return equation


def load_mvm():
    def if_not_immobile(tags: list[str]) -> bool:
        return 'immobile' not in tags

    return M.MovementBuilder(
        taus=[np.double(2/3), np.double(1/3)],
        clause_compiler=lambda ctx: C.Sequence([
            C.GeneralClause.by_row(
                ctx=ctx,
                name="Commuters",
                predicate=C.Predicates.everyday(step=0),
                returns=TickDelta(0, 1),
                equation=sparse_movement(ctx),
                compartment_tag_predicate=if_not_immobile
            ),
            C.Return(ctx)
        ])
    )
