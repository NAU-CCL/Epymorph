from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import movement as M
from clock import Tick
from sim_context import SimContext
from util import is_square


def disperser_movement(commuters: NDArray[np.int_], theta: float) -> M.RowEquation:
    """Pei-style random dispersers."""
    assert 0 <= theta, "Theta must be not less than zero."
    assert is_square(commuters), "Commuters matrix must be square."

    # Pre-compute the average commuters between node pairs.
    commuters_avg = np.zeros(commuters.shape)
    for i in range(commuters.shape[0]):
        for j in range(i + 1, commuters.shape[1]):
            nbar = (commuters[i, j] + commuters[j, i]) // 2
            commuters_avg[i, j] = nbar
            commuters_avg[j, i] = nbar

    def equation(sim: SimContext, tick: Tick, src_idx: int) -> NDArray[np.int_]:
        return sim.rng.poisson(commuters_avg[src_idx] * theta)

    return equation
