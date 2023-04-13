from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import epymorph.movement as M
from epymorph.clock import Tick
from epymorph.sim_context import SimContext
from epymorph.util import is_square


def commuter_movement(commuters: NDArray[np.int_], move_control: float) -> M.RowEquation:
    """Pei-style normal commuters."""
    assert 0 <= move_control <= 1.0, "Move Control must be in the range [0,1]."
    assert is_square(commuters), "Commuters matrix must be square."

    # Total commuters living in each state.
    commuters_by_state = commuters.sum(axis=1, dtype=np.int_)
    # Commuters as a ratio to the total commuters living in that state.
    commuting_prob = commuters / \
        commuters.sum(axis=1, keepdims=True, dtype=np.int_)

    def equation(sim: SimContext, tick: Tick, src_idx: int) -> NDArray[np.int_]:
        # Binomial draw with probability `move_control` to modulate total number of commuters.
        typical = commuters_by_state[src_idx]
        actual = sim.rng.binomial(typical, move_control)
        # Multinomial draw for destination.
        return sim.rng.multinomial(actual, commuting_prob[src_idx])

    return equation
