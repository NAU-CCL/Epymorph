import numpy as np

import epymorph.movement as M
import epymorph.movement_clause as C
from epymorph.clock import TickDelta
from epymorph.model.mvm_commuter import commuter_movement
from epymorph.model.mvm_disperser import disperser_movement


# Movement Model
# (Roughly equivalent to...)
# [move-steps: per-day=2; duration=[2/3, 1/3]]
# [mtype: daily; days=[m,t,w,th,f,st,sn]; leave-step=1; return=0d.step2; <equation for commuters>]
# [mtype: daily; days=[m,t,w,th,f,st,sn]; leave-step=1; return=0d.step2; <equation for dispersers>]
def load_mvm():
    return M.MovementBuilder(
        # First step is day: 2/3 tau
        # Second step is night: 1/3 tau
        taus=[np.double(2/3), np.double(1/3)],
        clause_compiler=lambda ctx: C.Sequence([
            # Main commuters: on step 0
            C.GeneralClause.by_row(
                ctx=ctx,
                name="Commuters",
                predicate=C.Predicates.everyday(step=0),
                returns=TickDelta(0, 1),  # returns today on step 1
                equation=commuter_movement(ctx)
            ),
            # Random dispersers: also on step 0, cumulative effect.
            C.GeneralClause.by_row(
                ctx=ctx,
                name="Dispersers",
                predicate=C.Predicates.everyday(step=0),
                returns=TickDelta(0, 1),  # returns today on step 1
                equation=disperser_movement(ctx)
            ),
            # Return: always triggers, but only moves pops whose return time is now.
            C.Return(ctx)
        ])
    )
