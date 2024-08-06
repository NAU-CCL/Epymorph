from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from epymorph.data_shape import SimDimensions
from epymorph.data_type import AttributeArray, SimDType
from epymorph.database import (Database, DatabaseWithFallback, ModuleNamespace,
                               NamePattern)
from epymorph.error import MmCompileException
from epymorph.event import (MovementEventsMixin, OnMovementClause,
                            OnMovementFinish, OnMovementStart)
from epymorph.movement.compile import compile_spec
from epymorph.movement.movement_model import (MovementContext, MovementModel,
                                              TravelClause)
from epymorph.rume import Rume
from epymorph.simulation import (NamespacedAttributeResolver, Tick, gpm_strata,
                                 resolve_tick_delta)
from epymorph.simulator.world import World
from epymorph.util import row_normalize


def calculate_travelers(
    # General movement model info.
    ctx: MovementContext,
    # Clause info.
    clause: TravelClause,
    clause_mobility: NDArray[np.bool_],
    tick: Tick,
    local_cohorts: NDArray[SimDType],
) -> OnMovementClause:
    """
    Calculate the number of travelers resulting from this movement clause for this tick.
    This evaluates the requested number movers, modulates that based on the available movers,
    then selects exactly which individuals (by compartment) should move.
    Returns an (N,N,C) array; from-source-to-destination-by-compartment.
    """
    _, N, C, _ = ctx.dim.TNCE

    clause_movers = clause.requested(ctx, tick)
    np.fill_diagonal(clause_movers, 0)
    clause_sum = clause_movers.sum(axis=1, dtype=SimDType)

    available_movers = local_cohorts * clause_mobility
    available_sum = available_movers.sum(axis=1, dtype=SimDType)

    # If clause requested total is greater than the total available,
    # use mvhg to select as many as possible.
    if not np.any(clause_sum > available_sum):
        throttled = False
        requested_movers = clause_movers
        requested_sum = clause_sum
    else:
        throttled = True
        requested_movers = clause_movers.copy()
        for src in range(N):
            if clause_sum[src] > available_sum[src]:
                requested_movers[src, :] = ctx.rng.multivariate_hypergeometric(
                    colors=requested_movers[src, :],
                    nsample=available_sum[src]
                )
        requested_sum = requested_movers.sum(axis=1, dtype=SimDType)

    # The probability a mover from a src will go to a dst.
    requested_prb = row_normalize(requested_movers, requested_sum, dtype=SimDType)

    travelers_cs = np.zeros((N, N, C), dtype=SimDType)
    for src in range(N):
        if requested_sum[src] == 0:
            continue

        # Select which individuals will be leaving this node.
        mover_cs = ctx.rng.multivariate_hypergeometric(
            available_movers[src, :],
            requested_sum[src]
        ).astype(SimDType)

        # Select which location they are each going to.
        # (Each row contains the compartments for a destination.)
        travelers_cs[src, :, :] = ctx.rng.multinomial(
            mover_cs,
            requested_prb[src, :]
        ).T.astype(SimDType)

    return OnMovementClause(
        tick.sim_index,
        tick.day,
        tick.step,
        clause.name,
        clause_movers,
        travelers_cs,
        requested_sum.sum(),
        throttled,
    )


class _Ctx(NamedTuple):
    dim: SimDimensions
    rng: np.random.Generator
    data: NamespacedAttributeResolver


class _StrataInfo(NamedTuple):
    model: MovementModel
    mobility: NDArray[np.bool_]
    ctx: _Ctx


class MovementExecutor:
    """Movement model execution specifically for multi-strata simulations."""

    _rume: Rume
    """the RUME"""
    _world: World
    """the world state"""
    _rng: np.random.Generator
    """the simulation RNG"""
    _event_target: MovementEventsMixin

    _data: Database[AttributeArray]
    _strata: dict[str, _StrataInfo]

    def __init__(
        self,
        rume: Rume,
        world: World,
        db: Database[AttributeArray],
        rng: np.random.Generator,
        event_target: MovementEventsMixin,
    ):
        # Introduce a new data layer so we have a place to store predefs
        data = DatabaseWithFallback({}, db)

        self._rume = rume
        self._world = world
        self._data = data
        self._rng = rng
        self._event_target = event_target
        self._strata = {
            strata: _StrataInfo(
                # Compile movement model
                model=compile_spec(mm, rng),
                # Get compartment mobility for this strata
                mobility=rume.compartment_mobility(strata),
                # Assemble a context with a resolver for this strata
                ctx=_Ctx(
                    dim=rume.dim,
                    rng=rng,
                    data=NamespacedAttributeResolver(
                        data=data,
                        dim=rume.dim,
                        namespace=ModuleNamespace(gpm_strata(strata), "mm"),
                    ),
                ),
            )
            for strata, mm in rume.mms.items()
        }
        self._compute_predefs()

    def _compute_predefs(self) -> None:
        """Compute predefs and store results to our database."""
        for strata, (model, _, ctx) in self._strata.items():
            result = model.predef(ctx)
            if not isinstance(result, dict):
                msg = f"Movement predef: did not return a dictionary result (got: {type(result)})"
                raise MmCompileException(msg)
            for key, value in result.items():
                if not isinstance(value, np.ndarray):
                    msg = f"Movement predef: key '{key}' invalid; it is not a numpy array."
                pattern = NamePattern(gpm_strata(strata), "mm", key)
                self._data.update(pattern, value.copy())

    def apply(self, tick: Tick) -> None:
        """Applies movement for this tick, mutating the world state."""

        self._event_target.on_movement_start.publish(
            OnMovementStart(tick.sim_index, tick.day, tick.step))

        # Process travel clauses.
        total = 0
        for model, mobility, ctx in self._strata.values():
            for clause in model.clauses:
                if not clause.predicate(ctx, tick):
                    continue
                local_array = self._world.get_local_array()

                clause_event = calculate_travelers(
                    ctx, clause, mobility, tick, local_array)
                self._event_target.on_movement_clause.publish(clause_event)
                travelers = clause_event.actual

                returns = clause.returns(ctx, tick)
                return_tick = resolve_tick_delta(ctx.dim, tick, returns)
                self._world.apply_travel(travelers, return_tick)
                total += travelers.sum()

        # Process return clause.
        return_movers = self._world.apply_return(tick, return_stats=True)
        return_total = return_movers.sum()
        total += return_total

        self._event_target.on_movement_clause.publish(
            OnMovementClause(
                tick.sim_index,
                tick.day,
                tick.step,
                "return",
                return_movers,
                return_movers,
                return_total,
                False,
            )
        )

        self._event_target.on_movement_finish.publish(
            OnMovementFinish(tick.sim_index, tick.day, tick.step, total))
