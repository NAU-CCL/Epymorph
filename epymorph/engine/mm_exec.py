"""
Movement executor classes handle the logic for processing the movement step of the simulation.
"""
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from epymorph.data_type import SimDType
from epymorph.engine.world import World
from epymorph.error import AttributeException, MmCompileException
from epymorph.event import (MovementEventsMixin, OnMovementClause,
                            OnMovementFinish, OnMovementStart)
from epymorph.movement.compile import compile_spec
from epymorph.movement.movement_model import (MovementContext, MovementModel,
                                              PredefData, TravelClause)
from epymorph.movement.parser import MovementSpec
from epymorph.simulation import Tick, resolve_tick
from epymorph.util import row_normalize


class MovementExecutor(ABC):
    """
    Abstract interface responsible for advancing the simulation state due to the MM.
    """

    @abstractmethod
    def apply(self, world: World, tick: Tick) -> None:
        """
        Applies movement for this tick, mutating the world state.
        """


def calculate_travelers(
    # General movement model info.
    ctx: MovementContext,
    predef: PredefData,
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

    clause_movers = clause.requested(ctx, predef, tick)
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
        tick.index,
        tick.day,
        tick.step,
        clause.name,
        clause_movers,
        travelers_cs,
        requested_sum.sum(),
        throttled,
    )


############################################################
# StandardMovementExecutor
############################################################


class StandardMovementExecutor(MovementEventsMixin, MovementExecutor):
    """The standard implementation of movement model execution."""

    _ctx: MovementContext
    _model: MovementModel
    _clause_masks: dict[TravelClause, NDArray[np.bool_]]
    _predef: PredefData = {}
    _predef_hash: int | None = None

    def __init__(
        self,
        ctx: MovementContext,
        mm: MovementSpec,
    ):
        MovementEventsMixin.__init__(self)
        self._model = compile_spec(mm, ctx.rng)
        self._predef = {}
        self._predef_hash = None
        self._ctx = ctx
        self._clause_masks = {c: c.mask(ctx) for c in self._model.clauses}
        self._check_predef()

    def _check_predef(self) -> None:
        """Check if predefs need to be re-calc'd, and if so, do so."""
        curr_hash = self._model.predef_context_hash(self._ctx)
        if curr_hash != self._predef_hash:
            try:
                self._predef = self._model.predef(self._ctx)
                self._predef_hash = curr_hash
            except KeyError as e:
                # NOTE: catching KeyError here will be necessary (to get nice error messages)
                # until we can properly validate the MM clauses.
                msg = f"Missing attribute {e} required by movement model predef."
                raise AttributeException(msg) from None

            if not isinstance(self._predef, dict):
                msg = f"Movement predef: did not return a dictionary result (got: {type(self._predef)})"
                raise MmCompileException(msg)

    def apply(self, world: World, tick: Tick) -> None:
        """Applies movement for this tick, mutating the world state."""

        self.on_movement_start.publish(
            OnMovementStart(tick.index, tick.day, tick.step))

        self._check_predef()

        # Process travel clauses.
        total = 0
        for clause in self._model.clauses:
            if not clause.predicate(self._ctx, tick):
                continue
            local_array = world.get_local_array()

            clause_event = calculate_travelers(
                self._ctx, self._predef,
                clause, self._clause_masks[clause], tick, local_array,
            )
            self.on_movement_clause.publish(clause_event)
            travelers = clause_event.actual

            returns = clause.returns(self._ctx, tick)
            return_tick = resolve_tick(self._ctx.dim, tick, returns)
            world.apply_travel(travelers, return_tick)
            total += travelers.sum()

        # Process return clause.
        return_movers = world.apply_return(tick, return_stats=True)
        return_total = return_movers.sum()
        total += return_total

        self.on_movement_clause.publish(
            OnMovementClause(
                tick.index,
                tick.day,
                tick.step,
                "return",
                return_movers,
                return_movers,
                return_total,
                False,
            )
        )

        self.on_movement_finish.publish(
            OnMovementFinish(tick.index, tick.day, tick.step, total))
