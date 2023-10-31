"""
Movement executor classes handle the logic for processing the movement step of the simulation.
"""
from abc import ABC, abstractmethod
from logging import DEBUG, Logger, getLogger

import numpy as np
from numpy.typing import NDArray

from epymorph.engine.context import ExecutionContext
from epymorph.engine.world import World
from epymorph.movement_model import TravelClause
from epymorph.simulation import SimDType, Tick
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


############################################################
# StandardMovementExecutor
############################################################


class StandardMovementExecutor(MovementExecutor):
    """The standard implementation of movement model execution."""

    _ctx: ExecutionContext
    _log: Logger
    _clause_masks: dict[TravelClause, NDArray[np.bool_]]

    def __init__(self, ctx: ExecutionContext):
        self._ctx = ctx
        self._log = getLogger('movement')
        self._clause_masks = {c: c.mask(ctx) for c in ctx.mm.clauses}

    def apply(self, world: World, tick: Tick) -> None:
        """Applies movement for this tick, mutating the world state."""
        self._log.debug('Processing movement for day %s, step %s', tick.day, tick.step)

        # Process travel clauses.
        for clause in self._ctx.mm.clauses:
            if not clause.predicate(self._ctx, tick):
                continue
            local_array = world.get_local_array()
            travelers = self._travelers(clause, tick, local_array)
            returns = clause.returns(self._ctx, tick)
            return_tick = self._ctx.resolve_tick(tick, returns)
            world.apply_travel(travelers, return_tick)

        # Process return clause.
        return_movers = world.apply_return(tick)
        self._log.getChild('Return').debug("moved %d", return_movers)

    def _travelers(self, clause: TravelClause, tick: Tick, local_cohorts: NDArray[SimDType]) -> NDArray[SimDType]:
        """
        Calculate the number of travelers resulting from this movement clause for this tick.
        This evaluates the requested number movers, modulates that based on the available movers,
        then selects exactly which individuals (by compartment) should move.
        Returns an (N,N,C) array; from-source-to-destination-by-compartment.
        """
        clause_log = self._log.getChild(clause.name)
        _, N, C, _ = self._ctx.dim.TNCE

        requested_movers = clause.requested(self._ctx, tick)
        np.fill_diagonal(requested_movers, 0)
        requested_sum = requested_movers.sum(axis=1, dtype=SimDType)

        available_movers = local_cohorts * self._clause_masks[clause]
        available_sum = available_movers.sum(axis=1, dtype=SimDType)

        # If requested total is greater than the total available,
        # use mvhg to select as many as possible.
        throttled = False
        for src in range(N):
            if requested_sum[src] > available_sum[src]:
                msg = "<WARNING> movement throttled for insufficient population at %d"
                clause_log.debug(msg, src)
                requested_movers[src, :] = self._ctx.rng.multivariate_hypergeometric(
                    colors=requested_movers[src, :],
                    nsample=available_sum[src]
                )
                throttled = True

        # Update sum if it changed in the previous step.
        if throttled:
            requested_sum = requested_movers.sum(axis=1, dtype=SimDType)

        # The probability a mover from a src will go to a dst.
        requested_prb = row_normalize(requested_movers, requested_sum, dtype=SimDType)

        clause_log.debug("requested_movers:\n%s", requested_movers)

        travelers_cs = np.zeros((N, N, C), dtype=SimDType)
        for src in range(N):
            if requested_sum[src] == 0:
                continue

            # Select which individuals will be leaving this node.
            mover_cs = self._ctx.rng.multivariate_hypergeometric(
                available_movers[src, :],
                requested_sum[src]
            ).astype(SimDType)

            # Select which location they are each going to.
            # (Each row contains the compartments for a destination.)
            travelers_cs[src, :, :] = self._ctx.rng.multinomial(
                mover_cs,
                requested_prb[src, :]
            ).T.astype(SimDType)

        if clause_log.isEnabledFor(DEBUG):
            clause_log.debug("moved %d", requested_sum.sum())

        return travelers_cs
