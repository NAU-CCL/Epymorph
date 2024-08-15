import numpy as np
from numpy.typing import NDArray

from epymorph.data_type import AttributeArray, SimDType
from epymorph.database import Database, ModuleNamespace
from epymorph.error import MmSimException
from epymorph.event import (MovementEventsMixin, OnMovementClause,
                            OnMovementFinish, OnMovementStart)
from epymorph.rume import Rume
from epymorph.simulation import (NamespacedAttributeResolver, Tick, gpm_strata,
                                 resolve_tick_delta)
from epymorph.simulator.world import World
from epymorph.util import row_normalize


def calculate_travelers(
    clause_name: str,
    clause_mobility: NDArray[np.bool_],
    requested_movers: NDArray[SimDType],
    available_movers: NDArray[SimDType],
    tick: Tick,
    rng: np.random.Generator
) -> OnMovementClause:
    """
    Calculate the number of travelers resulting from this movement clause for this tick.
    This evaluates the requested number movers, modulates that based on the available movers,
    then selects exactly which individuals (by compartment) should move.
    Returns an (N,N,C) array; from-source-to-destination-by-compartment.
    """
    # Extract number of nodes and cohorts from the provided array.
    (N, C) = available_movers.shape

    initial_requested_movers = requested_movers
    np.fill_diagonal(requested_movers, 0)
    requested_sum = requested_movers.sum(axis=1, dtype=SimDType)

    available_movers = available_movers * clause_mobility
    available_sum = available_movers.sum(axis=1, dtype=SimDType)

    # If clause requested total is greater than the total available,
    # use mvhg to select as many as possible.
    if not np.any(requested_sum > available_sum):
        throttled = False
        # requested_movers = requested_movers
        # requested_sum = requested_sum
    else:
        throttled = True
        requested_movers = requested_movers.copy()
        for src in range(N):
            if requested_sum[src] > available_sum[src]:
                requested_movers[src, :] = rng.multivariate_hypergeometric(
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
        mover_cs = rng.multivariate_hypergeometric(
            available_movers[src, :],
            requested_sum[src]
        ).astype(SimDType)

        # Select which location they are each going to.
        # (Each row contains the compartments for a destination.)
        travelers_cs[src, :, :] = rng.multinomial(
            mover_cs,
            requested_prb[src, :]
        ).T.astype(SimDType)

    return OnMovementClause(
        tick.sim_index,
        tick.day,
        tick.step,
        clause_name,
        initial_requested_movers,
        travelers_cs,
        requested_sum.sum(),
        throttled,
    )


class MovementExecutor:
    """Movement model execution specifically for multi-strata simulations."""

    _rume: Rume
    """the RUME"""
    _world: World
    """the world state"""
    _rng: np.random.Generator
    """the simulation RNG"""

    _event_target: MovementEventsMixin
    _strata_data: dict[str, NamespacedAttributeResolver]

    def __init__(
        self,
        rume: Rume,
        world: World,
        db: Database[AttributeArray],
        rng: np.random.Generator,
        event_target: MovementEventsMixin,
    ):
        self._rume = rume
        self._world = world
        self._rng = rng
        self._event_target = event_target

        # Create a per-strata attribute resolver for MM clauses.
        self._strata_data = {
            strata: NamespacedAttributeResolver(
                data=db,
                dim=rume.dim,
                namespace=ModuleNamespace(gpm_strata(strata), "mm"),
            )
            for strata, mm in rume.mms.items()
        }

    def apply(self, tick: Tick) -> None:
        """Applies movement for this tick, mutating the world state."""

        self._event_target.on_movement_start.publish(
            OnMovementStart(tick.sim_index, tick.day, tick.step))

        # Process travel clauses.
        total = 0
        for strata, model in self._rume.mms.items():
            for clause in model.clauses:
                if not clause.is_active(tick):
                    continue

                available_movers = self._world.get_local_array()

                try:
                    requested_movers = clause.evaluate_in_context(
                        self._strata_data[strata],
                        self._rume.dim,
                        self._rume.scope,
                        self._rng,
                        tick
                    )
                except Exception as e:
                    # NOTE: catching exceptions here is necessary to get nice error messages
                    # for some value error cause by incorrect parameter and/or clause definition
                    msg = f"Error from applying clause '{clause.__class__.__name__}': see exception trace"
                    raise MmSimException(msg) from e

                clause_event = calculate_travelers(
                    clause.__class__.__name__,
                    self._rume.compartment_mobility[strata],
                    requested_movers,
                    available_movers,
                    tick,
                    self._rng
                )
                self._event_target.on_movement_clause.publish(clause_event)
                travelers = clause_event.actual

                return_tick = resolve_tick_delta(self._rume.dim, tick, clause.returns)
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
