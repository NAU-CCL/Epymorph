from __future__ import annotations

import logging
from abc import ABC
from collections.abc import Sequence as abcSequence
from inspect import signature
from typing import Callable, cast

import numpy as np
from numpy.typing import NDArray

from epymorph.clock import Tick, TickDelta
from epymorph.context import SimContext
from epymorph.parser.move_clause import ALL_DAYS, DayOfWeek
from epymorph.world import HOME_TICK, Population, World


class Clause(ABC):
    """Movement clause base class. Transforms World at a given time step."""

    def apply(self, world: World, tick: Tick) -> None:
        pass


CompartmentPredicate = Callable[[list[str]], bool]
ClausePredicate = Callable[[Tick], bool]
ClauseFunction = Callable[[Tick], NDArray[np.int_]]
ExtendedClauseFunction = ClauseFunction |\
    Callable[[Tick, int], NDArray[np.int_]] |\
    Callable[[Tick, int, int], NDArray[np.int_]]


def normalize_clause_function(ctx: SimContext, cf: ExtendedClauseFunction) -> ClauseFunction:
    NxN = (ctx.nodes, ctx.nodes)
    cf_sig = signature(cf)
    match len(cf_sig.parameters):
        case 3:
            return cast(ClauseFunction, cf)

        case 2:
            cf = cast(Callable[[Tick, int], NDArray[np.int_]], cf)

            def norm_cf(tick: Tick) -> NDArray[np.int_]:
                arr = np.zeros(NxN, dtype=int)
                for src in range(ctx.nodes):
                    arr[src, :] = cf(tick, src)
                return arr
            return norm_cf

        case 1:
            cf = cast(Callable[[Tick, int, int], NDArray[np.int_]], cf)

            def norm_cf(tick: Tick) -> NDArray[np.int_]:
                arr = np.zeros(NxN, dtype=np.int_)
                for src, dst in np.ndindex(NxN):
                    arr[src, dst] = cf(tick, src, dst)
                return arr
            return norm_cf

        case _:
            raise Exception(
                f"Unable to normalize movement function with signature: {cf_sig}")


class FunctionalClause(Clause):
    """
    A general-purpose movement clause triggered by `predicate`, calculating requested node emigrants using `equation`,
    which return according to `returns`. If there are not enough locals to cover the requested number of movers,
    movement will be canceled for the source node this tick.
    """

    ctx: SimContext
    name: str
    predicate: ClausePredicate
    compartment_tag_predicate: CompartmentPredicate
    movement_mask: NDArray[np.bool_]
    clause_function: ClauseFunction
    returns: TickDelta
    logger: logging.Logger

    def __init__(self,
                 ctx: SimContext,
                 name: str,
                 predicate: ClausePredicate,
                 compartment_tag_predicate: CompartmentPredicate,
                 clause_function: ExtendedClauseFunction,
                 returns: TickDelta):
        self.ctx = ctx
        self.name = name
        self.predicate = predicate
        self.compartment_tag_predicate = compartment_tag_predicate
        self.movement_mask = np.array([compartment_tag_predicate(ts)
                                       for ts in ctx.compartment_tags], dtype=bool)
        self.clause_function = normalize_clause_function(ctx, clause_function)
        self.returns = returns
        self.logger = logging.getLogger(f'movement.{name}')

    def apply(self, world: World, tick: Tick) -> None:
        if not self.predicate(tick):
            return

        return_tick = self.ctx.clock.tick_plus(tick, self.returns)

        # TODO: test affect of keeping world as an array
        all_locals = world.all_locals()
        available_movers = all_locals * self.movement_mask

        # TODO: test affect of writing to the same array every tick rather than allocating a new array
        requested_movers = self.clause_function(tick)
        np.fill_diagonal(requested_movers, 0)

        available_sum = available_movers.sum(axis=1)
        requested_sum = requested_movers.sum(axis=1)

        for src in range(self.ctx.nodes):
            # If requested total is greater than the total available,
            # use mvhg to select as many as possible.
            if requested_sum[src] > available_sum[src]:
                self.logger.debug(
                    "<WARNING> movement throttled for insufficient population at %d", src)
                requested_movers[src, :] = self.ctx.rng.multivariate_hypergeometric(
                    colors=requested_movers[src, :],
                    nsample=available_sum[src]
                )

        self.logger.debug("requested_movers: %s", requested_movers)

        # Update sum in case it changed in the previous step.
        requested_sum = requested_movers.sum(axis=1)
        # The probability a mover from a src will go to a dst.
        requested_prb = requested_movers / requested_sum[:, np.newaxis]

        for src in range(self.ctx.nodes):
            if requested_sum[src] == 0:
                continue

            # Select which individuals will be leaving this node.
            mover_cs = self.ctx.rng.multivariate_hypergeometric(
                available_movers[src, :],
                requested_sum[src]
            )

            # Select which location they are each going to.
            # (Each row contains the compartments for a destination.)
            split_cs = self.ctx.rng.multinomial(
                mover_cs,
                requested_prb[src, :]
            ).T

            split_sum = split_cs.sum(axis=1)

            # Create new pops.
            for dst in range(self.ctx.nodes):
                if src == dst:
                    world.locations[src].locals.compartments -= mover_cs
                else:
                    if split_sum[dst] > 0:
                        p = Population(split_cs[dst, :], src, return_tick)
                        world.locations[dst].pops.append(p)

        world.normalize()

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("moved %d", requested_sum.sum())


class Noop(Clause):
    """A movement clause that does nothing."""

    def apply(self, world: World, tick: Tick) -> None:
        return None


class Return(Clause):
    """A movement clause to return people to their home if it's time."""
    logger: logging.Logger

    ctx: SimContext

    def __init__(self, ctx: SimContext):
        self.ctx = ctx
        self.logger = logging.getLogger('movement.Return')

    def apply(self, world: World, tick: Tick) -> None:
        total_movers = 0
        new_pops = [list[Population]() for _ in range(self.ctx.nodes)]
        for loc in world.locations:
            for pop in loc.pops:
                if pop.return_tick == tick.index:
                    # pop ready to go home
                    pop.return_tick = HOME_TICK
                    new_pops[pop.return_location].append(pop)
                    total_movers += pop.total
                else:
                    # pop staying
                    new_pops[loc.index].append(pop)
        for i, ps in enumerate(new_pops):
            Population.normalize(ps)
            world.locations[i].pops = ps
        self.logger.debug("moved %d (t:%d,%d)",
                          total_movers, tick.day, tick.step)


class Sequence(Clause):
    """A movement clause which executes a list of child clauses in sequence."""

    clauses: abcSequence[Clause]

    def __init__(self, clauses: abcSequence[Clause]):
        self.clauses = clauses

    def apply(self, world: World, tick: Tick) -> None:
        for clause in self.clauses:
            clause.apply(world, tick)


class Predicates:
    @ staticmethod
    def daylist(days: list[DayOfWeek], step: int | None = None) -> ClausePredicate:
        day_indices = [i for i in range(len(ALL_DAYS)) if ALL_DAYS[i] in days]
        return lambda tick: tick.date.weekday() in day_indices and \
            (step is None or step == tick.step)

    @ staticmethod
    def everyday(step: int | None = None) -> ClausePredicate:
        if step == None:
            return lambda _: True
        else:
            return lambda tick: tick.step == step

    # Example of some other predicate helper functions,
    # but many more are possible.
    # TODO: These could be cleaned up a bit.

    @ staticmethod
    def weekdays(step: int | None = None) -> ClausePredicate:
        if step == None:
            return lambda tick: 0 <= tick.date.weekday() < 5
        else:
            return lambda tick: 0 <= tick.date.weekday() < 5 and tick.step == step

    @ staticmethod
    def weekends(step: int | None = None) -> ClausePredicate:
        if step == None:
            return lambda tick: 4 < tick.date.weekday() <= 6
        else:
            return lambda tick: 4 < tick.date.weekday() <= 6 and tick.step == step
