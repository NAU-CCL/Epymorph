from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence as abcSequence
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from epymorph.clock import Tick, TickDelta
from epymorph.context import SimContext
from epymorph.parser.move_clause import ALL_DAYS, DayOfWeek
from epymorph.world import HOME_TICK, Population, World

ClausePred = Callable[[Tick], bool]
RowEquation = Callable[[Tick, int], NDArray[np.int_]]
CrossEquation = Callable[[Tick, int, int], np.int_]


class Clause(ABC):
    """Movement clause base class. Transforms World at a given time step."""

    def apply(self, world: World, tick: Tick) -> None:
        pass


class ConditionalClause(Clause):
    """Movement clause triggered only when `predicate` is true."""

    def __init__(self, predicate: ClausePred):
        self.predicate = predicate

    @abstractmethod
    def _execute(self, world: World, tick: Tick) -> None:
        pass

    def apply(self, world: World, tick: Tick) -> None:
        if self.predicate(tick):
            self._execute(world, tick)


class GeneralClause(ConditionalClause):
    """
    A general-purpose movement clause triggered by `predicate`, calculating requested node emigrants using `equation`,
    which return according to `returns`. If there are not enough locals to cover the requested number of movers,
    movement will be canceled for the source node this tick.
    """

    @classmethod
    def by_row(cls, ctx: SimContext, name: str, predicate: ClausePred, returns: TickDelta, equation: RowEquation) -> GeneralClause:
        return cls(ctx, name, predicate, returns, equation)

    @classmethod
    def by_cross(cls, ctx: SimContext, name: str, predicate: ClausePred, returns: TickDelta, equation: CrossEquation) -> GeneralClause:
        def e(tick: Tick, src_idx: int) -> NDArray[np.int_]:
            row = np.zeros(ctx.nodes, dtype=np.int_)
            for dst_idx in range(ctx.nodes):
                row[dst_idx] = 0 if src_idx == dst_idx else \
                    equation(tick, src_idx, dst_idx)
            return row
        return cls(ctx, name, predicate, returns, e)

    ctx: SimContext
    name: str
    returns: TickDelta
    equation: RowEquation
    logger: logging.Logger

    def __init__(self, ctx: SimContext, name: str, predicate: ClausePred, returns: TickDelta, equation: RowEquation):
        super().__init__(predicate)
        self.ctx = ctx
        self.name = name
        self.returns = returns
        self.equation = equation
        self.logger = logging.getLogger(f'movement.{name}')

    def _execute(self, world: World, tick: Tick) -> None:
        total_movers = 0
        return_tick = self.ctx.clock.tick_plus(tick, self.returns)
        for src_idx, src in enumerate(world.locations):
            locals = src.locals
            # movers from src to all destinations:
            requested_arr = self.equation(tick, src_idx)
            requested_arr[src_idx] = 0
            self.logger.debug("requested[%d]: %s", src_idx, requested_arr)
            requested_tot = np.sum(requested_arr)
            if requested_tot > locals.total:
                self.logger.debug("   actual[%d]: <WARNING> skipped for insufficient population",
                                  src_idx)
            elif requested_tot > 0:
                actual_arr = np.zeros_like(requested_arr)
                for dst_idx, n in enumerate(requested_arr):
                    if src_idx == dst_idx or n == 0:
                        continue
                    actual, movers = locals.split(
                        self.ctx, src_idx, dst_idx, return_tick, n)
                    world.locations[dst_idx].pops.append(movers)
                    actual_arr[dst_idx] = actual
                    total_movers += actual
                if self.logger.isEnabledFor(logging.DEBUG):
                    if not np.array_equal(actual_arr, requested_arr):
                        self.logger.debug("   actual[%d]: %s",
                                          src_idx, actual_arr)
        world.normalize()
        self.logger.debug("moved %d (t:%d,%d)",
                          total_movers, tick.day, tick.step)


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
    @staticmethod
    def daylist(days: list[DayOfWeek], step: int | None = None) -> ClausePred:
        day_indices = [i for i in range(len(ALL_DAYS)) if ALL_DAYS[i] in days]
        return lambda tick: tick.date.weekday() in day_indices and \
            (step is None or step == tick.step)

    @staticmethod
    def everyday(step: int | None = None) -> ClausePred:
        if step == None:
            return lambda _: True
        else:
            return lambda tick: tick.step == step

    # Example of some other predicate helper functions,
    # but many more are possible.
    # TODO: These could be cleaned up a bit.

    @staticmethod
    def weekdays(step: int | None = None) -> ClausePred:
        if step == None:
            return lambda tick: 0 <= tick.date.weekday() < 5
        else:
            return lambda tick: 0 <= tick.date.weekday() < 5 and tick.step == step

    @staticmethod
    def weekends(step: int | None = None) -> ClausePred:
        if step == None:
            return lambda tick: 4 < tick.date.weekday() <= 6
        else:
            return lambda tick: 4 < tick.date.weekday() <= 6 and tick.step == step
