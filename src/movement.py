from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence as abcSequence
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from clock import Tick, TickDelta
from sim_context import SimContext
from world import Population, Timer, World

ClausePred = Callable[[Tick], bool]
RowEquation = Callable[[SimContext, Tick, int], NDArray[np.int_]]
CrossEquation = Callable[[SimContext, Tick, int, int], np.int_]


def log(logger: logging.Logger, total_movers: np.int_, tick: Tick) -> None:
    logger.debug(f"moved {total_movers} (t:{tick.day},{tick.step})")


class Clause(ABC):
    """Movement clause base class. Transforms World at a given time step."""

    @abstractmethod
    def apply(self, sim: SimContext, world: World, tick: Tick) -> None:
        pass


class ConditionalClause(Clause):
    """Movement clause triggered only when `predicate` is true."""

    def __init__(self, predicate: ClausePred):
        self.predicate = predicate

    @abstractmethod
    def _execute(self, sim: SimContext, world: World, tick: Tick) -> None:
        pass

    def apply(self, sim: SimContext, world: World, tick: Tick) -> None:
        if self.predicate(tick):
            self._execute(sim, world, tick)


class GeneralClause(ConditionalClause):
    """
    A general-purpose movement clause triggered by `predicate`, calculating requested node emigrants using `equation`,
    which return according to `returns`. If there are not enough locals to cover the requested number of movers,
    movement will be canceled for the source node this tick.
    """

    @classmethod
    def byRow(cls, name: str, predicate: ClausePred, returns: TickDelta, equation: RowEquation) -> GeneralClause:
        return GeneralClause(name, predicate, returns, equation)

    @classmethod
    def byCross(cls, name: str, predicate: ClausePred, returns: TickDelta, equation: CrossEquation) -> GeneralClause:
        def e(sim: SimContext, tick: Tick, src_idx: int) -> NDArray[np.int_]:
            row = np.zeros(sim.nodes, dtype=np.int_)
            for dst_idx in range(sim.nodes):
                row[dst_idx] = 0 if src_idx == dst_idx else \
                    equation(sim, tick, src_idx, dst_idx)
            return row
        return GeneralClause(name, predicate, returns, e)

    def __init__(self, name: str, predicate: ClausePred, returns: TickDelta, equation: RowEquation):
        super().__init__(predicate)
        self.name = name
        self.returns = returns
        self.equation = equation
        self.logger = logging.getLogger(f'movement.{name}')

    def _execute(self, sim: SimContext, world: World, tick: Tick) -> None:
        total_movers = np.int_(0)
        return_tick = sim.clock.tickPlus(tick, self.returns)
        for src_idx, src in enumerate(world.locations):
            locals = src.locals
            # movers from src to all destinations:
            requested_arr = self.equation(sim, tick, src_idx)
            requested_tot = np.sum(requested_arr)
            if 0 < requested_tot <= locals.total:
                for dst_idx, n in enumerate(requested_arr):
                    if src_idx == dst_idx or n == 0:
                        continue
                    actual, movers = locals.split(
                        sim, src_idx, dst_idx, return_tick, n)
                    world.locations[dst_idx].pops.append(movers)
                    total_movers += actual
        world.normalize()
        log(self.logger, total_movers, tick)


class Noop(Clause):
    """A movement clause that does nothing."""

    def apply(self, sim: SimContext, world: World, tick: Tick) -> None:
        return None


class Return(Clause):
    """A movement clause to return people to their home if it's time."""
    log = logging.getLogger('movement.Return')

    def apply(self, sim: SimContext, world: World, tick: Tick) -> None:
        total_movers = np.int_(0)
        new_pops: list[list[Population]] = [[] for _ in range(sim.nodes)]
        for loc in world.locations:
            for pop in loc.pops:
                if pop.timer == tick.index:
                    # pop ready to go home
                    pop.timer = Timer.Home
                    new_pops[pop.dest].append(pop)
                    total_movers += pop.total
                else:
                    # pop staying
                    new_pops[loc.index].append(pop)
        for i, ps in enumerate(new_pops):
            Population.normalize(ps)
            world.locations[i].pops = ps
        log(Return.log, total_movers, tick)


class Sequence(Clause):
    """A movement clause which executes a list of child clauses in sequence."""

    def __init__(self, clauses: abcSequence[Clause]):
        self.clauses = clauses

    def apply(self, sim: SimContext, world: World, tick: Tick) -> None:
        for clause in self.clauses:
            clause.apply(sim, world, tick)


class Movement:
    def __init__(self, taus: list[np.double], clause: Clause):
        assert len(taus) > 0, "Must supply at least one tau step."
        assert np.sum(taus) == np.double(1), "Tau steps must sum to 1."
        self.taus = taus
        self.clause = clause


class Predicates:
    @classmethod
    def everyday(cls, step: int | None = None) -> ClausePred:
        if step == None:
            return lambda _: True
        else:
            return lambda tick: tick.step == step

    # Example of some other predicate helper functions,
    # but many more are possible.
    # TODO: These could be cleaned up a bit.

    @classmethod
    def weekdays(cls, step: int | None = None) -> ClausePred:
        if step == None:
            return lambda tick: 0 <= tick.date.weekday() < 5
        else:
            return lambda tick: 0 <= tick.date.weekday() < 5 and tick.step == step

    @classmethod
    def weekends(cls, step: int | None = None) -> ClausePred:
        if step == None:
            return lambda tick: 4 < tick.date.weekday() <= 6
        else:
            return lambda tick: 4 < tick.date.weekday() <= 6 and tick.step == step
