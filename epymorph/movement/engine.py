from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from attr import dataclass

from epymorph.clock import Tick
from epymorph.context import SimContext
from epymorph.movement.clause import (ArrayClause, CellClause, Clause,
                                      ReturnClause, RowClause)
from epymorph.movement.world import Location, World
from epymorph.util import Compartments


@dataclass
class Movement:
    """
    The movement model divides a day into simulation parts (tau steps) under the assumption
    that each day part will have movement characteristics relevant to the simulation.
    That is: there is no reason to have tau steps smaller than 1 day unless it's relevant to movement.
    """
    taus: list[float]
    """The tau steps for the simulation."""
    clauses: list[Clause]
    """The clauses which express the movement model"""


class MovementBuilder(ABC):
    taus: list[float]

    @abstractmethod
    def verify(self, ctx: SimContext) -> None:
        pass

    @abstractmethod
    def build(self, ctx: SimContext) -> Movement:
        pass


class MovementEngine(World, ABC):
    # movement engine encapsulates the model of the world
    ctx: SimContext
    movement: Movement

    def __init__(self, ctx: SimContext, movement: Movement, initial_compartments: list[Compartments]):
        self.ctx = ctx
        self.movement = movement

    def apply(self, tick: Tick) -> None:
        for clause in self.movement.clauses:
            if not clause.predicate(tick):
                continue
            match clause:
                case ReturnClause():
                    self._apply_return(clause, tick)
                case ArrayClause():
                    self._apply_array(clause, tick)
                case RowClause():
                    self._apply_row(clause, tick)
                case CellClause():
                    self._apply_cell(clause, tick)

    @abstractmethod
    def _apply_return(self, clause: ReturnClause, tick: Tick) -> None:
        pass

    @abstractmethod
    def _apply_array(self, clause: ArrayClause, tick: Tick) -> None:
        pass

    @abstractmethod
    def _apply_row(self, clause: RowClause, tick: Tick) -> None:
        pass

    @abstractmethod
    def _apply_cell(self, clause: CellClause, tick: Tick) -> None:
        pass
