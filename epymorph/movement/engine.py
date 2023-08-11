"""
Abstract base classes for the components of the movement system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from attr import dataclass

from epymorph.clock import Tick
from epymorph.context import Compartments, SimContext
from epymorph.movement.clause import (ArrayClause, CellClause, Clause,
                                      ReturnClause, RowClause)
from epymorph.movement.world import World


@dataclass
class Movement:
    """
    The movement model divides a day into simulation parts (tau steps) under the assumption
    that each day part will have movement characteristics relevant to the simulation.
    That is: there is no reason to have tau steps smaller than 1 day unless it's relevant
    to movement.
    """
    taus: list[float]
    """The tau steps for the simulation."""
    clauses: list[Clause]
    """The clauses which express the movement model"""


class MovementBuilder(ABC):
    """
    A class which can construct a movement model for a given SimContext,
    as well as verify that the movement model would be valid for that
    context.
    """
    taus: list[float]

    @abstractmethod
    def verify(self, ctx: SimContext) -> None:
        """Check the movement model for validity under the given context."""

    @abstractmethod
    def build(self, ctx: SimContext) -> Movement:
        """Build the movement model for a context."""


class MovementEngine(World, ABC):
    """
    Movement engine encapsulates the model of the world.
    (Each implementation is expected to have tradeoffs between performance and
    machine requirements.)
    """
    ctx: SimContext
    movement: Movement

    def __init__(self, ctx: SimContext, movement: Movement,
                 initial_compartments: list[Compartments]):
        self.ctx = ctx
        self.movement = movement
        # initial_compartments should be utilized by subclasses

    def apply(self, tick: Tick) -> None:
        """Apply the movement model for a given tick."""
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

    def shutdown(self) -> None:
        """Let the movement engine know we're done and it can clean up."""

    @abstractmethod
    def _apply_return(self, clause: ReturnClause, tick: Tick) -> None:
        """Apply a return clause."""

    @abstractmethod
    def _apply_array(self, clause: ArrayClause, tick: Tick) -> None:
        """Apply an array clause."""

    @abstractmethod
    def _apply_row(self, clause: RowClause, tick: Tick) -> None:
        """Apply a row clause."""

    @abstractmethod
    def _apply_cell(self, clause: CellClause, tick: Tick) -> None:
        """Apply a cell clause."""
