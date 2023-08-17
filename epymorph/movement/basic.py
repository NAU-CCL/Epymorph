from __future__ import annotations

from dataclasses import dataclass
from logging import DEBUG, Logger, getLogger
from operator import attrgetter
from typing import Generator, Iterable, Self

import numpy as np
from numpy.typing import NDArray

from epymorph.clock import Tick
from epymorph.context import Compartments, SimContext, SimDType
from epymorph.movement.clause import (ArrayClause, CellClause, ReturnClause,
                                      RowClause, TravelClause)
from epymorph.movement.engine import Movement, MovementEngine
from epymorph.movement.world import Location
from epymorph.util import row_normalize

# WORLD MODEL


HOME_TICK = -1
"""The value of a population's `return_tick` when the population is home."""


@dataclass
class Cohort:
    """
    Represents a group of individuals, divided into IPM compartments as appropriate for the simulation.
    These individuals share the same "home location" and a time at which they should return there.

    These are somewhat abstract concepts, however; a completely nomadic group doesn't really have a home location, merely the next
    location in a chain of movements.
    """

    @staticmethod
    def can_merge(a: Cohort, b: Cohort) -> bool:
        return a.return_tick == b.return_tick and a.return_location == b.return_location

    @staticmethod
    def merge(intoc: Cohort, fromc: Cohort) -> None:
        intoc.compartments += fromc.compartments

    compartments: Compartments
    return_location: int
    return_tick: int

    # Note: when a population is "home",
    # its `return_location` is the same as their current location,
    # and its `return_tick` is set to -1 (the "Never" placeholder value).


@dataclass
class BasicLocation(Location):
    index: int
    cohorts: list[Cohort]

    @classmethod
    def create(cls, index: int, initial_compartments: Compartments) -> Self:
        return cls(
            index=index,
            cohorts=[Cohort(initial_compartments, index, HOME_TICK)],
        )

    def get_index(self) -> int:
        return self.index

    def get_compartments(self) -> Compartments:
        return np.sum([c.compartments for c in self.cohorts], axis=0, dtype=SimDType)

    def get_cohorts(self) -> Compartments:
        return np.array([c.compartments for c in self.cohorts], dtype=SimDType)

    def update_cohorts(self, deltas: Compartments) -> None:
        for i, c in enumerate(self.cohorts):
            c.compartments += deltas[i]

    def normalize(self) -> None:
        """
        Sorts a list of populations and combines mergeable populations (in place).
        Lists of populations should be normalized after creation and any modification.
        """
        self.cohorts.sort(key=attrgetter('return_tick', 'return_location'))
        # Iterate over all sequential pairs, starting from (0,1)
        j = 1
        while j < len(self.cohorts):
            prev = self.cohorts[j - 1]
            curr = self.cohorts[j]
            if Cohort.can_merge(prev, curr):
                Cohort.merge(prev, curr)
                del self.cohorts[j]
            else:
                j += 1


class BasicEngine(MovementEngine):
    clause_loggers: dict[str, Logger]
    locations: list[BasicLocation]

    def __init__(self, ctx: SimContext, movement: Movement, initial_compartments: Compartments) -> None:
        super().__init__(ctx, movement, initial_compartments)
        self.clause_loggers = {c.name: getLogger(f'movement.{c.name}')
                               for c in movement.clauses}
        self.locations = [BasicLocation.create(i, cs)
                          for (i, cs) in enumerate(initial_compartments)]

    # Implement World

    def _normalize(self) -> None:
        for loc in self.locations:
            loc.normalize()

    def _all_cohorts(self) -> Generator[Cohort, None, None]:
        for loc in self.locations:
            for cohort in loc.cohorts:
                yield cohort

    def get_locations(self) -> Iterable[BasicLocation]:
        return self.locations

    def get_locals(self) -> NDArray[SimDType]:
        # NOTE: this only works if `cohorts` is normalized after modification
        loc_locals = [loc.cohorts[0].compartments
                      for loc in self.locations]
        return np.array(loc_locals, dtype=SimDType)

    def get_travelers(self) -> Compartments:
        # NOTE: this only works if `cohorts` is normalized after modification
        _, N, C, _ = self.ctx.TNCE
        loc_travelers = np.zeros((N, C), dtype=SimDType)
        for loc in self.locations:
            if len(loc.cohorts) > 1:
                cohorts = [c.compartments for c in loc.cohorts[1:]]
                total = np.array(cohorts, dtype=SimDType)\
                    .sum(axis=0, dtype=SimDType)
                loc_travelers[loc.get_index(), :] = total
        return loc_travelers

    def get_travelers_by_home(self) -> Compartments:
        # NOTE: this only works if `cohorts` is normalized after modification
        _, N, C, _ = self.ctx.TNCE
        home_travelers = np.zeros((N, C), dtype=SimDType)
        for c in self._all_cohorts():
            if c.return_tick != HOME_TICK:
                home_travelers[c.return_location, :] += c.compartments
        return home_travelers

    # Implement MovementEngine

    def apply(self, tick: Tick) -> None:
        getLogger('movement').debug("Processing movement for day %s, step %s",
                                    tick.day, tick.step)
        # Defer to base-class implementation.
        return super().apply(tick)

    def _apply_return(self, clause: ReturnClause, tick: Tick) -> None:
        logger = self.clause_loggers[clause.name]
        total_movers = 0
        new_cohorts = [list[Cohort]() for _ in range(self.ctx.nodes)]
        for loc in self.locations:
            for cohort in loc.cohorts:
                if cohort.return_tick == tick.index:
                    # cohort ready to go home
                    cohort.return_tick = HOME_TICK
                    new_cohorts[cohort.return_location].append(cohort)
                    total_movers += cohort.compartments.sum()
                else:
                    # cohort staying
                    new_cohorts[loc.get_index()].append(cohort)

        for loc, cohorts in zip(self.locations, new_cohorts):
            loc.cohorts = cohorts

        self._normalize()

        logger.debug("moved %d", total_movers)

    def _apply_travel(self, clause: TravelClause, tick: Tick, requested_movers: NDArray[SimDType]) -> None:
        logger = self.clause_loggers[clause.name]
        return_tick = self.ctx.clock.tick_plus(tick, clause.returns)

        all_locals = self.get_locals()
        available_movers = all_locals * clause.movement_mask

        available_sum = available_movers.sum(axis=1, dtype=SimDType)
        requested_sum = requested_movers.sum(axis=1, dtype=SimDType)

        for src in range(self.ctx.nodes):
            # If requested total is greater than the total available,
            # use mvhg to select as many as possible.
            if requested_sum[src] > available_sum[src]:
                logger.debug(
                    "<WARNING> movement throttled for insufficient population at %d", src)
                requested_movers[src, :] = self.ctx.rng.multivariate_hypergeometric(
                    colors=requested_movers[src, :],
                    nsample=available_sum[src]
                )

        logger.debug("requested_movers: %s", requested_movers)

        # Update sum in case it changed in the previous step.
        requested_sum = requested_movers.sum(axis=1, dtype=SimDType)
        # The probability a mover from a src will go to a dst.
        requested_prb = row_normalize(
            requested_movers, requested_sum, dtype=SimDType)

        for src in range(self.ctx.nodes):
            if requested_sum[src] == 0:
                continue

            # Select which individuals will be leaving this node.
            mover_cs = self.ctx.rng.multivariate_hypergeometric(
                available_movers[src, :],
                requested_sum[src]
            ).astype(SimDType)

            # Select which location they are each going to.
            # (Each row contains the compartments for a destination.)
            split_cs = self.ctx.rng.multinomial(
                mover_cs,
                requested_prb[src, :]
            ).T.astype(SimDType)

            split_sum = split_cs.sum(axis=1, dtype=SimDType)

            # Create new cohorts.
            for dst in range(self.ctx.nodes):
                if src == dst:
                    loc = self.locations[src]
                    loc.cohorts[0].compartments -= mover_cs
                else:
                    if split_sum[dst] > 0:
                        p = Cohort(split_cs[dst, :], src, return_tick)
                        loc = self.locations[dst]
                        loc.cohorts.append(p)

        self._normalize()

        if logger.isEnabledFor(DEBUG):
            logger.debug("moved %d", requested_sum.sum())

    def _apply_array(self, clause: ArrayClause, tick: Tick) -> None:
        requested = clause.apply(tick)
        np.fill_diagonal(requested, 0)
        self._apply_travel(clause, tick, requested)

    def _apply_row(self, clause: RowClause, tick: Tick) -> None:
        N = self.ctx.nodes
        requested = np.zeros((N, N), dtype=SimDType)
        for i in range(self.ctx.nodes):
            requested[i, :] = clause.apply(tick, i)
        np.fill_diagonal(requested, 0)
        self._apply_travel(clause, tick, requested)

    def _apply_cell(self, clause: CellClause, tick: Tick) -> None:
        N = self.ctx.nodes
        requested = np.empty((N, N), dtype=SimDType)
        for i, j in np.ndindex(self.ctx.nodes, self.ctx.nodes):
            requested[i, j] = clause.apply(tick, i, j)
        np.fill_diagonal(requested, 0)
        self._apply_travel(clause, tick, requested)
