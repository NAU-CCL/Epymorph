from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from math import ceil
from typing import Iterable

import numpy as np
import psutil
from numpy.typing import NDArray

import epymorph.movement.world as world
from epymorph.clock import Tick
from epymorph.context import Compartments, SimContext, SimDType
from epymorph.movement.clause import (ArrayClause, CellClause, ReturnClause,
                                      RowClause, TravelClause)
from epymorph.movement.engine import Movement, MovementEngine


def to_gib(bytes: int) -> float:
    return bytes / (1024 * 1024 * 1024)


def _mem_check(ctx: SimContext) -> None:
    """Will this simulation fit in memory?"""
    T, N, C, _ = ctx.TNCE
    required = np.dtype(SimDType).itemsize * ((T * N * N * C) + (2 * N * C))
    available = psutil.virtual_memory().available

    if available < required:
        msg = f"""\
Insufficient memory: the simulation is too large (using HypercubeEngine).
  T:{T}, N:{N}, C:{C} requires {to_gib(required):.1f} GiB;
  available memory is {to_gib(available):.1f} GiB"""
        raise Exception(msg)


def batches(items, workers) -> list[range]:
    size = int(ceil(items / workers))
    return [range(i * size, min(items, (i+1) * size))
            for i in range(workers)]


class HypercubeEngine(MovementEngine):
    # TODO: document how time offset/frontier work
    time_offset: int
    time_frontier: int
    home: NDArray[SimDType]  # NxC
    vstr: NDArray[SimDType]  # NxC
    ldgr: NDArray[SimDType]  # TxNxNxC
    locations: list[HLocation]

    def __init__(self, ctx: SimContext, movement: Movement, initial_compartments: list[Compartments]):
        _mem_check(ctx)
        super().__init__(ctx, movement, initial_compartments)
        T, N, C, _ = ctx.TNCE
        self.time_offset = 0
        self.time_frontier = 0
        self.home = np.array(initial_compartments, dtype=SimDType)
        self.vstr = np.zeros((N, C), dtype=SimDType)
        self.ldgr = np.zeros((T, N, N, C), dtype=SimDType)
        # each location has a set of views to the main arrays
        self.locations = [self.HLocation(self, index)
                          for index in range(ctx.nodes)]
        
        self.workers = min(N, 4)
        self.executor = ThreadPoolExecutor(self.workers)
        # TODO: seed_seq should probably be in context, rather than this approach
        seed_seq = np.random.SeedSequence(self.ctx.rng.integers(10_000_000))
        self.rngs = [np.random.default_rng(s) for s in seed_seq.spawn(self.workers)]

        
    def close(self) -> None:
        self.executor.shutdown(wait=True)


    # Implement World

    def get_locations(self) -> Iterable[HLocation]:
        return self.locations

    def get_locals(self) -> Compartments:
        return self.home

    def get_travelers(self) -> Compartments:
        return self.vstr

    def get_travelers_by_home(self) -> Compartments:
        return self.ldgr[self.time_offset:self.time_frontier].sum(axis=(0, 2), dtype=SimDType)

    # Implement MovementEngine

    def apply(self, tick: Tick) -> None:
        super().apply(tick)
        self.time_offset=tick.index + 1

    def _apply_return(self, clause: ReturnClause, tick: Tick) -> None:
        self.home += self.ldgr[tick.index, :, :, :].sum(axis=1, dtype=SimDType)
        self.vstr -= self.ldgr[tick.index, :, :, :].sum(axis=0, dtype=SimDType)

    def _apply_travel(self, clause: TravelClause, tick: Tick, requested_movers: NDArray[SimDType]) -> None:
        T, N, C, E = self.ctx.TNCE
        mover_cs = np.empty((N, C), dtype=SimDType)
        split_cs = np.empty((N, N, C), dtype=SimDType)
        
        # requested_movers (N,N)
        available_movers = self.home * clause.movement_mask  # (N,C)

        def process(rng: np.random.Generator, work_range: range) -> None:
            for src in work_range:
                available_sum = available_movers[src, :].sum(dtype=SimDType) # S
                requested_src = requested_movers[src, :] # (N,)
                requested_sum = requested_src.sum(dtype=SimDType) # S

                # If requested total is greater than the total available,
                # use mvhg to select as many as possible.
                if requested_sum > available_sum:
                    requested_src = self.ctx.rng.multivariate_hypergeometric(
                        colors=requested_src,
                        nsample=available_sum
                    ).astype(SimDType)
                    requested_sum = available_sum

                # Select which individuals will be leaving this node. (C,)
                mover_cs[src, :] = rng.multivariate_hypergeometric(
                    colors=available_movers[src, :],
                    nsample=requested_sum
                ).astype(SimDType)

                # Select which location they are each going to. (N,C)
                # (Each row contains the compartments for a destination.)
                split_cs[src, :, :] = rng.multinomial(
                    n=mover_cs[src, :],
                    # The probability a mover from a src will go to a dst. (N,)
                    pvals=requested_src / max(1, requested_sum)
                ).T.astype(SimDType)

        futures = [self.executor.submit(process, self.rngs[i], work_range)
                   for i, work_range in enumerate(batches(N, self.workers))]
        wait(futures)

        return_tick = self.ctx.clock.tick_plus(tick, clause.returns)
        # Subtract from home.
        self.home -= mover_cs
        # Add to vstr and ldgr.
        self.vstr += split_cs.sum(axis=0)
        self.ldgr[return_tick, :, :, :] += split_cs
        # Update frontier.
        self.time_frontier = max(self.time_frontier, return_tick + 1)

    def _apply_array(self, clause: ArrayClause, tick: Tick) -> None:
        requested = clause.apply(tick)
        np.fill_diagonal(requested, 0)
        self._apply_travel(clause, tick, requested)

    def _apply_row(self, clause: RowClause, tick: Tick) -> None:
        requested = np.zeros((self.ctx.nodes, self.ctx.nodes), dtype=SimDType)
        for i in range(self.ctx.nodes):
            requested[:, i] = clause.apply(tick, i)
        np.fill_diagonal(requested, 0)
        self._apply_travel(clause, tick, requested)

    def _apply_cell(self, clause: CellClause, tick: Tick) -> None:
        requested: NDArray[SimDType] = np.fromfunction(
            lambda i, j: clause.apply(tick, i, j),  # type: ignore
            shape=(self.ctx.nodes, self.ctx.nodes),
            dtype=SimDType)
        np.fill_diagonal(requested, 0)
        self._apply_travel(clause, tick, requested)

    class HLocation(world.Location):
        """HLocation is essentially a fancy accessor for the engine's data; it doesn't keep any data itself."""

        engine: HypercubeEngine
        index: int

        home: NDArray[SimDType]
        """(C,) the people whose home is here who are currently here"""

        vstr: NDArray[SimDType]
        """(C,) the people whose home is elsewhere who are currently here"""

        ldgr: NDArray[SimDType]
        """(T,N,C) the ledger for all visitors to this location"""

        def __init__(self, engine: HypercubeEngine, index: int):
            self.engine = engine
            self.index = index
            self.home = engine.home[index, :]
            self.vstr = engine.vstr[index, :]
            self.ldgr = engine.ldgr[:, :, index, :]

        def get_index(self) -> int:
            return self.index

        def get_compartments(self) -> Compartments:
            return self.home + self.vstr

        def _ldgr_slice(self) -> tuple[slice, int]:
            t_start = self.engine.time_offset
            t_end = self.engine.time_frontier
            return slice(t_start, t_end, 1), t_end - t_start

        # TODO: maybe there's a smarter API design here, that doesn't force us to make array copies
        def get_cohorts(self) -> Compartments:
            T, N, C, _ = self.engine.ctx.TNCE
            ts, dt = self._ldgr_slice()
            cohorts = self.ldgr[ts, :, :]
            cohorts = cohorts.reshape((dt * N, C))
            cohorts = np.insert(cohorts, 0, self.home, axis=0)
            return cohorts

        def update_cohorts(self, deltas: Compartments) -> None:
            T, N, C, _ = self.engine.ctx.TNCE
            ts, dt = self._ldgr_slice()
            home_deltas = deltas[0, :]
            ldgr_deltas = deltas[1:, :].reshape((dt, N, C))
            vstr_deltas = ldgr_deltas.sum(axis=(0, 1), dtype=SimDType)
            self.home += home_deltas
            self.vstr += vstr_deltas
            self.ldgr[ts, :, :] += ldgr_deltas
