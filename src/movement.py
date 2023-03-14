import functools
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence as abcSequence
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from clock import Tick
from geo import ParamNN
from sim_context import SimContext
from util import is_square
from world import Population, Timer, World


class Clause(ABC):
    """Movement clause base class. Transforms World at a given time step."""

    log: logging.Logger

    @abstractmethod
    def apply(self, sim: SimContext, world: World, tick: Tick) -> World:
        pass


class MatrixClause(Clause):
    """
    Abstract movement clause; assumes each source is going to emit a certain number
    of movers from the local population per tick to each other destination, and we 
    will use a multinomial draw to determine which compartments they came from.
    """

    def __init__(self, duration: int):
        self.duration = duration

    @abstractmethod
    def _calc_movers_from(self, sim: SimContext, src_idx: int, tick: Tick) -> NDArray[np.int_]:
        pass

    def apply(self, sim: SimContext, world: World, tick: Tick) -> World:
        total_movers = np.int_(0)
        for k, src in enumerate(world.locations):
            movers_by_dest = self._calc_movers_from(sim, k, tick)
            assert movers_by_dest[k] == 0, "MatrixClause must not include self-movement."
            num_movers = np.sum(movers_by_dest, dtype=np.int_)

            locals = src.locals()
            num_locals = locals.total
            if num_locals < num_movers:
                continue

            # Draw compartments for all destinations.
            cs_probability = locals.compartments / num_locals
            cs_by_dest = sim.rng.multinomial(movers_by_dest, cs_probability)
            # Column sum is everyone leaving by compartment.
            locals.compartments -= np.sum(cs_by_dest, axis=0)

            for n, dest in enumerate(world.locations):
                cs = cs_by_dest[n]
                if np.sum(cs) == 0:
                    continue
                p = Population(cs, k, tick.time + self.duration)
                dest.pops.append(p)

            total_movers += num_movers
        self.__class__.log.debug(
            f"moved {total_movers} (tic {tick.time}, {tick.tausum})")
        world.normalize()
        return world


class Noop(Clause):
    """A movement clause that does nothing."""
    log = logging.getLogger('movement.Noop')

    def apply(self, sim: SimContext, world: World, tick: Tick) -> World:
        return world


class Return(Clause):
    """A movement clause to return people to their home if it's time."""
    log = logging.getLogger('movement.Return')

    def apply(self, sim: SimContext, world: World, tick: Tick) -> World:
        total_movers = np.int_(0)
        new_pops: list[list[Population]] = [[] for _ in range(sim.nodes)]
        for loc in world.locations:
            for pop in loc.pops:
                if pop.timer == tick.time:
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
        Return.log.debug(
            f"moved {total_movers} (tic {tick.time}, {tick.tausum})")
        return world


class FixedCommuteMatrix(MatrixClause):
    """Movement clause which distributes commuters exactly using an NxN commuter matrix."""
    log = logging.getLogger('movement.FixedCommuteMatrix')

    def __init__(self, duration: int, commuters: NDArray[np.int_]):
        super().__init__(duration)
        assert is_square(commuters), "Commuters matrix must be square."
        self.commuters = commuters

    def _calc_movers_from(self, sim: SimContext, src_idx: int, tick: Tick) -> NDArray[np.int_]:
        movers = self.commuters[src_idx]
        movers[src_idx] = 0
        return movers


class StochasticCommuteMatrix(MatrixClause):
    """Movement clause which randomizes movement with close adherence to an NxN commuter matrix."""
    log = logging.getLogger('movement.StochasticCommuteMatrix')

    def __init__(self, duration: int, commuters: NDArray[np.int_], move_control: float):
        super().__init__(duration)
        assert 0 <= move_control <= 1.0, "Move Control must be in the range [0,1]."
        assert is_square(commuters), "Commuters matrix must be square."
        # Total commuters which live in one state (first axis) and work in another (second axis).
        self.commuters = commuters
        # Total commuters living in each state.
        self.commuters_by_state = commuters.sum(axis=1, dtype=np.int_)
        # Commuters as a ratio to the total commuters living in that state.
        self.commuting_prob = commuters / \
            commuters.sum(axis=1, keepdims=True, dtype=np.int_)
        self.move_control = move_control

    def _calc_movers_from(self, sim: SimContext, src_idx: int, tick: Tick) -> NDArray[np.int_]:
        # Binomial draw with probability `move_control` to modulate total number of commuters.
        typical = self.commuters_by_state[src_idx]
        current = sim.rng.binomial(typical, self.move_control)
        # Multinomial draw for destination, then zero out same-state.
        movers = sim.rng.multinomial(current, self.commuting_prob[src_idx])
        movers[src_idx] = 0
        return movers


class RandomDispersersMatrix(MatrixClause):
    """Movement clause which emits a randomized movement proportional to an NxN commuter matrix."""
    log = logging.getLogger('movement.RandomDisperersMatrix')

    def __init__(self, duration: int, commuters: NDArray[np.int_], theta: float):
        super().__init__(duration)
        assert 0 <= theta, "Theta must be not less than zero."
        assert is_square(commuters), "Commuters matrix must be square."
        self.commuters = commuters
        # Pre-compute the average commuters between node pairs.
        avg = np.zeros(commuters.shape)
        for i in range(commuters.shape[0]):
            for j in range(i + 1, commuters.shape[1]):
                nbar = (commuters[i, j] + commuters[j, i]) // 2
                avg[i, j] = nbar
                avg[j, i] = nbar
        self.commuters_avg = avg
        self.theta = theta

    def _calc_movers_from(self, sim: SimContext, src_idx: int, tick: Tick) -> NDArray[np.int_]:
        movers = np.zeros(sim.nodes, dtype=np.int_)
        for dst_idx in range(sim.nodes):
            if src_idx == dst_idx:
                continue
            n = self.commuters_avg[src_idx, dst_idx]
            movers[dst_idx] = sim.rng.poisson(n * self.theta)
        return movers


class Sequence(Clause):
    """A movement clause which executes a list of child clauses in sequence."""
    log = logging.getLogger('movement.Sequence')

    def __init__(self, clauses: abcSequence[Clause]):
        self.clauses = clauses

    def apply(self, sim: SimContext, world: World, tick: Tick) -> World:
        return functools.reduce(
            lambda curr, f: f.apply(sim, curr, tick),
            self.clauses,
            world
        )


class Step(NamedTuple):
    tau: np.double
    clause: Clause


class Movement:
    def __init__(self, steps: list[Step]):
        assert len(steps) > 0, "Must supply at least one movement step."
        self.taus = [s.tau for s in steps]
        assert np.sum(self.taus) == np.double(1), "Tau steps must sum to 1."
        self.steps = steps
