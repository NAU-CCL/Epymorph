import unittest
from copy import deepcopy
from datetime import date

import numpy as np

import movement as M
from clock import Clock, TickDelta
from sim_context import SimContext
from tests.world_test import p, w
from world import Location, Timer, World


def testSimContext(num_nodes: int) -> SimContext:
    clock = Clock.init(date(2023, 1, 1), 50, [np.double(1)])
    labels = [f'node{n}' for n in range(num_nodes)]
    return SimContext(3, 3, num_nodes, labels, clock, np.random.default_rng(1))


class TestNoopMovement(unittest.TestCase):
    def test_noop(self):
        clause = M.Noop()
        sim = testSimContext(3)
        exp = w([30000, 20000, 10000])
        act = deepcopy(exp)
        clause.apply(sim, act, sim.clock.ticks[0])
        self.assertEqual(act, exp)


class TestReturnMovement(unittest.TestCase):
    def test_return(self):
        clause = M.Return()
        sim = testSimContext(3)
        ini1 = World([
            Location(0, [
                p(25000, 0, Timer.Home),
                p(2500, 1, 1),
            ]),
            Location(1, [
                p(15000, 1, Timer.Home),
                p(5000, 0, 2),
            ]),
            Location(2, [
                p(10000, 2, Timer.Home),
                p(2500, 1, 1),
            ]),
        ])

        exp1 = deepcopy(ini1)
        exp2 = World([
            Location(0, [
                p(25000, 0, Timer.Home),
            ]),
            Location(1, [
                p(20000, 1, Timer.Home),
                p(5000, 0, 2),
            ]),
            Location(2, [
                p(10000, 2, Timer.Home),
            ]),
        ])
        exp3 = World([
            Location(0, [
                p(30000, 0, Timer.Home),
            ]),
            Location(1, [
                p(20000, 1, Timer.Home),
            ]),
            Location(2, [
                p(10000, 2, Timer.Home),
            ]),
        ])

        act1 = deepcopy(ini1)
        clause.apply(sim, act1, sim.clock.ticks[0])
        self.assertEqual(act1, exp1)

        act2 = deepcopy(act1)
        clause.apply(sim, act2, sim.clock.ticks[1])
        self.assertEqual(act2, exp2)

        act3 = deepcopy(act2)
        clause.apply(sim, act3, sim.clock.ticks[2])
        self.assertEqual(act3, exp3)


class TestCrosswalkMovement(unittest.TestCase):
    def test_crosswalk(self):
        sim = testSimContext(4)

        # Every tick, send 100 people to each other node.
        # Self-movement is ignored, so no need to zero it out.
        clause = M.GeneralClause(
            name="Crosswalk",
            predicate=M.Predicates.everyday(),
            returns=TickDelta(2, 0),
            equation=lambda *_: np.array([100, 100, 100, 100])
        )

        # The last node doesn't have enough locals, so it will receive movers but not send any.
        act = World([
            Location(0, [
                p(25000, 0, Timer.Home),
            ]),
            Location(1, [
                p(15000, 1, Timer.Home),
            ]),
            Location(2, [
                p(10000, 2, Timer.Home),
            ]),
            Location(3, [
                p(50, 3, Timer.Home),
            ]),
        ])

        exp = World([
            Location(0, [
                p(24700, 0, Timer.Home),
                p(100, 1, 2),
                p(100, 2, 2),
            ]),
            Location(1, [
                p(14700, 1, Timer.Home),
                p(100, 0, 2),
                p(100, 2, 2),
            ]),
            Location(2, [
                p(9700, 2, Timer.Home),
                p(100, 0, 2),
                p(100, 1, 2),
            ]),
            Location(3, [
                p(50, 3, Timer.Home),
                p(100, 0, 2),
                p(100, 1, 2),
                p(100, 2, 2),
            ]),
        ])

        clause.apply(sim, act, sim.clock.ticks[0])
        self.assertEqual(act, exp)


class TestSequenceMovement(unittest.TestCase):
    def test_sequence(self):
        sim = testSimContext(3)
        world = w([30000, 20000, 15000])
        initial = deepcopy(world)

        def count_pops(world: World) -> int:
            return sum([len(loc.pops) for loc in world.locations])

        clause = M.Sequence([
            M.GeneralClause(
                name="Crosswalk",
                predicate=M.Predicates.everyday(),
                returns=TickDelta(2, 0),
                equation=lambda *_: np.array([100, 100, 100])
            ),
            M.Return()
        ])

        running_pops = [count_pops(world)]
        for i in range(0, 20):
            clause.apply(sim, world, sim.clock.ticks[i])
            running_pops.append(count_pops(world))

        # t=0: start with 3 pops (3)
        # t=1: each pop gains 2 subpops (9)
        # t=2: each pop gains 2 subpops (15)
        # t>2: each pop gains and loses 2 subpops (15)
        exp_pops = [3, 9] + ([15] * 19)

        self.assertEqual(running_pops, exp_pops)

        # now do only returns for 2 steps and verify everyone goes home
        M.Return().apply(sim, world, sim.clock.ticks[20])
        M.Return().apply(sim, world, sim.clock.ticks[21])

        self.assertEqual(world, initial)
