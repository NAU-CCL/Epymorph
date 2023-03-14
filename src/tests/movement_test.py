import unittest
from copy import deepcopy
from datetime import date

import numpy as np

import movement as M
from clock import Clock, Tick
from sim_context import SimContext
from tests.world_test import p, w
from world import Location, Timer, World


class TestNoopMovement(unittest.TestCase):
    def test_noop(self):
        clause = M.Noop()
        tick = Tick(0, 0, date(2023, 1, 1), 0, np.double(0))
        sim = SimContext(3, 3, 3, ['A', 'B', 'C'], np.random.default_rng(1))
        ini = w([30000, 20000, 10000])
        exp = deepcopy(ini)
        act = clause.apply(sim, ini, tick)
        self.assertEqual(act, exp)


class TestReturnMovement(unittest.TestCase):
    def test_return(self):
        clause = M.Return()
        sim = SimContext(3, 3, 3, ['A', 'B', 'C'], np.random.default_rng(1))
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

        clock = Clock(date(2023, 1, 1), 3, [np.double(1)])

        act1 = clause.apply(sim, ini1, clock.ticks[0])
        self.assertEqual(act1, exp1)

        act2 = clause.apply(sim, act1, clock.ticks[1])
        self.assertEqual(act2, exp2)

        act3 = clause.apply(sim, act2, clock.ticks[2])
        self.assertEqual(act3, exp3)


class TestFixedCrosswalkMovement(unittest.TestCase):
    def test_fcross(self):
        sim = SimContext(3, 3, 4, ['A', 'B', 'C', 'D'],
                         np.random.default_rng(1))
        clause = M.FixedCommuteMatrix(
            duration=2, commuters=np.full((sim.nodes, sim.nodes), 100))

        ini = World([
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

        clock = Clock(date(2023, 1, 1), 1, [np.double(1)])

        act = clause.apply(sim, ini, clock.ticks[0])
        self.assertEqual(act, exp)


class TestSequenceMovement(unittest.TestCase):
    def test_fcross_return(self):
        sim = SimContext(3, 3, 3, ['A', 'B', 'C'], np.random.default_rng(1))
        curr = w([30000, 20000, 15000])
        exp = deepcopy(curr)

        def count_pops(world: World) -> int:
            num_pops = 0
            for loc in world.locations:
                for pop in loc.pops:
                    num_pops += 1
            return num_pops

        clause = M.Sequence([
            M.FixedCommuteMatrix(duration=2, commuters=np.full(
                (sim.nodes, sim.nodes), 100)),
            M.Return()
        ])

        clock = Clock(date(2023, 1, 1), 22, [np.double(1)])
        running_pops = [count_pops(curr)]
        for i in range(1, 20):
            clause.apply(sim, curr, clock.ticks[i])
            running_pops.append(count_pops(curr))

        # t=0: start with 3 pops (3)
        # t=1: each pop gains 2 subpops (9)
        # t=2: each pop gains 2 subpops (15)
        # t>2: each pop gains and loses 2 subpops (15)
        exp_pops = [3, 9] + ([15] * 18)

        self.assertEqual(running_pops, exp_pops)

        # now do only returns for 2 steps and verify everyone goes home
        M.Return().apply(sim, curr, clock.ticks[20])
        M.Return().apply(sim, curr, clock.ticks[21])

        self.assertEqual(curr, exp)


if __name__ == '__main__':
    unittest.main()
