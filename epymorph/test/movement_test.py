import unittest
from copy import deepcopy
from datetime import date

import numpy as np

from epymorph.clock import Clock, TickDelta
from epymorph.context import SimContext
from epymorph.movement_clause import (FunctionalClause, Noop, Predicates,
                                      Return, Sequence)
from epymorph.test.world_test import p, w
from epymorph.util import constant
from epymorph.world import HOME_TICK, Location, World


def test_sim_context(num_nodes: int) -> SimContext:
    return SimContext(
        nodes=num_nodes,
        labels=[f'node{n}' for n in range(num_nodes)],
        geo={},
        compartments=1,
        compartment_tags=[[]],
        events=0,
        param={},
        clock=Clock.init(date(2023, 1, 1), 50, [np.double(1)]),
        rng=np.random.default_rng(1)
    )


class TestNoopMovement(unittest.TestCase):
    def test_noop(self):
        clause = Noop()
        ctx = test_sim_context(3)
        exp = w([30000, 20000, 10000])
        act = deepcopy(exp)
        clause.apply(act, ctx.clock.ticks[0])
        self.assertEqual(act, exp)


class TestReturnMovement(unittest.TestCase):
    def test_return(self):
        ctx = test_sim_context(3)
        clause = Return(ctx)
        ini1 = World([
            Location(0, [
                p(25000, 0, HOME_TICK),
                p(2500, 1, 1),
            ]),
            Location(1, [
                p(15000, 1, HOME_TICK),
                p(5000, 0, 2),
            ]),
            Location(2, [
                p(10000, 2, HOME_TICK),
                p(2500, 1, 1),
            ]),
        ])

        exp1 = deepcopy(ini1)
        exp2 = World([
            Location(0, [
                p(25000, 0, HOME_TICK),
            ]),
            Location(1, [
                p(20000, 1, HOME_TICK),
                p(5000, 0, 2),
            ]),
            Location(2, [
                p(10000, 2, HOME_TICK),
            ]),
        ])
        exp3 = World([
            Location(0, [
                p(30000, 0, HOME_TICK),
            ]),
            Location(1, [
                p(20000, 1, HOME_TICK),
            ]),
            Location(2, [
                p(10000, 2, HOME_TICK),
            ]),
        ])

        act1 = deepcopy(ini1)
        clause.apply(act1, ctx.clock.ticks[0])
        self.assertEqual(act1, exp1)

        act2 = deepcopy(act1)
        clause.apply(act2, ctx.clock.ticks[1])
        self.assertEqual(act2, exp2)

        act3 = deepcopy(act2)
        clause.apply(act3, ctx.clock.ticks[2])
        self.assertEqual(act3, exp3)


class TestCrosswalkMovement(unittest.TestCase):
    def test_crosswalk(self):
        ctx = test_sim_context(4)

        # Every tick, send 100 people to each other node.
        # There is a random factor involved in who winds up where,
        # but exactly 300 people should leave each location if possible.
        # Self-movement is ignored, so no need to zero it out.
        clause = FunctionalClause(
            ctx=ctx,
            name="Crosswalk",
            predicate=Predicates.everyday(),
            compartment_tag_predicate=constant(True),
            clause_function=lambda tick, src: np.array([100, 100, 100, 100]),
            returns=TickDelta(2, 0),
        )

        people = np.array([25000, 15000, 10000, 50], dtype=int)
        expected_nontravelers = np.maximum(0, people - 300)
        expected_travelers = np.minimum(300, people)

        world = World([
            Location(i, [p(people[i], i, HOME_TICK)])
            for i in range(4)
        ])

        clause.apply(world, ctx.clock.ticks[0])

        # Check that we have the expected number of locals remaining.
        self.assertTrue(
            np.array_equal(
                world.all_locals(),  # returns in column form
                expected_nontravelers.reshape((4, 1))
            ),
            f"Locals array does not match expected. Received: {world.all_locals()}"
        )

        # For each location:
        for i in range(4):
            # Check that we have the expected number of pops.
            n = len(world.locations[i].pops)
            self.assertEqual(
                4, n, f"Expected to find 4 pops at {i} but found {n}")

            # Check that the sum of all travelers from this node matches.
            # We don't want to create or delete people.
            e = expected_travelers[i]
            t = sum(
                p.compartments.sum()
                for l in world.locations
                for p in l.pops
                if p.return_location == i and p.return_tick != HOME_TICK
            )
            self.assertEqual(
                e, t, f"Expected to find {e} travelers from {i} but found {t}")

            # Also check the sum of all people from each location.
            e = people[i]
            t = sum(
                p.compartments.sum()
                for l in world.locations
                for p in l.pops
                if p.return_location == i
            )
            self.assertEqual(
                e, t, f"Expected to find {e} total people from {i} but found {t}")


class TestSequenceMovement(unittest.TestCase):
    def test_sequence(self):
        ctx = test_sim_context(3)
        clock = ctx.clock
        world = w([30000, 20000, 15000])
        initial = deepcopy(world)

        def count_pops(world: World) -> int:
            return sum([len(loc.pops) for loc in world.locations])

        return_clause = Return(ctx)
        clause = Sequence([
            FunctionalClause(
                ctx=ctx,
                name="Crosswalk",
                predicate=Predicates.everyday(),
                compartment_tag_predicate=constant(True),
                clause_function=lambda tick, src: np.array([100, 100, 100]),
                returns=TickDelta(2, 0)
            ),
            return_clause
        ])

        running_pops = [count_pops(world)]
        for i in range(0, 20):
            clause.apply(world, clock.ticks[i])
            running_pops.append(count_pops(world))

        # t=0: start with 3 pops (3)
        # t=1: each pop gains 2 subpops (9)
        # t=2: each pop gains 2 subpops (15)
        # t>2: each pop gains and loses 2 subpops (15)
        exp_pops = [3, 9] + ([15] * 19)

        self.assertEqual(running_pops, exp_pops)

        # now do only returns for 2 steps and verify everyone goes home
        return_clause.apply(world, clock.ticks[20])
        return_clause.apply(world, clock.ticks[21])

        self.assertEqual(world, initial)
