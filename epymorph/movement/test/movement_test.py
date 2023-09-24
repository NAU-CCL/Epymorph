import unittest
from datetime import date

import numpy as np

from epymorph.clock import NEVER, Clock, Tick, TickDelta
from epymorph.context import Compartments, SimContext, SimDType
from epymorph.geo import StaticGeo
from epymorph.movement.basic import BasicEngine
from epymorph.movement.clause import RETURN, ArrayClause
from epymorph.movement.engine import Movement, MovementEngine
from epymorph.movement.hypercube import HypercubeEngine


def test_sim_context(pops: list[int]) -> SimContext:
    geo = StaticGeo.from_values({
        'population': np.array(pops, dtype=SimDType),
        'label': np.array([f'node{n}' for n in range(len(pops))], dtype=np.str_),
    })
    return SimContext(
        geo=geo,
        compartments=1,
        compartment_tags=[[]],
        events=0,
        param={},
        clock=Clock(date(2023, 1, 1), 50, [1.0]),
        rng=np.random.default_rng(1)
    )


def to_cs(pops: list[int]) -> Compartments:
    return np.array(pops, dtype=SimDType)[:, np.newaxis]


# Some clauses for testing


class NoClause(ArrayClause):
    name = 'Noop'
    returns = NEVER
    ctx: SimContext

    def __init__(self, ctx: SimContext):
        self.ctx = ctx
        self.movement_mask = np.full(ctx.compartments, True, dtype=bool)

    def predicate(self, tick: Tick) -> bool:
        return True

    def apply(self, tick: Tick) -> Compartments:
        # If we return zeros for all nodes, no movement should happen.
        return np.zeros((self.ctx.nodes, self.ctx.nodes), dtype=SimDType)


class TinyClause(ArrayClause):
    name = 'Tiny'
    returns = TickDelta(2, 0)
    ctx: SimContext

    def __init__(self, ctx: SimContext):
        self.ctx = ctx
        self.movement_mask = np.full(ctx.compartments, True, dtype=bool)

    def predicate(self, tick: Tick) -> bool:
        return True

    def apply(self, tick: Tick) -> Compartments:
        # The first pop sends 100 people to the second pop and that's it.
        req = np.zeros((self.ctx.nodes, self.ctx.nodes), dtype=SimDType)
        req[0, 1] = 100
        return req


class CrosswalkClause(ArrayClause):
    name = 'Crosswalk'
    returns = TickDelta(2, 0)
    ctx: SimContext

    def __init__(self, ctx: SimContext):
        self.ctx = ctx
        self.movement_mask = np.full(ctx.compartments, True, dtype=bool)

    def predicate(self, tick: Tick) -> bool:
        return True

    def apply(self, tick: Tick) -> Compartments:
        # Send 100 people to each other node.
        # Self-movement is ignored, so no need to zero it out.
        return np.full((self.ctx.nodes, self.ctx.nodes), 100, dtype=SimDType)


# Test cases


class TestNoMovement(unittest.TestCase):

    def _test(self, engine_cls: type[MovementEngine]):
        cs0 = [30_000, 20_000, 10_000]
        ctx = test_sim_context(cs0)

        movement = Movement(ctx.clock.taus, [NoClause(ctx)])
        engine = engine_cls(ctx, movement, to_cs(cs0))

        engine.apply(ctx.clock.ticks[0])

        cs1 = [loc.get_compartments().sum()
               for loc in engine.get_locations()]

        self.assertEqual(cs0, cs1)

    # Test case for BasicEngine
    def test_no_basic(self):
        self._test(BasicEngine)

    # Test case for HypercubeEngine
    def test_no_hypercube(self):
        self._test(HypercubeEngine)


class TestTinyMovement(unittest.TestCase):
    def _test(self, engine_cls: type[MovementEngine]):
        cs0 = [10000, 20000]
        ctx = test_sim_context(cs0)

        movement = Movement(ctx.clock.taus, [TinyClause(ctx), RETURN])

        engine = engine_cls(ctx, movement, to_cs(cs0))

        engine.apply(ctx.clock.ticks[0])

        self.assertEqual(
            engine.get_locals().squeeze().tolist(),
            [9900, 20000]
        )

        self.assertEqual(
            engine.get_travelers().squeeze().tolist(),
            [0, 100]
        )

        engine.apply(ctx.clock.ticks[1])

        self.assertEqual(
            engine.get_locals().squeeze().tolist(),
            [9800, 20000]
        )

        self.assertEqual(
            engine.get_travelers().squeeze().tolist(),
            [0, 200]
        )

        engine.apply(ctx.clock.ticks[2])

        self.assertEqual(
            engine.get_locals().squeeze().tolist(),
            [9800, 20000]
        )

        self.assertEqual(
            engine.get_travelers().squeeze().tolist(),
            [0, 200]
        )

    # Test case for BasicEngine
    def test_tiny_basic(self):
        self._test(BasicEngine)

    # Test case for HypercubeEngine
    def test_tiny_hypercube(self):
        self._test(HypercubeEngine)


class TestCrosswalkMovement(unittest.TestCase):

    def _test(self, engine_cls: type[MovementEngine]):
        cs0 = [25000, 15000, 10000, 50]
        ctx = test_sim_context(cs0)

        # Every tick, send 100 people to each other node.
        # There is a random factor involved in who winds up where,
        # but exactly 300 people should leave each location if possible.
        people = np.array(cs0, dtype=SimDType)
        expected_travelers = np.minimum(300, people, dtype=SimDType)
        expected_nontravelers = people - expected_travelers

        movement = Movement(ctx.clock.taus, [CrosswalkClause(ctx)])

        engine = engine_cls(ctx, movement, to_cs(cs0))
        engine.apply(ctx.clock.ticks[0])

        # Check that we have the expected number of locals remaining.
        a = engine.get_locals().squeeze()
        self.assertTrue(
            np.array_equal(a, expected_nontravelers),
            f"Locals array does not match expected. Received: {a}"
        )

        # Check that the sum of travelers from each location matches expected.
        # We don't want to create or delete people.
        a = engine.get_travelers_by_home().squeeze()
        self.assertTrue(
            np.array_equal(a, expected_travelers),
            f"Travelers array does not match expected. Received {a}"
        )

        # Check that we have the expected number of pops at each location.
        for loc in engine.get_locations():
            here_counts = loc.get_cohorts().sum(axis=1)
            # There is some difference between the engines to account for here:
            # "Basic" doesn't track cohorts without people in them (except for the home cohort!), but
            # "Hypercube" tracks all cohorts for all time.
            # So: the "effective number of cohorts" is the sum of the non-zero cohorts returned,
            # plus 1 if the first cohort ("home") is 0.
            n = np.count_nonzero(here_counts)
            if here_counts[0] == 0:
                n += 1
            self.assertEqual(
                4, n, f"Expected to find 4 pops at {loc.get_index()} but found {n}")

    # Test case for BasicEngine
    def test_crosswalk_basic(self):
        self._test(BasicEngine)

    # Test case for HypercubeEngine
    def test_crosswalk_hypercube(self):
        self._test(HypercubeEngine)


class TestSequenceMovement(unittest.TestCase):
    def _test(self, engine_cls: type[MovementEngine]):
        cs0 = [30000, 20000, 15000]
        ctx = test_sim_context(cs0)

        # Let's run a sequence of ticks, returning the `travelers_by_home`
        # once at the start and after each tick.
        def test_sequence():
            movement = Movement(ctx.clock.taus, [CrosswalkClause(ctx), RETURN])
            engine = engine_cls(ctx, movement, to_cs(cs0))

            # Return initial state
            yield engine.get_travelers_by_home().squeeze()  # tick -1

            # Run 5 ticks with Crosswalk and Return
            for t in ctx.clock.ticks[0:5]:
                engine.apply(t)
                yield engine.get_travelers_by_home().squeeze()

            # Then run 3 ticks with just Returns
            # (Of course we wouldn't normally change the movement model in the middle of a sim,
            # this is just for testing purposes.)
            engine.movement = Movement(ctx.clock.taus, [RETURN])

            for t in ctx.clock.ticks[5:8]:
                engine.apply(t)
                yield engine.get_travelers_by_home().squeeze()

        expected = [
            np.array([0, 0, 0]),  # tick -1
            np.array([200, 200, 200]),  # typical movement
            np.array([400, 400, 400]),
            np.array([400, 400, 400]),
            np.array([400, 400, 400]),
            np.array([400, 400, 400]),
            np.array([200, 200, 200]),  # returns only
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
        ]

        for i, (act, exp) in enumerate(zip(test_sequence(), expected)):
            self.assertTrue(
                np.array_equal(act, exp),
                f"Travelers array does not match expected on tick {i-1}. Received {act}"
            )

    # Test case for BasicEngine
    def test_sequence_basic(self):
        self._test(BasicEngine)

    # Test case for HypercubeEngine
    def test_sequence_hypercube(self):
        self._test(HypercubeEngine)
