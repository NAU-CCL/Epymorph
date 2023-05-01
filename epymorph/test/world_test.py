import unittest
from copy import deepcopy

import numpy as np

from epymorph.world import Population, World, _home_tick


# Test case constructor for World
def w(initial_pops: list[int]) -> World:
    return World.initialize([np.array([p], dtype=np.int_) for p in initial_pops])


# Test case constructor for Population
def p(count: int, dest: int, timer: int) -> Population:
    return Population(np.array([count], dtype=np.int_), dest, timer)


class TestPopulation(unittest.TestCase):
    def test_normalize(self):
        a = p(100, 0, _home_tick)
        b = p(200, 0, _home_tick)
        ab = p(300, 0, _home_tick)
        c = p(50, 1, 2)
        d = p(75, 2, 2)

        exp1 = deepcopy([ab])
        act1 = deepcopy([a, b])
        Population.normalize(act1)
        self.assertEqual(exp1, act1)

        exp2 = deepcopy([ab, c, d])
        act2 = deepcopy([b, d, c, a])
        Population.normalize(act2)
        self.assertEqual(exp2, act2)
