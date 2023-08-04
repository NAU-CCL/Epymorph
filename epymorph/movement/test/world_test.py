# TODO: adapt or delete this file?
# import unittest
# from copy import deepcopy

# import numpy as np

# from epymorph.movement.world import HOME_TICK, Population, World


# # Test case constructor for World
# def w(initial_pops: list[int]) -> World:
#     return World.initialize([np.array([p], dtype=np.int_) for p in initial_pops])


# # Test case constructor for Population
# def p(count: int, dest: int, timer: int) -> Population:
#     return Population(np.array([count], dtype=np.int_), dest, timer)


# class TestPopulation(unittest.TestCase):
#     def test_normalize(self):
#         a = p(100, 0, HOME_TICK)
#         b = p(200, 0, HOME_TICK)
#         ab = p(300, 0, HOME_TICK)
#         c = p(50, 1, 2)
#         d = p(75, 2, 2)

#         exp1 = deepcopy([ab])
#         act1 = deepcopy([a, b])
#         Population.normalize(act1)
#         self.assertEqual(exp1, act1)

#         exp2 = deepcopy([ab, c, d])
#         act2 = deepcopy([b, d, c, a])
#         Population.normalize(act2)
#         self.assertEqual(exp2, act2)

#     def test_normalize_2(self):
#         a = p(100, 0, HOME_TICK)
#         b = p(100, 0, HOME_TICK)
#         c = p(100, 0, HOME_TICK)
#         d = p(100, 0, HOME_TICK)
#         e = p(100, 0, HOME_TICK)

#         exp = [p(500, 0, HOME_TICK)]
#         act = [a, b, c, d, e]
#         Population.normalize(act)

#         self.assertEqual(exp, act)
