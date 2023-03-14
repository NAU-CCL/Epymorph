import unittest

from example import sum


class TestSum(unittest.TestCase):
    def test_sum(self):
        self.assertEquals(3, sum(1, 2))
