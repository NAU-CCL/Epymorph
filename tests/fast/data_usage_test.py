# ruff: noqa: PT009,PT027
import unittest

from epymorph.data_usage import AvailableDataEstimate, estimate_total


class DataUsageTest(unittest.TestCase):
    def test_estimate_total_1(self):
        e1 = AvailableDataEstimate(
            name="Thing One",
            cache_key="thing-one",
            new_network_bytes=179,
            max_bandwidth=None,
            new_cache_bytes=89,
            total_cache_bytes=307,
        )

        e2 = AvailableDataEstimate(
            name="Thing Two",
            cache_key="thing-two",
            new_network_bytes=283,
            max_bandwidth=7,
            new_cache_bytes=59,
            total_cache_bytes=619,
        )

        actual = estimate_total(
            estimates=[e1, e2],
            max_bandwidth=13,
        )

        self.assertEqual(actual.new_network_bytes, 179 + 283)
        self.assertEqual(actual.new_cache_bytes, 89 + 59)
        self.assertEqual(actual.total_cache_bytes, 307 + 619)
        self.assertAlmostEqual(actual.download_time, (179 / 13) + (283 / 7))

    def test_estimate_total_2(self):
        e1 = AvailableDataEstimate(
            name="Thing One A",
            cache_key="same-key",
            new_network_bytes=179,
            max_bandwidth=None,
            new_cache_bytes=89,
            total_cache_bytes=307,
        )

        e2 = AvailableDataEstimate(
            name="Thing Two",
            cache_key="different-key",
            new_network_bytes=283,
            max_bandwidth=7,
            new_cache_bytes=59,
            total_cache_bytes=619,
        )

        # because this duplicates the key of e1,
        # it will not be counted towards the total
        e3 = AvailableDataEstimate(
            name="Thing One B",
            cache_key="same-key",
            new_network_bytes=179,
            max_bandwidth=None,
            new_cache_bytes=89,
            total_cache_bytes=307,
        )

        actual = estimate_total(
            estimates=[e1, e2, e3],
            max_bandwidth=13,
        )

        self.assertEqual(actual.new_network_bytes, 179 + 283)
        self.assertEqual(actual.new_cache_bytes, 89 + 59)
        self.assertEqual(actual.total_cache_bytes, 307 + 619)
        self.assertAlmostEqual(actual.download_time, (179 / 13) + (283 / 7))
