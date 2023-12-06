# pylint: disable=missing-docstring
import unittest
from unittest.mock import MagicMock

from epymorph.geo.abstract import ProxyUnsetException, geo, proxy_geo
from epymorph.geo.geo import Geo


class TestProxyGeo(unittest.TestCase):

    def test_context_property(self):
        with self.assertRaises(ProxyUnsetException):
            geo.nodes

        mock_geo = MagicMock(spec=Geo)
        mock_geo.nodes = 6

        with proxy_geo(mock_geo):
            self.assertEqual(geo.nodes, 6,
                             "Accessing nodes property should return the correct value.")

    def test_context_data(self):
        mock_geo = MagicMock(spec=Geo)
        mock_val = {'key1': 123, 'key2': 456}
        mock_geo.__getitem__.side_effect = lambda key: mock_val[key]

        with proxy_geo(mock_geo):
            self.assertEqual(geo['key1'], 123,
                             "Accessing data should return the correct value for key1")
            self.assertEqual(geo['key2'], 456,
                             "Accessing data should return the correct value for key2")


if __name__ == '__main__':
    unittest.main()
