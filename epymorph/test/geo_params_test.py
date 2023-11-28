import unittest
from unittest.mock import MagicMock

from epymorph.geo.abstract import _ProxyGeo
from epymorph.geo.geo import Geo


class TestProxyGeo(unittest.TestCase):

    def test_singleton_instance(self):
        proxy1 = _ProxyGeo()
        proxy2 = _ProxyGeo()
        self.assertIs(proxy1, proxy2, "Two instances of ProxyGeo should be the same")

    def test_set_actual_geo(self):
        proxy = _ProxyGeo()
        mock_geo = MagicMock(spec=Geo)
        proxy.set_actual_geo(mock_geo)
        self.assertIs(proxy._actual_geo, mock_geo, "Actual geo should be set correctly")

    def test_get_item(self):
        proxy = _ProxyGeo()
        data = {'key1': 123, 'key2': 456}
        proxy.set_actual_geo(data)

        self.assertEqual(proxy['key1'], 123,
                         "Accessing data should return the correct value for key1")
        self.assertEqual(proxy['key2'], 456,
                         "Accessing data should return the correct value for key2")


if __name__ == '__main__':
    unittest.main()
