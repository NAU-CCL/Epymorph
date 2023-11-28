from abc import ABC, abstractmethod

from numpy.typing import NDArray

from epymorph.geo.geo import Geo


class ProxyGeoProtocol(ABC):
    @abstractmethod
    def __getitem__(self, key) -> NDArray:
        pass


class _ProxyGeo(ProxyGeoProtocol):
    _actual_geo: Geo
    _instance = None

    def __new__(cls):
        # Ensure only one instance is created
        if cls._instance is None:
            cls._instance = super(_ProxyGeo, cls).__new__(cls)
        return cls._instance

    def __getitem__(self, key):
        if self._actual_geo is None:
            raise ValueError("Geo infomration has not been set.")
        return self._actual_geo[key]

    def set_actual_geo(self, geo):
        """
        Set the actual_geo for the Singleton instance. For internal use only.
        """
        self._actual_geo = geo


proxy: ProxyGeoProtocol = _ProxyGeo()
