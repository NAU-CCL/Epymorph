"""
Proxy instances to epymorph models. Intended for use when evalution must be deferred,
e.g., when constructing simulation parameters as Python functions. We may wish to
define a function which depends on geo values, but before the geo itself has been defined.
"""
from abc import abstractmethod
from contextlib import contextmanager
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray


class ProxyUnsetException(Exception):
    """Exception for when a proxy is accessed but does not currently have a valid subject."""


class GeoProtocol(Protocol):
    """A Geo-compatible protocol for proxy usage."""

    nodes: int
    """The number of nodes in this Geo."""

    labels: NDArray[np.str_]
    """The labels for every node in this geo."""

    @abstractmethod
    def __getitem__(self, name: str) -> NDArray:
        pass


class _ProxyGeoSingleton:
    _subject: GeoProtocol | None = None

    def set_actual_geo(self, actual_geo: GeoProtocol | None):
        """Sets or clears the proxy subject. Internal use only please."""
        self._subject = actual_geo

    @property
    def nodes(self) -> int:
        """Proxy to `geo.nodes`"""
        if self._subject is None:
            raise ProxyUnsetException("Proxy geo has not been set.")
        return self._subject.nodes

    @property
    def labels(self) -> NDArray[np.str_]:
        """Proxy to `geo.labels`"""
        if self._subject is None:
            raise ProxyUnsetException("Proxy geo has not been set.")
        return self._subject.labels

    def __getitem__(self, name: str) -> NDArray:
        """Proxy to `geo[name]`"""
        if self._subject is None:
            raise ProxyUnsetException("Proxy geo has not been set.")
        return self._subject[name]


_proxy_geo_singleton = _ProxyGeoSingleton()
"""Internal-use proxy geo singleton instance."""


@contextmanager
def proxy_geo(actual_geo: GeoProtocol):
    """For the duration of this context, set the geo which the proxy geo should defer to."""
    _proxy_geo_singleton.set_actual_geo(actual_geo)
    yield
    _proxy_geo_singleton.set_actual_geo(None)


geo = cast(GeoProtocol, _proxy_geo_singleton)
"""
A proxy to the simulation geo.
You can use this proxy for function definitions and it will defer
to the simulation's geo at evaluation time.
"""
