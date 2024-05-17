"""
Proxy instances to epymorph models. Intended for use when evalution must be deferred,
e.g., when constructing simulation parameters as Python functions. We may wish to
define a function which depends on geo values, but before the geo itself has been defined.
"""
from typing import Any, cast

from epymorph.data_shape import SimDimensions
from epymorph.geo.geo import Geo


class ProxyAccessException(Exception):
    """Exception for when a proxy is accessed instead of the true object."""
    # This is intended to never happen -- all references to these proxy classes
    # (when used correctly) should be swapped for their true counterparts
    # before evaluation. (Typically through AST manipulation.)


class _Proxy:
    _proxy_name: str

    def __init__(self, name: str):
        self._proxy_name = name

    def __getattr__(self, name: str) -> Any:
        raise ProxyAccessException(f"Proxy {self._proxy_name} accessed.")

    def __getitem__(self, name: str) -> Any:
        raise ProxyAccessException(f"Proxy {self._proxy_name} accessed.")


geo = cast(Geo, _Proxy("geo"))
"""
A proxy to the simulation geo.
You can use this proxy for function definitions and it will defer
to the simulation's geo at evaluation time.
"""

dim = cast(SimDimensions, _Proxy("dim"))
"""
A proxy to the simulation dimensions.
You can use this proxy for function definitions and it will defer
to the simulation's dimensions at evaluation time.
"""
