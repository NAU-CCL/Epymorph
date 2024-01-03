"""
Effectively disable type checking for sympy.
See: https://github.com/microsoft/pyright/issues/945
"""
from typing import Any

# This comment stops isort and autopep8 from fighting with each other.


def __getattr__(_name) -> Any: ...
