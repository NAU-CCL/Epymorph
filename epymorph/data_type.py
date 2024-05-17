"""
Types for source data and attributes in epymorph.
"""
from typing import Any, Callable, Union

import numpy as np
from numpy.typing import NDArray

_DataPyBasic = int | float | str
_DataPyTuple = tuple[_DataPyBasic, ...]
# support recursively-nested lists
_DataPyList = list[Union[_DataPyBasic, _DataPyTuple, '_DataPyList']]
_DataPy = _DataPyBasic | _DataPyTuple | _DataPyList

_DataNpScalar = np.int64 | np.float64 | np.str_ | np.void
_DataNpArray = NDArray[_DataNpScalar]
_DataNp = _DataNpScalar | _DataNpArray

DataPyScalar = _DataPyBasic | _DataPyTuple
DataScalar = _DataPyBasic | _DataPyTuple | _DataNpScalar
"""The allowed scalar types (either python or numpy equivalents)."""

DataDType = type[_DataPyBasic] | list[tuple[str, type[_DataPyBasic]]]
"""The allowed dtype declarations for use with numpy."""


def dtype_as_np(dtype: DataDType) -> np.dtype:
    """Return a python-style dtype as its numpy-equivalent."""
    if dtype == int:
        return np.dtype(np.int64)
    if dtype == float:
        return np.dtype(np.float64)
    if dtype == str:
        return np.dtype(np.str_)
    if isinstance(dtype, list):
        return np.dtype(dtype)
    raise ValueError(f"Unsupported dtype: {dtype}")


def dtype_str(dtype: DataDType) -> str:
    """Return a human-readable description of the given dtype."""
    if dtype == int:
        return "int"
    if dtype == float:
        return "float"
    if dtype == str:
        return "str"
    if isinstance(dtype, list):
        values = (f"({x[0]}, {dtype_str(x[1])})" for x in dtype)
        return f"[{', '.join(values)}]"
    raise ValueError(f"Unsupported dtype: {dtype}")


def dtype_check(dtype: DataDType, value: Any) -> bool:
    """Checks that a value conforms to a given dtype. (Python types only.)"""
    if dtype in (int, float, str):
        return isinstance(value, dtype)
    if isinstance(dtype, list):
        if not isinstance(value, tuple):
            return False
        if len(value) != len(dtype):
            return False
        return all((
            dtype_check(vtype, v)
            for ((_, vtype), v) in zip(dtype, value)
        ))
    raise ValueError(f"Unsupported dtype: {dtype}")


ParamFunction = Callable[[int, int], DataScalar]
"""
Params may be defined as functions of time (day) and geo node (index),
returning a python or numpy scalar value.
"""

RawParam = _DataPy | _DataNp | ParamFunction
"""
Types for raw parameter values. Users can supply any of these forms when constructing
simulation parameters.
"""

AttributeScalar = _DataNpScalar
AttributeArray = _DataNpArray
"""
The type of all data attributes, whether in geo or params (after normalization).
"""

# SimDType being centrally-located means we can change it reliably.
SimDType = np.int64
"""
This is the numpy datatype that should be used to represent internal simulation data.
Where segments of the application maintain compartment and/or event counts,
they should take pains to use this type at all times (if possible).
"""

SimArray = NDArray[SimDType]

CentroidDType: DataDType = [('longitude', float), ('latitude', float)]
"""Structured numpy dtype for long/lat coordinates."""
