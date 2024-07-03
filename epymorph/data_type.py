"""
Types for source data and attributes in epymorph.
"""
from typing import Any, Sequence

import numpy as np
from numpy.typing import DTypeLike, NDArray

# _DataPyBasic = int | float | str
# _DataPyTuple = tuple[_DataPyBasic, ...]
# support recursively-nested lists
# _DataPyList = Sequence[Union[_DataPyBasic, _DataPyTuple, '_DataPyList']]
# _DataPy = _DataPyBasic | _DataPyTuple | _DataPyList

# DataPyScalar = _DataPyBasic | _DataPyTuple
# DataScalar = _DataPyBasic | _DataPyTuple | _DataNpScalar
# """The allowed scalar types (either python or numpy equivalents)."""

# Types for attribute declarations:
# these are expressed as Python types for simplicity.

ScalarType = type[int | float | str]
StructType = Sequence[tuple[str, ScalarType]]
AttributeType = ScalarType | StructType
"""The allowed type declarations for epymorph attributes."""

ScalarValue = int | float | str
StructValue = tuple[ScalarValue, ...]
AttributeValue = ScalarValue | StructValue
"""The allowed types for epymorph attribute values (specifically: default values)."""

ScalarDType = np.int64 | np.float64 | np.str_
StructDType = np.void
AttributeDType = ScalarDType | StructDType
"""The subset of numpy dtypes for use in epymorph: these map 1:1 with AttributeType."""

AttributeArray = NDArray[AttributeDType]


def dtype_as_np(dtype: AttributeType) -> np.dtype:
    """Return a python-style dtype as its numpy-equivalent."""
    if dtype == int:
        return np.dtype(np.int64)
    if dtype == float:
        return np.dtype(np.float64)
    if dtype == str:
        return np.dtype(np.str_)
    if isinstance(dtype, list):
        return np.dtype(dtype)
    if isinstance(dtype, Sequence):
        return np.dtype(list(dtype))
    raise ValueError(f"Unsupported dtype: {dtype}")


def dtype_str(dtype: AttributeType) -> str:
    """Return a human-readable description of the given dtype."""
    if dtype == int:
        return "int"
    if dtype == float:
        return "float"
    if dtype == str:
        return "str"
    if isinstance(dtype, Sequence):
        values = (f"({x[0]}, {dtype_str(x[1])})" for x in dtype)
        return f"[{', '.join(values)}]"
    raise ValueError(f"Unsupported dtype: {dtype}")


def dtype_check(dtype: AttributeType, value: Any) -> bool:
    """Checks that a value conforms to a given dtype. (Python types only.)"""
    if dtype in (int, float, str):
        return isinstance(value, dtype)
    if isinstance(dtype, Sequence):
        if not isinstance(value, tuple):
            return False
        if len(value) != len(dtype):
            return False
        return all((
            dtype_check(vtype, v)
            for ((_, vtype), v) in zip(dtype, value)
        ))
    raise ValueError(f"Unsupported dtype: {dtype}")


# ParamFunction = Callable[[int, int], DataScalar]
# """
# Params may be defined as functions of time (day) and geo node (index),
# returning a python or numpy scalar value.
# """

# RawParam = _DataPy | _DataNp | ParamFunction
# """
# Types for raw parameter values. Users can supply any of these forms when constructing
# simulation parameters.
# """

# AttributeScalar = _DataNpScalar
# AttributeArray = _DataNpArray
# """
# The type of all data attributes, whether in geo or params (after normalization).
# """


CentroidType: AttributeType = [('longitude', float), ('latitude', float)]
"""Structured epymorph type declaration for long/lat coordinates."""
CentroidDType: DTypeLike = [('longitude', float), ('latitude', float)]
"""Structured numpy dtype for long/lat coordinates."""

# SimDType being centrally-located means we can change it reliably.
SimDType = np.int64
"""
This is the numpy datatype that should be used to represent internal simulation data.
Where segments of the application maintain compartment and/or event counts,
they should take pains to use this type at all times (if possible).
"""

SimArray = NDArray[SimDType]

__all__ = [
    'AttributeType',
    'AttributeArray',
    'CentroidType',
    'SimDType',
]
