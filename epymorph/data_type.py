"""
Types for source data and attributes in epymorph.
"""

from datetime import date
from typing import Any

import numpy as np
from numpy.typing import DTypeLike, NDArray

# Types for attribute declarations:
# these are expressed as Python types for simplicity.

# NOTE: In epymorph, we express structured types as tuples-of-tuples;
# this way they're hashable, which is important for AttributeDef.
# However numpy expresses them as lists-of-tuples, so we have to convert;
# thankfully we had an infrastructure for this sort of thing already.

ScalarType = type[int | float | str | date]
StructType = tuple[tuple[str, ScalarType], ...]
AttributeType = ScalarType | StructType
"""The allowed type declarations for epymorph attributes."""

ScalarValue = int | float | str | date
StructValue = tuple[ScalarValue, ...]
AttributeValue = ScalarValue | StructValue
"""The allowed types for epymorph attribute values (specifically: default values)."""

ScalarDType = np.int64 | np.float64 | np.str_ | np.datetime64
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
    if dtype == date:
        return np.dtype(np.datetime64)
    if isinstance(dtype, tuple):
        fields = list(dtype)
        if len(fields) == 0:
            raise ValueError(f"Unsupported dtype: {dtype}")
        try:
            return np.dtype(
                [
                    (field_name, dtype_as_np(field_dtype))
                    for field_name, field_dtype in fields
                ]
            )
        except (TypeError, ValueError):
            raise ValueError(f"Unsupported dtype: {dtype}") from None
    raise ValueError(f"Unsupported dtype: {dtype}")


def dtype_str(dtype: AttributeType) -> str:
    """Return a human-readable description of the given dtype."""
    if dtype == int:
        return "int"
    if dtype == float:
        return "float"
    if dtype == str:
        return "str"
    if dtype == date:
        return "date"
    if isinstance(dtype, tuple):
        fields = list(dtype)
        if len(fields) == 0:
            raise ValueError(f"Unsupported dtype: {dtype}")
        try:
            values = [
                f"({field_name}, {dtype_str(field_dtype)})"
                for field_name, field_dtype in fields
            ]
            return f"[{', '.join(values)}]"
        except (TypeError, ValueError):
            raise ValueError(f"Unsupported dtype: {dtype}") from None
    raise ValueError(f"Unsupported dtype: {dtype}")


def dtype_check(dtype: AttributeType, value: Any) -> bool:
    """Checks that a value conforms to a given dtype. (Python types only.)"""
    if dtype in (int, float, str, date):
        return isinstance(value, dtype)
    if isinstance(dtype, tuple):
        fields = list(dtype)
        if not isinstance(value, tuple):
            return False
        if len(value) != len(fields):
            return False
        return all(
            (
                dtype_check(field_dtype, field_value)
                for ((_, field_dtype), field_value) in zip(fields, value)
            )
        )
    raise ValueError(f"Unsupported dtype: {dtype}")


CentroidType: AttributeType = (("longitude", float), ("latitude", float))
"""Structured epymorph type declaration for long/lat coordinates."""
CentroidDType: DTypeLike = dtype_as_np(CentroidType)
"""The numpy equivalent of `CentroidType` (structured dtype for long/lat coordinates)."""

# SimDType being centrally-located means we can change it reliably.
SimDType = np.int64
"""
This is the numpy datatype that should be used to represent internal simulation data.
Where segments of the application maintain compartment and/or event counts,
they should take pains to use this type at all times (if possible).
"""

SimArray = NDArray[SimDType]

__all__ = [
    "AttributeType",
    "AttributeArray",
    "CentroidType",
    "SimDType",
]
