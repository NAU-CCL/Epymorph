"""
In order to assist with proper data type handling in epymorph, we adopt a set of
conventions that somewhat narrow the possibilities for expressing simulation data
attributes. This module describes those conventions and provides some utilities for
working with them. The goal is to simplify and remove certain categories of errors,
like numerical overflow when simulating reasonable numbers of individuals.
"""

from datetime import date
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Types for attribute declarations:
# these are expressed as Python types for simplicity.

# NOTE: In epymorph, we express structured types as tuples-of-tuples;
# this way they're hashable, which is important for AttributeDef.
# However numpy expresses them as lists-of-tuples, so we have to convert;
# thankfully we had an infrastructure for this sort of thing already.

ScalarType = type[int | float | str | date]
"""Supported scalar value types."""
StructType = tuple[tuple[str, ScalarType], ...]
"""Supported structured types."""
AttributeType = ScalarType | StructType
"""The allowed type declarations for epymorph attributes."""

ScalarValue = int | float | str | date
"""Supported scalar values."""
StructValue = tuple[ScalarValue, ...]
"""Supported structured values."""
AttributeValue = ScalarValue | StructValue
"""
The allowed values types for epymorph attribute values
(most notably used as attribute defaults).
"""

ScalarDType = np.int64 | np.float64 | np.str_ | np.datetime64
"""Numpy equivalents to supported scalar values."""
StructDType = np.void
"""Numpy equivalents to supported structured values."""
AttributeDType = ScalarDType | StructDType
"""The allowed numpy dtypes for use in epymorph: these map 1:1 with `AttributeType`."""

AttributeArray = NDArray[AttributeDType]
"""A type describing all supported numpy array forms for attribute data."""


def dtype_as_np(dtype: AttributeType) -> np.dtype:
    """
    Convert a python-style dtype to its numpy-equivalent using epymorph typing
    conventions.

    Parameters
    ----------
    dtype :
        The attribute type in Python form, e.g., `int`

    Returns
    -------
    :
        The numpy equivalent, e.g., `np.int64`
    """
    if dtype is int:
        return np.dtype(np.int64)
    if dtype is float:
        return np.dtype(np.float64)
    if dtype is str:
        return np.dtype(np.str_)
    if dtype is date:
        return np.dtype("datetime64[D]")
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
    """
    Return a human-readable description of the given attribute data type.

    Parameters
    ----------
    dtype :
        The attribute type in Python form, e.g., `int`

    Returns
    -------
    :
        The friendly string representation of the type.
    """
    if dtype is int:
        return "int"
    if dtype is float:
        return "float"
    if dtype is str:
        return "str"
    if dtype is date:
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
    """
    Check that a singular Python value conforms to the given attribute data type.
    This is not intended to check numpy arrays, only scalars and tuples.

    Parameters
    ----------
    dtype :
        The attribute type in Python form, e.g., `int`
    value :
        A value to check.

    Returns
    -------
    :
        True if the values matches the dtype.
    """
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
"""Structured epymorph type declaration for longitude/latitude coordinates."""
CentroidDType: np.dtype[np.void] = dtype_as_np(CentroidType)
"""
The numpy equivalent of `CentroidType`
(structured dtype for longitude/latitude coordinates).
"""

SimDType = np.int64
"""
This is the numpy datatype that should be used to represent internal simulation data.
Where segments of the application maintain compartment and/or event counts,
they should take pains to use this type at all times (if possible).
"""
# NOTE: SimDType being centrally-located means we can change it reliably.

SimArray = NDArray[SimDType]
"""Type alias for a numpy array of `SimDType`."""

__all__ = [
    "AttributeType",
    "AttributeArray",
    "CentroidType",
    "SimDType",
]
