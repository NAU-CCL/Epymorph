"""
In order to assist with proper data type handling in epymorph, we adopt a set of
conventions that somewhat narrow the possibilities for expressing simulation data
attributes. This module describes those conventions and provides some utilities for
working with them. The goal is to simplify and remove certain categories of errors,
like numerical overflow when simulating reasonable numbers of individuals.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Generic, TypeVar

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
"""A type describing all supported single-stratum forms of attribute data."""


T = TypeVar("T", bound=AttributeDType)


@dataclass(frozen=True)
class StratifiedAttributeArray(Generic[T]):
    """
    A container for attributes which have data for multiple strata.

    Parameters
    ----------
    values :
        The underlying array, whose first dimension corresponds to the number of strata.
    strata :
        The list of strata names in order, if known, or else None to use
        positional indexing only.

    Raises
    ------
    ValueError :
        For invalid or mismatched values and strata information. The values array must
        be at least one-dimensional, and if strata names are given, their number must
        match the values array's first dimension.
    """

    values: NDArray[T]
    strata: list[str] | None = field(default=None)
    num_strata: int = field(init=False)
    # TODO: should we have strata names here? Not yet convinced this is necessary.

    def __post_init__(self):
        shape = self.values.shape
        if len(shape) < 1:
            err = (
                "Invalid values: the array's first dimension must correspond to "
                "the number of strata."
            )
            raise ValueError(err)
        setattr(self, "num_strata", shape[0])
        if self.strata is not None and len(self.strata) != self.num_strata:
            err = (
                "Invalid strata: the number of names given must match the "
                "number of strata in the values array, judging the length of its "
                "first dimension."
            )
            raise ValueError(err)

    @staticmethod
    def _calc_num_strata():
        pass

    def values_for(self, stratum: str | int) -> NDArray[T]:
        """
        Retrieve the values for a specific stratum.

        Parameters
        ----------
        stratum :
            The name or index of the stratum to retrieve.

        Returns
        -------
        :
            The values for the requested stratum.

        Raises
        ------
        ValueError :
            If the stratum is invalid. If an index is given but it's out of bounds,
            or if a name is given but this object was not constructed with a list of
            names.
        """
        if isinstance(stratum, str):
            if self.strata is None:
                err = "Strata not specified by name cannot be accessed by name."
                raise ValueError(err)
            else:
                try:
                    stratum = self.strata.index(stratum)
                except ValueError:
                    err = f"Stratum '{stratum}' not found in attribute array."
                    raise ValueError(err) from None
        return self.values[stratum, ...]


AnyAttributeArray = AttributeArray | StratifiedAttributeArray
"""A type describing all supported forms of attribute data."""
# TODO: this is a terrible name...


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
    "StratifiedAttributeArray",
    "AnyAttributeArray",
    "CentroidType",
    "SimDType",
]
