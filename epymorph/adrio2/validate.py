"""ADRIO result validation utilities."""

from dataclasses import dataclass
from typing import Callable, TypeGuard

import numpy as np

from epymorph.data_shape import DataShape
from epymorph.data_type import AttributeData, StratifiedAttributeArray
from epymorph.simulation import Context
from epymorph.util import extract_date_value, is_date_value_array


class ValidationError(Exception):
    """Errors during data validation."""


def is_numpy(data: AttributeData) -> TypeGuard[np.ndarray]:
    """
    Checks that data is a numpy array.

    Parameters
    ----------
    data :
        The data to check.

    Raises
    ------
    ValidationError
        If data is not a numpy array.

    Returns
    -------
    :
        True if valid, as a type guard.
    """
    if isinstance(data, np.ndarray):
        err = "result was not a numpy array"
        raise ValidationError(err)
    return True


def is_stratified(data: AttributeData) -> TypeGuard[StratifiedAttributeArray]:
    """
    Checks that data is a stratified value.

    Parameters
    ----------
    data :
        The data to check.

    Raises
    ------
    ValidationError
        If data is not a stratified value.

    Returns
    -------
    :
        True if valid, as a type guard.
    """
    if isinstance(data, StratifiedAttributeArray):
        err = "result was not a stratified value"
        raise ValidationError(err)
    return True


def values_in_range(
    data: np.ndarray,
    minimum: int | float | None,
    maximum: int | float | None,
) -> None:
    """
    Check that numeric values fall within the specified range.

    Assumes the values are not structured; if you wish to validate structured data
    combine this function with a wrapper like `on_date_values` or `on_structured`.

    Parameters
    ----------
    minimum :
        The minimum valid value, or `None` if there is no minimum.
    maximum :
        The maximum valid value, or `None` if there is no maximum.

    Raises
    ------
    ValidationError
        If the values are not within the specified range.
    """
    match (minimum, maximum):
        case (None, None):
            return  # shortcut: no range to check
        case (minimum, None):
            invalid = data < minimum
        case (None, maximum):
            invalid = data > maximum
        case (minimum, maximum):
            invalid = (data < minimum) | (data > maximum)
    if np.any(invalid):
        invalid_values = np.sort(data[invalid].flatten())
        err = f"result contains invalid values\ne.g., {invalid_values}"
        raise ValidationError(err)


def is_shape(data: np.ndarray, shape: tuple[int, ...]) -> None:
    """
    Check that the given data has the specified shape.

    Parameters
    ----------
    shape :
        The expected result shape.

    Raises
    ------
    ValidationError
        If data is not the correct array shape.
    """
    if -1 in shape:
        err = (
            "Cannot check result shapes containing arbitrary axes; "
            "the ADRIO should specify the expected shape as a tuple of axis lengths."
        )
        raise ValueError(err)
    if data.shape != shape:
        err = f"result was an invalid shape:\ngot {data.shape}, expected {shape}"
        raise ValidationError(err)


def is_shape_unchecked_arbitrary(data: np.ndarray, shape: tuple[int, ...]) -> None:
    """
    Checks that the given data has the specified shape with the special exception that
    if an axis is specified as -1, any length (one or greater) is permitted. There must
    still be the same number of dimensions in the result.

    Parameters
    ----------
    shape :
        The expected result shape.

    Raises
    ------
    ValidationError
        If data is not the correct array shape.
    """
    if len(data.shape) != len(shape):
        err = f"result was an invalid shape:\ngot {data.shape}, expected {shape}"
        raise ValidationError(err)
    for actual_length, expected_length in zip(data.shape, shape, strict=True):
        if expected_length != -1 and expected_length != actual_length:
            err = f"result was an invalid shape:\ngot {data.shape}, expected {shape}"
            raise ValidationError(err)


def is_dtype(data: np.ndarray, dtype: np.dtype | type[np.generic]) -> None:
    """
    Checks the dtype of the given data.

    Assumes the values are not structured; if you wish to validate structured data
    combine this function with a wrapper like `on_date_values` or `on_structured`.

    Parameters
    ----------
    dtype :
        The expected result dtype.

    Raises
    ------
    ValidationError
        If data is not the correct array shape.
    """
    if not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)
    values_dtype = np.dtype(data.dtype)
    if dtype.kind == "U":
        # for strings, ignore length
        valid = values_dtype.kind == "U"
    else:
        # for other types, match dtype exactly
        valid = values_dtype == dtype
    if not valid:
        err = (
            "result was not the expected data type\n"
            f"got {np.dtype(data.dtype)}, expected {(np.dtype(dtype))}"
        )
        raise ValidationError(err)


Validator = Callable[[np.ndarray], None]
"""
A validator function on a numpy array. Raises ValidationError if the check fails.
"""


def on_date_values(validator: Validator) -> Validator:
    """
    Wrap a validator function so that it can check the values of a date/value array.

    Parameters
    ----------
    validator :
        The validator function to wrap.

    Returns
    -------
    :
        The validator function adapted to work for data/value arrays.
    """

    def _on_date_values(date_values: np.ndarray) -> None:
        if not is_date_value_array(date_values):
            err = "result was not a date/value pair as expected"
            raise ValidationError(err)
        _, values = extract_date_value(date_values)
        validator(values)

    return _on_date_values


def on_structured(validator: Validator) -> Validator:
    """
    Wrap a validator function so that it can check the values of structured arrays.
    This presumes that the validator can be applied to all elements of the structured
    array in the same way. That is: it will likely work for homogenous types but not
    heterogenous types.

    Parameters
    ----------
    validator :
        The validator function to wrap.

    Returns
    -------
    :
        The validator function adapated to work for structured arrays.
    """

    def _on_structured(values: np.ndarray) -> None:
        if values.dtype.names is None:
            err = "result was not a structured array as expected"
            raise ValidationError(err)
        for name in values.dtype.names:
            validator(values[name])

    return _on_structured


@dataclass(frozen=True)
class ResultFormat:
    """
    Describes the properties of the expected result of evaluating an ADRIO.

    Parameters
    ----------
    shape :
        The expected shape of the result array.
    dtype :
        The dtype describing the result array.
    """

    shape: DataShape
    """The expected shape of the result array."""
    dtype: np.dtype
    """The dtype describing the result array."""

    def __init__(self, shape: DataShape, dtype: np.dtype | type[np.generic]):
        if not isinstance(dtype, np.dtype):
            dtype = np.dtype(dtype)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", dtype)


def is_result_format(
    data: np.ndarray,
    result_format: ResultFormat,
    context: Context,
) -> None:
    """
    Check the data matches the given result format.

    This is a shortcut for checking `is_shape` and `is_dtype`.

    Assumes the values are not structured; if you wish to validate structured data
    combine this function with a wrapper like `on_date_values` or `on_structured`.

    Parameters
    ----------
    data :
        The data to validate.
    result_format :
        The expected result format.
    context :
        The simulation context.

    Raises
    ------
    ValidationError
        If data does not match the expected result format.
    """
    exp_shape = result_format.shape.to_tuple(context.dim)
    is_shape(data, exp_shape)
    is_dtype(data, result_format.dtype)
