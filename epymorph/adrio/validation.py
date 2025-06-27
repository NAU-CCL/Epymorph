"""ADRIO result validation utilities."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.data_shape import DataShape
from epymorph.simulation import Context
from epymorph.util import extract_date_value, is_date_value_array


class ValidationResult(ABC):
    """
    The result of a data validation check.

    See Also
    --------
    [epymorph.adrio.validation.Valid][] and [epymorph.adrio.validation.Invalid][].
    """

    @property
    @abstractmethod
    def is_valid(self) -> bool:
        """True if this is a valid result, False for invalid."""

    @abstractmethod
    def __and__(self, other: "ValidationResult") -> "ValidationResult":
        """
        Combine two results by returning the first invalid result or else `Valid` if
        both are valid.

        Parameters
        ----------
        other :
            The other validation result to consider.

        Returns
        -------
        :
            The first invalid result or else `Valid` if both are valid.
        """
        # NOTE: this is `&` not `and`


@dataclass(frozen=True)
class Valid(ValidationResult):
    """The result of passing a validation check."""

    @property
    @override
    def is_valid(self) -> Literal[True]:
        return True

    @override
    def __and__(self, other: "ValidationResult") -> "ValidationResult":
        return other


@dataclass(frozen=True)
class Invalid(ValidationResult):
    """The result of failing a validation check."""

    error: str
    """The reason why this check failed.."""

    @property
    @override
    def is_valid(self) -> Literal[False]:
        return False

    @override
    def __and__(self, other: "ValidationResult") -> "ValidationResult":
        return self


VALID = Valid()
"""A singleton instance of `Valid`."""


Validator = Callable[[NDArray], ValidationResult]
"""The type for a function which validates a numpy result."""


def validate_pipe(*validators: Validator) -> Validator:
    """
    Create a validator by chaining multiple other validators. Validators are evaluated
    such that they short-circuit on the first invalid result, in which case only that
    invalid result is returned. If all validators pass, `Valid` is returned.

    Parameters
    ----------
    *validators :
        The validator functions in evaluation order.

    Returns
    -------
    :
        A validator function.
    """

    def _validate_pipe(values: NDArray) -> ValidationResult:
        for validate in validators:
            v = validate(values)
            if isinstance(v, Invalid):
                return v
        return VALID

    return _validate_pipe


def validate_numpy() -> Validator:
    """
    Create a validator which checks that the result is a numpy array.

    Returns
    -------
    :
        A validator function.
    """

    def _validate_numpy(values: NDArray) -> ValidationResult:
        if not isinstance(values, np.ndarray):
            return Invalid("result was not a numpy array")
        return VALID

    return _validate_numpy


def validate_values_in_range(
    minimum: int | float | None,
    maximum: int | float | None,
) -> Validator:
    """
    Create a validator which checks that numeric values fall within the specified
    range.

    Assumes the values are not structured; if you wish to validate structured data
    combine this function with a wrapper like `on_date_values` or `on_structured`.

    Parameters
    ----------
    minimum :
        The minimum valid value, or `None` if there is no minimum.
    maximum :
        The maximum valid value, or `None` if there is no maximum.

    Returns
    -------
    :
        A validator function.
    """

    def _validate_values_in_range(values: NDArray) -> ValidationResult:
        nonlocal minimum, maximum
        match (minimum, maximum):
            case (None, None):
                invalid = np.zeros_like(values, dtype=np.bool_)
            case (minimum, None):
                invalid = values < minimum
            case (None, maximum):
                invalid = values > maximum
            case (minimum, maximum):
                invalid = (values < minimum) | (values > maximum)
        if np.any(invalid):
            invalid_values = np.sort(values[invalid].flatten())
            return Invalid(f"result contains invalid values\ne.g., {invalid_values}")
        return VALID

    return _validate_values_in_range


def validate_shape(shape: tuple[int, ...]) -> Validator:
    """
    Create a validator which checks the given shape.

    Parameters
    ----------
    shape :
        The expected result shape.

    Returns
    -------
    :
        A validator function.
    """
    if -1 in shape:
        err = (
            "Cannot check result shapes containing arbitrary axes; "
            "the ADRIO should specify the expected shape as a tuple of axis lengths."
        )
        raise ValueError(err)

    def _validate_shape(values: NDArray) -> ValidationResult:
        if values.shape != shape:
            return Invalid(
                f"result was an invalid shape:\ngot {values.shape}, expected {shape}"
            )
        return VALID

    return _validate_shape


def validate_shape_unchecked_arbitrary(shape: tuple[int, ...]) -> Validator:
    """
    Create a validator which checks the given shape with the special exception that
    if an axis is specified as -1, any length (one or greater) is permitted. There must
    still be the same number of dimensions in the result.

    Parameters
    ----------
    shape :
        The expected result shape.

    Returns
    -------
    :
        A validator function.
    """

    def _validate_shape(values: NDArray) -> ValidationResult:
        err = "result was an invalid shape:\ngot {}, expected {}"
        if len(values.shape) != len(shape):
            return Invalid(err.format(values.shape, shape))
        for actual_length, expected_length in zip(values.shape, shape, strict=True):
            if expected_length != -1 and expected_length != actual_length:
                return Invalid(err.format(values.shape, shape))
        return VALID

    return _validate_shape


def validate_dtype(dtype: np.dtype | type[np.generic]) -> Validator:
    """
    Create a validator which checks the given dtype.

    Assumes the values are not structured; if you wish to validate structured data
    combine this function with a wrapper like `on_date_values` or `on_structured`.

    Parameters
    ----------
    dtype :
        The expected result dtype.

    Returns
    -------
    :
        A validator function.
    """

    def _validate_dtype(values: NDArray) -> ValidationResult:
        nonlocal dtype
        if not isinstance(dtype, np.dtype):
            dtype = np.dtype(dtype)
        values_dtype = np.dtype(values.dtype)
        if dtype.kind == "U":
            # for strings, ignore length
            valid = values_dtype.kind == "U"
        else:
            # for other types, match dtype exactly
            valid = values_dtype == dtype
        if not valid:
            return Invalid(
                "result was not the expected data type\n"
                f"got {np.dtype(values.dtype)}, expected {(np.dtype(dtype))}"
            )
        return VALID

    return _validate_dtype


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

    def _on_date_values(date_values: NDArray) -> ValidationResult:
        if not is_date_value_array(date_values):
            return Invalid("result was not a date/value pair as expected")
        _, values = extract_date_value(date_values)
        return validator(values)

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

    def _on_structured(values: NDArray) -> ValidationResult:
        if values.dtype.names is None:
            return Invalid("result was not a structured array as expected")
        for name in values.dtype.names:
            v = validator(values[name])
            if isinstance(v, Invalid):
                return v
        return VALID

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

    def __init__(self, shape: DataShape, dtype: np.dtype | type[np.generic]):
        if not isinstance(dtype, np.dtype):
            dtype = np.dtype(dtype)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", dtype)

    shape: DataShape
    """The expected shape of the result array."""
    dtype: np.dtype
    """The dtype describing the result array."""


def validate_result(rformat: ResultFormat, context: Context) -> Validator:
    """
    Create a validator for the given result format declaration. This is a shortcut
    for chaining `validate_shape` and `validate_dtype`.

    Assumes the values are not structured; if you wish to validate structured data
    combine this function with a wrapper like `on_date_values` or `on_structured`.

    Parameters
    ----------
    rformat :
        The expected result format.
    context :
        The simulation context.

    Returns
    -------
    :
        A validator function.
    """
    vshape = validate_shape(rformat.shape.to_tuple(context.dim))
    vdtype = validate_dtype(rformat.dtype)

    def _validate_result(values: NDArray) -> ValidationResult:
        if isinstance(v := vshape(values), Invalid):
            return v
        return vdtype(values)

    return _validate_result
