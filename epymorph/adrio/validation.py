"""Simply a bundle of ADRIO validation functions."""

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray

from epymorph.util import extract_date_value, is_date_value_array


@dataclass(frozen=True)
class Valid:
    kind: Literal["valid"] = field(init=False, default="valid")


VALID = Valid()


@dataclass(frozen=True)
class Invalid:
    kind: Literal["invalid"] = field(init=False, default="invalid")
    error: str


ValidationResult = Valid | Invalid

Validator = Callable[[NDArray], ValidationResult]


def validate_pipe(*validators: Validator) -> Validator:
    def _validate_pipe(values: NDArray) -> ValidationResult:
        for validate in validators:
            v = validate(values)
            if v.kind == "invalid":
                return v
        return VALID

    return _validate_pipe


def validate_numpy() -> Validator:
    def _validate_numpy(values: NDArray) -> ValidationResult:
        if not isinstance(values, np.ndarray):
            return Invalid("result was not a numpy array")
        return VALID

    return _validate_numpy


def format_invalid_values(values: NDArray, invalid_mask: NDArray[np.bool_]) -> str:
    return (
        "result contains invalid values\n"
        f"e.g., {np.sort(values[invalid_mask].flatten())}"
    )


def validate_values_in_range(
    minimum: int | float | None,
    maximum: int | float | None,
) -> Validator:
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
            return Invalid(format_invalid_values(values, invalid))
        return VALID

    return _validate_values_in_range


def validate_shape(shape: tuple[int, ...]) -> Validator:
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


def validate_dtype(dtype: np.dtype | type[np.generic]) -> Validator:
    def _validate_dtype(values: NDArray) -> ValidationResult:
        nonlocal dtype
        if not isinstance(dtype, np.dtype):
            dtype = np.dtype(dtype)
        if np.dtype(values.dtype) != dtype:
            return Invalid(
                "result was not the expected data type\n"
                f"got {np.dtype(values.dtype)}, expected {(np.dtype(dtype))}"
            )
        return VALID

    return _validate_dtype


def on_date_values(validator: Validator) -> Validator:
    def _on_date_values(date_values: NDArray) -> ValidationResult:
        if not is_date_value_array(date_values):
            return Invalid("result was not a date/value pair as expected")
        _, values = extract_date_value(date_values)
        return validator(values)

    return _on_date_values


def on_structured(validator: Validator) -> Validator:
    def _on_structured(values: NDArray) -> ValidationResult:
        if values.dtype.names is None:
            return Invalid("result was not a structured array as expected")
        for name in values.dtype.names:
            v = validator(values[name])
            if v.kind == "invalid":
                return v
        return VALID

    return _on_structured
