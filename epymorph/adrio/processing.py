import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial, reduce
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    Mapping,
    NamedTuple,
    Protocol,
    Self,
    Sequence,
    TypeVar,
    cast,
)

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.util import DateValueType, to_date_value_array

DataT = TypeVar("DataT", bound=np.generic)
"""A numpy array dtype."""


class HasRandomness(Protocol):
    """Protocol for an object containing a numpy random number generator."""

    @property
    @abstractmethod
    def rng(self) -> np.random.Generator:
        """The random number generator instance."""


class Fix(ABC, Generic[DataT]):
    """
    A method for fixing data issues as part of a DataPipeline. Fix instances act as
    functions (they have call semantics).

    Fix is an abstract base class. It is generic in the dtype of the data it fixes
    (`DataT`).
    """

    @abstractmethod
    def __call__(
        self,
        rng: HasRandomness,
        replace: DataT,
        columns: tuple[str, ...],
        data_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply this fix to some data.

        Parameters
        ----------
        rng : HasRandomness
            A source of randomness.
        replace : DataT
            The value to replace.
        columns : tuple[str, ...]
            The names of the columns to fix.
        data_df : DataFrame
            The data to fix.

        Returns
        -------
        DataFrame
            The data with the fix applied (a copy if modified).
        """

    @staticmethod
    def of_int64(
        fix: "Fix[np.int64] | int | Callable[[], int] | Literal[False]",
    ) -> "Fix[np.int64]":
        """
        Convenience constructor for a Fix for int64 data. The type of Fix
        returned depends on the type of argument provided.

        Parameters
        ----------
        fix : Fix[np.int64] | int | Callable[[], int] | Literal[False]
            A value which implies the type of fix to apply.
            - `Fix[np.int64]`: is returned unchanged (no-op)
            - `int`: returns a `ConstantFix`, to replace bad values with a constant
            - `Callable[[], int]`: return a `FunctionFix`, to replace bad values
              with values obtained from the given callable
            - `False`: return a `DontFix`, indicating not to replace bad values

        Returns
        -------
        Fix[np.int64]
            A Fix instance as determined by the type of the argument.
        """
        if fix is False:
            return DontFix()
        elif isinstance(fix, Fix):
            return fix  # for convenience, no-op if arg is a Fix
        elif isinstance(fix, int):
            return ConstantFix[np.int64](fix)  # type: ignore
        elif callable(fix):
            return FunctionFix[np.int64](fix)  # type: ignore
        raise ValueError("Not a valid fix.")

    @staticmethod
    def of_float64(
        fix: "Fix[np.float64] | float | Callable[[], float] | Literal[False]",
    ) -> "Fix[np.float64]":
        """
        Convenience constructor for a Fix for float64 data. The type of Fix
        returned depends on the type of argument provided.

        Parameters
        ----------
        fix : Fix[np.float64] | float | Callable[[], float] | Literal[False]
            A value which implies the type of fix to apply.
            - `Fix[np.float64]`: is returned unchanged (no-op)
            - `float`: returns a `ConstantFix`, to replace bad values with a constant
            - `Callable[[], float]`: return a `FunctionFix`, to replace bad values
              with values obtained from the given callable
            - `False`: return a `DontFix`, indicating not to replace bad values

        Returns
        -------
        Fix[np.float64]
            A Fix instance as determined by the type of the argument.
        """
        if fix is False:
            return DontFix()
        elif isinstance(fix, Fix):
            return fix  # for convenience, no-op if arg is a Fix
        elif isinstance(fix, float):
            return ConstantFix[np.float64](fix)  # type: ignore
        elif callable(fix):
            return FunctionFix[np.float64](fix)  # type: ignore
        raise ValueError("Not a valid fix.")


@dataclass(frozen=True)
class ConstantFix(Fix[DataT]):
    """
    A Fix which replaces values with a constant value.

    ConstantFix is a frozen dataclass. It is generic in the dtype of the data it fixes.
    """

    with_value: DataT
    """The value to use to replace bad values."""

    @override
    def __call__(self, rng, replace, columns, data_df):
        return data_df.assign(
            **{col: data_df[col].replace(replace, self.with_value) for col in columns}
        )


@dataclass(frozen=True)
class FunctionFix(Fix[DataT]):
    """
    A Fix which replaces values with values generated by the given function.

    FunctionFix is a frozen dataclass. It is generic in the dtype of the data it fixes.
    """

    with_function: Callable[[], DataT]
    """The function that generates replacement values."""

    @override
    def __call__(self, rng, replace, columns, data_df):
        return FunctionFix.apply(data_df, replace, columns, self.with_function)

    @staticmethod
    def apply(
        data_df: pd.DataFrame,
        replace: DataT,
        columns: tuple[str, ...],
        with_function: Callable[[], DataT],
    ) -> pd.DataFrame:
        """
        A static method that performs the work of a FunctionFix. This method can be
        useful in creating other Fix instances, when their replacement value logic
        can be expressed as a no-parameter function.

        Parameters
        ----------
        data_df : DataFrame
            The data to fix.
        replace : DataT
            The value to replace.
        with_function : Callable[[], DataT]
            The function used to generate replacement values.

        Returns
        -------
        DataFrame
            A copy of the data with bad values fixed.
        """

        def replace_col(col: str) -> pd.Series:
            is_replace = data_df[col] == replace
            num_replace = is_replace.sum()
            replacements = np.array([with_function() for _ in range(num_replace)])
            updated = data_df[col].copy()
            updated[is_replace] = replacements
            return updated

        return data_df.assign(**{col: replace_col(col) for col in columns})


@dataclass(frozen=True)
class RandomFix(Fix[DataT]):
    """
    A Fix which replaces values with randomly-generated values.

    RandomFix is a frozen dataclass. It is generic in the dtype of the data it fixes.
    """

    with_random: Callable[[np.random.Generator], DataT]
    """
    A function for generating replacement values using the given numpy random number
    generator.
    """

    @override
    def __call__(self, rng, replace, columns, data_df):
        random_fn = partial(self.with_random, rng.rng)
        return FunctionFix.apply(data_df, replace, columns, random_fn)

    @staticmethod
    def from_range(low: int, high: int) -> "RandomFix[np.int64]":
        """
        Convenience constructor for a RandomFix which replaces values
        with values sampled uniformly from a discrete range of integers.

        Parameters
        ----------
        low : int
            The lowest replacement value.
        high : int
            The highest replacement value.

        Returns
        -------
        RandomFix[np.int64]
            The RandomFix instance.
        """
        return RandomFix(
            lambda rng: rng.integers(low, high, endpoint=True),  # type: ignore
        )

    @staticmethod
    def from_range_float(low: float, high: float) -> "RandomFix[np.float64]":
        """
        Convenience constructor for a RandomFix which replaces values
        with values sampled uniformly from a continuous range.

        Parameters
        ----------
        low : float
            The low end of the range of replacement values.
        high : float
            The high end of the range of replacement values.
            (Not included in the possible values.)

        Returns
        -------
        RandomFix[np.float64]
            The RandomFix instance.
        """
        return RandomFix(
            lambda rng: rng.uniform(low, high),  # type: ignore
        )


@dataclass(frozen=True)
class DontFix(Fix[Any]):
    """
    A special Fix which does not fix values and simply returns the data as-is (no-op).

    DontFix is a frozen dataclass.
    """

    @override
    def __call__(self, rng, replace, columns, data_df):
        return data_df


class Fill(ABC, Generic[DataT]):
    """
    A method for filling-in missing data as part of a DataPipeline. Fill instances act
    as functions (they have call semantics).

    Fill is an abstract base class. It is generic in the dtype of the data it fixes
    (`DataT`).
    """

    @abstractmethod
    def __call__(
        self,
        rng: HasRandomness,
        data_np: NDArray[DataT],
        missing_mask: NDArray[np.bool_],
    ) -> tuple[NDArray[DataT], NDArray[np.bool_] | None]:
        """
        Apply this fix to some data.

        Parameters
        ----------
        rng : HasRandomness
            A source of randomness.
        data_np : NDArray[DataT]
            The data to fix.
        missing_mask : NDArray[np.bool_]
            A mask indicating values which should be considered missing.

        Returns
        -------
        tuple[NDArray[DataT], NDArray[np.bool_] | None]
            A tuple containing two values:
            - a copy of the data with the fill applied (a copy if modified),
            - an updated missing values mask. Fill methods may or may not
              replace all missing values; if all missing values have been filled,
              this will be None.
        """

    @staticmethod
    def of_int64(
        fill: "Fill[np.int64] | int | Callable[[], int] | Literal[False]",
    ) -> "Fill[np.int64]":
        """
        Convenience constructor for a Fill for int64 data. The type of Fill
        returned depends on the type of argument provided.

        Parameters
        ----------
        fill : Fill[np.int64] | int | Callable[[], int] | Literal[False]
            A value which implies the type of fix to apply.
            - `Fill[np.int64]`: is returned unchanged (no-op)
            - `int`: returns a `ConstantFill`, to replace missing values with a constant
            - `Callable[[], int]`: return a `FunctionFill`, to replace missing values
              with values obtained from the given callable
            - `False`: return a `DontFill`, indicating not to replace missing values

        Returns
        -------
        Fill[np.int64]
            A Fill instance as determined by the type of the argument.
        """
        if fill is False:
            return DontFill()
        elif isinstance(fill, Fill):
            return fill  # for convenience, no-op if arg is a Fill
        elif isinstance(fill, int):
            return ConstantFill[np.int64](fill)  # type: ignore
        elif callable(fill):
            return FunctionFill[np.int64](fill)  # type: ignore
        raise ValueError("Not a valid fill.")

    @staticmethod
    def of_float64(
        fill: "Fill[np.float64] | float | int | Callable[[], float] | Literal[False]",
    ) -> "Fill[np.float64]":
        """
        Convenience constructor for a Fill for float64 data. The type of Fill
        returned depends on the type of argument provided.

        Parameters
        ----------
        fill : Fill[np.float64] | float | Callable[[], float] | Literal[False]
            A value which implies the type of fix to apply.
            - `Fill[np.float64]`: is returned unchanged (no-op)
            - `float` or `int`: returns a `ConstantFill`, to replace missing values
              with a constant
            - `Callable[[], float]`: return a `FunctionFill`, to replace missing values
              with values obtained from the given callable
            - `False`: return a `DontFill`, indicating not to replace missing values

        Returns
        -------
        Fill[np.float64]
            A Fill instance as determined by the type of the argument.
        """
        if fill is False:
            return DontFill()
        elif isinstance(fill, Fill):
            return fill  # for convenience, no-op if arg is a Fill
        elif isinstance(fill, float | int):
            return ConstantFill[np.float64](float(fill))  # type: ignore
        elif callable(fill):
            return FunctionFill[np.float64](fill)  # type: ignore
        raise ValueError("Not a valid fill.")


@dataclass(frozen=True)
class ConstantFill(Fill[DataT]):
    """
    A Fill which replaces missing values with a constant value.

    ConstantFill is a frozen dataclass. It is generic in the dtype of the data it fixes.
    """

    with_value: DataT
    """The value to use to replace missing values."""

    @override
    def __call__(
        self, rng, data_np, missing_mask
    ) -> tuple[NDArray[DataT], NDArray[np.bool_] | None]:
        if missing_mask.any():
            result_np = data_np.copy()
            result_np[missing_mask] = self.with_value
        else:
            result_np = data_np
        return result_np, None


@dataclass(frozen=True)
class FunctionFill(Fill[DataT]):
    """
    A Fill which replaces missing values with values generated by the given function.

    FunctionFill is a frozen dataclass. It is generic in the dtype of the data it fixes.
    """

    with_function: Callable[[], DataT]
    """The function that generates replacement values."""

    @override
    def __call__(
        self, rng, data_np, missing_mask
    ) -> tuple[NDArray[DataT], NDArray[np.bool_] | None]:
        return FunctionFill.apply(data_np, missing_mask, self.with_function)

    @staticmethod
    def apply(
        data_np: NDArray[DataT],
        missing_mask: NDArray[np.bool_],
        with_function: Callable[[], DataT],
    ) -> tuple[NDArray[DataT], NDArray[np.bool_] | None]:
        """
        A static method that performs the work of a FunctionFill. This method can be
        useful in creating other Fill instances, when their replacement value logic
        can be expressed as a no-parameter function.

        Parameters
        ----------
        data_np : NDArray[DataT]
            The data to fix.
        missing_mask : NDArray[np.bool_]
            A mask indicating values which should be considered missing.
        with_function : Callable[[], DataT]
            The function used to generate replacement values.

        Returns
        -------
        tuple[NDArray[DataT], NDArray[np.bool_] | None]
            The fill apply result (updated data and missing mask).
        """
        num_replace = missing_mask.sum()
        replacements = np.array([with_function() for _ in range(num_replace)])
        result_np = data_np.copy()
        result_np[missing_mask] = replacements
        return result_np, None


@dataclass(frozen=True)
class RandomFill(Fill[DataT]):
    """
    A Fill which replaces missing values with randomly-generated values.

    RandomFill is a frozen dataclass. It is generic in the dtype of the data it fixes.
    """

    with_random: Callable[[np.random.Generator], DataT]
    """
    A function for generating replacement values using the given numpy random number
    generator.
    """

    @override
    def __call__(self, rng, data_np, missing_mask):
        random_fn = partial(self.with_random, rng.rng)
        return FunctionFill.apply(data_np, missing_mask, random_fn)

    @staticmethod
    def from_range(low: int, high: int) -> "RandomFill[np.int64]":
        """
        Convenience constructor for a RandomFill which replaces values
        with values sampled uniformly from a discrete range of integers.

        Parameters
        ----------
        low : int
            The lowest replacement value.
        high : int
            The highest replacement value.

        Returns
        -------
        RandomFill[np.int64]
            The RandomFill instance.
        """
        return RandomFill(
            lambda rng: rng.integers(low, high, endpoint=True),  # type: ignore
        )

    @staticmethod
    def from_range_float(low: float, high: float) -> "RandomFill[np.float64]":
        """
        Convenience constructor for a RandomFill which replaces values
        with values sampled uniformly from a continuous range.

        Parameters
        ----------
        low : float
            The low end of the range of replacement values.
        high : float
            The high end of the range of replacement values.
            (Not included in the possible values.)

        Returns
        -------
        RandomFill[np.float64]
            The RandomFill instance.
        """
        return RandomFill(
            lambda rng: rng.uniform(low, high),  # type: ignore
        )


@dataclass(frozen=True)
class DontFill(Fill[DataT]):
    """
    A special Fill which does not replace missing values and simply returns the data
    as-is (no-op).

    DontFill is a frozen dataclass. It is generic in the dtype of the data it fixes.
    """

    @override
    def __call__(
        self, rng, data_np, missing_mask
    ) -> tuple[NDArray[DataT], NDArray[np.bool_] | None]:
        return data_np, missing_mask


def _add_issue(
    issues: Mapping[str, NDArray[np.bool_]],
    issue_name: str,
    issue_mask: NDArray[np.bool_] | None,
) -> Mapping[str, NDArray[np.bool_]]:
    """
    Utility function for adding an issue to a list of issues,
    but only if the mask is not None and not "no mask" or all-False.
    """
    if issue_mask is not None and issue_mask.any():
        return {**issues, issue_name: issue_mask}
    return issues


def _all_issues_mask(issues: Mapping[str, NDArray[np.bool_]]):
    """
    Utility function for computing the logical union of the masks of a set of issues.
    """
    return reduce(
        np.logical_or,
        [m for _, m in issues.items()],
        np.ma.nomask,
    )


@dataclass(frozen=True)
class PipelineResult(Generic[DataT]):
    """
    An object containing the result of processing data through a DataPipeline.

    PipelineResult is a frozen dataclass. It is generic in the dtype of the
    resulting numpy array (`DataT`).
    """

    value: NDArray[DataT]
    """
    The resulting numpy array. In this form, the array will never masked, even if there
    are issues. If you want a masked array, see the `value_as_masked` property.
    """
    issues: Mapping[str, NDArray[np.bool_]]
    """
    The set of outstanding issues in the underlying data, with issue-specific masks.
    """

    def __post_init__(self):
        if np.ma.is_masked(self.value):
            err = "PipelineResult `value` should not be masked directly."
            raise ValueError(err)

    @property
    def value_as_masked(self) -> NDArray[DataT]:
        """
        The resulting numpy array which will be masked if-and-only-if there are issues.
        The mask is computed as the logical union of the individual issue masks.
        """
        if len(self.issues) == 0:
            return self.value
        mask = _all_issues_mask(self.issues)
        if not np.any(mask):
            return self.value
        return np.ma.masked_array(data=self.value, mask=mask)

    def with_issue(
        self,
        issue_name: str,
        issue_mask: NDArray[np.bool_] | None,
    ) -> Self:
        """
        Updates the result by adding a data issue.

        Parameters
        ----------
        issue_name : str
            The name of the issue.
        issue_mask : NDArray[np.bool_] | None,
            The mask indicating which values are affected by the issue. For convenience,
            the mask may be None or "no mask" to indicate the data does not have the
            named issue in fact, in which case the issue will not be added.

        Returns
        -------
        PipelineResult
            The updated result (copy).
        """
        return dataclasses.replace(
            self,
            issues=_add_issue(self.issues, issue_name, issue_mask),
        )

    def to_date_value(
        self,
        dates: NDArray[np.datetime64],
    ) -> "PipelineResult[DateValueType]":
        """
        Converts the result to a date-value-tuple array.

        Parameters
        ----------
        dates : NDArray[np.datetime64]
            The one-dimensional array of dates.

        Returns
        -------
        PipelineResult[DateValueType]
            The updated result (copy).

        See Also
        --------
        [`to_date_value_array`](`epymorph.util.to_date_value_array`) for more detail
        on how dates and values are combined,
        and [`extract_date_value`](`epymorph.util.extract_date_value`) for a convenient
        way to separate the dates and values when needed.
        """
        value = to_date_value_array(dates, self.value)
        return PipelineResult(value, self.issues)

    @staticmethod
    def sum(
        left: "PipelineResult[DataT]",
        right: "PipelineResult[DataT]",
        *,
        left_prefix: str,
        right_prefix: str,
    ) -> "PipelineResult[DataT]":
        """
        Combines two PipelineResults by summing unmasked data values.
        The result will include both lists of data issues by prefixing the issue names.

        Parameters
        ----------
        left : PipelineResult[DataT]
            The first addend.
        right : PipelineResult[DataT]
            The second addend.
        left_prefix : str
            A prefix to assign to any left-side issues.
        right_prefix : str
            A prefix to assign to any right-side issues.

        Returns
        -------
        PipelineResult[DataT]
            The combined result (copy).
        """
        if left.value.shape != right.value.shape:
            err = "When summing PipelineResults, their data shapes must be the same."
            raise ValueError(err)
        if np.dtype(left.value.dtype) != np.dtype(right.value.dtype):
            err = "When summing PipelineResults, their dtypes must be the same."
            raise ValueError(err)
        if not (
            np.issubdtype(left.value.dtype, np.integer)
            or np.issubdtype(left.value.dtype, np.floating)
        ):
            err = (
                "When summing PipelineResults, their dtypes must be integer or "
                "floating point numbers."
            )
            raise ValueError(err)

        new_issues = {
            **{f"{left_prefix}{iss}": m for iss, m in left.issues.items()},
            **{f"{right_prefix}{iss}": m for iss, m in right.issues.items()},
        }
        unmasked = ~_all_issues_mask(new_issues)
        new_value = np.zeros_like(left.value)
        new_value[unmasked] = left.value[unmasked] + right.value[unmasked]
        return PipelineResult(value=new_value, issues=new_issues)


class PivotAxis(NamedTuple):
    """
    Describes an axis on which a DataFrame will be pivoted to become a numpy array.

    PivotAxis is a NamedTuple.
    """

    column: str
    """The name of the column in a DataFrame."""
    values: list | NDArray
    """
    The set of values we expect to find in the column. This will be used to expand and
    reorder the resulting pivot table. If values are in this set and not in the data,
    the table will contain missing values -- which is better than not knowing which
    values are missing!
    """


class _PipelineState(NamedTuple):
    """
    The state of a data pipeline.

    PipelineState is a NamedTuple.
    """

    data_df: pd.DataFrame
    """The data being worked on."""
    issues: Mapping[str, NDArray[np.bool_]] = {}
    """A map of outstanding issues that have been discovered in the data."""

    def next(self, updated_df: pd.DataFrame) -> "_PipelineState":
        """
        Advances the state by updating the working data. Pipeline steps should avoid
        mutating the previous state's data and instead work on a new copy.

        Parameters
        ----------
        updated_df : DataFrame
            The updated working data.

        Returns
        -------
        PipelineState
            The updated state (copy).
        """
        return _PipelineState(updated_df, self.issues)

    def next_with_issue(
        self,
        updated_df: pd.DataFrame,
        issue_name: str,
        issue_mask: NDArray[np.bool_] | None,
    ) -> "_PipelineState":
        """
        Advances the state by updating the working data and adding a data issue.
        Pipeline steps should avoid mutating the previous state's data and instead
        work on a new copy.

        Parameters
        ----------
        updated_df : DataFrame
            The updated working data.
        issue_name : str
            The name of the issue.
        issue_mask : NDArray[np.bool_] | None,
            The mask indicating which values are affected by the issue. For convenience,
            the mask may be None or "no mask" to indicate the data does not have the
            named issue in fact, in which case the issue will not be added.

        Returns
        -------
        PipelineState
            The updated state (copy).
        """
        return _PipelineState(
            data_df=updated_df,
            issues=_add_issue(self.issues, issue_name, issue_mask),
        )


_PipelineStep = Callable[[_PipelineState], _PipelineState]
"""
A step in a data pipeline is just a function from one state to another.
"""


@dataclass(frozen=True)
class DataPipeline(Generic[DataT]):
    """
    DataPipeline is a factory class for assembling data processing pipelines.

    Using builder-style syntax you define the processing steps that the data should
    flow through. Finalizing the pipeline yields a function that takes a DataFrame,
    executes the pipeline steps in sequence, and returns a PipelineResult containing
    the processed data and any unresolved data issues discovered along the way.
    The DataPipeline instance itself can be discarded after the processing function
    is finalized.

    DataPipeline was designed to produce arrays with one or two dimensions. When
    there is more than one value in the "columns" dimension, it's obvious we should
    have a 2D array. But when there's only one column, a 1D or 2D array layout are
    both valid. Because of this ambiguity, it's up to you to provide the number
    of dimensions you expect. If you specify `ndims` as 1 and the data has more than
    one column, this will result in an error.

    DataPipeline is a frozen dataclass. It is generic in the dtype of the data is
    processes.
    """

    axes: tuple[PivotAxis, PivotAxis]
    """
    The definition of the axes which will be used to tabulate the data.
    The first axis represents rows in the result, and the second columns.

    """
    ndims: Literal[1, 2]
    """The number of dimensions expected in the result: 1 or 2."""
    dtype: type[DataT]
    """The dtype of the data values in the result."""
    rng: HasRandomness
    """A source of randomness."""

    pipeline_steps: Sequence[_PipelineStep] = field(default_factory=list)
    """The accumulated pipeline steps."""

    def __post_init__(self):
        if self.ndims == 1 and len(self.axes[1].values) > 1:
            err = (
                "If `ndims` is 1, the second axis definition in `axes` must contain "
                "exactly one value."
            )
            raise ValueError(err)
        if any(len(x.values) == 0 for x in self.axes):
            err = "All axis definitions in `axes` must contain at least one value."
            raise ValueError(err)

    def _and_then(self, f: _PipelineStep) -> Self:
        """Returns a copy of this pipeline with a pipeline step appended."""
        return dataclasses.replace(self, pipeline_steps=[*self.pipeline_steps, f])

    def _process(self, data_df: pd.DataFrame) -> _PipelineState:
        """Apply all of the pipeline steps to an input."""
        state = _PipelineState(data_df)
        for f in self.pipeline_steps:
            state = f(state)
        return state

    def map_series(
        self,
        column: str,
        map_fn: Callable[[pd.Series], pd.Series] | None = None,
    ) -> Self:
        """
        Add a pipeline step that transforms a column of the DataFrame
        by applying a mapping function to the series.

        Parameters
        ----------
        column : str
            The name of the column to transform.
        map_fn : Callable[[Series], Series], optional
            The series mapping function. As a convenience you may pass None,
            in which case this is a no-op.

        Returns
        -------
        DataPipeline
            A copy of this pipeline with the step added.
        """
        if map_fn is None:
            return self

        def map_series(state: _PipelineState) -> _PipelineState:
            series = map_fn(state.data_df[column])
            data_df = state.data_df.assign(**{column: series})
            return state.next(data_df)

        return self._and_then(map_series)

    def map_column(
        self,
        column: str,
        map_fn: Callable | None = None,
    ) -> Self:
        """
        Add a pipeline step that transforms a column of the DataFrame
        by applying a mapping function to all values in the series.

        Parameters
        ----------
        column : str
            The name of the column to transform.
        map_fn : Callable, optional
            The value mapping function. As a convenience you may pass None,
            in which case this is a no-op.

        Returns
        -------
        DataPipeline
            A copy of this pipeline with the step added.
        """
        if map_fn is None:
            return self

        return self.map_series(
            column=column,
            map_fn=lambda xs: xs.apply(map_fn),
        )

    def strip_sentinel(
        self,
        sentinel_name: str,
        sentinel_value: DataT,
        fix: "Fix[DataT]",
    ) -> Self:
        """
        Add a pipeline step for dealing with sentinel values in the DataFrame.
        First we apply the given Fix, then check for any remaining sentinel values.
        If sentinel values still remain in the data, these are recorded as a data
        issue with an associated mask.

        Parameters
        ----------
        sentinel_name : str
            The name used for the data issue if any sentinel values remain.
        sentinel_value : Any
            The value considered a sentinel.
        fix : Fix[DataT]
            The fix to apply to attempt to replace sentinel values.

        Returns
        -------
        DataPipeline
            A copy of this pipeline with the step added.
        """

        def strip_sentinel(state: _PipelineState) -> _PipelineState:
            # Apply fix.
            work_df = fix(self.rng, sentinel_value, ("value",), state.data_df)

            # Compute the mask for any remaining sentinels.
            is_sentinel = work_df["value"] == sentinel_value
            mask = (
                work_df.loc[is_sentinel]
                .pivot_table(
                    index=self.axes[0].column,
                    columns=self.axes[1].column,
                    values="value",
                    aggfunc=lambda _: True,
                    fill_value=False,
                )
                .reindex(
                    index=self.axes[0].values,
                    columns=self.axes[1].values,
                    fill_value=False,
                )
                .to_numpy(dtype=np.bool_)
            )

            # pivot_table produces 2D results; adjust if we expect a 1D result.
            if self.ndims == 1:
                if mask.shape[1] > 1:
                    err = (
                        "You specified `ndims` as 1, but the data contains more than "
                        "one column so it is impossible to convert it to this form."
                    )
                    raise Exception(err)
                mask = mask[:, 0]

            # Strip out sentinel values.
            work_df = work_df.loc[~is_sentinel]
            return state.next_with_issue(work_df, sentinel_name, mask)

        return self._and_then(strip_sentinel)

    def strip_na_as_sentinel(
        self,
        column: str,
        sentinel_name: str,
        sentinel_value: DataT,
        fix: "Fix[DataT]",
    ) -> Self:
        """
        Add a pipeline step for dealing with NaN/NA/null values in the DataFrame.
        First replace NA values with a user-defined sentinel value, then apply the
        given Fix. Finally check for any remaining such values. If sentinel values
        still remain in the data, these are recorded as a data issue with an associated
        mask.

        Parameters
        ----------
        column : str
            The name of the column to transform.
        sentinel_name : str
            The name used for the data issue if any NA/sentinel values remain.
        sentinel_value : Any
            The value to use to replace NA values. We want to replace NAs so that
            we can universally convert the data column to the desired type --
            np.int64 doesn't support NA values like np.float64 does, so this allows
            the input DataFrame to start with something like Pandas' "Int64" data type
            while the pipeline produces np.int64 results.
            The sentinel value chosen for this must not already exist in the data.
        fix : Fix[DataT]
            The fix to apply to attempt to replace NA/sentinel values.

        Returns
        -------
        DataPipeline
            A copy of this pipeline with the step added.

        Raises
        ------
        Exception
            If the data naturally contains the chosen sentinel value.
        """

        def replace_na(state: _PipelineState) -> _PipelineState:
            series = state.data_df[column]
            if (series == sentinel_value).any():
                # the below S608 Ruff warning is a false positive;
                # apparently it thinks this error message looks like a SQL statement
                err = (
                    "The data contains the proposed artificial sentinel value "  # noqa: S608
                    f"({sentinel_value}), so if we replace NA values with this value "
                    "we'll get missing values mixed up with others. Please choose a "
                    "sentinel value that is guaranteed not to be in the data naturally."
                )
                raise Exception(err)

            series = series.fillna(sentinel_value).astype(self.dtype)  # type: ignore
            data_df = state.data_df.assign(**{column: series})
            return state.next(data_df)

        return self._and_then(replace_na).strip_sentinel(
            sentinel_name, sentinel_value, fix
        )

    def finalize(
        self,
        fill_missing: "Fill[DataT]",
    ) -> Callable[[pd.DataFrame], PipelineResult[DataT]]:
        """
        Completes construction of the pipeline.

        Parameters
        ----------
        fill_missing : Fill[DataT]
            A method for filling in missing data.

        Returns
        -------
        Callable[[DataFrame], PipelineResult[DataT]]
            The prepared pipeline: a function that processes a DataFrame and produces a
            result.
        """

        def pipeline(data_df: pd.DataFrame) -> PipelineResult[DataT]:
            # Run the accumulated processing pipeline.
            state = self._process(data_df)

            # Tabulate data; reindexing step can expose missing values.
            result_np = cast(
                NDArray[DataT],
                (
                    state.data_df.pivot_table(
                        index=self.axes[0].column,
                        columns=self.axes[1].column,
                        values="value",
                        aggfunc="sum",
                        fill_value=0,
                    )
                    .reindex(
                        index=self.axes[0].values,
                        columns=self.axes[1].values,
                        fill_value=0,
                    )
                    .to_numpy(dtype=self.dtype)
                ),
            )
            missing_mask = cast(
                NDArray[np.bool_],
                (
                    state.data_df.assign(value=False)
                    .pivot_table(
                        index=self.axes[0].column,
                        columns=self.axes[1].column,
                        values="value",
                        aggfunc=lambda _: False,
                        fill_value=True,
                    )
                    .reindex(
                        index=self.axes[0].values,
                        columns=self.axes[1].values,
                        fill_value=True,
                    )
                ).to_numpy(dtype=np.bool_),
            )

            # pivot_table produces 2D results; adjust if we expect a 1D result.
            if self.ndims == 1:
                if missing_mask.shape[1] > 1 or result_np.shape[1] > 1:
                    err = (
                        "You specified `ndims` as 1, but the data contains more than "
                        "one column so it is impossible to convert it to this form."
                    )
                    raise Exception(err)
                result_np = result_np[:, 0]
                missing_mask = missing_mask[:, 0]

            # If a value is present in an issue mask it's not missing,
            # but for calculation purposes we may have removed it
            # (e.g., stripping sentinels.). Correct that now.
            all_issues_mask = reduce(
                np.logical_or,
                [m for _, m in state.issues.items()],
                np.ma.nomask,
            )
            missing_mask = missing_mask & ~all_issues_mask

            # Discover and fill missing data.
            result_np, missing_mask = fill_missing(self.rng, result_np, missing_mask)
            state = state.next_with_issue(state.data_df, "missing", missing_mask)
            return PipelineResult(result_np, state.issues)

        return pipeline
