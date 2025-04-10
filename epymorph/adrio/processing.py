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


class HasRandomness(Protocol):
    @property
    @abstractmethod
    def rng(self) -> np.random.Generator:
        pass


class Fix(ABC, Generic[DataT]):
    @abstractmethod
    def __call__(
        self,
        rng: HasRandomness,
        replace: DataT,
        data_df: pd.DataFrame,
    ) -> pd.DataFrame:
        pass

    @staticmethod
    def of_int64(
        fix: "Fix[np.int64] | int | Callable[[], int] | Literal[False]",
    ) -> "Fix[np.int64]":
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
    with_value: DataT

    @override
    def __call__(self, rng, replace, data_df):
        to_replace = {replace: self.with_value}
        values = data_df["value"].replace(to_replace)
        return data_df.assign(value=values)


@dataclass(frozen=True)
class FunctionFix(Fix[DataT]):
    with_function: Callable[[], DataT]

    @override
    def __call__(self, rng, replace, data_df):
        return FunctionFix.apply(rng, data_df, replace, self.with_function)

    @staticmethod
    def apply(
        rng: HasRandomness,
        data_df: pd.DataFrame,
        replace: DataT,
        with_function: Callable[[], DataT],
    ) -> pd.DataFrame:
        is_replace = data_df["value"] == replace
        num_replace = is_replace.sum()
        replacements = np.array([with_function() for _ in range(num_replace)])
        values = data_df["value"].copy()
        values[is_replace] = replacements
        return data_df.assign(value=values)


@dataclass(frozen=True)
class RandomFix(Fix[DataT]):
    with_random: Callable[[np.random.Generator], DataT]

    @override
    def __call__(self, rng, replace, data_df):
        random_fn = partial(self.with_random, rng.rng)
        return FunctionFix.apply(rng, data_df, replace, random_fn)

    @staticmethod
    def from_range(low: int, high: int) -> "RandomFix[np.int64]":
        return RandomFix(
            lambda rng: rng.integers(low, high, endpoint=True),  # type: ignore
        )


@dataclass(frozen=True)
class DontFix(Fix[Any]):
    @override
    def __call__(self, rng, replace, data_df):
        return data_df


class Fill(ABC, Generic[DataT]):
    @abstractmethod
    def __call__(
        self,
        rng: HasRandomness,
        data_np: NDArray[DataT],
        missing_mask: NDArray[np.bool_],
    ) -> tuple[NDArray[DataT], NDArray[np.bool_] | None]:
        pass

    @staticmethod
    def of_int64(
        fill: "Fill[np.int64] | int | Callable[[], int] | Literal[False]",
    ) -> "Fill[np.int64]":
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
    with_value: DataT

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
    with_function: Callable[[], DataT]

    @override
    def __call__(
        self, rng, data_np, missing_mask
    ) -> tuple[NDArray[DataT], NDArray[np.bool_] | None]:
        num_replace = missing_mask.sum()
        replacements = np.array([self.with_function() for _ in range(num_replace)])
        result_np = data_np.copy()
        result_np[missing_mask] = replacements
        return result_np, None


@dataclass(frozen=True)
class DontFill(Fill[DataT]):
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
    if issue_mask is not None and issue_mask.any():
        return {**issues, issue_name: issue_mask}
    return issues


@dataclass(frozen=True)
class ProcessResult(Generic[DataT]):
    value: NDArray[DataT]
    issues: Mapping[str, NDArray[np.bool_]]

    def with_issue(
        self,
        issue_name: str,
        issue_mask: NDArray[np.bool_] | None,
    ) -> Self:
        return dataclasses.replace(
            self,
            issues=_add_issue(self.issues, issue_name, issue_mask),
        )

    def to_date_value(
        self,
        dates: NDArray[np.datetime64],
    ) -> "ProcessResult[DateValueType]":
        value = to_date_value_array(dates, self.value)
        return ProcessResult(value, self.issues)

    @staticmethod
    def sum(
        left: "ProcessResult[DataT]",
        right: "ProcessResult[DataT]",
        *,
        left_prefix: str,
        right_prefix: str,
    ) -> "ProcessResult[DataT]":
        """
        Combines two ProcessResults by summing unmasked data values.
        The result will include both lists of data issues by prefixing the issue names.
        """
        if left.value.shape != right.value.shape:
            err = "When summing ProcessResults, their data shapes must be the same."
            raise ValueError(err)
        if np.dtype(left.value.dtype) != np.dtype(right.value.dtype):
            err = "When summing ProcessResults, their dtypes must be the same."
            raise ValueError(err)
        if not (
            np.issubdtype(left.value.dtype, np.integer)
            or np.issubdtype(left.value.dtype, np.floating)
        ):
            err = (
                "When summing ProcessResults, their dtypes must be integer or "
                "floating point numbers."
            )
            raise ValueError(err)

        new_issues = {
            **{f"{left_prefix}{iss}": m for iss, m in left.issues.items()},
            **{f"{right_prefix}{iss}": m for iss, m in right.issues.items()},
        }
        unmasked = ~reduce(
            np.logical_or,
            [m for _, m in new_issues.items()],
            np.ma.nomask,
        )
        new_value = np.zeros_like(left.value)
        new_value[unmasked] = left.value[unmasked] + right.value[unmasked]
        return ProcessResult(value=new_value, issues=new_issues)


class PivotAxis(NamedTuple):
    column: str
    values: list | NDArray


class PipelineState(NamedTuple):
    data_df: pd.DataFrame
    issues: Mapping[str, NDArray[np.bool_]] = {}

    def next(self, updated_df: pd.DataFrame) -> "PipelineState":
        return PipelineState(updated_df, self.issues)

    def next_with_issue(
        self,
        updated_df: pd.DataFrame,
        issue_name: str,
        issue_mask: NDArray[np.bool_] | None,
    ) -> "PipelineState":
        return PipelineState(
            data_df=updated_df,
            issues=_add_issue(self.issues, issue_name, issue_mask),
        )


_PipelineStep = Callable[[PipelineState], PipelineState]


@dataclass(frozen=True)
class DataPipeline(Generic[DataT]):
    axes: tuple[PivotAxis, PivotAxis]
    ndims: Literal[1, 2]
    dtype: type[DataT]
    rng: HasRandomness

    pipeline_steps: Sequence[_PipelineStep] = field(default_factory=list)

    def _and_then(self, f: _PipelineStep) -> Self:
        return dataclasses.replace(self, pipeline_steps=[*self.pipeline_steps, f])

    def _process(self, data_df: pd.DataFrame) -> PipelineState:
        state = PipelineState(data_df)
        for f in self.pipeline_steps:
            state = f(state)
        return state

    def map_series(
        self,
        column: str,
        map_fn: Callable[[pd.Series], pd.Series] | None = None,
        dtype: type[np.generic] | None = None,
    ) -> Self:
        if map_fn is None and dtype is None:
            return self

        def map_series(state: PipelineState) -> PipelineState:
            series = state.data_df[column]
            if map_fn is not None:
                series = map_fn(series)
            if dtype is not None:
                series = series.astype(dtype)
            data_df = state.data_df.assign(**{column: series})
            return state.next(data_df)

        return self._and_then(map_series)

    def map_column(
        self,
        column: str,
        map_fn: Callable | None = None,
        dtype: type[np.generic] | None = None,
    ) -> Self:
        map_series_fn = None if map_fn is None else lambda xs: xs.apply(map_fn)
        return self.map_series(column=column, map_fn=map_series_fn, dtype=dtype)

    def strip_sentinel(
        self,
        sentinel_name: str,
        sentinel_value: Any,
        fix: "Fix[DataT]",
    ) -> Self:
        def strip_sentinel(state: PipelineState) -> PipelineState:
            # Apply fix.
            work_df = fix(self.rng, sentinel_value, state.data_df)

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
            if self.ndims == 1:
                if mask.shape[1] > 1:
                    raise Exception("?")  # TODO
                mask = mask[:, 0]

            # Strip out sentinel values.
            work_df = work_df.loc[~is_sentinel]
            return state.next_with_issue(work_df, sentinel_name, mask)

        return self._and_then(strip_sentinel)

    def strip_na_as_sentinel(
        self,
        sentinel_name: str,
        sentinel_value: Any,
        fix: "Fix[DataT]",
    ) -> Self:
        def replace_na(state: PipelineState) -> PipelineState:
            # TODO: if value already contains the proposed sentinel,
            # maybe we should raise an error
            value = state.data_df["value"].fillna(sentinel_value).astype(np.int64)
            work_df = state.data_df.assign(value=value)
            return state.next(work_df)

        pipe = self._and_then(replace_na)
        return pipe.strip_sentinel(sentinel_name, sentinel_value, fix)

    def finalize(
        self,
        fill_missing: "Fill[DataT]",
    ) -> Callable[[pd.DataFrame], ProcessResult[DataT]]:
        def pipeline(data_df: pd.DataFrame) -> ProcessResult[DataT]:
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
            if self.ndims == 1:
                if missing_mask.shape[1] > 1 or result_np.shape[1] > 1:
                    raise Exception("?")  # TODO
                result_np = result_np[:, 0]
                missing_mask = missing_mask[:, 0]

            # Discover and fill missing data.
            result_np, missing_mask = fill_missing(self.rng, result_np, missing_mask)
            state = state.next_with_issue(state.data_df, "missing", missing_mask)
            return ProcessResult(result_np, state.issues)

        return pipeline
