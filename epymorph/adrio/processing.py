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
    Self,
    Sequence,
    TypeVar,
    cast,
)

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.simulation import Context
from epymorph.util import DateValueType, to_date_value_array

DataT = TypeVar("DataT", bound=np.generic)


class Fix(ABC, Generic[DataT]):
    @abstractmethod
    def __call__(
        self,
        context: Context,
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
    def __call__(
        self,
        context: Context,
        replace: DataT,
        data_df: pd.DataFrame,
    ) -> pd.DataFrame:
        to_replace = {replace: self.with_value}
        values = data_df["value"].replace(to_replace)
        return data_df.assign(value=values)


@dataclass(frozen=True)
class FunctionFix(Fix[DataT]):
    with_function: Callable[[], DataT]

    @override
    def __call__(
        self,
        context: Context,
        replace: DataT,
        data_df: pd.DataFrame,
    ) -> pd.DataFrame:
        return FunctionFix.apply(context, data_df, replace, self.with_function)

    @staticmethod
    def apply(
        context: Context,
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
    def __call__(
        self,
        context: Context,
        replace: DataT,
        data_df: pd.DataFrame,
    ) -> pd.DataFrame:
        random_fn = partial(self.with_random, context.rng)
        return FunctionFix.apply(context, data_df, replace, random_fn)

    @staticmethod
    def from_range(low: int, high: int) -> "RandomFix[np.int64]":
        return RandomFix(
            lambda rng: rng.integers(low, high, endpoint=True),  # type: ignore
        )


@dataclass(frozen=True)
class DontFix(Fix[Any]):
    @override
    def __call__(
        self,
        context: Context,
        replace: Any,
        data_df: pd.DataFrame,
    ) -> pd.DataFrame:
        return data_df


class Fill(ABC, Generic[DataT]):
    @abstractmethod
    def __call__(
        self,
        context: Context,
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
        elif fill == 0:
            return AlreadyFilled[np.int64]()
        elif isinstance(fill, int):
            return ConstantFill[np.int64](fill)  # type: ignore
        elif callable(fill):
            return FunctionFill[np.int64](fill)  # type: ignore
        raise ValueError("Not a valid fill.")

    @staticmethod
    def of_float64(
        fill: "Fill[np.float64] | float | Callable[[], float] | Literal[False]",
    ) -> "Fill[np.float64]":
        if fill is False:
            return DontFill()
        elif isinstance(fill, Fill):
            return fill  # for convenience, no-op if arg is a Fill
        elif fill == 0:
            return AlreadyFilled[np.float64]()
        elif isinstance(fill, float):
            return ConstantFill[np.float64](fill)  # type: ignore
        elif callable(fill):
            return FunctionFill[np.float64](fill)  # type: ignore
        raise ValueError("Not a valid fill.")


@dataclass(frozen=True)
class AlreadyFilled(Fill[DataT]):
    @override
    def __call__(
        self,
        context: Context,
        data_np: NDArray[DataT],
        missing_mask: NDArray[np.bool_],
    ) -> tuple[NDArray[DataT], NDArray[np.bool_] | None]:
        return data_np, None


@dataclass(frozen=True)
class ConstantFill(Fill[DataT]):
    with_value: DataT

    @override
    def __call__(
        self,
        context: Context,
        data_np: NDArray[DataT],
        missing_mask: NDArray[np.bool_],
    ) -> tuple[NDArray[DataT], NDArray[np.bool_] | None]:
        result_np = data_np.copy()
        result_np[missing_mask] = self.with_value
        return result_np, None


@dataclass(frozen=True)
class FunctionFill(Fill[DataT]):
    with_function: Callable[[], DataT]

    @override
    def __call__(
        self,
        context: Context,
        data_np: NDArray[DataT],
        missing_mask: NDArray[np.bool_],
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
        self,
        context: Context,
        data_np: NDArray[DataT],
        missing_mask: NDArray[np.bool_],
    ) -> tuple[NDArray[DataT], NDArray[np.bool_] | None]:
        return data_np, missing_mask


@dataclass(frozen=True)
class ProcessResult(Generic[DataT]):
    raw: NDArray[DataT]
    issues: Sequence[tuple[str, NDArray[np.bool_]]]

    def to_date_value(
        self,
        dates: NDArray[np.datetime64],
    ) -> "ProcessResult[DateValueType]":
        return ProcessResult(
            raw=to_date_value_array(dates, self.raw),
            issues=self.issues,
        )

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
        if left.raw.shape != right.raw.shape:
            err = "When summing ProcessResults, their data shapes must be the same."
            raise ValueError(err)
        if np.dtype(left.raw.dtype) != np.dtype(right.raw.dtype):
            err = "When summing ProcessResults, their dtypes must be the same."
            raise ValueError(err)
        if not (
            np.issubdtype(left.raw.dtype, np.integer)
            or np.issubdtype(left.raw.dtype, np.floating)
        ):
            err = (
                "When summing ProcessResults, their dtypes must be integer or "
                "floating point numbers."
            )
            raise ValueError(err)

        new_issues = [
            *[(f"{left_prefix}{iss}", m) for iss, m in left.issues],
            *[(f"{right_prefix}{iss}", m) for iss, m in right.issues],
        ]
        unmasked = ~reduce(np.logical_or, [m for _, m in new_issues], np.ma.nomask)
        new_raw = np.zeros_like(left.raw)
        new_raw[unmasked] = left.raw[unmasked] + right.raw[unmasked]
        return ProcessResult(raw=new_raw, issues=new_issues)


class PivotAxis(NamedTuple):
    column: str
    values: list | NDArray


class PipelineResult(NamedTuple):
    result: NDArray
    issues: Mapping[str, NDArray[np.bool_]]


@dataclass(frozen=True)
class DataPipeline:  # TODO: Generic[DataT] ?
    context: Context  # TODO: do we really need context now, or is rng sufficient?
    data_df: pd.DataFrame
    axes: tuple[PivotAxis, PivotAxis]
    ndims: Literal[1, 2]
    dtype: type[np.generic]
    issues: Mapping[str, NDArray[np.bool_]] = field(default_factory=dict)

    def _add_issue(
        self,
        issue_name: str,
        issue_mask: NDArray[np.bool_] | None,
    ) -> Mapping[str, NDArray[np.bool_]]:
        if issue_mask is not None and issue_mask.any():
            return {**self.issues, issue_name: issue_mask}
        return self.issues

    def map_column(
        self,
        column: str,
        dtype: type[np.generic] | None,
        map_fn: Callable[[Any], Any] | None,
    ) -> Self:
        if map_fn is None and dtype is None:
            return self
        series = self.data_df[column]
        if map_fn is not None:
            series = series.apply(map_fn)
        if dtype is not None:
            series = series.astype(dtype)
        data_df = self.data_df.assign(**{column: series})
        return dataclasses.replace(self, data_df=data_df)

    def strip_sentinel(
        self,
        sentinel_name: str,
        sentinel_value: Any,
        fix: "Fix",
    ) -> Self:
        # Apply fix.
        work_df = fix(self.context, sentinel_value, self.data_df)

        # Compute the mask for any remaining sentinels.
        is_sentinel = work_df["value"] == sentinel_value
        mask = (
            work_df.pivot_table(
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
            mask = mask[:, 0]

        # Strip out sentinel values.
        work_df = work_df.loc[~is_sentinel]
        return dataclasses.replace(
            self,
            data_df=work_df,
            issues=self._add_issue(sentinel_name, mask),
        )

    def strip_na_as_sentinel(
        self,
        sentinel_name: str,
        sentinel_value: Any,
        fix: "Fix",
    ) -> Self:
        # TODO: if value already contains the proposed sentinel,
        # maybe we should raise an error
        value = self.data_df["value"].fillna(sentinel_value).astype(np.int64)
        work_df = self.data_df.assign(value=value)
        pipeline = dataclasses.replace(self, data_df=work_df)
        return pipeline.strip_sentinel(sentinel_name, sentinel_value, fix)

    def finalize(self, fill_missing: "Fill") -> PipelineResult:
        # Tabulate data.
        result_np = cast(
            NDArray,  # [DataT],
            (
                self.data_df.pivot_table(
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
        mask = cast(
            NDArray[np.bool_],
            (
                self.data_df.assign(value=False)
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
            result_np = result_np[:, 0]
            mask = mask[:, 0]

        # Discover and fill missing data.
        result_np, mask = fill_missing(self.context, result_np, mask)

        return PipelineResult(result_np, self._add_issue("missing", mask))

    def finalize_as_date_value(
        self,
        fill_missing: "Fill",
        dates: NDArray[np.datetime64],
    ) -> PipelineResult:
        result_np, issues = self.finalize(fill_missing)
        result_np = to_date_value_array(dates, result_np)
        return PipelineResult(result_np, issues)


def _mask_pivot(
    mask_df: pd.DataFrame,
    defined_value: bool,
    axis1: PivotAxis,
    axis2: PivotAxis,
    value_column: str,
) -> NDArray[np.bool_]:
    """
    Pivots the given long DataFrame into wide format,
    so as to create a boolean mask that is
    `defined_value` everywhere that `mask_df` is defined,
    and `not defined_value` everywhere `mask_df` is missing.
    Also reindexes the two axes against a known set of keys
    so that the table is complete and properly sorted.
    """
    return (
        mask_df.pivot_table(
            index=axis1.column,
            columns=axis2.column,
            values=value_column,
            aggfunc=lambda _: defined_value,
            fill_value=not defined_value,
        )
        .reindex(
            index=axis1.values,
            columns=axis2.values,
            fill_value=not defined_value,
        )
        .to_numpy(dtype=np.bool_)
    )


def strip_sentinel_n(
    context: Context,
    result_df: pd.DataFrame,
    sentinel: int,
    value_name: str,
) -> tuple[pd.DataFrame, NDArray[np.bool_]]:
    is_sentinel = result_df["value"] == sentinel
    sentinel_mask = _mask_pivot(
        result_df.loc[is_sentinel],
        defined_value=True,
        axis1=PivotAxis("geoid", context.scope.node_ids),
        axis2=PivotAxis("variable", [value_name]),
        value_column="value",
    )
    return result_df.loc[~is_sentinel], sentinel_mask[:, 0]


def strip_sentinel_nxa(
    context: Context,
    result_df: pd.DataFrame,
    sentinel: int,
    value_names: list[str],
) -> tuple[pd.DataFrame, NDArray[np.bool_]]:
    is_sentinel = result_df["value"] == sentinel
    sentinel_mask = _mask_pivot(
        result_df.loc[is_sentinel],
        defined_value=True,
        axis1=PivotAxis("geoid", context.scope.node_ids),
        axis2=PivotAxis("variable", value_names),
        value_column="value",
    )
    return result_df.loc[~is_sentinel], sentinel_mask


def strip_sentinel_txn(
    time_series: NDArray[np.datetime64],
    context: Context,
    result_df: pd.DataFrame,
    sentinel: int,
) -> tuple[pd.DataFrame, NDArray[np.bool_]]:
    is_sentinel = result_df["value"] == sentinel
    sentinel_mask = _mask_pivot(
        result_df.loc[is_sentinel],
        defined_value=True,
        axis1=PivotAxis("date", time_series),
        axis2=PivotAxis("geoid", context.scope.node_ids),
        value_column="value",
    )
    return result_df.loc[~is_sentinel], sentinel_mask


def strip_sentinel_nxn(
    context: Context,
    result_df: pd.DataFrame,
    sentinel: int,
) -> tuple[pd.DataFrame, NDArray[np.bool_]]:
    is_sentinel = result_df["value"] == sentinel
    sentinel_mask = _mask_pivot(
        result_df.loc[is_sentinel],
        defined_value=True,
        axis1=PivotAxis("geoid_src", context.scope.node_ids),
        axis2=PivotAxis("geoid_dst", context.scope.node_ids),
        value_column="value",
    )
    return result_df.loc[~is_sentinel], sentinel_mask


def tabulate_data_n(
    context: Context,
    result_df: pd.DataFrame,
    dtype: type[DataT],
    value_name: str,
) -> tuple[NDArray[DataT], NDArray[np.bool_]]:
    # result_df must be a long data frame with columns
    # geoid, variable, and value
    # (though variable is expected to contain only one unique value)
    # sentinel values have been removed
    # groups will be aggregated by sum
    data = cast(
        NDArray[DataT],
        (
            # pivot_table serves to group and agg duplicate geoids, if any
            result_df.pivot_table(
                index="geoid",
                columns="variable",
                values="value",
                aggfunc="sum",
                fill_value=0,
            )
            .reindex(
                index=context.scope.node_ids,
                columns=[value_name],
                fill_value=0,
            )
            .to_numpy(dtype=dtype)
        ),
    )
    missing_mask = cast(
        NDArray[np.bool_],
        (
            result_df.assign(value=False)
            .pivot_table(
                index="geoid",
                columns="variable",
                values="value",
                aggfunc=lambda _: False,
                fill_value=True,
            )
            .reindex(
                index=context.scope.node_ids,
                columns=[value_name],
                fill_value=True,
            )
        ).to_numpy(dtype=np.bool_),
    )
    return data[:, 0], missing_mask[:, 0]


def tabulate_data_nxa(
    context: Context,
    result_df: pd.DataFrame,
    dtype: type[DataT],
    value_names: list[str],
) -> tuple[NDArray[DataT], NDArray[np.bool_]]:
    # result_df must be a long data frame with columns
    # geoid, variable, and value
    # sentinel values have been removed
    # groups will be aggregated by sum
    data = cast(
        NDArray[DataT],
        (
            # pivot_table serves to group and agg duplicate geoids, if any
            result_df.pivot_table(
                index="geoid",
                columns="variable",
                values="value",
                aggfunc="sum",
                fill_value=0,
            )
            .reindex(
                index=context.scope.node_ids,
                columns=value_names,
                fill_value=0,
            )
            .to_numpy(dtype=dtype)
        ),
    )
    missing_mask = cast(
        NDArray[np.bool_],
        (
            result_df.assign(value=False)
            .pivot_table(
                index="geoid",
                columns="variable",
                values="value",
                aggfunc=lambda _: False,
                fill_value=True,
            )
            .reindex(
                index=context.scope.node_ids,
                columns=value_names,
                fill_value=True,
            )
        ).to_numpy(dtype=np.bool_),
    )
    return data, missing_mask


def tabulate_data_txn(
    time_series: NDArray[np.datetime64],
    context: Context,
    result_df: pd.DataFrame,
    dtype: type[DataT],
) -> tuple[NDArray[DataT], NDArray[np.bool_]]:
    # result_df must be a long data frame with columns
    # date, geoid, and value
    # sentinel values have been removed
    # groups will be aggregated by sum
    data = cast(
        NDArray[DataT],
        (
            result_df.pivot_table(
                index="date",
                columns="geoid",
                values="value",
                aggfunc="sum",
                fill_value=0,
            )
            .reindex(
                index=time_series,
                columns=context.scope.node_ids,
                fill_value=0,
            )
            .to_numpy(dtype=dtype)
        ),
    )
    missing_mask = cast(
        NDArray[np.bool_],
        (
            result_df.assign(value=False)
            .pivot_table(
                index="date",
                columns="geoid",
                values="value",
                aggfunc=lambda _: False,
                fill_value=True,
            )
            .reindex(
                index=time_series,
                columns=context.scope.node_ids,
                fill_value=True,
            )
        ).to_numpy(dtype=np.bool_),
    )
    return data, missing_mask


def tabulate_data_nxn(
    context: Context,
    result_df: pd.DataFrame,
    dtype: type[DataT],
) -> tuple[NDArray[DataT], NDArray[np.bool_]]:
    # result_df must be a long data frame with columns
    # geoid_src, geoid_dst, and value
    # sentinel values have been removed
    # groups will be aggregated by sum
    data = cast(
        NDArray[DataT],
        (
            result_df.pivot_table(
                index="geoid_src",
                columns="geoid_dst",
                values="value",
                aggfunc="sum",
                fill_value=0,
            )
            .reindex(
                index=context.scope.node_ids,
                columns=context.scope.node_ids,
                fill_value=0,
            )
            .to_numpy(dtype=dtype)
        ),
    )
    missing_mask = cast(
        NDArray[np.bool_],
        (
            result_df.assign(value=False)
            .pivot_table(
                index="geoid_src",
                columns="geoid_dst",
                values="value",
                aggfunc=lambda _: False,
                fill_value=True,
            )
            .reindex(
                index=context.scope.node_ids,
                columns=context.scope.node_ids,
                fill_value=True,
            )
        ).to_numpy(dtype=np.bool_),
    )
    return data, missing_mask


class FixSentinel(NamedTuple, Generic[DataT]):
    name: str
    value: DataT
    fix: Fix[DataT]


def process_n(
    *,
    sentinels: list[FixSentinel],
    fix_missing: Fill[DataT],
    dtype: type[DataT],
    context: Context,
    data_df: pd.DataFrame,
    value_name: str,
) -> ProcessResult[DataT]:
    if pd.isna(data_df["value"]).any():
        err = "Please remove all NA-like values prior to processing."
        raise ValueError(err)
    if data_df["value"].dtype == "Int64":
        err = (
            "Please convert Int64 (nullable) columns to (non-nullable) int64 columns "
            "prior to processing."
        )
        raise ValueError(err)

    issues = []
    work_df = data_df

    for sentinel_name, sentinel_value, fix_sentinel in sentinels:
        work_df = fix_sentinel(context, sentinel_value, work_df)
        work_df, mask = strip_sentinel_n(
            context,
            work_df,
            sentinel_value,
            value_name=value_name,
        )
        if mask is not None and mask.any():
            issues.append((sentinel_name, mask))

    tab_np, missing_mask = tabulate_data_n(context, work_df, dtype, value_name)
    data_np, missing_mask = fix_missing(context, tab_np, missing_mask)

    if missing_mask is not None and missing_mask.any():
        issues.append(("missing", missing_mask))

    return ProcessResult(data_np, issues)


def process_nxa(
    *,
    sentinels: list[FixSentinel],
    fix_missing: Fill[DataT],
    dtype: type[DataT],
    context: Context,
    data_df: pd.DataFrame,
    value_names: list[str],
) -> ProcessResult[DataT]:
    if pd.isna(data_df["value"]).any():
        err = "Please remove all NA-like values prior to processing."
        raise ValueError(err)
    if data_df["value"].dtype == "Int64":
        err = (
            "Please convert Int64 (nullable) columns to (non-nullable) int64 columns "
            "prior to processing."
        )
        raise ValueError(err)

    issues = []
    work_df = data_df

    for sentinel_name, sentinel_value, fix_sentinel in sentinels:
        work_df = fix_sentinel(context, sentinel_value, work_df)
        work_df, mask = strip_sentinel_nxa(
            context,
            work_df,
            sentinel_value,
            value_names=value_names,
        )
        if mask is not None and mask.any():
            issues.append((sentinel_name, mask))

    tab_np, missing_mask = tabulate_data_nxa(context, work_df, dtype, value_names)
    data_np, missing_mask = fix_missing(context, tab_np, missing_mask)

    if missing_mask is not None and missing_mask.any():
        issues.append(("missing", missing_mask))

    return ProcessResult(data_np, issues)


def process_txn(
    *,
    sentinels: list[FixSentinel],
    fix_missing: Fill[DataT],
    dtype: type[DataT],
    context: Context,
    data_df: pd.DataFrame,
    time_series: NDArray[np.datetime64],
) -> ProcessResult[DataT]:
    if pd.isna(data_df["value"]).any():
        err = "Please remove all NA-like values prior to processing."
        raise ValueError(err)
    if data_df["value"].dtype == "Int64":
        err = (
            "Please convert Int64 (nullable) columns to (non-nullable) int64 columns "
            "prior to processing."
        )
        raise ValueError(err)

    issues = []
    work_df = data_df

    for sentinel_name, sentinel_value, fix_sentinel in sentinels:
        work_df = fix_sentinel(context, sentinel_value, work_df)
        work_df, mask = strip_sentinel_txn(
            time_series,
            context,
            work_df,
            sentinel_value,
        )
        if mask is not None and mask.any():
            issues.append((sentinel_name, mask))

    tab_np, missing_mask = tabulate_data_txn(time_series, context, work_df, dtype)
    data_np, missing_mask = fix_missing(context, tab_np, missing_mask)

    if missing_mask is not None and missing_mask.any():
        issues.append(("missing", missing_mask))

    return ProcessResult(data_np, issues)


def process_nxn(
    *,
    sentinels: list[FixSentinel],
    fix_missing: Fill[DataT],
    dtype: type[DataT],
    context: Context,
    data_df: pd.DataFrame,
) -> ProcessResult[DataT]:
    if pd.isna(data_df["value"]).any():
        err = "Please remove all NA-like values prior to processing."
        raise ValueError(err)
    if data_df["value"].dtype == "Int64":
        err = (
            "Please convert Int64 (nullable) columns to (non-nullable) int64 columns "
            "prior to processing."
        )
        raise ValueError(err)

    issues = []
    work_df = data_df

    for sentinel_name, sentinel_value, fix_sentinel in sentinels:
        work_df = fix_sentinel(context, sentinel_value, work_df)
        work_df, mask = strip_sentinel_nxn(
            context,
            work_df,
            sentinel_value,
        )
        if mask is not None and mask.any():
            issues.append((sentinel_name, mask))

    tab_np, missing_mask = tabulate_data_nxn(context, work_df, dtype)
    data_np, missing_mask = fix_missing(context, tab_np, missing_mask)

    if missing_mask is not None and missing_mask.any():
        issues.append(("missing", missing_mask))

    return ProcessResult(data_np, issues)
