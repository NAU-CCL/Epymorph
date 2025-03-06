from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from typing import Callable, Generic, Literal, Sequence, TypeVar, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.adrio.soda import (
    And,
    Ascending,
    DateBetween,
    In,
    Select,
    SocrataResource,
    query_csv,
)
from epymorph.attribute import AbsoluteName
from epymorph.compartment_model import BaseCompartmentModel
from epymorph.data_shape import Dimensions
from epymorph.database import DataResolver
from epymorph.error import DataResourceError
from epymorph.geography.scope import GeoScope
from epymorph.geography.us_census import CountyScope, StateScope
from epymorph.time import TimeFrame, iso8601

T = TypeVar("T", bound=np.generic)


@dataclass(frozen=True)
class Context:
    name: AbsoluteName
    data: DataResolver
    scope: GeoScope
    time_frame: TimeFrame
    ipm: BaseCompartmentModel
    rng: np.random.Generator
    dim: Dimensions


@dataclass(frozen=True)
class Data(Generic[T]):
    raw: NDArray[T]
    issues: Sequence[tuple[str, NDArray[np.bool_]]]


class ADRIOPrototype(ABC, Generic[T]):
    @abstractmethod
    def _validate_context(self, context: Context) -> None:
        pass

    @abstractmethod
    def _fetch(self, context: Context) -> pd.DataFrame:
        pass

    @abstractmethod
    def _process(self, context: Context, data: pd.DataFrame) -> Data:
        pass

    def _format(self, context: Context, data: Data) -> NDArray[T]:
        if len(data.issues) > 0:
            mask = reduce(np.logical_and, [m for _, m in data.issues])
            return np.ma.masked_array(data.raw, mask)
        return data.raw

    @abstractmethod
    def _validate_result(self, context: Context, result: NDArray[T]) -> None:
        pass

    def evaluate(self, context: Context) -> NDArray[T]:
        self._validate_context(context)
        source_df = self._fetch(context)
        data = self._process(context, source_df)
        result_np = self._format(context, data)
        self._validate_result(context, result_np)
        return result_np


def range_mask_fn(
    *,
    minimum: T | None,
    maximum: T | None,
) -> Callable[[NDArray[T]], NDArray[np.bool_]]:
    match (minimum, maximum):
        case (None, None):
            return lambda xs: np.ones_like(xs, dtype=np.bool_)
        case (minimum, None):
            return lambda xs: xs >= minimum
        case (None, maximum):
            return lambda xs: xs < maximum
        case (minimum, maximum):
            return lambda xs: xs >= minimum & xs < maximum


def validate_time_frame(
    time_range: NDArray[np.datetime64],
    time_frame: TimeFrame,
) -> None:
    start = time_range[0]
    end = time_range[-1]
    if time_frame.start_date < start or time_frame.end_date >= end:
        err = f"This ADRIO is only valid for time frames between {start} and {end}."
        raise DataResourceError(err)


def filter_time_range(
    time_range: NDArray[np.datetime64],
    time_frame: TimeFrame,
) -> NDArray[np.datetime64]:
    after_start = time_range >= time_frame.start_date
    before_end = time_range < time_frame.end_date
    return time_range[after_start & before_end]


def strip_sentinel_txn(
    valid_dates: NDArray[np.datetime64],
    context: Context,
    result_df: pd.DataFrame,
    sentinel: int,
) -> tuple[pd.DataFrame, NDArray[np.bool_]]:
    expected_time_frame = filter_time_range(valid_dates, context.time_frame)

    is_sentinel = result_df["value"] == sentinel
    sentinel_df = result_df.loc[is_sentinel]
    stripped_df = result_df.loc[~is_sentinel]

    sentinel_mask = cast(
        NDArray[np.bool_],
        (
            sentinel_df.pivot_table(
                index="date",
                columns="geoid",
                values="value",
                aggfunc=lambda _: True,
                fill_value=False,
            )
            .reindex(
                index=expected_time_frame,
                columns=context.scope.node_ids,
                fill_value=False,
            )
            .to_numpy(dtype=np.bool_)
        ),
    )

    return stripped_df, sentinel_mask


class Fix(ABC):
    @abstractmethod
    def __call__(self, data_df: pd.DataFrame) -> pd.DataFrame:
        pass

    @staticmethod
    def of(
        bad_value: int,
        fix: int | Callable[[], int] | Literal[False],
    ) -> "Fix":
        if fix is False:
            return DontFix()
        elif isinstance(fix, int):
            return ConstantFix(bad_value, fix)
        elif callable(fix):
            return FunctionFix(bad_value, fix)
        raise ValueError("Not a valid fix.")


@dataclass(frozen=True)
class ConstantFix(Fix):
    replace: int
    with_value: int

    @override
    def __call__(self, data_df: pd.DataFrame) -> pd.DataFrame:
        to_replace = {self.replace: self.with_value}
        values = data_df["value"].replace(to_replace)
        return data_df.assign(value=values)


@dataclass(frozen=True)
class FunctionFix(Fix):
    replace: int
    with_function: Callable[[], int]

    @override
    def __call__(self, data_df: pd.DataFrame) -> pd.DataFrame:
        is_replace = data_df["value"] == self.replace
        num_replace = is_replace.sum()
        replacements = np.array([self.with_function() for _ in range(num_replace)])
        values = data_df["value"].copy()
        values[is_replace] = replacements
        return data_df.assign(value=values)


@dataclass(frozen=True)
class DontFix(Fix):
    @override
    def __call__(self, data_df: pd.DataFrame) -> pd.DataFrame:
        return data_df


class Fill(ABC):
    @abstractmethod
    def __call__(
        self,
        data_np: NDArray[np.int64],
        missing_mask: NDArray[np.bool_],
    ) -> tuple[NDArray[np.int64], NDArray[np.bool_] | None]:
        pass

    @staticmethod
    def of(
        fill: int | Callable[[], int] | Literal[False],
    ) -> "Fill":
        if fill is False:
            return DontFill()
        elif fill == 0:
            return AlreadyFilled()
        elif isinstance(fill, int):
            return ConstantFill(fill)
        elif callable(fill):
            return FunctionFill(fill)
        raise ValueError("Not a valid fill.")


@dataclass(frozen=True)
class AlreadyFilled(Fill):
    @override
    def __call__(
        self,
        data_np: NDArray[np.int64],
        missing_mask: NDArray[np.bool_],
    ) -> tuple[NDArray[np.int64], NDArray[np.bool_] | None]:
        return data_np, None


@dataclass(frozen=True)
class ConstantFill(Fill):
    with_value: int

    @override
    def __call__(
        self,
        data_np: NDArray[np.int64],
        missing_mask: NDArray[np.bool_],
    ) -> tuple[NDArray[np.int64], NDArray[np.bool_] | None]:
        result_np = data_np.copy()
        result_np[missing_mask] = self.with_value
        return result_np, None


@dataclass(frozen=True)
class FunctionFill(Fill):
    with_function: Callable[[], int]

    @override
    def __call__(
        self,
        data_np: NDArray[np.int64],
        missing_mask: NDArray[np.bool_],
    ) -> tuple[NDArray[np.int64], NDArray[np.bool_] | None]:
        num_replace = missing_mask.sum()
        replacements = np.array([self.with_function() for _ in range(num_replace)])
        result_np = data_np.copy()
        result_np[missing_mask] = replacements
        return result_np, None


@dataclass(frozen=True)
class DontFill(Fill):
    @override
    def __call__(
        self,
        data_np: NDArray[np.int64],
        missing_mask: NDArray[np.bool_],
    ) -> tuple[NDArray[np.int64], NDArray[np.bool_] | None]:
        return data_np, missing_mask


def tabulate_data_txn_int64(
    valid_dates: NDArray[np.datetime64],
    context: Context,
    result_df: pd.DataFrame,
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    # result_df must be a long data frame with columns
    # date, geoid, and value
    # sentinel values have been removed
    # groups will be aggregated by sum

    expected_time_frame = filter_time_range(valid_dates, context.time_frame)

    data = cast(
        NDArray[np.int64],
        (
            result_df.pivot_table(
                index="date",
                columns="geoid",
                values="value",
                aggfunc="sum",
                fill_value=0,
            )
            .reindex(
                index=expected_time_frame,
                columns=context.scope.node_ids,
                fill_value=0,
            )
            .to_numpy(dtype=np.int64)
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
                index=expected_time_frame,
                columns=context.scope.node_ids,
                fill_value=True,
            )
        ).to_numpy(dtype=np.bool_),
    )

    return data, missing_mask


class InfluenzaFacilityHospitalization(ADRIOPrototype[np.int64]):
    # TODO: use a different type to represent time ranges
    # _TIME_RANGE = DateRange(iso8601("2019-12-29"), iso8601("2024-04-21"), 7)  # noqa: E501, ERA001

    # time range from inspection of data
    _TIME_RANGE = np.arange(
        iso8601("2019-12-29"),
        iso8601("2024-04-21"),
        step=np.timedelta64(1, "W"),
    )
    _VALUE_RANGE = range_mask_fn(minimum=np.int64(0), maximum=None)
    _REDACTED_VALUE = -999999

    fix_redacted: Fix
    fix_missing: Fill

    def __init__(
        self,
        *,
        fix_redacted: int | Callable[[], int] | Literal[False] = False,
        fix_missing: int | Callable[[], int] | Literal[False] = False,
    ):
        try:
            self.fix_redacted = Fix.of(self._REDACTED_VALUE, fix_redacted)
        except ValueError:
            raise ValueError("Invalid value for `fix_redacted`")
        try:
            self.fix_missing = Fill.of(fix_missing)
        except ValueError:
            raise ValueError("Invalid value for `fix_missing`")

    @override
    def _validate_context(self, context: Context):
        if not isinstance(context.scope, StateScope | CountyScope):
            err = "US State or County geo scope required."
            raise DataResourceError(err)

        validate_time_frame(
            InfluenzaFacilityHospitalization._TIME_RANGE,
            context.time_frame,
        )

    @override
    def _fetch(self, context: Context) -> pd.DataFrame:
        res = SocrataResource(domain="healthdata.gov", id="anag-cw7u")
        value_col = "total_patients_hospitalized_confirmed_influenza_7_day_sum"
        return query_csv(
            resource=res,
            select=(
                Select("collection_week", "date", as_name="date"),
                Select("fips_code", "str", as_name="geoid"),
                Select(value_col, "int", as_name="value"),
            ),
            where=And(
                In("fips_code", context.scope.node_ids),
                DateBetween(
                    "collection_week",
                    context.time_frame.start_date,
                    context.time_frame.end_date,
                ),
            ),
            order_by=(
                Ascending("collection_week"),
                Ascending("fips_code"),
                Ascending(":id"),
            ),
        )

    def _process(self, context: Context, data: pd.DataFrame) -> Data:
        time_range = InfluenzaFacilityHospitalization._TIME_RANGE
        redacted = InfluenzaFacilityHospitalization._REDACTED_VALUE

        dates = filter_time_range(time_range, context.time_frame)

        fxrd_df = self.fix_redacted(data)
        unrd_df, redacted_mask = strip_sentinel_txn(dates, context, fxrd_df, redacted)
        data_np, missing_mask = tabulate_data_txn_int64(dates, context, unrd_df)
        data_np, missing_mask = self.fix_missing(data_np, missing_mask)

        issues = []
        if redacted_mask is not None and redacted_mask.any():
            issues.append(("redacted", redacted_mask))
        if missing_mask is not None and missing_mask.any():
            issues.append(("missing", missing_mask))
        return Data(data_np, issues)

    @override
    def _validate_result(self, context: Context, result: NDArray[np.int64]) -> None:
        # NOTE: validation only checks non-masked values
        is_valid = InfluenzaFacilityHospitalization._VALUE_RANGE
        if not is_valid(result).all():
            raise ValueError("invalid values")  # TODO

        time_range = InfluenzaFacilityHospitalization._TIME_RANGE
        T = len(filter_time_range(time_range, context.time_frame))
        if result.shape != (T, context.scope.nodes):
            raise ValueError("invalid shape")  # TODO


def processor_txn(
    time_range: NDArray[np.datetime64],
    sentinels: list[tuple[str, int, Fix]],
    fix_missing: Fill,
):
    def f(context: Context, data_df: pd.DataFrame) -> Data:
        dates = filter_time_range(time_range, context.time_frame)

        issues = []
        work_df = data_df

        for sentinel_name, sentinel_value, fix_sentinel in sentinels:
            work_df = fix_sentinel(data_df)
            work_df, mask = strip_sentinel_txn(
                dates,
                context,
                work_df,
                sentinel_value,
            )
            if mask is not None and mask.any():
                issues.append((sentinel_name, mask))

        data_np, missing_mask = fix_missing(
            *tabulate_data_txn_int64(dates, context, work_df)
        )

        if missing_mask is not None and missing_mask.any():
            issues.append(("missing", missing_mask))

        return Data(data_np, issues)

    return f
