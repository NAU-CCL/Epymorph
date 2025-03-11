"""
Implements the base class for all ADRIOs, as well as some general-purpose
ADRIO implementations.
"""

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from functools import partial, reduce
from time import perf_counter
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    NamedTuple,
    Sequence,
    TypeVar,
    cast,
    final,
)

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.attribute import NAME_PLACEHOLDER, AbsoluteName, AttributeDef
from epymorph.compartment_model import BaseCompartmentModel
from epymorph.data_shape import Shapes
from epymorph.data_type import AttributeArray
from epymorph.data_usage import DataEstimate, EmptyDataEstimate
from epymorph.database import DataResolver, evaluate_param
from epymorph.error import DataResourceError
from epymorph.event import ADRIOProgress, DownloadActivity, EventBus
from epymorph.geography.scope import GeoScope
from epymorph.simulation import Context, SimulationFunction
from epymorph.time import DateRange, TimeFrame

ResultDType = TypeVar("ResultDType", bound=np.generic)
"""The result type of an Adrio."""

ProgressCallback = Callable[[float, DownloadActivity | None], None]

_events = EventBus()


class ADRIO(SimulationFunction[NDArray[ResultDType]]):
    """
    ADRIO (or Abstract Data Resource Interface Object) are functions which are intended
    to load data from external sources for epymorph simulations. This may be from
    web APIs, local files or database, or anything imaginable.
    """

    def estimate_data(self) -> DataEstimate:
        """Estimate the data usage for this ADRIO in a RUME.
        If a reasonable estimate cannot be made, return EmptyDataEstimate."""
        return EmptyDataEstimate(self.class_name)

    @abstractmethod
    def evaluate_adrio(self) -> NDArray[ResultDType]:
        """Implement this method to provide logic for the function.
        Use self methods and properties to access the simulation context or defer
        processing to another function.
        """

    @override
    @final
    def evaluate(self) -> NDArray[ResultDType]:
        """The ADRIO parent class overrides this to provide ADRIO-specific
        functionality. ADRIO implementations should override `evaluate_adrio`."""
        _events.on_adrio_progress.publish(
            ADRIOProgress(
                adrio_name=self.class_name,
                attribute=self.name,
                final=False,
                ratio_complete=0,
                download=None,
                duration=None,
            )
        )
        t0 = perf_counter()
        result = self.evaluate_adrio()
        t1 = perf_counter()
        _events.on_adrio_progress.publish(
            ADRIOProgress(
                adrio_name=self.class_name,
                attribute=self.name,
                final=True,
                ratio_complete=1,
                download=None,
                duration=t1 - t0,
            )
        )
        return result

    @final
    def progress(
        self,
        ratio_complete: float,
        download: DownloadActivity | None = None,
    ) -> None:
        """Emit a progress event."""
        _events.on_adrio_progress.publish(
            ADRIOProgress(
                adrio_name=self.class_name,
                attribute=self.name,
                final=False,
                ratio_complete=ratio_complete,
                download=download,
                duration=None,
            )
        )


@evaluate_param.register
def _(
    value: ADRIO,
    name: AbsoluteName,
    data: DataResolver,
    scope: GeoScope | None,
    time_frame: TimeFrame | None,
    ipm: BaseCompartmentModel | None,
    rng: np.random.Generator | None,
) -> AttributeArray:
    # depth-first evaluation guarantees `data` has our dependencies.
    sim_func = value.with_context_internal(name, data, scope, time_frame, ipm, rng)
    return np.asarray(sim_func.evaluate())


AdrioClassT = TypeVar("AdrioClassT", bound=type[ADRIO])


def adrio_cache(cls: AdrioClassT) -> AdrioClassT:
    """Adrio class decorator to add result-caching behavior."""

    orig_with_context = cls.with_context_internal
    cached_instance: AdrioClassT | None = None
    cached_hash: int | None = None

    @functools.wraps(orig_with_context)
    def with_context_internal(
        self,
        name: AbsoluteName = NAME_PLACEHOLDER,
        data: DataResolver | None = None,
        scope: GeoScope | None = None,
        time_frame: TimeFrame | None = None,
        ipm: BaseCompartmentModel | None = None,
        rng: np.random.Generator | None = None,
    ):
        if data is None:
            req_hashes = ()
        else:
            req_hashes = (
                data.resolve(name.with_id(r.name), r).tobytes()
                for r in self.requirements
            )
        ipm_hash = None
        if ipm is not None:
            C = ipm.num_compartments
            E = ipm.num_events
            ipm_hash = (ipm.__class__.__name__, C, E)
        curr_hash = hash(tuple([str(name), scope, time_frame, ipm_hash, *req_hashes]))
        nonlocal cached_instance, cached_hash
        if cached_instance is None or cached_hash != curr_hash:
            cached_instance = orig_with_context(
                self, name, data, scope, time_frame, ipm, rng
            )
            cached_hash = curr_hash
        return cached_instance

    cls.with_context_internal = with_context_internal
    return cls


######################
# ADRIO V2 PROTOTYPE #
######################


DataT = TypeVar("DataT", bound=np.generic)


@dataclass(frozen=True)
class ProcessResult(Generic[DataT]):
    raw: NDArray[DataT]
    issues: Sequence[tuple[str, NDArray[np.bool_]]]


class ADRIOPrototype(SimulationFunction[NDArray[DataT]]):
    """
    ADRIO (or Abstract Data Resource Interface Object) are functions which are intended
    to load data from external sources for epymorph simulations. This may be from
    web APIs, local files or database, or anything imaginable.
    """

    @abstractmethod
    def _validate_context(self, context: Context) -> None:
        pass

    @abstractmethod
    def _fetch(self, context: Context) -> pd.DataFrame:
        pass

    @abstractmethod
    def _process(self, context: Context, data_df: pd.DataFrame) -> ProcessResult[DataT]:
        pass

    def _format(self, context: Context, data: ProcessResult[DataT]) -> NDArray[DataT]:
        if len(data.issues) > 0:
            mask = reduce(np.logical_and, [m for _, m in data.issues])
            return np.ma.masked_array(data.raw, mask)
        return data.raw

    @abstractmethod
    def _validate_result(self, context: Context, result: NDArray[DataT]) -> None:
        pass

    def evaluate(self) -> NDArray[DataT]:
        context = self.context
        self._validate_context(context)
        self.report_progress(0.0)
        t0 = perf_counter()
        source_df = self._fetch(context)
        proc_res = self._process(context, source_df)
        result_np = self._format(context, proc_res)
        t1 = perf_counter()
        self._validate_result(context, result_np)
        self.report_complete(t1 - t0)
        return result_np

    def estimate_data(self) -> DataEstimate:
        """Estimate the data usage for this ADRIO in a RUME.
        If a reasonable estimate cannot be made, return EmptyDataEstimate."""
        return EmptyDataEstimate(self.class_name)

    @final
    def report_progress(
        self,
        ratio: float,
        *,
        download: DownloadActivity | None = None,
    ) -> None:
        """Emit a progress event."""
        _events.on_adrio_progress.publish(
            ADRIOProgress(
                adrio_name=self.class_name,
                attribute=self.name,
                final=False,
                ratio_complete=min(ratio, 1.0),
                download=download,
                duration=None,
            )
        )

    @final
    def report_complete(
        self,
        duration: float,
        *,
        download: DownloadActivity | None = None,
    ) -> None:
        """Emit a progress event."""
        _events.on_adrio_progress.publish(
            ADRIOProgress(
                adrio_name=self.class_name,
                attribute=self.name,
                final=True,
                ratio_complete=1.0,
                download=download,
                duration=duration,
            )
        )


def range_mask_fn(
    *,
    minimum: DataT | None,
    maximum: DataT | None,
) -> Callable[[NDArray[DataT]], NDArray[np.bool_]]:
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
    time_range: DateRange,
    time_frame: TimeFrame,
) -> None:
    start = time_range.start_date
    end = time_range.end_date
    if time_frame.start_date < start or time_frame.end_date > end:
        err = f"This ADRIO is only valid for time frames between {start} and {end}."
        raise DataResourceError(err)


def strip_sentinel_txn(
    time_series: NDArray[np.datetime64],
    context: Context,
    result_df: pd.DataFrame,
    sentinel: int,
) -> tuple[pd.DataFrame, NDArray[np.bool_]]:
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
                index=time_series,
                columns=context.scope.node_ids,
                fill_value=False,
            )
            .to_numpy(dtype=np.bool_)
        ),
    )
    return stripped_df, sentinel_mask


def strip_sentinel_nxn(
    context: Context,
    result_df: pd.DataFrame,
    sentinel: int,
) -> tuple[pd.DataFrame, NDArray[np.bool_]]:
    is_sentinel = result_df["value"] == sentinel
    sentinel_df = result_df.loc[is_sentinel]
    stripped_df = result_df.loc[~is_sentinel]
    sentinel_mask = cast(
        NDArray[np.bool_],
        (
            sentinel_df.pivot_table(
                index="geoid_src",
                columns="geoid_dst",
                values="value",
                aggfunc=lambda _: True,
                fill_value=False,
            )
            .reindex(
                index=context.scope.node_ids,
                columns=context.scope.node_ids,
                fill_value=False,
            )
            .to_numpy(dtype=np.bool_)
        ),
    )
    return stripped_df, sentinel_mask


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


def process_txn(
    *,
    data_availability: DateRange,
    sentinels: list[FixSentinel],
    fix_missing: Fill[DataT],
    dtype: type[DataT],
    context: Context,
    data_df: pd.DataFrame,
) -> ProcessResult[DataT]:
    time_series_dr = data_availability.between(
        context.time_frame.start_date,
        context.time_frame.end_date - timedelta(days=1),
    )
    if time_series_dr is None:
        raise ValueError("Invalid time frame for this data.")
    time_series = time_series_dr.to_numpy()

    issues = []
    work_df = data_df

    for sentinel_name, sentinel_value, fix_sentinel in sentinels:
        work_df = fix_sentinel(context, sentinel_value, data_df)
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
    issues = []
    work_df = data_df

    for sentinel_name, sentinel_value, fix_sentinel in sentinels:
        work_df = fix_sentinel(context, sentinel_value, data_df)
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


##################
# UTILITY ADRIOS #
##################


class NodeID(ADRIO[np.str_]):
    """An ADRIO that provides the node IDs as they exist in the geo scope."""

    @override
    def evaluate_adrio(self) -> NDArray:
        return self.scope.node_ids


class Scale(ADRIO[np.float64]):
    """Scales the result of another ADRIO by multiplying values by the given factor."""

    _parent: ADRIO[np.float64]
    _factor: float

    def __init__(self, parent: ADRIO[np.float64], factor: float):
        """
        Initializes scaling with the ADRIO to be scaled and with the factor to multiply
        those resulting ADRIO values by.

        Parameters
        ----------
        parent : Adrio[np.int64 | np.float64]
            The ADRIO to scale all values for.
        factor : float
            The factor to multiply all resulting ADRIO values by.
        """
        self._parent = parent
        self._factor = factor

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        return self.defer(self._parent).astype(dtype=np.float64) * self._factor


class PopulationPerKM2(ADRIO[np.float64]):
    """
    Calculates population density by combining the values from attributes named
    `population` and `land_area_km2`. You must provide those attributes
    separately.
    """

    POPULATION = AttributeDef("population", int, Shapes.N)
    LAND_AREA_KM2 = AttributeDef("land_area_km2", float, Shapes.N)

    requirements = [POPULATION, LAND_AREA_KM2]

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        pop = self.data(self.POPULATION)
        area = self.data(self.LAND_AREA_KM2)
        return (pop / area).astype(dtype=np.float64)
