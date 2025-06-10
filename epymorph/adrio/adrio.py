"""
Implements the base class for all ADRIOs, as well as some general-purpose
ADRIO implementations.
"""

import functools
from abc import abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import (
    Callable,
    Generic,
    Mapping,
    Sequence,
    TypeVar,
    final,
)
from urllib.error import HTTPError

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sparklines import sparklines
from typing_extensions import deprecated, override

from epymorph.adrio.processing import PipelineResult
from epymorph.adrio.validation import (
    Invalid,
    ResultFormat,
    Validator,
    validate_dtype,
    validate_numpy,
    validate_pipe,
    validate_shape,
)
from epymorph.attribute import NAME_PLACEHOLDER, AbsoluteName, AttributeDef
from epymorph.compartment_model import BaseCompartmentModel
from epymorph.data_shape import DataShape, Shapes
from epymorph.data_type import AttributeArray
from epymorph.data_usage import DataEstimate, EmptyDataEstimate
from epymorph.database import DataResolver, evaluate_param
from epymorph.error import MissingContextError
from epymorph.event import ADRIOProgress, DownloadActivity, EventBus
from epymorph.geography.scope import GeoScope
from epymorph.simulation import Context, SimulationFunction
from epymorph.time import DateRange, TimeFrame
from epymorph.util import (
    dtype_name,
    extract_date_value,
    is_date_value_array,
    is_numeric,
)

ResultDType = TypeVar("ResultDType", bound=np.generic)
"""The result type of an ADRIO."""

ProgressCallback = Callable[[float, DownloadActivity | None], None]
"""
The type of a callback function used by ADRIO implementations to report data fetching
progress.
"""

_events = EventBus()


#################################################
# ADRIOLegacy: old-style ADRIOS (to be removed) #
#################################################


@deprecated("Prefer ADRIO.")
class ADRIOLegacy(SimulationFunction[NDArray[ResultDType]]):
    """
    ADRIO (or Abstract Data Resource Interface Object) are functions which are intended
    to load data from external sources for epymorph simulations. This may be from
    web APIs, local files or database, or anything imaginable.

    `ADRIO` is generic on the dtype of the numpy array which it produces
    (`ResultDType`).

    Implementation Notes
    --------------------
    Implement this class by overriding `evaluate_adrio`; your implementation should
    call `progress` to report download/processing progress if applicable.
    Do not override `evaluate` unless you need to alter the base behavior.
    Override `estimate_data` if it's possible to estimate data usage ahead of time.

    When evaluating an ADRIO, call `evaluate`, not `evaluate_adrio`.
    """

    def estimate_data(self) -> DataEstimate:
        """
        Estimate the data usage for this ADRIO in a RUME.

        Returns
        -------
        :
            The data usage estimate. If a reasonable estimate cannot be made,
            returns `EmptyDataEstimate`.
        """
        return EmptyDataEstimate(self.class_name)

    @abstractmethod
    def evaluate_adrio(self) -> NDArray[ResultDType]:
        """
        Implement this method to provide logic for the function.
        Use self methods and properties to access the simulation context or defer
        processing to another function.

        Returns
        -------
        :
            The data array.
        """

    @override
    def evaluate(self) -> NDArray[ResultDType]:
        """
        Fully evaluate the ADRIO.

        Returns
        -------
        :
            The data array.
        """
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
        """
        Emit a progress event.

        Parameters
        ----------
        ratio_complete :
            The ratio of how much work the ADRIO has completed in total;
            0 meaning no progress and 1 meaning it is finished.
        download :
            Describes current network activity. If there is no network activity
            to report or if it cannot be measured, provide `None`.
        """
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
    value: ADRIOLegacy,
    name: AbsoluteName,
    data: DataResolver,
    scope: GeoScope | None,
    time_frame: TimeFrame | None,
    ipm: BaseCompartmentModel | None,
    rng: np.random.Generator | None,
) -> AttributeArray:
    # depth-first evaluation guarantees `data` has our dependencies.
    ctx = Context.of(name, data, scope, time_frame, ipm, rng)
    sim_func = value.with_context_internal(ctx)
    return sim_func.evaluate()


_ADRIOLegacyClassT = TypeVar("_ADRIOLegacyClassT", bound=ADRIOLegacy)


def adrio_legacy_cache(cls: type[_ADRIOLegacyClassT]) -> type[_ADRIOLegacyClassT]:
    """
    `ADRIOLegacy` class decorator which adds result-caching behavior.

    Examples
    --------
    >>> @adrio_legacy_cache
    >>> class MyADRIO(ADRIOLegacy[np.int64]):
    >>>     # Now this ADRIO will cache its results.
    >>>     # ...
    """

    orig_with_context = cls.with_context_internal
    orig_evaluate = cls.evaluate
    ctx_cache_key = "__with_context_cache__"
    eval_cache_key = "__evaluate_cache__"

    @functools.wraps(orig_with_context)
    def with_context_internal(self, context: Context):
        curr_hash = context.hash(self.requirements)
        cached_hash, cached_instance = getattr(self, ctx_cache_key, (None, None))
        if cached_instance is None or cached_hash != curr_hash:
            cached_instance = orig_with_context(self, context)
            cached_hash = curr_hash
            setattr(self, ctx_cache_key, (cached_hash, cached_instance))
            setattr(self, eval_cache_key, None)
        return cached_instance

    @functools.wraps(orig_evaluate)
    def evaluate(self):
        cached_value = getattr(self, eval_cache_key, None)
        if cached_value is None:
            cached_value = orig_evaluate(self)
            setattr(self, eval_cache_key, cached_value)
        return cached_value

    cls.with_context_internal = with_context_internal
    cls.evaluate = evaluate
    return cls


_ADRIOClassT = TypeVar("_ADRIOClassT", bound="ADRIO")


def adrio_cache(cls: type[_ADRIOClassT]) -> type[_ADRIOClassT]:
    """
    `ADRIO` class decorator to add result-caching behavior.

    Examples
    --------
    >>> @adrio_cache
    >>> class MyADRIO(ADRIO[np.int64]):
    >>>     # Now this ADRIO will cache its results.
    >>>     # ...
    """

    orig_with_context = cls.with_context_internal
    orig_evaluate = cls.evaluate
    ctx_cache_key = "__with_context_cache__"
    eval_cache_key = "__evaluate_cache__"

    @functools.wraps(orig_with_context)
    def with_context_internal(self, context: Context):
        curr_hash = context.hash(self.requirements)
        cached_hash, cached_instance = getattr(self, ctx_cache_key, (None, None))
        if cached_instance is None or cached_hash != curr_hash:
            cached_instance = orig_with_context(self, context)
            cached_hash = curr_hash
            setattr(self, ctx_cache_key, (cached_hash, cached_instance))
            setattr(self, eval_cache_key, None)
        return cached_instance

    @functools.wraps(orig_evaluate)
    def evaluate(self):
        cached_value = getattr(self, eval_cache_key, None)
        if cached_value is None:
            cached_value = orig_evaluate(self)
            setattr(self, eval_cache_key, cached_value)
        return cached_value

    cls.with_context_internal = with_context_internal
    cls.evaluate = evaluate
    return cls


#########
# ADRIO #
#########


def _adrio_name(adrio: "ADRIO", context: Context) -> str:
    if context.name == NAME_PLACEHOLDER:
        return adrio.class_name
    else:
        return f"{context.name} ({adrio.name})"


class ADRIOError(Exception):
    """
    Error while loading or processing data with an ADRIO.

    Parameters
    ----------
    adrio :
        The ADRIO being evaluated.
    context :
        The evaluation context.
    message :
        An error description.
    """

    adrio: "ADRIO"
    """The ADRIO being evaluated."""
    context: Context
    """The evaluation context."""

    def __init__(self, adrio: "ADRIO", context: Context, message: str):
        self.adrio = adrio
        self.context = context
        # If message contains "{adrio_name}", fill it in.
        message = message.format(adrio_name=_adrio_name(adrio, context))
        super().__init__(message)


class ADRIOContextError(ADRIOError):
    """
    Error if the simulation context is invalid for evaluating the ADRIO.

    Parameters
    ----------
    adrio :
        The ADRIO being evaluated.
    context :
        The evaluation context.
    message :
        An error description, or else a default message will be used.
    """

    def __init__(
        self,
        adrio: "ADRIO",
        context: Context,
        message: str | None = None,
    ):
        if message is None:
            message = "the ADRIO encountered an unexpected error"
        message = "Invalid context for {adrio_name}: " + message
        super().__init__(adrio, context, message)


class ADRIOCommunicationError(ADRIOError):
    """
    Error if the ADRIO could not communicate with the external resource.

    Parameters
    ----------
    adrio :
        The ADRIO being evaluated.
    context :
        The evaluation context.
    message :
        An error description, or else a default message will be used.
    """

    def __init__(
        self,
        adrio: "ADRIO",
        context: Context,
        message: str | None = None,
    ):
        if message is None:
            message = "the ADRIO was unable to communicate with the external resource"
        message = "Error loading {adrio_name}: " + message
        super().__init__(adrio, context, message)


class ADRIOProcessingError(ADRIOError):
    """
    An unexpected error occurred while processing ADRIO data.

    Parameters
    ----------
    adrio :
        The ADRIO being evaluated.
    context :
        The evaluation context.
    message :
        An error description, or else a default message will be used.
    """

    def __init__(
        self,
        adrio: "ADRIO",
        context: Context,
        message: str | None = None,
    ):
        if message is None:
            message = "the ADRIO encountered an unexpected error processing results"
        message = "Error processing {adrio_name}: " + message
        super().__init__(adrio, context, message)


ResultT = TypeVar("ResultT", bound=np.generic)
"""The dtype of an ADRIO result."""
ValueT = TypeVar("ValueT", bound=np.generic)
"""The dtype of an ADRIO result's values, which may differ from the result type."""


@dataclass(frozen=True)
class InspectResult(Generic[ResultT, ValueT]):
    """
    Inspection is the process by which an ADRIO fetches data and analyzes its quality.

    The result encapsulates the source data, the processed result data, and any
    outstanding data issues. ADRIOs will provide methods for correcting these issues
    as is appropriate for the task, but often these will be optional. A result which
    contains unresolved data issues will be represented as a masked numpy array. Values
    which are not impacted by any of the data issues will be unmasked. Individual issues
    are tracked along with masks specific to the issue.

    For example: if data is not available for every geo node requested, some values will
    be represented as missing. Missing values will be masked in the result, and an issue
    will be included (likely called "missing") with a boolean mask indicating the
    missing values. The ADRIO will likely provide a fill method option which allows
    users the option to fill missing values, for instance with zeros.
    Providing a fill method and inspecting the ADRIO a second time should resolve the
    "missing" issue and, assuming no other issues remain, produce a non-masked numpy
    array as a result.

    `InspectResult` is generic on the result and value type (`ResultT` and `ValueT`) of
    the ADRIO.

    Parameters
    ----------
    adrio :
        A reference to the ADRIO which produced this result.
    source :
        The data as fetched from the source. This can be useful for debugging data
        issues.
    result :
        The final result produced by the ADRIO.
    dtype :
        The dtype of the data values.
    shape :
        The shape of the result.
    issues :
        The set of issues in the data along with a mask which indicates which values
        are impacted by the issue. The keys of this mapping are specific to the ADRIO,
        as ADRIOs tend to deal with unique data challenges.

    Examples
    --------
    --8<-- "docs/_examples/adrio_adrio_InspectResult.md"
    """

    adrio: "ADRIO[ResultT, ValueT]"
    """A reference to the ADRIO which produced this result."""
    source: pd.DataFrame | NDArray
    """
    The data as fetched from the source. This can be useful for debugging data issues.
    """
    result: NDArray[ResultT]
    """The final result produced by the ADRIO."""
    dtype: type[ValueT]
    """The dtype of the data values."""
    shape: DataShape
    """The shape of the result."""
    issues: Mapping[str, NDArray[np.bool_]]
    """
    The set of issues in the data along with a mask which indicates which values
    are impacted by the issue. The keys of this mapping are specific to the ADRIO,
    as ADRIOs tend to deal with unique data challenges.
    """

    def __post_init__(self):
        for issue_name, mask in self.issues.items():
            if mask.shape != self.result.shape:
                err = (
                    f"The shape of the mask for '{issue_name}' {mask.shape} did "
                    f"not match the shape of the result data {self.result.shape}."
                )
                raise ValueError(err)

    @functools.cached_property
    def values(self) -> NDArray[ValueT]:
        """
        The values in the result. If the result is date/value tuples, the values are
        first extracted.
        """
        values = self.result
        if is_date_value_array(values, value_dtype=self.dtype):
            _, values = extract_date_value(values, self.dtype)
        return values  # type: ignore

    @property
    def unmasked_count(self) -> int:
        """The number of unmasked values in the result."""
        vs = self.values  # NOTE: the values accessor unwraps date/values
        mask = np.ma.getmaskarray(vs)
        if mask.dtype.names is None:
            # values mask is not structured
            return np.ma.count(vs)

        # values mask is structured! (but not date/value)
        # consider a value masked if any part of it is masked (mask boolean or)
        combined_mask = np.zeros(vs.shape, dtype=np.bool_)
        for name in mask.dtype.names:
            combined_mask |= mask[name]
        return np.invert(combined_mask).sum()

    @property
    def quantify(self) -> Sequence[tuple[str, float]]:
        """
        Quantifies properties of the data: what percentage of the values are impacted by
        each data issue (if any), how many are zero, and how many are "unmasked" (that
        is, not affected by any issues). The value is a sequence of tuples which contain
        the name of the quality and the percentage of values.
        """
        vs = self.values
        size = vs.size
        unmasked_count = self.unmasked_count
        quant = []
        if unmasked_count > 0 and is_numeric(vs):
            zero_value = vs.dtype.type(0)
            quant.append(("zero", (vs == zero_value).sum() / size))
        for name, mask in self.issues.items():
            quant.append((name, mask.sum() / size))
        quant.append(("unmasked", unmasked_count / size))
        return quant

    def __str__(self) -> str:
        extra_info = []
        if is_date_value_array(self.result):
            # calc display values for date/value data
            dates, vs = extract_date_value(self.result)
            dtname = f"date/value ({dtype_name(np.dtype(vs.dtype))})"
            match len(dates):
                case 1:
                    extra_info.append(f"  Date range: {dates[0]}")
                case x if x > 1:
                    deltas = np.unique((dates[1:] - dates[:-1]))
                    period = str(deltas[0]) if len(deltas) == 1 else "irregular"
                    extra_info.append(
                        f"  Date range: {dates.min()} to {dates.max()}"
                        f", period: {period}"
                    )
                case _:
                    # might happen if there are zero data points
                    pass
        else:
            # calc display values for simple value data (not date/value)
            vs = self.result
            dtname = dtype_name(np.dtype(vs.dtype))

        # Value statistics and histogram: only possible with numeric data.
        stats = []
        if is_numeric(vs):
            if self.unmasked_count == 0:
                stats.extend(
                    [
                        "  Values:",
                        "    N/A (all values are masked)",
                    ]
                )
            else:
                # stats methods don't really support masked arrays
                stats_vs = vs if not np.ma.is_masked(vs) else np.ma.compressed(vs)
                qs = np.quantile(stats_vs, [0.25, 0.50, 0.75])
                qs_str = ", ".join(f"{q:.1f}" for q in qs)

                minimum = vs.min()
                maximum = vs.max()
                spark = sparklines(
                    np.histogram(vs, bins=20, range=(minimum, maximum))[0],
                    num_lines=1,
                )[0]
                histogram = f"{minimum} {spark} {maximum}"

                stats.extend(
                    [
                        "  Values:",
                        f"    histogram: {histogram}",
                        f"    quartiles: {qs_str} (IQR: {(qs[-1] - qs[0]):.1f})",
                        f"    std dev: {np.std(stats_vs):.1f}",
                    ]
                )

        lines = [
            f"ADRIO inspection for {self.adrio.class_name}:",
            f"  Result shape: {self.shape} {vs.shape}; dtype: {dtname}; size: {vs.size}",  # noqa: E501
            *extra_info,
            *stats,
            *[
                f"    percent {issue}: {percent:.1%}"
                for issue, percent in self.quantify
            ],
        ]
        return "\n".join(lines)


class ADRIO(SimulationFunction[NDArray[ResultT]], Generic[ResultT, ValueT]):
    """
    ADRIOs (or Abstract Data Resource Interface Objects) are functions which are
    intended to load data from external sources for epymorph simulations. This may be
    from web APIs, local files or databases, or anything imaginable.

    ADRIO is an abstract base class. It is generic in both the form of the result
    (`ResultT`) and the type of the values in the result (`ValueT`). Both represent
    numpy dtypes.

    When the ADRIO's result is simple, like a numpy array of 64-bit
    integers, both `ResultT` and `ValueT` will be the same -- `np.int64`. If the result
    is a structured type, however, like with numpy arrays containing date/value tuples,
    `ResultT` will reflect the "outer" structured type and `ValueT` will reflect type
    of the "inner" data values. As a common example, a date/value array with 64-bit
    integer values will have `ResultT` equal to
    `[("date", np.datetime64), ("value", np.int64)]` and `ValueT` equal to `np.int64`.
    (This complexity is necessary to work around weaknesses in Python's type system.)

    Implementation Notes
    --------------------
    Implement this class by overriding `result_format` to describe the expected results,
    `validate_context` to check the provided context (happens prior to loading data),
    and `inspect` to implement the data loading logic. Do not override `evaluate` unless
    you need to change the base behavior. Override `estimate_data` if it's possible to
    estimate data usage ahead of time.

    When evaluating an ADRIO, call `evaluate` or `inspect`.

    See Also
    --------
    You may prefer to extend [epymorph.adrio.adrio.FetchADRIO][], which provides more
    scaffolding for ADRIOs that fetch data from external sources like web APIs.
    """

    @property
    @abstractmethod
    def result_format(self) -> ResultFormat:
        """Information about the expected format of the ADRIO's resulting data."""

    @abstractmethod
    def validate_context(self, context: Context) -> None:
        """
        Validates the context before ADRIO evaluation.

        Parameters
        ----------
        context :
            The context to validate.

        Raises
        ------
        ADRIOContextError
            If this ADRIO cannot be evaluated in the given context.
        """

    def validate_result(
        self,
        context: Context,
        result: NDArray[ResultT],
    ) -> None:
        """
        Validates that the result of evaluating the ADRIO adheres to the
        expected result format.

        Parameters
        ----------
        context :
            The context in which the result has been evaluated.
        result :
            The result produced by the ADRIO.

        Raises
        ------
        ADRIOProcessingError
            If the result is invalid, indicating the processing logic has a bug.
        """
        adrio_validate_pipe(
            self,
            context,
            result,
            validate_numpy(),
            validate_shape(self.result_format.shape.to_tuple(context.dim)),
            validate_dtype(self.result_format.dtype),
        )

    def evaluate(self) -> NDArray[ResultT]:
        """
        Evaluates the ADRIO in the current context.

        Returns
        -------
        :
            The result value.
        """
        return self.inspect().result

    @abstractmethod
    def inspect(self) -> InspectResult[ResultT, ValueT]:
        """
        Produce an inspection of the ADRIO's data for the current context.

        When implementing an ADRIO, override this method to provide data fetching and
        processing logic. Use self methods and properties to access the simulation
        context or defer processing to another function.

        NOTE: if you are implementing this method, make sure to call `validate_context`
        first and `_validate_result` last.

        Returns
        -------
        :
            The data inspection results for the ADRIO's current context.
        """

    def estimate_data(self) -> DataEstimate:
        """
        Estimate the data usage for this ADRIO in the current context.

        Returns
        -------
        :
            The estimated data usage for this ADRIO's current context.
            If a reasonable estimate cannot be made, returns `EmptyDataEstimate`.
        """
        return EmptyDataEstimate(self.class_name)

    @final
    def _report_progress(
        self,
        ratio: float,
        *,
        download: DownloadActivity | None = None,
    ) -> None:
        """
        Emit an intermediate progress event.

        Parameters
        ----------
        ratio_complete :
            The ratio of how much work the ADRIO has completed in total;
            0 meaning no progress and 1 meaning it is finished.
        download :
            Describes current network activity. If there is no network activity
            to report or if it cannot be measured, provide `None`.
        """
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
    def _report_complete(
        self,
        duration: float,
        *,
        download: DownloadActivity | None = None,
    ) -> None:
        """
        Emit a final progress event.

        Parameters
        ----------
        duration :
            How long, in seconds, did processing take?
        download :
            Describes current network activity. If there is no network activity
            to report or if it cannot be measured, provide `None`.
        """
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
    ctx = Context.of(name, data, scope, time_frame, ipm, rng)
    sim_func = value.with_context_internal(ctx)
    return sim_func.evaluate()


class FetchADRIO(ADRIO[ResultT, ValueT]):
    """
    A specialization of `ADRIO` that adds structure for ADRIOs that load data from
    an external source, such as a web API.

    Implementation Notes
    --------------------
    `FetchADRIO` provides an implementation of `inspect`, and requires that you
    implement methods `_fetch` and `_process` instead.
    """

    @abstractmethod
    def _fetch(self, context: Context) -> pd.DataFrame:
        """
        Fetch the source data from the external source (or cache).

        Parameters
        ----------
        context :
            The evaluation context.

        Returns
        -------
        :
            A dataframe of the source data, as close to its original form as practical.
        """

    @abstractmethod
    def _process(
        self, context: Context, data_df: pd.DataFrame
    ) -> PipelineResult[ResultT]:
        """
        Process the source data through a data pipeline.

        Parameters
        ----------
        context :
            The evaluation context.
        data_df :
            The source data (from `_fetch`).

        Returns
        -------
        :
            The result of processing the data.

        See Also
        --------
        [epymorph.adrio.processing.DataPipeline][] which is a toolkit for writing
        standardizing data processing workflows.
        """

    def inspect(self) -> InspectResult[ResultT, ValueT]:
        """
        Produce an inspection of the ADRIO's data for the current context.

        Returns
        -------
        :
            The data inspection results for the ADRIO's current context.
        """
        ctx = self.context
        try:
            self.validate_context(ctx)
        except ADRIOError:
            raise
        except MissingContextError as e:
            raise ADRIOContextError(self, ctx, str(e))
        except Exception as e:
            raise ADRIOContextError(self, ctx) from e

        self._report_progress(0.0)
        start_time = perf_counter()

        try:
            source_df = self._fetch(ctx)
        except ADRIOCommunicationError as e:
            e2 = e.__cause__
            if isinstance(e2, HTTPError) and e2.code == 414:
                err = (
                    "the attempted request URI was too long to send. "
                    "The root cause for this can vary, but it usually suggests "
                    "your query involves too many locations."
                )
                raise ADRIOCommunicationError(e.adrio, e.context, err) from e2
            else:
                raise e
        except ADRIOError:
            raise
        except MissingContextError as e:
            raise ADRIOContextError(self, ctx, str(e))
        except Exception as e:
            raise ADRIOProcessingError(self, ctx) from e

        try:
            proc_res = self._process(ctx, source_df)
            result_np = proc_res.value_as_masked
            self.validate_result(ctx, result_np)
        except ADRIOError:
            raise
        except MissingContextError as e:
            raise ADRIOContextError(self, ctx, str(e))
        except Exception as e:
            raise ADRIOProcessingError(self, ctx) from e

        finish_time = perf_counter()
        self._report_complete(finish_time - start_time)

        return InspectResult[ResultT, ValueT](
            self,
            source_df,
            result_np,
            self.result_format.dtype.type,
            self.result_format.shape,
            proc_res.issues,
        )


def adrio_validate_pipe(
    adrio: ADRIO,
    context: Context,
    result: NDArray[ResultT],
    *validators: Validator,
) -> None:
    """
    Applies a sequence of validator function to the result of an ADRIO,
    using that ADRIO's context and raising an appropriate error if the
    result is invalid.

    Parameters
    ----------
    adrio :
        The ADRIO instance.
    context :
        The current simulation context.
    result :
        The ADRIO result array.
    *validators :
        The sequence of validation checks to apply.

    Raises
    ------
    ADRIOProcessingError
        If the result is invalid.
    """
    v = validate_pipe(*validators)(result)
    if isinstance(v, Invalid):
        raise ADRIOProcessingError(adrio, context, v.error)


def range_mask_fn(
    *,
    minimum: ValueT | None,
    maximum: ValueT | None,
) -> Callable[[NDArray[ValueT]], NDArray[np.bool_]]:
    """
    Constructs a validation function for checking that values are in a given range.

    This function is generic in the dtype of arrays it checks (`ValueT`).

    Parameters
    ----------
    minimum :
        The minimum valid value, or `None` if there is no minimum.
    maximum :
        The maximum valid value, or `None` is there is no maximum.

    Returns
    -------
    :
        The validation function.
    """
    match (minimum, maximum):
        case (None, None):
            return lambda xs: np.ones_like(xs, dtype=np.bool_)
        case (minimum, None):
            return lambda xs: xs >= minimum
        case (None, maximum):
            return lambda xs: xs <= maximum
        case (minimum, maximum):
            return lambda xs: (xs >= minimum) & (xs <= maximum)


def validate_time_frame(
    adrio: ADRIO,
    context: Context,
    time_range: DateRange,
) -> None:
    """
    Validates that the context time frame is within the specified `DateRange`.

    Parameters
    ----------
    adrio :
        The ADRIO instance doing the validation.
    context :
        The evaluation context.
    time_range :
        The valid range of dates.

    Raises
    ------
    ADRIOContextError
        If the context time frame is not valid.
    """
    start = time_range.start_date
    end = time_range.end_date
    tf = context.time_frame
    if tf.start_date < start or tf.end_date > end:
        err = f"This ADRIO is only valid for time frames between {start} and {end}."
        raise ADRIOContextError(adrio, context, err)
    if time_range.overlap(tf) is None:
        err = "The supplied time frame does not include any available dates."
        raise ADRIOContextError(adrio, context, err)


##################
# UTILITY ADRIOS #
##################


class NodeID(ADRIOLegacy[np.str_]):
    """An ADRIO that provides the node IDs as they exist in the geo scope."""

    @override
    def evaluate_adrio(self) -> NDArray:
        return self.scope.node_ids


class Scale(ADRIOLegacy[np.float64]):
    """Scales the result of another ADRIO by multiplying values by the given factor."""

    _parent: ADRIOLegacy[np.float64]
    _factor: float

    def __init__(self, parent: ADRIOLegacy[np.float64], factor: float):
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


class PopulationPerKM2(ADRIOLegacy[np.float64]):
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
