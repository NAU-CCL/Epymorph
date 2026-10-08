"""
Implements the base class for all ADRIOs, as well as some general-purpose
ADRIO implementations.
"""

import dataclasses
import functools
from abc import abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Protocol, Self, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from requests import HTTPError
from sparklines.sparklines import sparklines

from epymorph.adrio2.validate import ResultFormat, ValidationError
from epymorph.attribute import NAME_PLACEHOLDER, AbsoluteName
from epymorph.compartment_model import BaseCompartmentModel
from epymorph.data_type import AttributeData, StratifiedAttributeArray
from epymorph.data_usage import DataEstimate, EmptyDataEstimate
from epymorph.database import DataResolver, evaluate_param
from epymorph.error import MissingContextError
from epymorph.event import ADRIOProgress, DownloadActivity, EventBus
from epymorph.geography.scope import GeoScope
from epymorph.simulation import Context, SimulationFunction
from epymorph.time import TimeFrame
from epymorph.util import dtype_name, is_numeric


class ProgressCallback(Protocol):
    """
    The type of a callback function used by ADRIO implementations to report data fetching
    progress.
    """

    def __call__(
        self,
        ratio: float,
        download: DownloadActivity | None = None,
    ) -> None: ...


#########
# ADRIO #
#########


ADRIOClassT = TypeVar("ADRIOClassT", bound="ADRIO")


def adrio_cache(cls: type[ADRIOClassT]) -> type[ADRIOClassT]:
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

    def __init__(self, adrio: "ADRIO", context: Context, message: str | None):
        self.adrio = adrio
        self.context = context
        if message is None:
            message = "the ADRIO encountered an unexpected error"
        # If message contains "{adrio_name}", fill it in.
        if context.name == NAME_PLACEHOLDER:
            adrio_name = adrio.class_name
        else:
            adrio_name = f"{context.name} ({adrio.name})"
        message = message.format(adrio_name=adrio_name)
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


class DeferredADRIOError(Exception):
    """
    An error that occurred while evaluating an ADRIO, but in a place where we do not
    have access to the full context.

    Parameters
    ----------
    error_type :
        The type of the error that occurred, but that will need to be constructed later.
    message :
        An error description.
    """

    message: str | None

    def __init__(
        self,
        error_type: type[ADRIOError],
        message: str | None = None,
    ):
        self.error_type = error_type
        self.message = message
        super().__init__()


# TODO
# @dataclass(frozen=True)
# class InspectResult:
#     """
#     Inspection is the process by which an ADRIO fetches data and analyzes its quality.

#     The result encapsulates the source data, the processed result data, and any
#     outstanding data issues. ADRIOs will provide methods for correcting these issues
#     as is appropriate for the task, but often these will be optional. A result which
#     contains unresolved data issues will be represented as a masked numpy array. Values
#     which are not impacted by any of the data issues will be unmasked. Individual issues
#     are tracked along with masks specific to the issue.

#     For example: if data is not available for every geo node requested, some values will
#     be represented as missing. Missing values will be masked in the result, and an issue
#     will be included (likely called "missing") with a boolean mask indicating the
#     missing values. The ADRIO will likely provide a fill method option which allows
#     users the option to fill missing values, for instance with zeros.
#     Providing a fill method and inspecting the ADRIO a second time should resolve the
#     "missing" issue and, assuming no other issues remain, produce a non-masked numpy
#     array as a result.

#     `InspectResult` is generic on the result and value type (`ResultT` and `ValueT`) of
#     the ADRIO.

#     Parameters
#     ----------
#     adrio :
#         A reference to the ADRIO which produced this result.
#     source :
#         The data as fetched from the source. This can be useful for debugging data
#         issues.
#     result :
#         The final result produced by the ADRIO.
#     dtype :
#         The dtype of the data values.
#     shape :
#         The shape of the result.
#     issues :
#         The set of issues in the data along with a mask which indicates which values
#         are impacted by the issue. The keys of this mapping are specific to the ADRIO,
#         as ADRIOs tend to deal with unique data challenges.

#     Examples
#     --------
#     --8<-- "docs/_examples/adrio_adrio_InspectResult.md"
#     """


@dataclass(frozen=True)
class InspectResult:
    source: pd.DataFrame
    processed: pd.DataFrame
    issues: dict[str, NDArray[np.bool_]]
    result: AttributeData | None

    @classmethod
    def from_source(cls, source: pd.DataFrame) -> Self:
        return cls(
            source=source,
            processed=source.copy(),
            issues={},
            result=None,
        )

    def fix(
        self,
        issue_name: str,
        where: Callable[[pd.DataFrame], pd.DataFrame],
        replacement: int | None = None,
    ) -> Self:
        issue_mask = where(self.processed)
        if issue_mask.any(axis=None):
            if replacement is None:
                new_issues = {**self.issues, issue_name: issue_mask.to_numpy()}
                return dataclasses.replace(self, issues=new_issues)
            else:
                new_processed = self.processed.mask(issue_mask, replacement)
                return dataclasses.replace(self, processed=new_processed)
        return self

    def map(self, func: Callable[[pd.DataFrame], AttributeData]) -> Self:
        if self.issues:
            return self
        return dataclasses.replace(self, result=func(self.processed))


def adrio_inspection_report(adrio: "ADRIO", inspect_result: InspectResult) -> str:
    lines = [f"ADRIO inspection for {adrio.class_name}:"]

    # TODO
    # if is_date_value_array(inspect_result.result):
    #     # calc display values for date/value data
    #     dates, vs = extract_date_value(inspect_result.result)
    #     dtname = f"date/value ({dtype_name(np.dtype(vs.dtype))})"
    #     match len(dates):
    #         case 1:
    #             extra_info.append(f"  Date range: {dates[0]}")
    #         case x if x > 1:
    #             deltas = np.unique((dates[1:] - dates[:-1]))
    #             period = str(deltas[0]) if len(deltas) == 1 else "irregular"
    #             extra_info.append(
    #                 f"  Date range: {dates.min()} to {dates.max()}, period: {period}"
    #             )
    #         case _:
    #             # might happen if there are zero data points
    #             pass
    # else:
    #     # calc display values for simple value data (not date/value)
    #     vs = inspect_result.result
    #     dtname = dtype_name(np.dtype(vs.dtype))

    vs = inspect_result.result
    if isinstance(vs, StratifiedAttributeArray):
        vs = vs.values  # noqa: PD011

    if vs is not None:
        dtname = dtype_name(np.dtype(vs.dtype))
        lines.append(f"  Result shape: {vs.shape}; dtype: {dtname}; size: {vs.size}")

    size = 0
    unmasked_count = 0
    if vs is not None:
        size = vs.size
        mask = np.ma.getmaskarray(vs)
        if mask.dtype.names is None:
            # For unstructured values...
            unmasked_count = np.ma.count(vs)
        else:
            # For structured values...
            combined_mask = np.zeros(vs.shape, dtype=np.bool_)
            for name in mask.dtype.names:
                combined_mask |= mask[name]
            unmasked_count = np.invert(combined_mask).sum()

    # Value statistics and histogram: only possible with numeric data.
    if vs is not None and is_numeric(vs):
        if unmasked_count == 0:
            lines.extend(
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
                np.histogram(vs, bins=20, range=(minimum, maximum))[0],  # type: ignore
                num_lines=1,
            )[0]
            histogram = f"{minimum} {spark} {maximum}"

            lines.extend(
                [
                    "  Values:",
                    f"    histogram: {histogram}",
                    f"    quartiles: {qs_str} (IQR: {(qs[-1] - qs[0]):.1f})",
                    f"    std dev: {np.std(stats_vs):.1f}",
                ]
            )

    quant = []
    if unmasked_count > 0 and vs is not None and is_numeric(vs):
        zero_value = vs.dtype.type(0)
        quant.append(("zero", (vs == zero_value).sum() / size))
    for name, mask in inspect_result.issues.items():
        quant.append((name, mask.sum() / size))
    quant.append(("unmasked", unmasked_count / size))
    lines.extend(f"    percent {issue}: {percent:.1%}" for issue, percent in quant)

    return "\n".join(lines)


class ADRIO(SimulationFunction[AttributeData]):
    """
    ADRIOs (or Abstract Data Resource Interface Objects) are functions which are
    intended to load data from external sources for epymorph simulations. This may be
    from web APIs, local files or databases, or anything imaginable.

    When evaluating an ADRIO, call `evaluate` or `inspect`.

    Implementation Notes
    --------------------
    ADRIO is an abstract base class. Implement this class by overriding `result_format`
    to describe the expected results, and `inspect` to implement the data loading logic.
    Do not override `evaluate` unless you need to change the base behavior. Override
    `estimate_data` if it's possible to estimate data usage ahead of time.
    """

    @property
    @abstractmethod
    def result_format(self) -> ResultFormat:
        """Information about the expected format of the ADRIO's resulting data."""

    def evaluate(self) -> AttributeData:
        """
        Evaluate the ADRIO in the current context.

        Returns
        -------
        :
            The result value.
        """
        inspection = self.inspect()
        if inspection.result is None:
            # TODO: better handling?
            raise ADRIOProcessingError(
                self,
                self.context,
                "the ADRIO did not produce a result",
            )
        return inspection.result

    @abstractmethod
    def inspect(self) -> InspectResult:
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


@contextmanager
def adrio_progress(adrio: ADRIO) -> Generator[ProgressCallback, None, None]:
    # TODO: docstring
    events = EventBus()

    def _report_progress(
        ratio: float,
        download: DownloadActivity | None = None,
    ) -> None:
        """
        Emit an intermediate progress event.

        Parameters
        ----------
        ratio :
            The ratio of how much work the ADRIO has completed in total;
            0 meaning no progress and 1 meaning it is finished.
        download :
            Describes current network activity. If there is no network activity
            to report or if it cannot be measured, provide `None`.
        """
        events.on_adrio_progress.publish(
            ADRIOProgress(
                adrio_name=adrio.class_name,
                attribute=adrio.name,
                final=False,
                ratio_complete=min(ratio, 1.0),
                download=download,
                duration=None,
            )
        )

    _report_progress(0.0)
    start_time = perf_counter()
    yield _report_progress
    finish_time = perf_counter()
    duration = finish_time - start_time

    # Emit a final progress event.
    events.on_adrio_progress.publish(
        ADRIOProgress(
            adrio_name=adrio.class_name,
            attribute=adrio.name,
            final=True,
            ratio_complete=1.0,
            download=None,
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
) -> AttributeData:
    # depth-first evaluation guarantees `data` has our dependencies.
    ctx = Context.of(name, data, scope, time_frame, ipm, rng)
    sim_func = value.with_context_internal(ctx)
    return sim_func.evaluate()


@contextmanager
def adrio_exception_handling(adrio: ADRIO, ctx: Context):
    # TODO: docstring
    try:
        yield
    except ADRIOError:
        raise
    except DeferredADRIOError as e:
        raise e.error_type(adrio, ctx, e.message) from e.__cause__
    except MissingContextError as e:
        raise ADRIOContextError(adrio, ctx, str(e))
    except HTTPError as e:
        raise ADRIOCommunicationError(adrio, ctx, str(e)) from e
    except ValidationError as e:
        raise ADRIOProcessingError(adrio, ctx, str(e)) from e
    except Exception as e:
        raise ADRIOProcessingError(adrio, ctx) from e
