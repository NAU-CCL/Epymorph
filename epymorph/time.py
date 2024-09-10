from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import cached_property
from typing import Generic, Iterator, Literal, NamedTuple, Self, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import override


@dataclass(frozen=True)
class DateRange:
    """Like `range` but for dates, with an optional fixed interval between dates."""

    start_date: date
    stop_date: date | None = field(default=None)
    step: timedelta | None = field(default=None)

    def __iter__(self) -> Iterator[date]:
        step = self.step or timedelta(days=1)
        curr = self.start_date
        if self.stop_date is None:
            while True:
                yield curr
                curr += step
        else:
            while curr < self.stop_date:
                yield curr
                curr += step


def iso8601(value: date | str) -> date:
    """Adapt ISO 8601 strings to dates; leave dates as they are."""
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


@dataclass(frozen=True)
class TimeFrame:
    """The time frame of a simulation."""

    @classmethod
    def of(cls, start_date: date | str, duration_days: int) -> Self:
        """
        Alternate constructor for TimeFrame.
        If a date is passed as a string, it will be parsed using ISO-8601 format.
        """
        start_date = iso8601(start_date)
        return cls(start_date, duration_days)

    @classmethod
    def range(cls, start_date: date | str, end_date: date | str) -> Self:
        """
        Alternate constructor for TimeFrame, comprising the
        (endpoint inclusive) date range.
        If a date is passed as a string, it will be parsed using ISO-8601 format.
        """
        start_date = iso8601(start_date)
        end_date = iso8601(end_date)
        duration = (end_date - start_date).days + 1
        return cls(start_date, duration)

    @classmethod
    def rangex(cls, start_date: date | str, end_date_exclusive: date | str) -> Self:
        """
        Alternate constructor for TimeFrame, comprising the
        (endpoint exclusive) date range.
        If a date is passed as a string, it will be parsed using ISO-8601 format.
        """
        start_date = iso8601(start_date)
        end_date_exclusive = iso8601(end_date_exclusive)
        duration = (end_date_exclusive - start_date).days
        return cls(start_date, duration)

    @classmethod
    def year(cls, year: int) -> Self:
        """Alternate constructor for TimeFrame, comprising one full calendar year."""
        return cls.rangex(date(year, 1, 1), date(year + 1, 1, 1))

    start_date: date
    """The first date in the simulation."""
    duration_days: int
    """The number of days for which to run the simulation."""

    def __post_init__(self):
        if self.duration_days < 1:
            err = (
                "TimeFrame's end date cannot be before its start date. "
                "(Its duration in days must be at least 1.)"
            )
            raise ValueError(err)

    @cached_property
    def end_date(self) -> date:
        """The last date included in the simulation."""
        return self.start_date + timedelta(days=self.duration_days - 1)

    def is_subset(self, other: "TimeFrame") -> bool:
        """Is the given TimeFrame a subset of this one?"""
        return self.start_date <= other.start_date and self.end_date >= other.end_date

    def __iter__(self) -> Iterator[date]:
        """Iterates over the sequence of dates which are part of this time frame."""
        stop_date = self.start_date + timedelta(days=self.duration_days)
        yield from DateRange(self.start_date, stop_date)

    def __str__(self) -> str:
        if self.duration_days == 1:
            return f"{self.start_date} (1D)"
        return f"{self.start_date}/{self.end_date} ({self.duration_days}D)"

    def to_numpy(self) -> NDArray[np.datetime64]:
        """Returns a numpy array of the dates in this time frame."""
        return np.array(list(self), dtype=np.datetime64)

    @property
    def select(self) -> "TimeSelector":
        """Create a time-axis strategy from this time frame.

        In most cases, this will be used to process a simulation result
        and so you should use a selection on the time frame used in the
        RUME that produced the result."""
        return TimeSelector(self)


class EpiWeek(NamedTuple):
    """A CDC-defined system for labeling weeks of the year.
    See: https://www.cmmcp.org/mosquito-surveillance-data/pages/epi-week-calendars-2008-2024"""

    year: int
    """Four-digit year"""
    week: int
    """Week number, in the range [1,53]"""

    def __str__(self) -> str:
        return f"{self.year}-{self.week}"


def epi_year_first_day(year: int) -> pd.Timestamp:
    """Calculates the first day in an epi-year."""
    first_saturday = pd.Timestamp(year, 1, 1) + pd.offsets.Week(weekday=5)
    if first_saturday.day < 4:
        first_saturday = first_saturday + pd.offsets.Week(weekday=5)
    first_epi_day = first_saturday - pd.offsets.Week(weekday=6)
    return first_epi_day


def epi_week(check_date: date) -> EpiWeek:
    """Calculates which epi week the given date belongs to."""
    d = pd.Timestamp(check_date.year, check_date.month, check_date.day)
    last_year_day1 = epi_year_first_day(d.year - 1)
    this_year_day1 = epi_year_first_day(d.year)
    next_year_day1 = epi_year_first_day(d.year + 1)
    if d < this_year_day1:
        # in last years' epi weeks
        origin = last_year_day1
        year = d.year - 1
    elif d >= next_year_day1:
        # in next years' epi weeks
        origin = next_year_day1
        year = d.year + 1
    else:
        # in this years' epi weeks
        origin = this_year_day1
        year = d.year
    return EpiWeek(year, (d - origin).days // 7 + 1)


def epi_week_start(epi_week: EpiWeek) -> pd.Timestamp:
    """Returns the first date in a given epi week."""
    day1 = epi_year_first_day(epi_week.year)
    return day1 + pd.offsets.Week(n=epi_week.week - 1)


#####################################
# Time frame select/group/aggregate #
#####################################


D = TypeVar("D", bound=np.generic)


class Dim(NamedTuple):
    """Describes data dimensions for a time-grouping operation;
    i.e., after subselections and geo-grouping."""

    nodes: int
    days: int
    tau_steps: int


class TimeGrouping(ABC, Generic[D]):
    """Defines a time-axis grouping scheme. This is essentially a function that maps
    the simulation time axis info (ticks and dates) into a new series which describes
    the group membership of each time axis row."""

    group_format: Literal["date", "tick", "day", "other"]
    """What scale describes the result of the grouping?
    Are the group keys dates? Simulation ticks? Simulation days?
    Or some arbitrary other type?"""

    @abstractmethod
    def map(
        self,
        dim: Dim,
        ticks: NDArray[np.int64],
        dates: NDArray[np.datetime64],
    ) -> NDArray[D]:
        """Produce a column that describes the group membership of each "row",
        where each entry of `ticks` and `dates` describes a row of the time series.
        This column will be used as the basis of a `groupby` operation.

        The result must correspond element-wise to the given `ticks` and `dates` arrays.
        `dim` contains dimensional info relevant to this grouping operation.
        Note that we may have sub-selected the geo and/or the time frame, so these
        dimensions may differ from those of the simulation as a whole.

        `ticks` and `dates` will be `nodes * days * tau_steps` in length.
        Values will be in order, but each tick will be represented `nodes` times,
        and each date will be represented `nodes * tau_steps` times.
        Since we may be grouping a slice of the simulation time frame, ticks may start
        after 0 and end before the last tick of the simulation."""


class ByTick(TimeGrouping[np.int64]):
    """Group by simulation tick. (Effectively the same as no grouping.)"""

    group_format = "tick"

    @override
    def map(self, dim, ticks, dates):
        return ticks


class ByDate(TimeGrouping[np.datetime64]):
    """Group by date."""

    group_format = "date"

    @override
    def map(self, dim, ticks, dates):
        return dates


class ByWeek(TimeGrouping[np.datetime64]):
    """Group by week, using the given start of the week."""

    _NP_START_OF_WEEK: int = 3
    """Numpy starts its weeks on Thursday, because the unix epoch 0
    is on a Thursday."""

    group_format = "date"

    start_of_week: int
    """Which day of the week should we group on? 0=Monday through 6=Sunday"""

    def __init__(self, start_of_week: int = 0):
        if not (0 <= start_of_week <= 6):
            raise ValueError("Invalid day of the week.")
        self.start_of_week = start_of_week

    @override
    def map(self, dim, ticks, dates) -> NDArray[np.datetime64]:
        delta = ByWeek._NP_START_OF_WEEK - self.start_of_week
        result = (dates + delta).astype("datetime64[W]").astype("datetime64[D]") - delta
        return result  # type: ignore


class ByMonth(TimeGrouping[np.datetime64]):
    """Group by month, using the first day of the month."""

    group_format = "date"

    @override
    def map(self, dim, ticks, dates):
        return dates.astype("datetime64[M]").astype("datetime64[D]")


class EveryNDays(TimeGrouping[np.datetime64]):
    """Group every 'n' days from the start of the time range."""

    group_format = "date"

    days: int
    """How many days are in each group?"""

    def __init__(self, days: int):
        self.days = days

    @override
    def map(self, dim, ticks, dates):
        n = self.days * dim.tau_steps * dim.nodes
        # careful we don't return too many date values
        return (dates[::n].repeat(n))[0 : len(dates)]


class NBins(TimeGrouping[np.int64]):
    """Group the time series into a number of bins where bin boundaries
    must align with simulation days.
    If the time series is not evenly divisible into the given number of bins,
    you may get more bins than requested (data will not be truncated).
    You will never get more than one bin per day."""

    group_format = "other"

    bins: int
    """Approximately how many bins should be in the result?"""

    def __init__(self, bins: int):
        self.bins = bins

    @override
    def map(self, dim, ticks, dates):
        bin_size = max(1, dim.days // self.bins) * dim.tau_steps
        return (ticks - ticks[0]) // bin_size


class ByEpiWeek(TimeGrouping[np.str_]):
    """Group the time series by epi week."""

    group_format = "other"

    @override
    def map(self, dim, ticks, dates):
        return np.array([str(epi_week(d.astype(date))) for d in dates], dtype=np.str_)


_AggMethod = Literal["sum", "max", "min", "mean", "median", "last", "first"]
"""A method for aggregating time series data."""


class TimeAggMethod(NamedTuple):
    """A time-axis aggregation scheme. There may be one aggregation method for
    compartments and another for events."""

    compartments: _AggMethod
    events: _AggMethod


DEFAULT_TIME_AGG = TimeAggMethod("last", "sum")
"""By default, time series should be aggregated using 'last' for compartments
and 'sum' for events."""


@dataclass(frozen=True)
class TimeStrategy:
    """A strategy for dealing with the time axis, e.g., in processing results.

    Strategies can include selection of a subset, grouping, and aggregation."""

    time_frame: TimeFrame
    """The original time frame."""
    selection: slice
    """A slice for selection of a subset of the time frame."""
    grouping: TimeGrouping | None
    """A method for grouping the time series data."""
    aggregation: TimeAggMethod | None
    """A method for aggregating the time series data
    (if no grouping is specified, the time series is reduced to a scalar)."""

    @property
    @abstractmethod
    def group_format(self) -> Literal["date", "tick", "day", "other"]:
        """What scale describes the result of the grouping?
        Are the group keys dates? Simulation ticks? Simulation days?
        Or some arbitrary other type?"""

    @property
    def date_bounds(self) -> tuple[date, date]:
        """The bounds of the selection, given as the first and last date included."""
        start = self.selection.start or 0
        stop = self.selection.stop or self.time_frame.duration_days
        first_date = self.time_frame.start_date + timedelta(days=start)
        last_date = self.time_frame.start_date + timedelta(days=stop - 1)
        return (first_date, last_date)

    def to_time_frame(self) -> TimeFrame:
        """Creates a TimeFrame that has the same bounds as this TimeStrategy.
        NOTE: this does not mean the TimeFrame contains the same number of entries
        (group keys) as the result of applying this strategy -- groups can skip days
        whereas TimeFrames are contiguous."""
        first, last = self.date_bounds
        return TimeFrame.range(first, last)

    def selection_ticks(self, taus: int) -> slice:
        s = self.selection
        return slice(
            None if s.start is None else s.start * taus,
            None if s.stop is None else s.stop * taus,
            None if s.step is None else s.step * taus,
        )


@dataclass(frozen=True)
class TimeSelector:
    """A utility class for selecting a subset of a time frame."""

    time_frame: TimeFrame
    """The original time frame."""

    def all(self) -> "TimeSelection":
        """Select the entirety of the time frame."""
        return TimeSelection(self.time_frame, slice(None))

    def by_slice(self, start: int, stop: int | None = None) -> "TimeSelection":
        return TimeSelection(self.time_frame, slice(start, stop))

    def _to_selection(self, other: TimeFrame) -> "TimeSelection":
        if not self.time_frame.is_subset(other):
            err = "When selecting part of a time frame you must specify a subset."
            raise ValueError(err)
        from_index = (other.start_date - self.time_frame.start_date).days
        to_index = (other.end_date - self.time_frame.start_date).days + 1
        return self.by_slice(from_index, to_index)

    def range(self, from_date: date | str, to_date: date | str) -> "TimeSelection":
        """Subset the time frame by providing the start and end date (inclusive)."""
        other = TimeFrame.range(from_date, to_date)
        return self._to_selection(other)

    def rangex(self, from_date: date | str, until_date: date | str) -> "TimeSelection":
        """Subset the time frame by providing the start and end date (exclusive)."""
        other = TimeFrame.rangex(from_date, until_date)
        return self._to_selection(other)

    def year(self, year: int) -> "TimeSelection":
        """Subset the time frame to a specific year."""
        other = TimeFrame.year(year)
        return self._to_selection(other)


class _CanAggregate(TimeStrategy):
    def agg(
        self,
        compartments: _AggMethod = "last",
        events: _AggMethod = "sum",
    ) -> "TimeAggregation":
        return TimeAggregation(
            self.time_frame,
            self.selection,
            self.grouping,
            TimeAggMethod(compartments, events),
        )


@dataclass(frozen=True)
class TimeSelection(_CanAggregate, TimeStrategy):
    """Describes a sub-selection operation on a time frame
    (no grouping or aggregation)."""

    time_frame: TimeFrame
    """The original time frame."""
    selection: slice
    """A slice for selection of a subset of the time frame."""
    grouping: None = field(init=False, default=None)
    """A method for grouping the time series data."""
    aggregation: None = field(init=False, default=None)
    """A method for aggregating the time series data
    (if no grouping is specified, the time series is reduced to a scalar)."""

    @property
    @override
    def group_format(self) -> Literal["tick"]:
        return "tick"

    def group(self, grouping: TimeGrouping) -> "TimeGroup":
        """Groups the time series using the specified grouping."""
        return TimeGroup(self.time_frame, self.selection, grouping)


@dataclass(frozen=True)
class TimeGroup(_CanAggregate, TimeStrategy):
    """Describes a grouping operation on a time frame,
    with an optional sub-selection."""

    time_frame: TimeFrame
    """The original time frame."""
    selection: slice
    """A slice for selection of a subset of the time frame."""
    grouping: TimeGrouping
    """A method for grouping the time series data."""
    aggregation: None = field(init=False, default=None)
    """A method for aggregating the time series data
    (if no grouping is specified, the time series is reduced to a scalar)."""

    @property
    @override
    def group_format(self):
        return self.grouping.group_format


@dataclass(frozen=True)
class TimeAggregation(TimeStrategy):
    """Describes a group-and-aggregate operation on a time frame,
    with an optional sub-selection."""

    time_frame: TimeFrame
    """The original time frame."""
    selection: slice
    """A slice for selection of a subset of the time frame."""
    grouping: TimeGrouping | None
    """A method for grouping the time series data."""
    aggregation: TimeAggMethod
    """A method for aggregating the time series data
    (if no grouping is specified, the time series is reduced to a scalar)."""

    @property
    @override
    def group_format(self):
        return "tick" if self.grouping is None else self.grouping.group_format
