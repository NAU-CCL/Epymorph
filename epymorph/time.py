# ruff: noqa: A005
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Generic, Iterator, Literal, NamedTuple, Self, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import override


@dataclass(frozen=True)
class DateRange:
    """
    Like [`range`](:py:class:`range`) but for dates, with an optional fixed interval
    between dates.

    DateRange is a frozen dataclass.
    """

    start_date: date
    """The first date in the range."""
    stop_date: date | None = field(default=None)
    """The optional end of the date range (not included in the range)."""
    step: int | None = field(default=None)
    """The step between dates in the range, as a number of days.
    Must be 1 or greater."""

    def __post_init__(self):
        if self.step is not None and self.step < 1:
            raise ValueError("step must be 1 or greater")

    def __iter__(self) -> Iterator[date]:
        step = timedelta(days=(self.step or 1))
        curr = self.start_date
        while self.stop_date is None or curr < self.stop_date:
            yield curr
            curr += step


def iso8601(value: date | str) -> date:
    """Adapt ISO 8601 strings to dates; leave dates as they are."""
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


@dataclass(frozen=True)
class TimeFrame:
    """
    Describes a time frame as a contiguous set of calendar dates,
    primarily used to define the time frame of a simulation.

    TimeFrame is a frozen dataclass. TimeFrame is iterable, and produces
    the sequence of dates in the time frame.
    """

    @classmethod
    def of(cls, start_date: date | str, duration_days: int) -> Self:
        """
        Alternate constructor: start date and a given duration in days.

        Parameters
        ----------
        start_date : date | str
            The starting date.
            If a date is passed as a string, it will be parsed using ISO-8601 format.
        duration_days : int
            The number of days in the time frame, including the first day.
        """
        start_date = iso8601(start_date)
        return cls(start_date, duration_days)

    @classmethod
    def range(cls, start_date: date | str, end_date: date | str) -> Self:
        """
        Alternate constructor: start and end date (inclusive).

        Parameters
        ----------
        start_date : date | str
            The starting date.
            If a date is passed as a string, it will be parsed using ISO-8601 format.
        end_date : date | str
            The final date included in the time frame.
            If a date is passed as a string, it will be parsed using ISO-8601 format.
        """
        start_date = iso8601(start_date)
        end_date = iso8601(end_date)
        duration = (end_date - start_date).days + 1
        return cls(start_date, duration)

    @classmethod
    def rangex(cls, start_date: date | str, end_date_exclusive: date | str) -> Self:
        """
        Alternate constructor: start and end date (exclusive).

        Parameters
        ----------
        start_date : date | str
            The starting date.
            If a date is passed as a string, it will be parsed using ISO-8601 format.
        end_date_exclusive : date | str
            The stop date, which is to say the first date not in the time frame.
            If a date is passed as a string, it will be parsed using ISO-8601 format.
        """
        start_date = iso8601(start_date)
        end_date_exclusive = iso8601(end_date_exclusive)
        duration = (end_date_exclusive - start_date).days
        return cls(start_date, duration)

    @classmethod
    def year(cls, year: int) -> Self:
        """
        Alternate constructor: an entire calendar year.

        Parameters
        ----------
        year : int
            The year of the time frame, from January 1st through December 31st.
            Includes 365 days, or 366 days on a leap year.
        """
        return cls.rangex(date(year, 1, 1), date(year + 1, 1, 1))

    start_date: date
    """The first date in the time frame."""
    duration_days: int
    """The number of days for included in the time frame."""
    end_date: date = field(init=False)
    """The last date included in the time frame."""

    def __post_init__(self):
        if self.duration_days < 1:
            err = (
                "TimeFrame's end date cannot be before its start date. "
                "(Its duration in days must be at least 1.)"
            )
            raise ValueError(err)
        end_date = self.start_date + timedelta(days=self.duration_days - 1)
        object.__setattr__(self, "end_date", end_date)

    def is_subset(self, other: "TimeFrame") -> bool:
        """
        Is the given TimeFrame a subset of this one?

        Parameters
        ----------
        other : TimeFrame
            The other time frame to consider.

        Returns
        -------
        bool
            True if the other time frame is a subset of this time frame.
        """
        return self.start_date <= other.start_date and self.end_date >= other.end_date

    @property
    def days(self) -> int:
        """Alias for `duration_days`"""
        return self.duration_days

    def __iter__(self) -> Iterator[date]:
        """Iterates over the sequence of dates which are part of this time frame."""
        stop_date = self.start_date + timedelta(days=self.duration_days)
        yield from DateRange(self.start_date, stop_date)

    def __str__(self) -> str:
        if self.duration_days == 1:
            return f"{self.start_date} (1D)"
        return f"{self.start_date}/{self.end_date} ({self.duration_days}D)"

    def to_numpy(self) -> NDArray[np.datetime64]:
        """
        Returns a numpy array of the dates in this time frame.

        Returns
        -------
        NDArray[np.datetime64]
            The array in `np.datetime64` format.
        """
        return np.array(list(self), dtype=np.datetime64)

    @property
    def select(self) -> "TimeSelector":
        """Create a time-axis strategy from this time frame.

        In most cases, this will be used to process a simulation result
        and so you should use a selection on the time frame used in the
        RUME that produced the result."""
        return TimeSelector(self)


#############
# Epi Weeks #
#############


class EpiWeek(NamedTuple):
    """
    A CDC-defined system for labeling weeks of the year.

    EpiWeek is a NamedTuple.

    See Also
    --------
    [This reference on epi weeks.](https://www.cmmcp.org/mosquito-surveillance-data/pages/epi-week-calendars-2008-2024)
    """

    year: int
    """Four-digit year"""
    week: int
    """Week number, in the range [1,53]"""

    def __str__(self) -> str:
        return f"{self.year}-{self.week}"

    @property
    def start(self) -> pd.Timestamp:
        """The first date in this epi week."""
        day1 = epi_year_first_day(self.year)
        return day1 + pd.offsets.Week(n=self.week - 1)


def epi_year_first_day(year: int) -> pd.Timestamp:
    """
    Calculates the first day in an epi-year.

    Parameters
    ----------
    year : int
        The year to consider.

    Returns
    -------
    pd.Timestamp
        The first day of the first epi week in the given year.
    """
    first_saturday = pd.Timestamp(year, 1, 1) + pd.offsets.Week(weekday=5)
    if first_saturday.day < 4:
        first_saturday = first_saturday + pd.offsets.Week(weekday=5)
    first_epi_day = first_saturday - pd.offsets.Week(weekday=6)
    return first_epi_day


def epi_week(check_date: date) -> EpiWeek:
    """
    Calculates which epi week the given date belongs to.

    Parameters
    ----------
    check_date : date
        The date to consider.

    Returns
    -------
    EpiWeek
        The EpiWeek that contains the given date.
    """
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


#####################################
# Time frame select/group/aggregate #
#####################################


GroupKeyType = TypeVar("GroupKeyType", bound=np.generic)


class Dim(NamedTuple):
    """
    Describes data dimensions for a time-grouping operation;
    i.e., after subselections and geo-grouping.

    Dim is a NamedTuple.
    """

    nodes: int
    """The number of unique nodes or node-groups."""
    days: int
    """The number of days, after any sub-selection."""
    tau_steps: int
    """The number of tau steps per day."""


class TimeGrouping(ABC, Generic[GroupKeyType]):
    """
    Defines a time-axis grouping scheme. This is essentially a function that maps
    the simulation time axis info (ticks and dates) into a new series which describes
    the group membership of each time axis row.

    TimeGrouping is an abstract class.

    TimeGrouping is generic in the type of the key it uses for its
    time groups (`GroupKeyType`) -- e.g., a grouping that groups weeks
    using the Monday of the week is datetime64 typed, while a grouping that groups
    days into arbitrary buckets might use integers to identify groups.
    """

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
    ) -> NDArray[GroupKeyType]:
        """
        Produce a column that describes the group membership of each "row",
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
        after 0 and end before the last tick of the simulation.

        Parameters
        ----------
        dim : Dim
            The simulation dimensions for time grouping.
        ticks : NDArray[np.int64]
            The series of simulation ticks.
        dates : NDArray[np.datetime64]
            The series of calendar dates corresponding to simulation ticks.

        Returns
        -------
        NDArray[GroupKeyType]
            The group membership of each tick. For example, if the first three
            ticks were in group 0 and the next three ticks were in group 1, etc.,
            the returned array would contain `[0,0,0,1,1,1,...]`.
        """


class ByTick(TimeGrouping[np.int64]):
    """
    An instance of [`TimeGrouping`](`epymorph.time.TimeGrouping`).
    Group by simulation tick. (Effectively the same as no grouping.)
    """

    group_format = "tick"

    @override
    def map(self, dim, ticks, dates):
        return ticks


class ByDate(TimeGrouping[np.datetime64]):
    """
    An instance of [`TimeGrouping`](`epymorph.time.TimeGrouping`).
    Group by date.
    """

    group_format = "date"

    @override
    def map(self, dim, ticks, dates):
        return dates


class ByWeek(TimeGrouping[np.datetime64]):
    """
    An instance of [`TimeGrouping`](`epymorph.time.TimeGrouping`).
    Group by week, using the given start of the week.

    Parameters
    ----------
    start_of_week : int, default=0
        Which day of the week is the begins each weekly group?
        This uses the Python standard numbering for week days,
        so 0=Monday and 6=Sunday.
    """

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
    """
    An instance of [`TimeGrouping`](`epymorph.time.TimeGrouping`).
    Group by month, using the first day of the month.
    """

    group_format = "date"

    @override
    def map(self, dim, ticks, dates):
        return dates.astype("datetime64[M]").astype("datetime64[D]")


class EveryNDays(TimeGrouping[np.datetime64]):
    """
    An instance of [`TimeGrouping`](`epymorph.time.TimeGrouping`).
    Group every 'n' days from the start of the time range.

    Parameters
    ----------
    days : int
        How many days should be in each group?
    """

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
    """
    An instance of [`TimeGrouping`](`epymorph.time.TimeGrouping`).
    Group the time series into a number of bins where bin boundaries
    must align with simulation days.

    If the time series is not evenly divisible into the given number of bins,
    you may get more bins than requested (data will not be truncated).
    You will never get more than one bin per day.

    Parameters
    ----------
    bins : int
        Approximately how many bins should be in the result?
    """

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
    """
    An instance of [`TimeGrouping`](`epymorph.time.TimeGrouping`).
    Group the time series by epi week.
    """

    group_format = "other"

    @override
    def map(self, dim, ticks, dates):
        return np.array([str(epi_week(d.astype(date))) for d in dates], dtype=np.str_)


_AggMethod = Literal["sum", "max", "min", "mean", "median", "last", "first"]
"""
A method for aggregating time series data.

`Literal["sum", "max", "min", "mean", "median", "last", "first"]`
"""


class TimeAggMethod(NamedTuple):
    """
    A time-axis aggregation scheme. There may be one aggregation method for
    compartments and another for events.

    TimeAggMethod is a NamedTuple.
    """

    compartments: _AggMethod
    """The method for aggregating compartment values."""
    events: _AggMethod
    """The method for aggregating event values."""


DEFAULT_TIME_AGG = TimeAggMethod("last", "sum")
"""By default, time series should be aggregated using 'last' for compartments
and 'sum' for events."""


@dataclass(frozen=True)
class TimeStrategy:
    """
    A strategy for dealing with the time axis, e.g., in processing results.
    Strategies can include selection of a subset, grouping, and aggregation.

    Typically you will create one of these by calling methods on a
    [`TimeSelector`](`epymorph.time.TimeSelector`) instance.

    TimeStrategy is a frozen dataclass.

    Parameters
    ----------
    time_frame: TimeFrame
        The original simulation time frame.
    selection: tuple[slice, int | None]
        The selected subset of the time frame: described as a date slice
        and an optional tau step index.
    grouping: TimeGrouping | None
        A method for grouping the time series data.
    aggregation: TimeAggMethod | None
        A method for aggregating the time series data
        (if no grouping is specified, the time series is reduced to a scalar).
    """

    time_frame: TimeFrame
    """The original time frame."""
    selection: tuple[slice, int | None]
    """The selected subset of the time frame: described as a date slice
    and an optional tau step index."""
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
        date_slice, _ = self.selection
        start = date_slice.start or 0
        stop = date_slice.stop or self.time_frame.duration_days
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
        """Converts this into a slice for which ticks are selected (by index)."""
        day, tau_step = self.selection
        if tau_step is not None and tau_step >= taus:
            err = (
                "Invalid time-axis tau step selection: this model has "
                f"{taus} tau steps but you selected step index {tau_step} "
                "which is out of range."
            )
            raise ValueError(err)
        # There are two cases:
        if tau_step is not None:
            # If tau_step is specified, we want to select only the ticks
            # in the date range which correspond to that tau step.
            start = taus * (day.start or 0) + tau_step
            stop = None if day.stop is None else taus * day.stop + tau_step
            step = taus
            return slice(start, stop, step)
        else:
            # If tau_step is None, then we will select every tau step
            # in the date range.
            # This implies it is not possible to "step" over days --
            # the time series must be contiguous w.r.t. simulation days.
            return slice(
                None if day.start is None else day.start * taus,
                None if day.stop is None else day.stop * taus,
                # day.step is ignored if it is present
            )


class _CanAggregate(TimeStrategy):
    def agg(
        self,
        compartments: _AggMethod = "last",
        events: _AggMethod = "sum",
    ) -> "TimeAggregation":
        """
        Aggregates the time series using the specified methods.

        Paramters
        ---------
        compartments : Literal["sum", "max", "min", "mean", "median", "last", "first"], default="last"
            The method to use to aggregate compartment values.
        events : Literal["sum", "max", "min", "mean", "median", "last", "first"], default="sum"
            The method to use to aggregate event values.

        Returns
        -------
        TimeAggregation
            The aggregation strategy object.
        """  # noqa: E501
        return TimeAggregation(
            self.time_frame,
            self.selection,
            self.grouping,
            TimeAggMethod(compartments, events),
        )


@dataclass(frozen=True)
class TimeSelection(_CanAggregate, TimeStrategy):
    """
    A kind of [`TimeStrategy`](`epymorph.time.TimeStrategy`)
    describing a sub-selection of a time frame.
    A selection performs no grouping or aggregation.

    Typically you will create one of these by calling methods on a
    [`TimeSelector`](`epymorph.time.TimeSelector`) instance.

    TimeSelection is a frozen dataclass.

    Parameters
    ----------
    time_frame: TimeFrame
        The original simulation time frame.
    selection: tuple[slice, int | None]
        The selected subset of the time frame: described as a date slice
        and an optional tau step index.
    """

    time_frame: TimeFrame
    """The original time frame."""
    selection: tuple[slice, int | None]
    """The selected subset of the time frame: described as a date slice
    and an optional tau step index."""
    grouping: None = field(init=False, default=None)
    """A method for grouping the time series data."""
    aggregation: None = field(init=False, default=None)
    """A method for aggregating the time series data
    (if no grouping is specified, the time series is reduced to a scalar)."""

    @property
    @override
    def group_format(self) -> Literal["tick"]:
        return "tick"

    def group(
        self,
        grouping: Literal["day", "week", "epiweek", "month"] | TimeGrouping,
    ) -> "TimeGroup":
        """
        Groups the time series using the specified grouping.

        Parameters
        ----------
        grouping : Literal["day", "week", "epiweek", "month"] | TimeGrouping
            The grouping to use. You can specify a supported string value --
            all of which act as shortcuts for common TimeGrouping instances --
            or you can provide a TimeGrouping instance to perform custom grouping.

            The shortcut values are:

            - "day": equivalent to [`ByDate`](`epymorph.time.ByDate`)
            - "week": equivalent to [`ByWeek`](`epymorph.time.ByWeek`)
            - "epiweek": equivalent to [`ByEpiWeek`](`epymorph.time.ByEpiWeek`)
            - "month": equivalent to [`ByMonth`](`epymorph.time.ByMonth`)

        Returns
        -------
        TimeGroup
            The grouping strategy object.
        """
        match grouping:
            # String-based short-cuts:
            case "day":
                grouping = ByDate()
            case "week":
                grouping = ByWeek()
            case "epiweek":
                grouping = ByEpiWeek()
            case "month":
                grouping = ByMonth()
            # Otherwise grouping is a TimeGrouping instance
            case _:
                pass
        return TimeGroup(self.time_frame, self.selection, grouping)


@dataclass(frozen=True)
class TimeGroup(_CanAggregate, TimeStrategy):
    """
    A kind of [`TimeStrategy`](`epymorph.time.TimeStrategy`)
    describing a group operation on a time frame, with an optional sub-selection.

    Typically you will create one of these by calling methods on a
    [`TimeSelection`](`epymorph.time.TimeSelection`) instance.

    TimeGroup is a frozen dataclass.

    Parameters
    ----------
    time_frame: TimeFrame
        The original simulation time frame.
    selection: tuple[slice, int | None]
        The selected subset of the time frame: described as a date slice
        and an optional tau step index.
    grouping: TimeGrouping
        A method for grouping the time series data.
    """

    time_frame: TimeFrame
    """The original time frame."""
    selection: tuple[slice, int | None]
    """The selected subset of the time frame: described as a date slice
    and an optional tau step index."""
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
    """
    A kind of [`TimeStrategy`](`epymorph.time.TimeStrategy`)
    describing a group-and-aggregate operation on a time frame,
    with an optional sub-selection.

    Typically you will create one of these by calling methods on a
    [`TimeSelection`](`epymorph.time.TimeSelection`) or
    [`TimeGroup`](`epymorph.time.TimeGroup`) instance.

    TimeAggregation is a frozen dataclass.

    Parameters
    ----------
    time_frame: TimeFrame
        The original simulation time frame.
    selection: tuple[slice, int | None]
        The selected subset of the time frame: described as a date slice
        and an optional tau step index.
    grouping: TimeGrouping | None
        A method for grouping the time series data.
    aggregation: TimeAggMethod
        A method for aggregating the time series data
        (if no grouping is specified, the time series is reduced to a scalar).
    """

    time_frame: TimeFrame
    """The original time frame."""
    selection: tuple[slice, int | None]
    """The selected subset of the time frame: described as a date slice
    and an optional tau step index."""
    grouping: TimeGrouping | None
    """A method for grouping the time series data."""
    aggregation: TimeAggMethod
    """A method for aggregating the time series data
    (if no grouping is specified, the time series is reduced to a scalar)."""

    @property
    @override
    def group_format(self):
        return "tick" if self.grouping is None else self.grouping.group_format


@dataclass(frozen=True)
class TimeSelector:
    """
    A utility class for selecting a subset of a time frame.
    Most of the time you obtain one of these using
    [`TimeFrame`](`epymorph.time.TimeFrame`)'s `select` property.
    """

    time_frame: TimeFrame
    """The original time frame."""

    def all(self, step: int | None = None) -> TimeSelection:
        """Select the entirety of the time frame.

        Parameters
        ----------
        step : int, optional
            if given, narrow the selection to a specific tau step (by index) within
            the date range; by default include all steps
        """
        return TimeSelection(self.time_frame, (slice(None), step))

    def _to_selection(
        self,
        other: TimeFrame,
        step: int | None = None,
    ) -> TimeSelection:
        """
        Use a TimeFrame object (which must be a subset of the base TimeFrame)
        to create a TimeSelection.
        """
        if not self.time_frame.is_subset(other):
            err = "When selecting part of a time frame you must specify a subset."
            raise ValueError(err)
        from_index = (other.start_date - self.time_frame.start_date).days
        to_index = (other.end_date - self.time_frame.start_date).days + 1
        return TimeSelection(self.time_frame, (slice(from_index, to_index), step))

    def days(
        self,
        from_day: int,
        to_day: int,
        step: int | None = None,
    ) -> TimeSelection:
        """Subset the time frame by providing a start and end simulation day
        (inclusive).

        Parameters
        ----------
        from_day : int
            the starting simulation day of the range, as an index
        to_day : int
            the last included simulation day of the range, as an index
        step : int, optional
            if given, narrow the selection to a specific tau step (by index) within
            the date range; by default include all steps

        Returns
        -------
        TimeSelection
            The selection strategy object.
        """
        return TimeSelection(self.time_frame, (slice(from_day, to_day + 1), step))

    def range(
        self,
        from_date: date | str,
        to_date: date | str,
        step: int | None = None,
    ) -> TimeSelection:
        """Subset the time frame by providing the start and end date (inclusive).

        Parameters
        ----------
        from_date : date | str
            the starting date of the range, as a date object or an ISO-8601 string
        to_date : date | str
            the last included date of the range, as a date object or an ISO-8601 string
        step : int, optional
            if given, narrow the selection to a specific tau step (by index) within
            the date range; by default include all steps

        Returns
        -------
        TimeSelection
            The selection strategy object.
        """
        other = TimeFrame.range(from_date, to_date)
        return self._to_selection(other, step)

    def rangex(
        self,
        from_date: date | str,
        until_date: date | str,
        step: int | None = None,
    ) -> TimeSelection:
        """Subset the time frame by providing the start and end date (exclusive).

        Parameters
        ----------
        from_date : date | str
            the starting date of the range, as a date object or an ISO-8601 string
        until_date : date | str
            the stop date date of the range (the first date excluded)
            as a date object or an ISO-8601 string
        step : int, optional
            if given, narrow the selection to a specific tau step (by index) within
            the date range; by default include all steps

        Returns
        -------
        TimeSelection
            The selection strategy object.
        """
        other = TimeFrame.rangex(from_date, until_date)
        return self._to_selection(other, step)

    def year(
        self,
        year: int,
        step: int | None = None,
    ) -> TimeSelection:
        """Subset the time frame to a specific year.

        Parameters
        ----------
        year : int
            the year to include, from January 1st through December 31st
        step : int, optional
            if given, narrow the selection to a specific tau step (by index) within
            the date range; by default include all steps

        Returns
        -------
        TimeSelection
            The selection strategy object.
        """
        other = TimeFrame.year(year)
        return self._to_selection(other, step)
