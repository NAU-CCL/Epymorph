"""Modeling time in epymorph."""

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
    A sequence of calendar dates, with a fixed interval between dates (default 1 day).

    Parameters
    ----------
    start_date :
        The first date in the range.
    end_date :
        The last date in the range. Must be an exact multiple of steps after start_date.
    step :
        The step between dates in the range, as a number of days. Must be 1 or greater.
    """

    start_date: date
    """The first date in the range."""
    end_date: date
    """The last date in the range."""
    step: int = field(default=1)
    """The step between dates in the range, as a number of days."""

    @classmethod
    def until_date(cls, start_date: date, max_end_date: date, step: int) -> Self:
        """
        Alternative constructor for cases where you aren't sure of the precise end
        date: that is, you know roughly when the range ends but aren't sure if that
        date is an exact number of steps after start date.

        Parameters
        ----------
        start_date :
            The first date in the range.
        max_end_date :
            The latest possible date in the range. If `max_end_date` is already an exact
            multiple of steps away from `start_date`, it will be the `DateRange`'s end
            date. Otherwise, we will calculate the latest date that is before
            `max_end_date` and also an exact multiple of steps after start date.
        step :
            The interval between dates in the range, as a number of days.
            Must be 1 or greater.

        Returns
        -------
        :
            A new `DateRange`.
        """
        diff = (max_end_date - start_date).days % step
        end_date = max_end_date - timedelta(days=diff)
        return cls(start_date, end_date, step)

    def __post_init__(self):
        if self.start_date > self.end_date:
            raise ValueError("`start_date` must be before or equal to `end_date`")
        if self.step < 1:
            raise ValueError("`step` must be 1 or greater")
        if (self.end_date - self.start_date).days % self.step != 0:
            err = "`end_date` must be a multiple of `step` days from `start_date`"
            raise ValueError(err)

    def _next_date(self, from_date: date) -> date | None:
        """
        If `from_date` is in the range, return it.
        Otherwise return the next date that is in the range.
        Returns `None` if no date satisfies these conditions.
        """
        s = self.step
        diff = (s - (from_date - self.start_date).days % s) % s
        result = max(self.start_date, from_date + timedelta(days=diff))
        return result if result <= self.end_date else None

    def _prev_date(self, from_date: date) -> date | None:
        """
        If `from_date` is in the range, return it.
        Otherwise return the most recent date that is in the range.
        Returns `None` if no date satisfies these conditions.
        """
        diff = (from_date - self.start_date).days % self.step
        result = min(self.end_date, from_date - timedelta(days=diff))
        return result if result >= self.start_date else None

    def between(self, min_date: date, max_date: date) -> "DateRange | None":
        """
        Compute a new `DateRange` that represents the subset of dates in this range
        that are also between `min_date` and `max_date` (inclusive).

        Parameters
        ----------
        min_date :
            The earliest date to include in the subset.
        max_date :
            The latest date to include in the subset.

        Returns
        -------
        :
            The subset `DateRange`, or `None` if that subset would be empty -- when
            there's no overlap between this range and the min/max dates.
        """
        if min_date > max_date:
            raise ValueError("`min_date` must be before or equal to `max_date`")
        new_start_date = self._next_date(min_date)
        new_end_date = self._prev_date(max_date)
        if (
            new_start_date is None
            or new_end_date is None
            or new_start_date > new_end_date
        ):
            return None
        return DateRange(new_start_date, new_end_date, self.step)

    def overlap(self, time_frame: "TimeFrame") -> "DateRange | None":
        """
        Compute a new `DateRange` that represents the subset of dates in this range
        that are also in the given TimeFrame.

        Parameters
        ----------
        time_frame :
            The time frame to overlap.

        Returns
        -------
        :
            The subset DateRange, or None if that subset would be empty -- when there's
            no overlap between this DateRange and the time frame.
        """
        return self.between(time_frame.start_date, time_frame.end_date)

    def overlap_or_raise(self, time_frame: "TimeFrame") -> "DateRange":
        """
        Compute a new `DateRange` that represents the subset of dates in this range
        that are also in the given `TimeFrame`. If there is no overlap, raise an error.

        Parameters
        ----------
        time_frame :
            The time frame to overlap.

        Returns
        -------
        :
            The subset `DateRange`.

        Raises
        ------
        ValueError
            When there's no overlap between this `DateRange` and the time frame.
        """
        if (date_range := self.overlap(time_frame)) is None:
            err = "There is no overlap between the date range and time frame."
            raise ValueError(err)
        return date_range

    def __len__(self) -> int:
        delta_days = (self.end_date - self.start_date).days
        return (delta_days // self.step) + 1

    def to_pandas(self) -> pd.DatetimeIndex:
        """
        Convert the `DateRange` to a Pandas datetime index.

        Returns
        -------
        :
            The index containing all dates in the range in order.
        """
        return pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq=timedelta(days=self.step),
            inclusive="both",
        )

    def to_numpy(self) -> NDArray[np.datetime64]:
        """
        Convert the `DateRange` to a numpy datetime64 array.

        Returns
        -------
        :
            The one-dimensional array containing all dates in the range in order.
        """
        step = timedelta(days=self.step)
        return np.arange(
            start=self.start_date,
            stop=self.end_date + step,
            step=step,
            dtype="datetime64[D]",
        )


def iso8601(value: date | str) -> date:
    """
    Adapt ISO 8601 strings to dates; leave dates as they are.

    Parameters
    ----------
    value :
        The value to parse (if string) or pass through unchanged (if date).

    Returns
    -------
    :
        The equivalent date.
    """
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


@dataclass(frozen=True)
class TimeFrame:
    """
    Describes a time frame as a contiguous set of calendar dates,
    primarily used to define the time frame of a simulation.

    Parameters
    ----------
    start_date :
        The first date included in the time frame.
    duration_days :
        The number of days included in the time frame.

    Examples
    --------
    --8<-- "docs/_examples/time_TimeFrame.md"
    """

    @classmethod
    def of(cls, start_date: date | str, duration_days: int) -> Self:
        """
        Alternate constructor: start date (accepts ISO-8601 strings)
        and a given duration in days.

        Parameters
        ----------
        start_date :
            The starting date.
            If a date is passed as a string, it will be parsed using ISO-8601 format.
        duration_days :
            The number of days in the time frame, including the first day.

        Returns
        -------
        :
            The new `TimeFrame`.
        """
        start_date = iso8601(start_date)
        return cls(start_date, duration_days)

    @classmethod
    def range(cls, start_date: date | str, end_date: date | str) -> Self:
        """
        Alternate constructor: start and end date (inclusive).

        Parameters
        ----------
        start_date :
            The starting date.
            If a date is passed as a string, it will be parsed using ISO-8601 format.
        end_date :
            The final date included in the time frame.
            If a date is passed as a string, it will be parsed using ISO-8601 format.

        Returns
        -------
        :
            The new `TimeFrame`.
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
        start_date :
            The starting date.
            If a date is passed as a string, it will be parsed using ISO-8601 format.
        end_date_exclusive :
            The stop date, which is to say the first date not in the time frame.
            If a date is passed as a string, it will be parsed using ISO-8601 format.

        Returns
        -------
        :
            The new `TimeFrame`.
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
        year :
            The year of the time frame, from January 1st through December 31st.
            Includes 365 days, or 366 days on a leap year.

        Returns
        -------
        :
            The new `TimeFrame`.
        """
        return cls.rangex(date(year, 1, 1), date(year + 1, 1, 1))

    start_date: date
    """The first date in the time frame."""
    duration_days: int
    """The number of days included in the time frame."""
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
        Check if the given `TimeFrame` is a subset of this one.

        Parameters
        ----------
        other :
            The other time frame to consider.

        Returns
        -------
        :
            True if the other time frame is a subset of this time frame.
        """
        return self.start_date <= other.start_date and self.end_date >= other.end_date

    @property
    def days(self) -> int:
        """
        The number of days included in the time frame.

        Alias for `duration_days`.
        """
        return self.duration_days

    def __iter__(self) -> Iterator[date]:
        """Iterate over the sequence of dates in the time frame."""
        step = timedelta(days=1)
        curr = self.start_date
        for _ in range(self.duration_days):
            yield curr
            curr += step

    def __str__(self) -> str:
        if self.duration_days == 1:
            return f"{self.start_date} (1D)"
        return f"{self.start_date}/{self.end_date} ({self.duration_days}D)"

    def to_numpy(self) -> NDArray[np.datetime64]:
        """
        Return a numpy array of the dates in this time frame.

        Returns
        -------
        :
            The equivalent numpy array.
        """
        return np.array(list(self), dtype=np.datetime64)

    def to_date_range(self) -> DateRange:
        """
        Return a date range that corresponds to this time frame's
        start and end date.

        Returns
        -------
        :
            A matching date range.
        """
        return DateRange(
            start_date=self.start_date,
            end_date=self.end_date,
        )

    @property
    def select(self) -> "TimeSelector":
        """
        Create a time-axis strategy from this time frame.

        In most cases, this will be used to process a simulation result
        and so you should use a selection on the time frame used in the
        RUME that produced the result.
        """
        return TimeSelector(self)


#############
# Epi Weeks #
#############


class EpiWeek(NamedTuple):
    """
    Identifies a specific epi week, a CDC-defined system for labeling weeks of the year
    in a consistent way.

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
    Calculate the first day in an epi-year.

    Parameters
    ----------
    year :
        The year to consider.

    Returns
    -------
    :
        The first day of the first epi week in the given year.
    """
    first_saturday = pd.Timestamp(year, 1, 1) + pd.offsets.Week(weekday=5)
    if first_saturday.day < 4:
        first_saturday = first_saturday + pd.offsets.Week(weekday=5)
    first_epi_day = first_saturday - pd.offsets.Week(weekday=6)
    return first_epi_day


def epi_week(check_date: date) -> EpiWeek:
    """
    Calculate which epi week the given date belongs to.

    Parameters
    ----------
    check_date :
        The date to consider.

    Returns
    -------
    :
        The `EpiWeek` that contains the given date.
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
"""The numpy type used to represent the group keys of a `TimeGrouping`."""


class Dim(NamedTuple):
    """
    Describes data dimensions for a time-grouping operation;
    i.e., after subselections and geo-grouping.
    """

    nodes: int
    """The number of unique nodes or node-groups."""
    days: int
    """The number of days, after any sub-selection."""
    tau_steps: int
    """The number of tau steps per day."""


class TimeGrouping(ABC, Generic[GroupKeyType]):
    """
    Base class for time-axis grouping schemes. This is essentially a function that maps
    the simulation time axis info (ticks and dates) into a new series which describes
    the group membership of each time axis row.

    TimeGrouping is generic in the type of the key it uses for its
    time groups (`GroupKeyType`) -- e.g., a grouping that groups weeks
    using the Monday of the week is datetime64 typed, while a grouping that groups
    days into arbitrary buckets might use integers to identify groups.
    """

    group_format: Literal["date", "tick", "day", "other"]
    """
    What scale describes the result of the grouping?
    Are the group keys dates? Simulation ticks? Simulation days?
    Or some arbitrary other type?
    """

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
        dim :
            The simulation dimensions for time grouping.
        ticks :
            The series of simulation ticks.
        dates :
            The series of calendar dates corresponding to simulation ticks.

        Returns
        -------
        :
            The group membership of each tick. For example, if the first three
            ticks were in group 0 and the next three ticks were in group 1, etc.,
            the returned array would contain `[0,0,0,1,1,1,...]`.
        """


class ByTick(TimeGrouping[np.int64]):
    """
    A kind of `TimeGrouping` to group by simulation tick.
    Effectively the same as no grouping.
    """

    group_format = "tick"

    @override
    def map(self, dim, ticks, dates):
        return ticks


class ByDate(TimeGrouping[np.datetime64]):
    """A kind of `TimeGrouping` to group by date."""

    group_format = "date"

    @override
    def map(self, dim, ticks, dates):
        return dates


class ByWeek(TimeGrouping[np.datetime64]):
    """
    A kind of `TimeGrouping` to group by week, using a configurable start of the week.

    Parameters
    ----------
    start_of_week :
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
    """A kind `TimeGrouping` to group by month, using the first day of the month."""

    group_format = "date"

    @override
    def map(self, dim, ticks, dates):
        return dates.astype("datetime64[M]").astype("datetime64[D]")


class EveryNDays(TimeGrouping[np.datetime64]):
    """
    A kind of `TimeGrouping` to group every 'n' days from the start of the time range.

    Parameters
    ----------
    days :
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
    A kind of `TimeGrouping` to group the time series into a number of bins
    where bin boundaries must align with simulation days.

    If the time series is not evenly divisible into the given number of bins,
    you may get more bins than requested (data will not be truncated).
    You will never get more than one bin per day.

    Parameters
    ----------
    bins :
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
    """A kind of `TimeGrouping` to group the time series by epi week."""

    group_format = "other"

    @override
    def map(self, dim, ticks, dates):
        return np.array([str(epi_week(d.astype(date))) for d in dates], dtype=np.str_)


AggMethod = Literal["sum", "max", "min", "mean", "median", "last", "first"]
"""A method for aggregating time series data."""


class TimeAggMethod(NamedTuple):
    """
    A time-axis aggregation scheme. There may be one aggregation method for
    compartments and another for events.
    """

    compartments: AggMethod
    """The method for aggregating compartment values."""
    events: AggMethod
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
    `TimeSelector` instance.

    Parameters
    ----------
    time_frame :
        The original simulation time frame.
    selection :
        The selected subset of the time frame: described as a date slice
        and an optional tau step index.
    grouping :
        A method for grouping the time series data.
    aggregation :
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
    """
    A method for aggregating the time series data
    (if no grouping is specified, the time series is reduced to a scalar).
    """

    @property
    @abstractmethod
    def group_format(self) -> Literal["date", "tick", "day", "other"]:
        """
        The scale that describes the result of the grouping.

        Are the group keys dates? Simulation ticks? Simulation days?
        Or some arbitrary other type?
        """

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
        """
        Create a `TimeFrame` that has the same bounds as this `TimeStrategy`.

        NOTE: this does not mean the `TimeFrame` contains the same number of entries
        (group keys) as the result of applying this strategy -- groups can skip days
        whereas `TimeFrames` are contiguous.

        Returns
        -------
        :
            The corresponding time frame.
        """
        first, last = self.date_bounds
        return TimeFrame.range(first, last)

    def selection_ticks(self, taus: int) -> slice:
        """
        Convert this into a slice for which ticks are selected (by index).

        Parameters
        ----------
        taus :
            The total number of tau steps per day.

        Returns
        -------
        :
            The selection as an index slice.
        """
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
        compartments: AggMethod = "last",
        events: AggMethod = "sum",
    ) -> "TimeAggregation":
        """
        Aggregate the time series using the specified methods.

        Parameters
        ----------
        compartments :
            The method to use to aggregate compartment values.
        events :
            The method to use to aggregate event values.

        Returns
        -------
        :
            The aggregation strategy object.
        """
        return TimeAggregation(
            self.time_frame,
            self.selection,
            self.grouping,
            TimeAggMethod(compartments, events),
        )


@dataclass(frozen=True)
class TimeSelection(_CanAggregate, TimeStrategy):
    """
    A kind of `TimeStrategy` describing a sub-selection of a time frame.
    A selection performs no grouping or aggregation.

    Typically you will create one of these by calling methods on a
    `TimeSelector` instance.

    Parameters
    ----------
    time_frame :
        The original simulation time frame.
    selection :
        The selected subset of the time frame: described as a date slice
        and an optional tau step index.
    """

    time_frame: TimeFrame
    """The original time frame."""
    selection: tuple[slice, int | None]
    """
    The selected subset of the time frame: described as a date slice
    and an optional tau step index.
    """
    grouping: None = field(init=False, default=None)
    """A method for grouping the time series data."""
    aggregation: None = field(init=False, default=None)
    """
    A method for aggregating the time series data
    (if no grouping is specified, the time series is reduced to a scalar).
    """

    @property
    @override
    def group_format(self) -> Literal["tick"]:
        return "tick"

    def group(
        self,
        grouping: Literal["day", "week", "epiweek", "month"] | TimeGrouping,
    ) -> "TimeGroup":
        """
        Group the time series using the specified grouping.

        Parameters
        ----------
        grouping :
            The grouping to use. You can specify a supported string value --
            all of which act as shortcuts for common `TimeGrouping` instances --
            or you can provide a `TimeGrouping` instance to perform custom grouping.

            The shortcut values are:

            - "day": equivalent to `ByDate`
            - "week": equivalent to `ByWeek`
            - "epiweek": equivalent to `ByEpiWeek`
            - "month": equivalent to `ByMonth`

        Returns
        -------
        :
            The grouping strategy.
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
    A kind of `TimeStrategy` describing a group operation on a time frame,
    with an optional sub-selection.

    Typically you will create one of these by calling methods on a
    `TimeSelection` instance.

    Parameters
    ----------
    time_frame :
        The original simulation time frame.
    selection :
        The selected subset of the time frame: described as a date slice
        and an optional tau step index.
    grouping :
        A method for grouping the time series data.
    """

    time_frame: TimeFrame
    """The original time frame."""
    selection: tuple[slice, int | None]
    """
    The selected subset of the time frame: described as a date slice
    and an optional tau step index.
    """
    grouping: TimeGrouping
    """A method for grouping the time series data."""
    aggregation: None = field(init=False, default=None)
    """
    A method for aggregating the time series data
    (if no grouping is specified, the time series is reduced to a scalar).
    """

    @property
    @override
    def group_format(self):
        return self.grouping.group_format


@dataclass(frozen=True)
class TimeAggregation(TimeStrategy):
    """
    A kind of `TimeStrategy` describing a group-and-aggregate operation on a time frame,
    with an optional sub-selection.

    Typically you will create one of these by calling methods on a
    `TimeSelection` or `TimeGroup` instance.

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
    """
    The selected subset of the time frame: described as a date slice
    and an optional tau step index.
    """
    grouping: TimeGrouping | None
    """A method for grouping the time series data."""
    aggregation: TimeAggMethod
    """
    A method for aggregating the time series data
    (if no grouping is specified, the time series is reduced to a scalar).
    """

    @property
    @override
    def group_format(self):
        return "tick" if self.grouping is None else self.grouping.group_format


@dataclass(frozen=True)
class TimeSelector:
    """
    A utility class for selecting a subset of a time frame.
    Most of the time you obtain one of these using `TimeFrame`'s `select` property.
    """

    time_frame: TimeFrame
    """The original time frame."""

    def all(self, step: int | None = None) -> TimeSelection:
        """
        Select the entirety of the time frame.

        Parameters
        ----------
        step :
            If given, narrow the selection to a specific tau step (by index) within
            the date range; by default include all steps.

        Returns
        -------
        :
            The selection strategy object.
        """
        return TimeSelection(self.time_frame, (slice(None), step))

    def _to_selection(
        self,
        other: TimeFrame,
        step: int | None = None,
    ) -> TimeSelection:
        """
        Use a `TimeFrame` object (which must be a subset of the base `TimeFrame`)
        to create a `TimeSelection`.
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
        """
        Subset the time frame by providing a start and end simulation day (inclusive).

        Parameters
        ----------
        from_day :
            The starting simulation day of the range, as an index.
        to_day :
            The last included simulation day of the range, as an index.
        step :
            If given, narrow the selection to a specific tau step (by index) within
            the date range; by default include all steps.

        Returns
        -------
        :
            The selection strategy object.
        """
        return TimeSelection(self.time_frame, (slice(from_day, to_day + 1), step))

    def range(
        self,
        from_date: date | str,
        to_date: date | str,
        step: int | None = None,
    ) -> TimeSelection:
        """
        Subset the time frame by providing the start and end date (inclusive).

        Parameters
        ----------
        from_date :
            The starting date of the range, as a date object or an ISO-8601 string.
        to_date :
            The last included date of the range, as a date object or an ISO-8601 string.
        step :
            If given, narrow the selection to a specific tau step (by index) within
            the date range; by default include all steps.

        Returns
        -------
        :
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
        """
        Subset the time frame by providing the start and end date (exclusive).

        Parameters
        ----------
        from_date :
            The starting date of the range, as a date object or an ISO-8601 string.
        until_date :
            The stop date date of the range (the first date excluded)
            as a date object or an ISO-8601 string.
        step :
            If given, narrow the selection to a specific tau step (by index) within
            the date range; by default include all steps.

        Returns
        -------
        :
            The selection strategy object.
        """
        other = TimeFrame.rangex(from_date, until_date)
        return self._to_selection(other, step)

    def year(
        self,
        year: int,
        step: int | None = None,
    ) -> TimeSelection:
        """
        Subset the time frame to a specific year.

        Parameters
        ----------
        year :
            The year to include, from January 1st through December 31st.
        step :
            If given, narrow the selection to a specific tau step (by index) within
            the date range; by default include all steps.

        Returns
        -------
        :
            The selection strategy object.
        """
        other = TimeFrame.year(year)
        return self._to_selection(other, step)
