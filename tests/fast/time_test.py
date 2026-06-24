# ruff: noqa: PT009,PT027
import unittest
from datetime import date
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from epymorph.time import (
    DateRange,
    Dim,
    EpiWeek,
    NBins,
    TimeFrame,
    epi_week,
    epi_year_first_day,
)

#############
# DateRange #
#############


def test_date_range():
    # These are all valid ranges.
    DateRange(date(2020, 1, 1), date(2020, 1, 1), step=7)
    DateRange(date(2020, 1, 1), date(2020, 1, 8), step=7)
    DateRange(date(2020, 1, 1), date(2020, 1, 15), step=7)


def test_date_range_invalid():
    with pytest.raises(
        ValueError,
        match="`start_date` must be before or equal to `end_date`",
    ):
        DateRange(date(2021, 1, 1), date(2020, 1, 9), step=2)

    with pytest.raises(
        ValueError,
        match="`step` must be 1 or greater",
    ):
        DateRange(date(2020, 1, 1), date(2020, 1, 9), step=-1)

    with pytest.raises(
        ValueError,
        match="`end_date` must be a multiple of `step` days from `start_date`",
    ):
        DateRange(date(2020, 1, 1), date(2020, 1, 10), step=2)


def test_date_range_len():
    d = DateRange(date(2020, 1, 1), date(2020, 1, 9), step=2)
    assert len(d) == 5


def test_date_range_to_numpy():
    d = DateRange(date(2020, 1, 1), date(2020, 1, 9), step=2)
    actual = d.to_numpy()
    expected = np.array(
        [
            "2020-01-01",
            "2020-01-03",
            "2020-01-05",
            "2020-01-07",
            "2020-01-09",
        ],
        dtype=np.datetime64,
    )
    np.testing.assert_array_equal(actual, expected)


def test_date_range_to_pandas():
    d = DateRange(date(2020, 1, 1), date(2020, 1, 9), step=2)
    actual = d.to_pandas()
    expected = pd.date_range(date(2020, 1, 1), date(2020, 1, 9), freq="2D")
    pd.testing.assert_index_equal(actual, expected)


def test_date_range_between_same_dates():
    d = DateRange(date(2025, 1, 1), date(2025, 1, 29), step=7)
    actual = d.between(min_date=date(2025, 1, 1), max_date=date(2025, 1, 29))
    assert actual is not None
    assert actual.start_date == date(2025, 1, 1)
    assert actual.end_date == date(2025, 1, 29)
    assert actual.step == 7


def test_date_range_between_within_step():
    d = DateRange(date(2025, 1, 1), date(2025, 1, 29), step=7)
    actual = d.between(min_date=date(2025, 1, 8), max_date=date(2025, 1, 22))
    assert actual is not None
    assert actual.start_date == date(2025, 1, 8)
    assert actual.end_date == date(2025, 1, 22)
    assert actual.step == 7


def test_date_range_between_outside_step():
    d = DateRange(date(2025, 1, 1), date(2025, 1, 29), step=7)
    actual = d.between(min_date=date(2025, 1, 5), max_date=date(2025, 1, 24))
    assert actual is not None
    assert actual.start_date == date(2025, 1, 8)
    assert actual.end_date == date(2025, 1, 22)
    assert actual.step == 7


def test_date_range_between_no_change():
    d = DateRange(date(2025, 1, 1), date(2025, 1, 29), step=7)
    actual = d.between(min_date=date(2020, 1, 1), max_date=date(2030, 1, 1))
    assert actual is not None
    assert actual.start_date == date(2025, 1, 1)
    assert actual.end_date == date(2025, 1, 29)
    assert actual.step == 7


def test_date_range_between_min_date_after_max_date():
    d = DateRange(date(2025, 1, 1), date(2025, 1, 29), step=7)
    with pytest.raises(
        ValueError,
        match="`min_date` must be before or equal to `max_date`",
    ):
        d.between(min_date=date(2025, 1, 25), max_date=date(2025, 1, 20))


def test_date_range_between_resulting_invalid_range():
    d = DateRange(date(2025, 1, 1), date(2025, 1, 29), step=7)
    actual = d.between(min_date=date(2025, 1, 25), max_date=date(2025, 1, 27))
    assert actual is None


def test_date_range_between_no_overlap():
    d = DateRange(date(2025, 1, 1), date(2025, 1, 29), step=7)
    actual = d.between(min_date=date(2025, 2, 1), max_date=date(2025, 2, 10))
    assert actual is None


class TestTimeFrame(unittest.TestCase):
    def test_init_1(self):
        tf = TimeFrame(date(2020, 1, 1), 30)
        self.assertEqual(tf.start_date, date(2020, 1, 1))
        self.assertEqual(tf.duration_days, 30)
        self.assertEqual(tf.end_date, date(2020, 1, 30))

    def test_init_2(self):
        tf = TimeFrame.of("2020-01-01", 30)
        self.assertEqual(tf.start_date, date(2020, 1, 1))
        self.assertEqual(tf.duration_days, 30)
        self.assertEqual(tf.end_date, date(2020, 1, 30))

    def test_init_3(self):
        tf = TimeFrame.range("2020-01-01", "2020-01-30")
        self.assertEqual(tf.start_date, date(2020, 1, 1))
        self.assertEqual(tf.duration_days, 30)
        self.assertEqual(tf.end_date, date(2020, 1, 30))

    def test_init_4(self):
        tf = TimeFrame.rangex("2020-01-01", "2020-01-30")
        self.assertEqual(tf.start_date, date(2020, 1, 1))
        self.assertEqual(tf.duration_days, 29)
        self.assertEqual(tf.end_date, date(2020, 1, 29))

    def test_init_5(self):
        tf = TimeFrame.year(2020)
        self.assertEqual(tf.start_date, date(2020, 1, 1))
        self.assertEqual(tf.duration_days, 366)
        self.assertEqual(tf.end_date, date(2020, 12, 31))

    def test_init_6(self):
        # ERROR: negative duration
        with self.assertRaises(ValueError):
            TimeFrame(date(2020, 1, 1), -7)

    def test_init_7(self):
        # ERROR: negative duration
        with self.assertRaises(ValueError):
            TimeFrame.range(date(2020, 1, 1), date(1999, 1, 1))

    def test_subset_1(self):
        a = TimeFrame.rangex("2020-01-01", "2020-02-01")
        b = TimeFrame.rangex("2020-01-01", "2020-02-01")
        c = TimeFrame.rangex("2020-01-01", "2020-01-21")
        d = TimeFrame.rangex("2020-01-14", "2020-02-01")
        e = TimeFrame.rangex("2020-01-14", "2020-01-21")
        self.assertTrue(a.is_subset(b))
        self.assertTrue(a.is_subset(c))
        self.assertTrue(a.is_subset(d))
        self.assertTrue(a.is_subset(e))

    def test_subset_2(self):
        a = TimeFrame.rangex("2020-01-01", "2020-02-01")
        b = TimeFrame.rangex("2019-12-31", "2020-02-01")
        c = TimeFrame.rangex("2020-01-01", "2020-09-21")
        d = TimeFrame.rangex("2019-12-31", "2020-09-21")
        e = TimeFrame.rangex("2019-01-01", "2019-02-01")
        f = TimeFrame.rangex("2021-01-01", "2021-02-01")
        self.assertFalse(a.is_subset(b))
        self.assertFalse(a.is_subset(c))
        self.assertFalse(a.is_subset(d))
        self.assertFalse(a.is_subset(e))
        self.assertFalse(a.is_subset(f))


class EpiWeeksTest(unittest.TestCase):
    def test_first_epi_day(self):
        self.assertEqual(epi_year_first_day(2020), pd.Timestamp(2019, 12, 29))
        self.assertEqual(epi_year_first_day(2021), pd.Timestamp(2021, 1, 3))
        self.assertEqual(epi_year_first_day(2022), pd.Timestamp(2022, 1, 2))
        self.assertEqual(epi_year_first_day(2023), pd.Timestamp(2023, 1, 1))
        self.assertEqual(epi_year_first_day(2024), pd.Timestamp(2023, 12, 31))
        self.assertEqual(epi_year_first_day(2025), pd.Timestamp(2024, 12, 29))

    def test_epi_week(self):
        self.assertEqual(epi_week(date(2021, 1, 1)), EpiWeek(2020, 53))
        self.assertEqual(epi_week(date(2021, 1, 2)), EpiWeek(2020, 53))
        self.assertEqual(epi_week(date(2021, 1, 3)), EpiWeek(2021, 1))
        self.assertEqual(epi_week(date(2024, 1, 1)), EpiWeek(2024, 1))
        self.assertEqual(epi_week(date(2024, 1, 6)), EpiWeek(2024, 1))
        self.assertEqual(epi_week(date(2024, 1, 7)), EpiWeek(2024, 2))
        self.assertEqual(epi_week(date(2024, 3, 14)), EpiWeek(2024, 11))
        self.assertEqual(epi_week(date(2024, 12, 28)), EpiWeek(2024, 52))
        self.assertEqual(epi_week(date(2024, 12, 29)), EpiWeek(2025, 1))
        self.assertEqual(epi_week(date(2024, 12, 31)), EpiWeek(2025, 1))


class TestNBins(unittest.TestCase):
    def _do_test(
        self,
        *,
        bins: int,
        nodes: int,
        days: int,
        tau_steps: int,
        expected_bins: int,
        expected_ticks_per_bin: int,
        tick_offset: int = 0,  # tick indices don't have to start from zero!
    ) -> None:
        dim = Dim(nodes, days, tau_steps)
        t = np.arange(dim.days * dim.tau_steps).repeat(dim.nodes) + tick_offset
        d = Mock(np.array)  # NBins doesn't use dates, so mock is fine
        exp = np.arange(expected_bins).repeat(nodes).repeat(expected_ticks_per_bin)
        act = NBins(bins).map(dim, t, d)
        assert_array_equal(exp, act)

    def test_nbins_01(self):
        # Simple case
        self._do_test(
            bins=10,
            nodes=1,
            days=10,
            tau_steps=3,
            tick_offset=0,
            expected_bins=10,
            expected_ticks_per_bin=3,
        )

    def test_nbins_02(self):
        # Simple case with non-zero tick offset
        self._do_test(
            bins=10,
            nodes=1,
            days=10,
            tau_steps=3,
            tick_offset=3,
            expected_bins=10,
            expected_ticks_per_bin=3,
        )

    def test_nbins_03(self):
        # Simple case with more than one node
        self._do_test(
            bins=10,
            nodes=5,
            days=10,
            tau_steps=3,
            tick_offset=0,
            expected_bins=10,
            expected_ticks_per_bin=3,
        )

    def test_nbins_04(self):
        # We can wind up with more bins than asked for
        self._do_test(
            bins=10,  # not evenly divisible
            nodes=1,
            days=11,
            tau_steps=3,
            tick_offset=0,
            expected_bins=11,
            expected_ticks_per_bin=3,
        )

    def test_nbins_05(self):
        # We can wind up with less bins than asked for
        self._do_test(
            bins=25,  # more than one bin per day
            nodes=2,
            days=10,
            tau_steps=2,
            tick_offset=0,
            expected_bins=10,
            expected_ticks_per_bin=2,
        )


#####################
# TimeSelector.days #
#####################


@pytest.fixture
def ten_day_frame():
    return TimeFrame.of("2022-09-15", 10)


def test_selector_days_valid_full_range(ten_day_frame):
    sel = ten_day_frame.select.days(0, 9)
    day_slice, step = sel.selection
    assert day_slice == slice(0, 10)
    assert step is None


def test_selector_days_valid_partial_range(ten_day_frame):
    sel = ten_day_frame.select.days(2, 5)
    day_slice, _ = sel.selection
    assert day_slice == slice(2, 6)


def test_selector_days_valid_single_day(ten_day_frame):
    sel = ten_day_frame.select.days(4, 4)
    day_slice, _ = sel.selection
    assert day_slice == slice(4, 5)


def test_selector_days_out_of_range_to_day(ten_day_frame):
    with pytest.raises(ValueError, match="out of range"):
        ten_day_frame.select.days(0, 10)


def test_selector_days_out_of_range_both(ten_day_frame):
    with pytest.raises(ValueError, match="out of range"):
        ten_day_frame.select.days(80, 80)


def test_selector_days_out_of_range_from_day(ten_day_frame):
    with pytest.raises(ValueError, match="out of range"):
        ten_day_frame.select.days(10, 10)


def test_selector_days_negative_from_day(ten_day_frame):
    with pytest.raises(ValueError, match="must be non-negative"):
        ten_day_frame.select.days(-1, 5)


def test_selector_days_negative_to_day(ten_day_frame):
    with pytest.raises(ValueError, match="must be non-negative"):
        ten_day_frame.select.days(0, -1)


def test_selector_days_from_day_after_to_day(ten_day_frame):
    with pytest.raises(ValueError, match="from_day must be less than or equal"):
        ten_day_frame.select.days(5, 3)


####################
# TimeSelector.all #
####################


def test_selector_all(ten_day_frame):
    sel = ten_day_frame.select.all()
    day_slice, step = sel.selection
    assert day_slice == slice(None)
    assert step is None


def test_selector_all_with_step(ten_day_frame):
    sel = ten_day_frame.select.all(step=1)
    day_slice, step = sel.selection
    assert day_slice == slice(None)
    assert step == 1


######################
# TimeSelector.range #
######################


def test_selector_range_full(ten_day_frame):
    sel = ten_day_frame.select.range("2022-09-15", "2022-09-24")
    day_slice, step = sel.selection
    assert day_slice == slice(0, 10)
    assert step is None


def test_selector_range_partial(ten_day_frame):
    sel = ten_day_frame.select.range("2022-09-17", "2022-09-21")
    day_slice, _ = sel.selection
    assert day_slice == slice(2, 7)


def test_selector_range_date_objects(ten_day_frame):
    sel = ten_day_frame.select.range(date(2022, 9, 17), date(2022, 9, 21))
    day_slice, _ = sel.selection
    assert day_slice == slice(2, 7)


def test_selector_range_with_step(ten_day_frame):
    sel = ten_day_frame.select.range("2022-09-17", "2022-09-21", step=0)
    _, step = sel.selection
    assert step == 0


def test_selector_range_out_of_bounds(ten_day_frame):
    with pytest.raises(ValueError, match="must specify a subset"):
        ten_day_frame.select.range("2022-09-15", "2022-09-30")


def test_selector_range_before_start(ten_day_frame):
    with pytest.raises(ValueError, match="must specify a subset"):
        ten_day_frame.select.range("2022-09-01", "2022-09-20")


#######################
# TimeSelector.rangex #
#######################


def test_selector_rangex_full(ten_day_frame):
    # exclusive: "2022-09-25" is the day after the last day
    sel = ten_day_frame.select.rangex("2022-09-15", "2022-09-25")
    day_slice, step = sel.selection
    assert day_slice == slice(0, 10)
    assert step is None


def test_selector_rangex_partial(ten_day_frame):
    sel = ten_day_frame.select.rangex("2022-09-17", "2022-09-22")
    day_slice, _ = sel.selection
    assert day_slice == slice(2, 7)


def test_selector_rangex_with_step(ten_day_frame):
    sel = ten_day_frame.select.rangex("2022-09-17", "2022-09-22", step=1)
    _, step = sel.selection
    assert step == 1


def test_selector_rangex_out_of_bounds(ten_day_frame):
    with pytest.raises(ValueError, match="must specify a subset"):
        ten_day_frame.select.rangex("2022-09-15", "2022-09-30")


#####################
# TimeSelector.year #
#####################


@pytest.fixture
def year_2022_frame():
    return TimeFrame.year(2022)


def test_selector_year_valid(year_2022_frame):
    sel = year_2022_frame.select.year(2022)
    day_slice, step = sel.selection
    assert day_slice == slice(0, 365)
    assert step is None


def test_selector_year_valid_multi_year():
    multi_year_frame = TimeFrame.rangex("2021-01-01", "2024-01-01")
    sel = multi_year_frame.select.year(2022)
    day_slice, step = sel.selection
    assert day_slice == slice(365, 730)
    assert step is None


def test_selector_year_with_step(year_2022_frame):
    sel = year_2022_frame.select.year(2022, step=0)
    _, step = sel.selection
    assert step == 0


def test_selector_year_out_of_bounds(year_2022_frame):
    with pytest.raises(ValueError, match="must specify a subset"):
        year_2022_frame.select.year(2023)
