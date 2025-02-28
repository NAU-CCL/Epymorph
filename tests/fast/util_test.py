# ruff: noqa: PT009,PT027
import unittest

import numpy as np
import pytest

from epymorph import util
from epymorph.data_shape import DataShapeMatcher, Dimensions, Shapes
from epymorph.util import (
    Event,
    extract_date_value,
    is_date_value_array,
    is_numeric,
    progress,
    subscriptions,
    to_date_value_array,
)
from epymorph.util import match as m

######################
# FUNCTION UTILITIES #
######################


@pytest.mark.parametrize("test_input", [1, "hey", [1, 2, 3], {"foo": "bar"}])
def test_identity(test_input):
    assert test_input == util.identity(test_input)


########################
# COLLECTION UTILITIES #
########################


def test_filter_unique():
    act = util.filter_unique(["a", "b", "b", "c", "a"])
    exp = ["a", "b", "c"]
    assert act == exp


def test_list_not_none():
    act = util.list_not_none(["a", None, "b", None, None, "c", None])
    exp = ["a", "b", "c"]
    assert act == exp


###################
# NUMPY UTILITIES #
###################


def test_is_numeric():
    assert is_numeric(np.array([1, 2, 3]))
    assert is_numeric(np.array([1.0, 2.0, 3.0]))
    assert is_numeric(np.array([1, 2.0, 3]))  # mixed type: float
    assert not is_numeric(np.array([1 + 1j, 2 + 2j, 3 + 3j]))
    assert not is_numeric(np.array(["a", "b", "c"]))
    assert not is_numeric(np.array([True, False, True]))
    assert not is_numeric(np.array([1, "a", 3]))  # mixed type: object
    assert not is_numeric(  # structured types not allowed
        np.array(
            [(1.0, 1.0), (2.0, 2.0)],
            dtype=[("a", np.float64), ("b", np.float64)],
        )
    )
    # NOTE: empty arrays default to float so this would pass:
    #   `assert is_numeric(np.array([]))`
    # but we don't need to encode numpy quirks into our tests


def test_check_ndarray_01():
    # None of these should raise NumpyTypeError
    arr = np.array([1, 2, 3], dtype=np.int64)

    dim = Dimensions.of(T=10, N=3)

    util.check_ndarray(arr)
    util.check_ndarray(arr, dtype=m.dtype(np.int64))
    util.check_ndarray(arr, shape=DataShapeMatcher(Shapes.N, dim))
    util.check_ndarray(
        arr,
        dtype=m.dtype(np.int64),
        shape=DataShapeMatcher(
            Shapes.N,
            dim,
        ),
    )
    util.check_ndarray(
        arr,
        dtype=m.dtype(np.int64, np.float64),
        shape=DataShapeMatcher(Shapes.N, dim),
    )
    util.check_ndarray(
        arr,
        dtype=m.dtype(np.float64, np.int64),
        shape=DataShapeMatcher(Shapes.N, dim, exact=True),
    )
    util.check_ndarray(
        arr,
        dtype=m.dtype(np.int64, np.float64),
        shape=DataShapeMatcher(
            Shapes.TxN,
            dim,
        ),
    )


def test_check_ndarray_02():
    # Raises exception for anything that's not a numpy array
    with pytest.raises(util.NumpyTypeError):
        util.check_ndarray(None)
    with pytest.raises(util.NumpyTypeError):
        util.check_ndarray(1)
    with pytest.raises(util.NumpyTypeError):
        util.check_ndarray([1, 2, 3])
    with pytest.raises(util.NumpyTypeError):
        util.check_ndarray("foofaraw")


def test_check_ndarray_03():
    arr = np.arange(12).reshape((3, 4))

    # Doesn't raise...
    dim1 = Dimensions.of(T=3, N=4)
    util.check_ndarray(arr, shape=DataShapeMatcher(Shapes.TxN, dim1))

    # Does raise...
    dim2 = Dimensions.of(T=4, N=3)
    with pytest.raises(util.NumpyTypeError):
        util.check_ndarray(arr, shape=DataShapeMatcher(Shapes.TxN, dim2))

    with pytest.raises(util.NumpyTypeError):
        util.check_ndarray(arr, dtype=m.dtype(np.str_))


class TestHaversine(unittest.TestCase):
    test_coords = np.array(
        [(0, 0), (1, 0), (0, 1)],
        dtype=[("longitude", np.float64), ("latitude", np.float64)],
    )

    def test_01(self):
        # Test miles
        actual1 = util.pairwise_haversine(self.test_coords)
        actual2 = util.pairwise_haversine(self.test_coords, units="miles")
        actual3 = util.pairwise_haversine(self.test_coords, radius=3963.1906)
        expected = np.array(
            [
                [0.0, 69.17, 69.17],
                [69.17, 0.0, 97.82],
                [69.17, 97.82, 0.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_almost_equal(expected, actual1, decimal=2)
        np.testing.assert_array_almost_equal(expected, actual2, decimal=2)
        np.testing.assert_array_almost_equal(expected, actual3, decimal=2)

    def test_02(self):
        # Test kilometers
        actual1 = util.pairwise_haversine(self.test_coords, units="kilometers")
        actual2 = util.pairwise_haversine(self.test_coords, radius=6378.1370)
        expected = np.array(
            [
                [0.0, 111.32, 111.32],
                [111.32, 0.0, 157.43],
                [111.32, 157.43, 0.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_almost_equal(expected, actual1, decimal=2)
        np.testing.assert_array_almost_equal(expected, actual2, decimal=2)

    def test_03(self):
        # Test custom radius
        actual1 = util.pairwise_haversine(self.test_coords, radius=10)
        expected = np.array(
            [
                [0.0, 0.17, 0.17],
                [0.17, 0.0, 0.25],
                [0.17, 0.25, 0.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_almost_equal(expected, actual1, decimal=2)

    def test_04(self):
        # Test tuple input
        actual1 = util.pairwise_haversine((np.array([0, 1, 0]), np.array([0, 0, 1])))
        expected = np.array(
            [
                [0.0, 69.17, 69.17],
                [69.17, 0.0, 97.82],
                [69.17, 97.82, 0.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_almost_equal(expected, actual1, decimal=2)

    def test_05(self):
        # Test bad input
        with self.assertRaises(ValueError):
            util.pairwise_haversine([1, 2, 3])  # type: ignore


####################
# DATE/VALUE TYPES #
####################


def test_to_date_value_array():
    # Test with 1D values array
    dates = np.array(["2021-01-01", "2021-01-02"], dtype="datetime64[D]")
    values = np.array([10, 20], dtype=np.int64)
    actual = to_date_value_array(dates, values)
    expected = np.array(
        [("2021-01-01", 10), ("2021-01-02", 20)],
        dtype=[("date", "datetime64[D]"), ("value", np.int64)],
    )
    np.testing.assert_array_equal(actual, expected)

    # Test with 2D values array
    dates = np.array(["2021-01-01", "2021-01-02"], dtype="datetime64[D]")
    values = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int64)
    actual = to_date_value_array(dates, values)
    expected = np.array(
        [
            [("2021-01-01", 10), ("2021-01-01", 20), ("2021-01-01", 30)],
            [("2021-01-02", 40), ("2021-01-02", 50), ("2021-01-02", 60)],
        ],
        dtype=[("date", "datetime64[D]"), ("value", np.int64)],
    )
    np.testing.assert_array_equal(actual, expected)

    # Test with 3D values array
    dates = np.array(["2021-01-01", "2021-01-02"], dtype="datetime64[D]")
    values = np.array(
        [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]],
        dtype=np.int64,
    )
    actual = to_date_value_array(dates, values)
    expected = np.array(
        [
            [
                [("2021-01-01", 10), ("2021-01-01", 20), ("2021-01-01", 30)],
                [("2021-01-01", 40), ("2021-01-01", 50), ("2021-01-01", 60)],
            ],
            [
                [("2021-01-02", 70), ("2021-01-02", 80), ("2021-01-02", 90)],
                [("2021-01-02", 100), ("2021-01-02", 110), ("2021-01-02", 120)],
            ],
        ],
        dtype=[("date", "datetime64[D]"), ("value", np.int64)],
    )
    np.testing.assert_array_equal(actual, expected)

    # Error: mismatched number of dates
    with pytest.raises(ValueError):  # noqa: PT011
        to_date_value_array(
            np.array(["2021-01-01", "2021-01-02"], dtype="datetime64[D]"),
            np.array([10, 20, 30], dtype=np.int64),
        )

    # Error: 2D array of dates
    with pytest.raises(ValueError):  # noqa: PT011
        to_date_value_array(
            np.array(
                [["2021-01-01", "2021-01-02"], ["2021-01-03", "2021-01-04"]],
                dtype="datetime64[D]",
            ),
            np.array([10, 20, 30], dtype=np.int64),
        )


def test_is_date_value_array():  #
    # Test with valid date/value array - int64 values
    array = np.array(
        [("2021-01-01", 10), ("2021-01-02", 20)],
        dtype=[("date", "datetime64[D]"), ("value", np.int64)],
    )
    assert is_date_value_array(array)
    assert is_date_value_array(array, value_dtype=np.int64)
    assert not is_date_value_array(array, value_dtype=np.str_)

    # Test with valid date/value array - str_ values
    array = np.array(
        [("2021-01-01", "10"), ("2021-01-02", "20")],
        dtype=[("date", "datetime64[D]"), ("value", np.str_)],
    )
    assert is_date_value_array(array)
    assert is_date_value_array(array, value_dtype=np.str_)
    assert not is_date_value_array(array, value_dtype=np.int64)

    # Test incorrect date dtype
    array = np.array(
        [("2021-01-01T11:11:11", 10), ("2021-01-02T12:13:14", 20)],
        dtype=[("date", "datetime64[ns]"), ("value", np.int64)],
    )
    assert not is_date_value_array(array)

    # Test incorrect field names
    array = np.array(
        [("2021-01-01", 10), ("2021-01-02", 20)],
        dtype=[
            ("a_field_formerly_known_as_date", "datetime64[D]"),
            ("value", np.int64),
        ],
    )
    assert not is_date_value_array(array)

    # Test non-structured types
    assert not is_date_value_array(np.arange(9))
    assert not is_date_value_array(np.array([]))
    assert not is_date_value_array(np.int64(42))  # type: ignore


def test_extract_date_value():
    # Test 1D data
    date_values = np.array(
        [("2021-01-01", 10), ("2021-01-02", 20)],
        dtype=[("date", "datetime64[D]"), ("value", np.int64)],
    )
    exp_dates = np.array(["2021-01-01", "2021-01-02"], dtype="datetime64[D]")
    exp_values = np.array([10, 20])

    dates, values = extract_date_value(date_values, value_dtype=np.int64)
    np.testing.assert_array_equal(dates, exp_dates)
    np.testing.assert_array_equal(values, exp_values)

    dates, values = extract_date_value(date_values)
    np.testing.assert_array_equal(dates, exp_dates)
    np.testing.assert_array_equal(values, exp_values)

    # Test 2D data
    date_values = np.array(
        [
            [("2021-01-01", 10), ("2021-01-01", 20)],
            [("2021-01-02", 30), ("2021-01-02", 40)],
        ],
        dtype=[("date", "datetime64[D]"), ("value", np.int64)],
    )
    exp_dates = np.array(["2021-01-01", "2021-01-02"], dtype="datetime64[D]")
    exp_values = np.array([[10, 20], [30, 40]])

    dates, values = extract_date_value(date_values)
    np.testing.assert_array_equal(dates, exp_dates)
    np.testing.assert_array_equal(values, exp_values)

    # Test non-matching value_dtype
    date_values = np.array(
        [("2021-01-01", 10), ("2021-01-02", 20)],
        dtype=[("date", "datetime64[D]"), ("value", np.int64)],
    )
    with pytest.raises(ValueError, match="values did not match expected dtype"):
        extract_date_value(date_values, value_dtype=np.float64)


#######################
# CONSOLE DECORATIONS #
#######################


class TestProgress(unittest.TestCase):
    def test_zero_percent(self):
        self.assertEqual(progress(0), "|                    | 0% ")

    def test_full_percent(self):
        self.assertEqual(progress(1), "|####################| 100% ")

    def test_half_percent(self):
        self.assertEqual(progress(0.5), "|##########          | 50% ")

    def test_clamping_below_zero(self):
        self.assertEqual(progress(-0.1), "|                    | 0% ")

    def test_clamping_above_one(self):
        self.assertEqual(progress(1.5), "|####################| 100% ")

    def test_custom_length(self):
        self.assertEqual(progress(0.5, 10), "|#####     | 50% ")

    def test_small_length(self):
        self.assertEqual(progress(0.51, 1), "| | 51% ")

    def test_long_length(self):
        self.assertEqual(
            progress(0.7, 43), "|##############################             | 70% "
        )

    def test_invalid_length(self):
        """Test with invalid length (less than 1)."""
        with self.assertRaises(ValueError):
            progress(0.5, 0)


##################
# PUB-SUB EVENTS #
##################


class TestEvent(unittest.TestCase):
    def setUp(self):
        """Set up a new Event instance for each test."""
        self.event = Event[int]()

    def test_subscribe_adds_subscriber(self):
        """Test: subscribing adds a subscriber."""

        def handler(event: int):
            pass

        self.event.subscribe(handler)
        self.assertEqual(len(self.event._subscribers), 1)
        self.assertIn(handler, self.event._subscribers)

    def test_unsubscribe_removes_subscriber(self):
        """Test: unsubscribing removes the correct subscriber."""

        def handler(event: int):
            pass

        self.assertEqual(len(self.event._subscribers), 0)

        unsubscribe = self.event.subscribe(handler)
        self.assertEqual(len(self.event._subscribers), 1)

        unsubscribe()
        self.assertEqual(len(self.event._subscribers), 0)
        self.assertNotIn(handler, self.event._subscribers)

    def test_publish_calls_subscriber(self):
        """Test: publish calls the subscribed handler."""
        self.subscriber_called = False

        def handler(event: int):
            self.subscriber_called = True
            self.assertEqual(event, 42)

        self.event.subscribe(handler)
        self.event.publish(42)
        self.assertTrue(self.subscriber_called)

    def test_publish_multiple_subscribers(self):
        """Test: publish calls all subscribers."""
        self.subscriber1_called = False
        self.subscriber2_called = False

        def handler1(event: int):
            self.subscriber1_called = True

        def handler2(event: int):
            self.subscriber2_called = True

        self.event.subscribe(handler1)
        self.event.subscribe(handler2)
        self.event.publish(42)

        self.assertTrue(self.subscriber1_called)
        self.assertTrue(self.subscriber2_called)

    def test_unsubscribed_handler_not_called(self):
        """Test that unsubscribed handler is not called when event is published."""
        self.subscriber_called = False

        def handler(event: int):
            self.subscriber_called = True

        unsubscribe = self.event.subscribe(handler)
        unsubscribe()

        self.event.publish(42)
        self.assertFalse(self.subscriber_called)

    def test_has_subscribers_initially_false(self):
        """Test: has_subscribers is False initially."""
        self.assertFalse(self.event.has_subscribers)

    def test_has_subscribers_after_subscribe(self):
        """Test: has_subscribers becomes True after subscribing."""

        def handler(event: int):
            pass

        self.event.subscribe(handler)
        self.assertTrue(self.event.has_subscribers)

    def test_has_subscribers_after_unsubscribe(self):
        """Test: has_subscribers becomes False after unsubscribing all."""

        def handler(event: int):
            pass

        unsubscribe = self.event.subscribe(handler)
        unsubscribe()
        self.assertFalse(self.event.has_subscribers)

    def test_subscribe_multiple_times_same_handler(self):
        """Test: a handler can subscribe multiple times and all instances get called."""
        call_count = 0

        def handler(event: int):
            nonlocal call_count
            call_count += 1

        self.event.subscribe(handler)
        self.event.subscribe(handler)
        self.event.publish(42)

        self.assertEqual(call_count, 2)

    def test_unsubscribe_multiple_times_same_handler(self):
        """Test: multiple subs of the same handler can be individually unsub'd."""
        call_count = 0

        def handler(event: int):
            nonlocal call_count
            call_count += 1

        unsubscribe1 = self.event.subscribe(handler)
        unsubscribe2 = self.event.subscribe(handler)

        # Unsubscribe the first one
        unsubscribe1()
        self.event.publish(42)

        self.assertEqual(call_count, 1)

        # Unsubscribe the second one
        unsubscribe2()
        self.event.publish(42)

        self.assertEqual(call_count, 1)  # Should not be incremented again

    def test_publish_with_no_subscribers(self):
        """Test: publishing with no subscribers does nothing."""
        try:
            self.event.publish(42)
        except Exception as e:  # noqa: BLE001
            self.fail(f"publish raised an exception: {e}")


class TestSubscriptions(unittest.TestCase):
    def setUp(self):
        """Set up a new Event instance for each test."""
        self.event = Event[int]()

    def test_no_subs(self):
        """Test: no subscribing happened."""
        try:
            with subscriptions() as _sub:
                pass
        except Exception as e:  # noqa: BLE001
            self.fail(f"subscriptions raised an exception: {e}")

    def test_one_sub(self):
        """Test: one subscriber."""
        acc = 0

        def handler(event: int):
            nonlocal acc
            acc += event

        # Events values published during the context will accumulate into `acc`,
        # but not outside of the context.

        self.event.publish(3)

        with subscriptions() as sub:
            sub.subscribe(self.event, handler)
            self.assertTrue(self.event.has_subscribers)
            self.event.publish(7)
            self.event.publish(11)

        self.event.publish(13)

        self.assertEqual(acc, 18)  # 7 + 11
        self.assertFalse(self.event.has_subscribers)

    def test_multiple_sub(self):
        """Test: multiple subscribers."""
        acc = 0

        def handler1(event: int):
            nonlocal acc
            acc += event

        def handler2(event: int):
            nonlocal acc
            acc += event

        self.event.publish(3)

        with subscriptions() as sub:
            sub.subscribe(self.event, handler1)
            sub.subscribe(self.event, handler2)
            self.assertTrue(self.event.has_subscribers)
            self.event.publish(7)
            self.event.publish(11)

        self.event.publish(13)

        self.assertEqual(acc, 36)  # 2 * (7 + 11)
        self.assertFalse(self.event.has_subscribers)

    def test_before_sub(self):
        """Test: subscribers from before the context are untouched."""

        acc1, acc2 = 0, 0

        def handler_before(event: int):
            nonlocal acc1
            acc1 += event

        def handler_context(event: int):
            nonlocal acc2
            acc2 += event

        self.event.subscribe(handler_before)
        self.event.publish(3)

        with subscriptions() as sub:
            sub.subscribe(self.event, handler_context)
            self.event.publish(7)
            self.assertEqual(len(self.event._subscribers), 2)

        self.event.publish(13)

        self.assertEqual(acc1, 23)  # 3 + 7 + 13
        self.assertEqual(acc2, 7)  # 7
        self.assertEqual(len(self.event._subscribers), 1)

    def test_exception_in_context(self):
        """Test: subscribers are unsub'd even if an exception was thrown."""

        def handler(event: int):
            pass

        with self.assertRaises(Exception):
            with subscriptions() as sub:
                sub.subscribe(self.event, handler)
                raise Exception("ruh roh")

        self.assertFalse(self.event.has_subscribers)
