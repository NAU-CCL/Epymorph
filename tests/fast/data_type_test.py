# ruff: noqa: PT009,PT027
import unittest
from datetime import date

import numpy as np
import pandas as pd
import pytest

from epymorph.data_type import (
    StratifiedAttributeArray,
    dtype_as_np,
    dtype_check,
    dtype_str,
)


class DataTypeTest(unittest.TestCase):
    def test_dtype_as_np(self):
        self.assertEqual(dtype_as_np(int), np.int64)
        self.assertEqual(dtype_as_np(float), np.float64)
        self.assertEqual(dtype_as_np(str), np.str_)
        self.assertEqual(dtype_as_np(date), np.dtype("datetime64[D]"))

        struct = (("foo", float), ("bar", int), ("baz", str), ("bux", date))
        self.assertEqual(
            dtype_as_np(struct),
            [
                ("foo", np.float64),
                ("bar", np.int64),
                ("baz", np.str_),
                ("bux", np.dtype("datetime64[D]")),
            ],
        )

    def test_dtype_str(self):
        self.assertEqual(dtype_str(int), "int")
        self.assertEqual(dtype_str(float), "float")
        self.assertEqual(dtype_str(str), "str")
        self.assertEqual(dtype_str(date), "date")

        struct = (("foo", float), ("bar", int), ("baz", str), ("bux", date))
        self.assertEqual(
            dtype_str(struct), "[(foo, float), (bar, int), (baz, str), (bux, date)]"
        )

    def test_dtype_invalid(self):
        with self.assertRaises(ValueError):
            dtype_as_np([int, float, str])  # type: ignore
        with self.assertRaises(ValueError):
            dtype_as_np([])  # type: ignore
        with self.assertRaises(ValueError):
            dtype_as_np(tuple())  # type: ignore
        with self.assertRaises(ValueError):
            dtype_as_np(("foo", "bar", "baz"))  # type: ignore

    def test_dtype_check(self):
        self.assertTrue(dtype_check(int, 1))
        self.assertTrue(dtype_check(int, -32))
        self.assertTrue(dtype_check(float, -0.1))
        self.assertTrue(dtype_check(float, 191827312.231234))
        self.assertTrue(dtype_check(str, "hi"))
        self.assertTrue(dtype_check(str, ""))
        self.assertTrue(dtype_check(date, date(2024, 1, 1)))
        self.assertTrue(dtype_check(date, date(1066, 10, 14)))
        self.assertTrue(dtype_check((("x", int), ("y", int)), (1, 2)))
        self.assertTrue(dtype_check((("a", str), ("b", float)), ("hi", 9273.3)))

        self.assertFalse(dtype_check(int, "hi"))
        self.assertFalse(dtype_check(int, 42.42))
        self.assertFalse(dtype_check(int, (1, 2, 3)))

        self.assertFalse(dtype_check(float, "hi"))
        self.assertFalse(dtype_check(float, 1))
        self.assertFalse(dtype_check(float, 8273))
        self.assertFalse(dtype_check(float, (32.0, 12.7, 99.9)))

        self.assertFalse(dtype_check(date, "2024-01-01"))
        self.assertFalse(dtype_check(date, 123))

        dt1 = (("x", int), ("y", int))
        self.assertFalse(dtype_check(dt1, 1))
        self.assertFalse(dtype_check(dt1, 78923.1))
        self.assertFalse(dtype_check(dt1, "hi"))
        self.assertFalse(dtype_check(dt1, ()))
        self.assertFalse(dtype_check(dt1, (1, 237.8)))
        self.assertFalse(dtype_check(dt1, (1, 2, 3)))


############################
# StratifiedAttributeArray #
############################


def test_stratified_num_strata():
    value = StratifiedAttributeArray(
        values=np.zeros((2, 3)),
        strata=None,
    )
    assert value.num_strata == 2

    value = StratifiedAttributeArray(
        values=np.zeros((2, 3)),
        strata=["a", "b"],
    )
    assert value.num_strata == 2


def test_stratified_invalid_values():
    with pytest.raises(ValueError, match="Invalid values"):
        StratifiedAttributeArray(np.array(1))


def test_stratified_invalid_strata():
    with pytest.raises(ValueError, match="Invalid strata"):
        StratifiedAttributeArray(np.zeros((2, 3)), strata=["a"])


def test_stratified_from_dict():
    values = {
        "bbb": np.array([1, 2]),
        "aaa": np.array([3, 4]),
    }

    array = StratifiedAttributeArray.from_dict(values)

    assert array.strata == ["bbb", "aaa"]
    assert array.num_strata == 2
    np.testing.assert_array_equal(array.values, [[1, 2], [3, 4]])

    broadcasted = StratifiedAttributeArray.from_dict(
        {
            "aaa": np.array([[1], [2]]),
            "bbb": np.array([[3, 4, 5]]),
        }
    )
    assert broadcasted.values.shape == (2, 2, 3)  # noqa: PD011 (false positive)
    np.testing.assert_array_equal(broadcasted.values[0], [[1, 1, 1], [2, 2, 2]])  # noqa: PD011 (false positive)
    np.testing.assert_array_equal(broadcasted.values[1], [[3, 4, 5], [3, 4, 5]])  # noqa: PD011 (false positive)


def test_stratified_from_dict_invalid():
    with pytest.raises(ValueError, match="must not be empty"):
        StratifiedAttributeArray.from_dict({})

    with pytest.raises(ValueError, match="broadcast together"):
        StratifiedAttributeArray.from_dict(
            {
                "aaa": np.array([1, 2]),
                "bbb": np.array([3, 4, 5]),
            }
        )

    with pytest.raises(ValueError, match="must have equivalent types"):
        StratifiedAttributeArray.from_dict(
            {
                "aaa": np.array(["1", "2"], dtype=np.str_),
                "bbb": np.array([3, 4], dtype=np.float64),
            }
        )


def test_stratified_from_dataframe_wide():
    df = pd.DataFrame(
        {
            "bbb": [10, 20],
            "aaa": [1, 2],
        }
    )

    array = StratifiedAttributeArray.from_dataframe(df)

    assert array.strata == ["bbb", "aaa"]
    np.testing.assert_array_equal(array.values, [[10, 20], [1, 2]])


def test_stratified_from_dataframe_long():
    df = pd.DataFrame(
        {
            "strata": ["bbb", "aaa", "aaa", "bbb"],
            "value": [20, 1, 2, 10],
        },
        index=pd.DatetimeIndex(
            ["2026-01-08", "2026-01-01", "2026-01-08", "2026-01-01"]
        ),
    )

    array = StratifiedAttributeArray.from_dataframe(
        df,
        columns="strata",
        values="value",
    )

    assert array.strata == ["aaa", "bbb"]
    np.testing.assert_array_equal(array.values, [[1, 2], [10, 20]])


def test_stratified_from_dataframe_long_with_index():
    df = pd.DataFrame(
        {
            "strata": ["aaa", "bbb", "aaa", "bbb"],
            "value": [1, 10, 2, 20],
            "obs": ["obs1", "obs1", "obs2", "obs2"],
        }
    )

    array = StratifiedAttributeArray.from_dataframe(
        df,
        columns="strata",
        values="value",
        index="obs",
    )

    assert array.strata == ["aaa", "bbb"]
    np.testing.assert_array_equal(array.values, [[1, 2], [10, 20]])


def test_stratified_from_dataframe_invalid_long():
    df = pd.DataFrame(
        {
            "strata": ["aaa", "aaa", "bbb"],
            "value": [1, 2, 20],
        }
    )

    with pytest.raises(
        ValueError,
        match="Both `columns` and `values` must be provided",
    ):
        StratifiedAttributeArray.from_dataframe(df, columns="strata")

    with pytest.raises(ValueError, match="columns must exist"):
        StratifiedAttributeArray.from_dataframe(df, columns="foo", values="value")

    with pytest.raises(ValueError, match="columns must be distinct"):
        StratifiedAttributeArray.from_dataframe(df, columns="value", values="value")

    with pytest.raises(ValueError, match="columns must exist"):
        StratifiedAttributeArray.from_dataframe(
            df,
            columns="strata",
            values="value",
            index="missing",
        )

    duplicate_columns = pd.DataFrame(
        [["aaa", 1, 2]],
        columns=["strata", "value", "value"],
    )
    with pytest.raises(ValueError, match="unique column names"):
        StratifiedAttributeArray.from_dataframe(
            duplicate_columns,
            columns="strata",
            values="value",
        )


def test_stratified_from_dataframe_long_rejects_missing_data():
    incomplete = pd.DataFrame(
        {
            "obs": [0, 0, 1],
            "strata": ["aaa", "bbb", "aaa"],
            "value": [1, 10, 2],
        }
    )
    with pytest.raises(ValueError, match="contains missing values"):
        StratifiedAttributeArray.from_dataframe(
            incomplete,
            columns="strata",
            values="value",
            index="obs",
        )

    missing_value = pd.DataFrame(
        {
            "obs": [0, 0],
            "strata": ["aaa", "bbb"],
            "value": [1, np.nan],
        }
    )
    with pytest.raises(ValueError, match="contains missing values"):
        StratifiedAttributeArray.from_dataframe(
            missing_value,
            columns="strata",
            values="value",
            index="obs",
        )


def test_stratified_values_for_by_index():
    values = np.array([[1, 2], [3, 4]])
    array = StratifiedAttributeArray(values)

    np.testing.assert_array_equal(array.values_for(0), [1, 2])
    np.testing.assert_array_equal(array.values_for(1), [3, 4])


def test_stratified_values_for_by_name():
    values = np.array([[1, 2], [3, 4]])
    array = StratifiedAttributeArray(values, strata=["aaa", "bbb"])

    np.testing.assert_array_equal(array.values_for("aaa"), [1, 2])
    np.testing.assert_array_equal(array.values_for("bbb"), [3, 4])


def test_stratified_values_for_invalid_stratum():
    unnamed = StratifiedAttributeArray(np.array([[1, 2]]))
    with pytest.raises(ValueError, match="not specified by name"):
        unnamed.values_for("aaa")

    named = StratifiedAttributeArray(np.array([[1, 2]]), strata=["aaa"])
    with pytest.raises(ValueError, match="not found"):
        named.values_for("missing")
    with pytest.raises(ValueError, match="out of bounds"):
        named.values_for(1)
