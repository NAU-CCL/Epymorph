import numpy as np
import pandas as pd
import pytest

from epymorph.adrio.processing import (
    ConstantFill,
    ConstantFix,
    DataPipeline,
    DontFill,
    DontFix,
    FunctionFill,
    FunctionFix,
    PipelineResult,
    PivotAxis,
    RandomFill,
    RandomFix,
)


class _Randomness:
    def __init__(self, seed=42):
        self._rng = np.random.default_rng(seed)

    @property
    def rng(self):
        return self._rng


@pytest.fixture
def rng():
    return _Randomness()


#######
# Fix #
#######


def test_constant_fix(rng):
    fix = ConstantFix(with_value=np.int64(9))
    in_df = pd.DataFrame(
        {"a": [1, 2, 1], "b": [1, 4, 2], "c": [0, 1, 3]},
        dtype=np.int64,
    )
    out_df = fix(
        rng,
        replace=np.int64(1),  # replace 1s
        columns=("a", "c"),
        data_df=in_df,
    )
    # original DF remains unchanged
    assert (in_df["a"] == [1, 2, 1]).all()
    assert (in_df["b"] == [1, 4, 2]).all()
    assert (in_df["c"] == [0, 1, 3]).all()
    # new DF has replaced values
    assert (out_df["a"] == [9, 2, 9]).all()
    assert (out_df["b"] == [1, 4, 2]).all()
    assert (out_df["c"] == [0, 9, 3]).all()


def test_function_fix(rng):
    value = 6

    def gen():
        nonlocal value
        value += 1
        return np.int64(value)

    fix = FunctionFix(with_function=gen)
    in_df = pd.DataFrame({"x": [0, 1, 0], "y": [1, 1, 1]})
    out_df = fix(
        rng,
        replace=np.int64(0),  # replace 0s
        data_df=in_df,
        columns=("x",),
    )
    assert (out_df["x"] == [7, 1, 8]).all()
    assert (out_df["y"] == [1, 1, 1]).all()


def test_random_fix():
    fix = RandomFix.from_range(10, 20)
    in_df = pd.DataFrame({"v": [2, 2, 3], "w": [5, 6, 2]})
    out1_df = fix(
        rng=_Randomness(),
        replace=np.int64(2),  # replace 2s
        data_df=in_df,
        columns=("v",),
    )
    assert 10 <= out1_df["v"][0] <= 20
    assert out1_df["v"][2] == 3
    assert 10 <= out1_df["v"][1] <= 20
    assert (out1_df["w"] == [5, 6, 2]).all()

    # with the same random seed, results should be reproducible
    out2_df = fix(
        rng=_Randomness(),
        replace=np.int64(2),
        data_df=in_df,
        columns=("v",),
    )
    np.testing.assert_array_equal(out1_df["v"], out2_df["v"])
    np.testing.assert_array_equal(out1_df["w"], out2_df["w"])


def test_dont_fix(rng):
    fix = DontFix()
    in_df = pd.DataFrame({"a": [1, 2], "b": [3, 1]})
    out_df = fix(
        rng,
        replace=np.int64(1),
        data_df=in_df,
        columns=("a", "b"),
    )
    assert out_df is in_df


########
# Fill #
########


def test_constant_fill(rng):
    fill = ConstantFill(with_value=np.int64(0))
    in_np = np.array([1, 2, 3], dtype=np.int64)
    missing = np.array([True, False, True])
    out_np, out_mask = fill(rng, data_np=in_np, missing_mask=missing)
    np.testing.assert_array_equal(
        out_np,
        np.array([0, 2, 0], dtype=np.int64),
    )
    assert out_mask is None


def test_function_fill(rng):
    value = 6

    def gen():
        nonlocal value
        value += 1
        return np.int64(value)

    fill = FunctionFill(with_function=gen)

    in_np = np.array([1, 2, 3], dtype=np.int64)
    missing = np.array([True, False, True])
    out_np, out_mask = fill(rng, data_np=in_np, missing_mask=missing)
    np.testing.assert_array_equal(
        out_np,
        np.array([7, 2, 8], dtype=np.int64),
    )
    assert out_mask is None


def test_random_fill():
    fill = RandomFill.from_range_float(1, 2)

    in_np = np.array([33.3, 66.6, 99.9], dtype=np.float64)
    missing = np.array([False, True, False])
    out1_np, out1_mask = fill(
        rng=_Randomness(),
        data_np=in_np,
        missing_mask=missing,
    )
    np.testing.assert_array_almost_equal(
        out1_np[[0, 2]],
        np.array([33.3, 99.9], dtype=np.float64),
    )
    assert 1.0 <= out1_np[1] < 2.0

    # with the same random seed, results should be reproducible
    out2_np, out2_mask = fill(
        rng=_Randomness(),
        data_np=in_np,
        missing_mask=missing,
    )
    np.testing.assert_array_almost_equal(out1_np, out2_np)
    assert out1_mask is None
    assert out2_mask is None


def test_dont_fill(rng):
    fill = DontFill()
    in_np = np.array([1, 2, 3], dtype=np.int64)
    missing = np.array([False, True, False])
    out_np, out_mask = fill(rng, data_np=in_np, missing_mask=missing)
    assert out_np is in_np
    assert out_mask is missing


def test_dont_fill_none_missing(rng):
    fill = DontFill()
    in_np = np.array([1, 2, 3], dtype=np.int64)
    missing = np.broadcast_to(np.ma.nomask, in_np.shape)
    out_np, out_mask = fill(rng, data_np=in_np, missing_mask=missing)
    assert out_np is in_np
    assert out_mask is missing


##################
# PipelineResult #
##################


def test_pipeline_init_error():
    with pytest.raises(ValueError, match="should not be masked"):
        PipelineResult(
            np.ma.masked_array(data=[1, 2, 3], mask=[1, 0, 0]),
            issues={},
        )


def test_pipeline_value_as_masked():
    data = np.array([1, 2, 3])
    res1 = PipelineResult(
        value=data,
        issues={},
    )
    assert not np.ma.is_masked(res1.value_as_masked)

    res2 = PipelineResult(
        value=data,
        issues={
            "x": np.array([1, 0, 0], dtype=np.bool_),
            "y": np.array([0, 0, 1], dtype=np.bool_),
        },
    )
    val2 = res2.value_as_masked
    assert np.ma.is_masked(val2)
    np.testing.assert_array_equal(
        np.ma.getmask(val2),
        np.array([1, 0, 1], dtype=np.bool_),
    )


def test_pipeline_result_sum():
    a = PipelineResult(
        value=np.arange(3 * 3, dtype=np.int64).reshape((3, 3)),
        issues={
            "x": np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.bool_),
            "y": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.bool_),
        },
    )
    b = PipelineResult(
        value=np.arange(3 * 3, dtype=np.int64).reshape((3, 3)) * 10,
        issues={
            "x": np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=np.bool_),
            "y": np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=np.bool_),
        },
    )
    c = PipelineResult.sum(a, b, left_prefix="a_", right_prefix="b_")

    expected_value = np.ma.masked_array(
        data=np.array([[0, 11, 22], [33, 44, 55], [66, 77, 88]], dtype=np.int64),
        mask=np.array([[0, 0, 0], [1, 0, 1], [1, 0, 1]], dtype=np.bool_),
    )

    np.testing.assert_array_equal(c.value_as_masked, expected_value)
    np.testing.assert_array_equal(c.issues["a_x"], a.issues["x"])
    np.testing.assert_array_equal(c.issues["a_y"], a.issues["y"])
    np.testing.assert_array_equal(c.issues["b_x"], b.issues["x"])
    np.testing.assert_array_equal(c.issues["b_y"], b.issues["y"])


################
# DataPipeline #
################


def test_data_pipeline_simple():
    pipeline = DataPipeline(
        axes=(PivotAxis("x", [1, 2, 3]), PivotAxis("y", [4, 5, 6])),
        ndims=2,
        dtype=np.int64,
        rng=_Randomness(),
    ).finalize(DontFill())

    out = pipeline(
        pd.DataFrame(
            {
                "x": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "y": [4, 5, 6, 4, 5, 6, 4, 5, 6],
                "value": [11, 22, 33, 44, 55, 66, 77, 88, 99],
            }
        )
    )

    assert len(out.issues) == 0
    np.testing.assert_array_equal(
        out.value_as_masked,
        np.array(
            [
                [11, 22, 33],
                [44, 55, 66],
                [77, 88, 99],
            ],
            dtype=np.int64,
        ),
    )


def test_data_pipeline_some_missing():
    pipeline = DataPipeline(
        axes=(PivotAxis("x", [1, 2, 3]), PivotAxis("y", [4, 5, 6])),
        ndims=2,
        dtype=np.int64,
        rng=_Randomness(),
    ).finalize(DontFill())

    out = pipeline(
        pd.DataFrame(
            {
                "x": [1, 1, 2, 2, 2, 3, 3],
                "y": [4, 5, 4, 5, 6, 4, 6],
                "value": [11, 22, 44, 55, 66, 77, 99],
            },
            dtype=np.int64,
        )
    )

    assert "missing" in out.issues
    np.testing.assert_array_equal(
        out.value_as_masked,
        np.ma.masked_array(
            data=np.array(
                [
                    [11, 22, -1],
                    [44, 55, 66],
                    [77, -1, 99],
                ],
                dtype=np.int64,
            ),
            mask=np.array(
                [
                    [0, 0, 1],
                    [0, 0, 0],
                    [0, 1, 0],
                ],
                dtype=np.bool_,
            ),
        ),
    )


def test_data_pipeline_map_series():
    pipeline = (
        DataPipeline(
            axes=(PivotAxis("x", [1, 2, 3]), PivotAxis("y", [4, 5, 6])),
            ndims=2,
            dtype=np.int64,
            rng=_Randomness(),
        )
        .map_series("value", lambda xs: xs // 2)
        .finalize(ConstantFill(np.int64(-1)))
    )

    out = pipeline(
        pd.DataFrame(
            {
                "x": [1, 1, 1, 2, 2, 3, 3, 3],
                "y": [4, 5, 6, 5, 6, 4, 5, 6],
                "value": [11, 22, 33, 55, 66, 77, 88, 99],
            },
            dtype=np.int64,
        )
    )

    assert len(out.issues) == 0
    np.testing.assert_array_equal(
        out.value_as_masked,
        np.array(
            [
                [5, 11, 16],
                [-1, 27, 33],
                [38, 44, 49],
            ],
            dtype=np.int64,
        ),
    )


def test_data_pipeline_map_column():
    pipeline = (
        DataPipeline(
            axes=(PivotAxis("x", [1, 2, 3]), PivotAxis("y", [4, 5, 6])),
            ndims=2,
            dtype=np.int64,
            rng=_Randomness(),
        )
        .map_column("value", lambda x: x // 2)
        .finalize(ConstantFill(np.int64(-1)))
    )

    out = pipeline(
        pd.DataFrame(
            {
                "x": [1, 1, 1, 2, 2, 3, 3, 3],
                "y": [4, 5, 6, 5, 6, 4, 5, 6],
                "value": [11, 22, 33, 55, 66, 77, 88, 99],
            },
            dtype=np.int64,
        )
    )

    assert len(out.issues) == 0
    np.testing.assert_array_equal(
        out.value_as_masked,
        np.array(
            [
                [5, 11, 16],
                [-1, 27, 33],
                [38, 44, 49],
            ],
            dtype=np.int64,
        ),
    )


def test_data_pipeline_strip_sentinel():
    pipeline = (
        DataPipeline(
            axes=(PivotAxis("x", [1, 2, 3]), PivotAxis("y", [4, 5, 6])),
            ndims=2,
            dtype=np.int64,
            rng=_Randomness(),
        )
        .strip_sentinel("foobar", np.int64(99), ConstantFix(np.int64(-2)))
        .finalize(ConstantFill(np.int64(-1)))
    )

    out = pipeline(
        pd.DataFrame(
            {
                "x": [1, 1, 1, 2, 2, 3, 3, 3],
                "y": [4, 5, 6, 5, 6, 4, 5, 6],
                "value": [11, 22, 99, 55, 66, 77, 88, 99],
            },
            dtype=np.int64,
        )
    )

    assert len(out.issues) == 0
    np.testing.assert_array_equal(
        out.value_as_masked,
        np.array(
            [
                [11, 22, -2],
                [-1, 55, 66],
                [77, 88, -2],
            ],
            dtype=np.int64,
        ),
    )


def test_data_pipeline_strip_sentinel_dont_fix():
    pipeline = (
        DataPipeline(
            axes=(PivotAxis("x", [1, 2, 3]), PivotAxis("y", [4, 5, 6])),
            ndims=2,
            dtype=np.int64,
            rng=_Randomness(),
        )
        .strip_sentinel("foobar", np.int64(99), DontFix())
        .finalize(DontFill())
    )

    out = pipeline(
        pd.DataFrame(
            {
                "x": [1, 1, 1, 2, 2, 3, 3, 3],
                "y": [4, 5, 6, 5, 6, 4, 5, 6],
                "value": [11, 22, 99, 55, 66, 77, 88, 99],
            },
            dtype=np.int64,
        )
    )

    np.testing.assert_array_equal(
        out.value_as_masked,
        np.ma.masked_array(
            data=np.array(
                [
                    [11, 22, -9],
                    [-9, 55, 66],
                    [77, 88, -9],
                ],
                dtype=np.int64,
            ),
            mask=np.array(
                [
                    [0, 0, 1],
                    [1, 0, 0],
                    [0, 0, 1],
                ],
                dtype=np.bool_,
            ),
        ),
    )
    assert len(out.issues) == 2
    np.testing.assert_array_equal(
        out.issues["foobar"],
        np.array(
            [
                [0, 0, 1],
                [0, 0, 0],
                [0, 0, 1],
            ],
            dtype=np.bool_,
        ),
    )
    np.testing.assert_array_equal(
        out.issues["missing"],
        np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 0, 0],
            ],
            dtype=np.bool_,
        ),
    )


def test_data_pipeline_strip_na_as_sentinel():
    pipeline = (
        DataPipeline(
            axes=(PivotAxis("x", [1, 2, 3]), PivotAxis("y", [4, 5, 6])),
            ndims=2,
            dtype=np.int64,
            rng=_Randomness(),
        )
        .strip_na_as_sentinel(
            column="value",
            sentinel_name="foobar",
            sentinel_value=np.int64(-9),
            fix=ConstantFix(np.int64(1)),
        )
        .finalize(ConstantFill(np.int64(-1)))
    )

    out = pipeline(
        pd.DataFrame(
            {
                "x": [1, 1, 1, 2, 2, 3, 3, 3],
                "y": [4, 5, 6, 5, 6, 4, 5, 6],
                "value": pd.Series(
                    [11, 22, pd.NA, 55, 66, pd.NA, 88, 99],
                    dtype="Int64",
                ),
            },
        )
    )

    assert len(out.issues) == 0
    np.testing.assert_array_equal(
        out.value_as_masked,
        np.array(
            [
                [11, 22, 1],
                [-1, 55, 66],
                [1, 88, 99],
            ],
            dtype=np.int64,
        ),
    )


def test_data_pipeline_strip_na_as_sentinel_dont_fix():
    pipeline = (
        DataPipeline(
            axes=(PivotAxis("x", [1, 2, 3]), PivotAxis("y", [4, 5, 6])),
            ndims=2,
            dtype=np.int64,
            rng=_Randomness(),
        )
        .strip_na_as_sentinel(
            column="value",
            sentinel_name="foobar",
            sentinel_value=np.int64(-9),
            fix=DontFix(),
        )
        .finalize(DontFill())
    )

    out = pipeline(
        pd.DataFrame(
            {
                "x": [1, 1, 1, 2, 2, 3, 3, 3],
                "y": [4, 5, 6, 5, 6, 4, 5, 6],
                "value": pd.Series(
                    [11, 22, pd.NA, 55, 66, pd.NA, 88, 99],
                    dtype="Int64",
                ),
            },
        )
    )

    np.testing.assert_array_equal(
        out.value_as_masked,
        np.ma.masked_array(
            data=np.array(
                [
                    [11, 22, -9],
                    [-9, 55, 66],
                    [-9, 88, 99],
                ],
                dtype=np.int64,
            ),
            mask=np.array(
                [
                    [0, 0, 1],
                    [1, 0, 0],
                    [1, 0, 0],
                ],
                dtype=np.bool_,
            ),
        ),
    )
    assert len(out.issues) == 2
    np.testing.assert_array_equal(
        out.issues["foobar"],
        np.array(
            [
                [0, 0, 1],
                [0, 0, 0],
                [1, 0, 0],
            ],
            dtype=np.bool_,
        ),
    )
    np.testing.assert_array_equal(
        out.issues["missing"],
        np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 0, 0],
            ],
            dtype=np.bool_,
        ),
    )
