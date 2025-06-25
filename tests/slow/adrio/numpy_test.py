from pathlib import Path
from typing import Callable, TypeVar
from uuid import uuid4

import numpy as np
import pytest
from numpy.typing import NDArray

from epymorph.adrio import numpy
from epymorph.adrio.adrio import ADRIOContextError, ADRIOProcessingError
from epymorph.data_shape import Shapes
from epymorph.geography.custom import CustomScope
from epymorph.time import TimeFrame

NpyType = TypeVar("NpyType", bound=NDArray)
NpzType = TypeVar("NpzType", bound=dict[str, NDArray])


@pytest.fixture
def npy(tmp_path: Path) -> Callable[[NpyType], tuple[NpyType, Path]]:
    def prepare_npy(data: NpyType) -> tuple[NpyType, Path]:
        path = tmp_path / f"{uuid4()}.npy"
        np.save(path, data)
        return data, path

    return prepare_npy


@pytest.fixture
def npz(tmp_path: Path) -> Callable[[NpzType], tuple[NpzType, Path]]:
    def prepare_npz(data: NpzType) -> tuple[NpzType, Path]:
        path = tmp_path / f"{uuid4()}.npz"
        np.savez(path, **data)
        return data, path

    return prepare_npz


def test_npy_01(npy):
    # Basic npy case
    expected, path = npy(np.array([1, 2, 3], dtype=np.int64))

    actual = (
        numpy.NumpyFile(file_path=path, shape=Shapes.N, dtype=np.int64)
        .with_context(
            scope=CustomScope(["A", "B", "C"]),
        )
        .evaluate()
    )

    np.testing.assert_array_equal(actual, expected)


def test_npy_02(npy):
    # Error: missing scope
    _, path = npy(np.array([1, 2, 3], dtype=np.int64))
    with pytest.raises(ADRIOContextError):
        numpy.NumpyFile(
            file_path=path,
            shape=Shapes.N,
            dtype=np.int64,
        ).with_context().evaluate()


def test_npy_03(npy):
    # Error: data is bad shape
    _, path = npy(np.array([1, 2, 3, 4], dtype=np.int64))
    with pytest.raises(ADRIOProcessingError, match="invalid shape"):
        numpy.NumpyFile(
            file_path=path,
            shape=Shapes.N,
            dtype=np.int64,
        ).with_context(
            scope=CustomScope(["A", "B", "C"]),
        ).evaluate()


def test_npy_04(npy):
    # Fix above error by slicing
    data, path = npy(np.array([1, 2, 3, 4], dtype=np.int64))
    expected = data[1:4]

    actual = (
        numpy.NumpyFile(
            file_path=path,
            shape=Shapes.N,
            dtype=np.int64,
            array_slice=slice(1, 4),
        )
        .with_context(
            scope=CustomScope(["A", "B", "C"]),
        )
        .evaluate()
    )

    np.testing.assert_array_equal(actual, expected)


def test_npy_05(npy):
    # Error: data is bad type
    _, path = npy(np.array([1, 2, 3], dtype=np.float64))
    with pytest.raises(ADRIOProcessingError, match="expected data type"):
        numpy.NumpyFile(
            file_path=path,
            shape=Shapes.N,
            dtype=np.int64,
        ).with_context(
            scope=CustomScope(["A", "B", "C"]),
        ).evaluate()


def test_npy_06(npy):
    # Test "select all" slice
    expected, path = npy(np.array([1, 2, 3], dtype=np.int64))

    actual = (
        numpy.NumpyFile(
            file_path=path,
            shape=Shapes.N,
            dtype=np.int64,
            array_slice=slice(None),
        )
        .with_context(
            scope=CustomScope(["A", "B", "C"]),
        )
        .evaluate()
    )

    np.testing.assert_array_equal(actual, expected)


def test_npy_07(tmp_path):
    # For an npy file, speciying array_name is not allowed
    with pytest.raises(ValueError, match="do not specify an `array_name`"):
        numpy.NumpyFile(
            file_path=tmp_path / "foo.npy",
            shape=Shapes.N,
            dtype=np.int64,
            array_name="foo",
        )


def test_npy_08(tmp_path):
    # Error: load a non-existent file
    with pytest.raises(ADRIOProcessingError, match="Error loading file"):
        numpy.NumpyFile(
            file_path=tmp_path / "not-a-file.npy",
            shape=Shapes.N,
            dtype=np.int64,
        ).with_context(
            scope=CustomScope(["A", "B", "C"]),
        ).evaluate()


def test_npy_09(tmp_path):
    # Error: load not an npy file
    path = tmp_path / "not-actually-npy-file.npy"
    path.write_text("hello")
    with pytest.raises(ADRIOProcessingError, match="Error loading file"):
        numpy.NumpyFile(
            file_path=path,
            shape=Shapes.N,
            dtype=np.int64,
        ).with_context(
            scope=CustomScope(["A", "B", "C"]),
        ).evaluate()


def test_npy_10(npy):
    # Error: bad slice
    expected, path = npy(np.array([1, 2, 3], dtype=np.int64))
    with pytest.raises(ADRIOProcessingError, match="array slice is invalid"):
        numpy.NumpyFile(
            file_path=path,
            shape=Shapes.N,
            dtype=np.int64,
            array_slice=np.s_[0:3, 0:3],
        ).with_context(
            scope=CustomScope(["A", "B", "C"]),
        ).evaluate()


def test_npy_11(npy):
    # Test TxN data
    expected, path = npy(np.arange((7 * 3), dtype=np.int64).reshape((7, 3)))

    actual = (
        numpy.NumpyFile(
            file_path=path,
            shape=Shapes.TxN,
            dtype=np.int64,
        )
        .with_context(
            scope=CustomScope(["A", "B", "C"]),
            time_frame=TimeFrame.range("2020-01-01", "2020-01-07"),
        )
        .evaluate()
    )

    np.testing.assert_array_equal(actual, expected)


def test_npy_12(npy):
    # Test AxN data
    expected1, path1 = npy(np.arange((12 * 3), dtype=np.int64).reshape((12, 3)))

    actual1 = (
        numpy.NumpyFile(file_path=path1, shape=Shapes.AxN, dtype=np.int64)
        .with_context(scope=CustomScope(["A", "B", "C"]))
        .evaluate()
    )

    np.testing.assert_array_equal(actual1, expected1)

    expected2, path2 = npy(np.arange((2 * 3), dtype=np.int64).reshape((2, 3)))

    actual2 = (
        numpy.NumpyFile(file_path=path2, shape=Shapes.AxN, dtype=np.int64)
        .with_context(scope=CustomScope(["A", "B", "C"]))
        .evaluate()
    )

    np.testing.assert_array_equal(actual2, expected2)


def test_npz_01(npz):
    # Basic npz case
    data, path = npz(
        {
            "foo": np.array([1, 2, 3], dtype=np.int64),
            "bar": np.array([4, 5, 6], dtype=np.float64),
        }
    )
    expected = data["foo"]

    actual = (
        numpy.NumpyFile(
            file_path=path,
            shape=Shapes.N,
            dtype=np.int64,
            array_name="foo",
        )
        .with_context(
            scope=CustomScope(["A", "B", "C"]),
        )
        .evaluate()
    )

    np.testing.assert_array_equal(actual, expected)


def test_npz_02(tmp_path):
    # Error: array_name not specified
    with pytest.raises(ValueError, match="specify the `array_name`"):
        numpy.NumpyFile(
            file_path=tmp_path / "foo.npz",
            shape=Shapes.N,
            dtype=np.int64,
        )


def test_npz_03(npz):
    # npz with array slice
    data, path = npz(
        {
            "foo": np.array([1, 2, 3], dtype=np.int64),
            "bar": np.array([4, 5, 6, 7, 8, 9], dtype=np.float64),
        }
    )
    expected = data["bar"][3:]

    actual = (
        numpy.NumpyFile(
            file_path=path,
            shape=Shapes.N,
            dtype=np.float64,
            array_name="bar",
            array_slice=np.s_[3:],
        )
        .with_context(
            scope=CustomScope(["A", "B", "C"]),
        )
        .evaluate()
    )

    np.testing.assert_array_equal(actual, expected)
