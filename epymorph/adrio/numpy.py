from os import PathLike
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.adrio.adrio import ADRIO
from epymorph.error import DataResourceError

_SliceLike = slice | type(Ellipsis)
_ArraySlice = _SliceLike | tuple[_SliceLike, ...]


class NPY(ADRIO[Any]):
    """Retrieves an array of data from a user-provided .npy file."""

    file_path: PathLike
    """The path to the .npy file containing data."""
    array_slice: _ArraySlice | None
    """Optional slice(s) of the array to load."""

    def __init__(
        self, file_path: PathLike, array_slice: _ArraySlice | None = None
    ) -> None:
        """
        Initializes the array of user-provided .npy file of data with a file path and
        optional array slice(s).

        Parameters
        ----------
        file_path : PathLike
            The path to the .npy file to retrieve the array data from.
        array_slice : _ArraySlice | None
            The optional slice or slices to apply to the array. If None, the entire
            array is returned (default is None).
        """
        if Path(file_path).suffix != ".npy":
            msg = (
                "Incorrect file type. Only .npy files can be loaded through NPY ADRIOs."
            )
            raise DataResourceError(msg)
        self.file_path = file_path
        self.array_slice = array_slice

    @override
    def evaluate_adrio(self) -> NDArray:
        try:
            data = cast(NDArray, np.load(self.file_path))
            if self.array_slice is not None:
                data = data[self.array_slice]
            return data
        except OSError as e:
            msg = "File not found."
            raise DataResourceError(msg) from e
        except ValueError as e:
            msg = "Object arrays cannot be loaded."
            raise DataResourceError(msg) from e
        except IndexError as e:
            msg = "Specified array slice is invalid for the shape of this data."
            raise DataResourceError(msg) from e


class NPZ(ADRIO[Any]):
    """Retrieves an array of data from a user-defined .npz file."""

    file_path: PathLike
    """The path to the .npz file containing data."""
    array_name: str
    """The name of the array in the .npz file to load."""
    array_slice: _ArraySlice | None
    """Optional slice(s) of the array to load."""

    def __init__(
        self,
        file_path: PathLike,
        array_name: str,
        array_slice: _ArraySlice | None = None,
    ) -> None:
        """
        Initializes the array of user-provided .npz file of data with a file path, array
        name, and optional array slice(s).

        Parameters
        ----------
        file_path : PathLike
            The path to the .npy file to retrieve the array data from.
        array_name: str
            The name of the array to retrieve from the user-given file.
        array_slice : _ArraySlice | None
            The optional slice or slices to apply to the array. If None, the entire
            array is returned (default is None).
        """
        if Path(file_path).suffix != ".npz":
            msg = (
                "Incorrect file type. Only .npz files can be loaded through NPZ ADRIOs."
            )
            raise DataResourceError(msg)
        self.file_path = file_path
        self.array_name = array_name
        self.array_slice = array_slice

    @override
    def evaluate_adrio(self) -> NDArray:
        try:
            data = cast(NDArray, np.load(self.file_path)[self.array_name])
            if self.array_slice is not None:
                data = data[self.array_slice]
            return data
        except OSError as e:
            msg = "File not found."
            raise DataResourceError(msg) from e
        except ValueError as e:
            msg = "Object arrays cannot be loaded."
            raise DataResourceError(msg) from e
        except IndexError as e:
            msg = "Specified array slice is invalid for the shape of this data."
            raise DataResourceError(msg) from e
