from os import PathLike
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.adrio.adrio import Adrio
from epymorph.error import DataResourceException

_SliceLike = slice | type(Ellipsis)
_ArraySlice = _SliceLike | tuple[_SliceLike, ...]


class NPY(Adrio[Any]):
    """Retrieves an array of data from a user-provided .npy file."""

    file_path: PathLike
    """The path to the .npy file containing data."""
    array_slice: _ArraySlice | None
    """Optional slice(s) of the array to load."""

    def __init__(
        self, file_path: PathLike, array_slice: _ArraySlice | None = None
    ) -> None:
        if Path(file_path).suffix != ".npy":
            msg = (
                "Incorrect file type. Only .npy files can be loaded through NPY ADRIOs."
            )
            raise DataResourceException(msg)
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
            raise DataResourceException(msg) from e
        except ValueError as e:
            msg = "Object arrays cannot be loaded."
            raise DataResourceException(msg) from e
        except IndexError as e:
            msg = "Specified array slice is invalid for the shape of this data."
            raise DataResourceException(msg) from e


class NPZ(Adrio[Any]):
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
        if Path(file_path).suffix != ".npz":
            msg = (
                "Incorrect file type. Only .npz files can be loaded through NPZ ADRIOs."
            )
            raise DataResourceException(msg)
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
            raise DataResourceException(msg) from e
        except ValueError as e:
            msg = "Object arrays cannot be loaded."
            raise DataResourceException(msg) from e
        except IndexError as e:
            msg = "Specified array slice is invalid for the shape of this data."
            raise DataResourceException(msg) from e
