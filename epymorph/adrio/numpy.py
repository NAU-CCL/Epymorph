from os import PathLike
from typing import Any

import numpy as np
from numpy.typing import NDArray

from epymorph.adrio.adrio import Adrio
from epymorph.error import DataResourceException


class NPY(Adrio[Any]):
    """Retrieves an array of data from a user-provided .npy file."""

    file_path: PathLike
    """The path to the .npy file containing data."""
    arr_slice: list[slice] | None
    """Optional slice(s) of the array to load."""

    def __init__(self, file_path: PathLike, slice: list[slice] | None = None) -> None:
        self.file_path = file_path
        self.slice = slice

    def evaluate(self) -> NDArray:
        data = np.load(self.file_path)

        data = np.array(data)

        if self.arr_slice is not None:
            if len(self.arr_slice) != data.ndim:
                msg = 'One slice is required for each array axis.'
                raise DataResourceException(msg)
            axis = 0
            for x in self.arr_slice:
                if x is not None and isinstance(x, slice):
                    data = data.take(indices=range(x.start, x.stop), axis=axis)
                axis += 1

        return data


class NPZ(Adrio[Any]):
    """Retrieves an array of data from a user-defined .npz file."""

    file_path: PathLike
    """The path to the .npz file containing data."""
    array_name: str
    """The name of the array in the .npz file to load."""
    arr_slice: list[slice] | None
    """Optional slice(s) of the array to load."""

    def __init__(self, file_path: PathLike, array_name: str, arr_slice: list[slice] | None = None) -> None:
        self.file_path = file_path
        self.array_name = array_name
        self.arr_slice = arr_slice

    def evaluate(self) -> NDArray:
        data = np.load(self.file_path)
        data = np.array(data[self.array_name])

        if self.arr_slice is not None:
            if len(self.arr_slice) != data.ndim:
                msg = 'One slice is required for each array axis.'
                raise DataResourceException(msg)
            axis = 0
            for x in self.arr_slice:
                if x is not None and isinstance(x, slice):
                    data = data.take(indices=range(x.start, x.stop), axis=axis)
                axis += 1

        return data
