from os import PathLike
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.adrio.adrio import Adrio
from epymorph.error import DataResourceException


class NPY(Adrio[Any]):
    """Retrieves an array of data from a user-provided .npy file."""

    file_path: PathLike
    """The path to the .npy file containing data."""
    array_slice: tuple[slice, ...] | None
    """Optional slice(s) of the array to load."""

    def __init__(self, file_path: PathLike, array_slice: tuple[slice, ...] | None = None) -> None:
        self.file_path = file_path
        self.array_slice = array_slice

    @override
    def evaluate(self) -> NDArray:
        if Path(self.file_path).suffix != '.npy':
            msg = 'Incorrect file type. Only .npy files can be loaded through NPY ADRIOs.'
            raise DataResourceException(msg)

        try:
            data = cast(NDArray, np.load(self.file_path))
        except OSError as e:
            msg = 'File not found.'
            raise DataResourceException(msg) from e
        except ValueError as e:
            msg = 'Object arrays cannot be loaded.'
            raise DataResourceException(msg) from e

        if self.array_slice is not None and len(self.array_slice) != 0:
            if len(self.array_slice) != data.ndim:
                msg = 'One slice is required for each array axis.'
                raise DataResourceException(msg)

            data = data[self.array_slice]

        return data


class NPZ(Adrio[Any]):
    """Retrieves an array of data from a user-defined .npz file."""

    file_path: PathLike
    """The path to the .npz file containing data."""
    array_name: str
    """The name of the array in the .npz file to load."""
    array_slice: tuple[slice, ...] | None
    """Optional slice(s) of the array to load."""

    def __init__(self, file_path: PathLike, array_name: str, array_slice: tuple[slice, ...] | None = None) -> None:
        self.file_path = file_path
        self.array_name = array_name
        self.array_slice = array_slice

    def evaluate(self) -> NDArray:
        if Path(self.file_path).suffix != '.npz':
            msg = 'Incorrect file type. Only .npz files can be loaded through NPZ ADRIOs.'
            raise DataResourceException(msg)

        try:
            data = cast(NDArray, np.load(self.file_path)[self.array_name])
        except OSError as e:
            msg = 'File not found.'
            raise DataResourceException(msg) from e
        except ValueError as e:
            msg = 'Object arrays cannot be loaded.'
            raise DataResourceException(msg) from e

        if self.array_slice is not None and len(self.array_slice) != 0:
            if len(self.array_slice) != data.ndim:
                msg = 'One slice is required for each array axis.'
                raise DataResourceException(msg)

            data = data[self.array_slice]

        return data
