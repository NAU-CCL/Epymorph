"""ADRIO for loading data from a compressed numpy data file (.npy or .npz)."""

from pathlib import Path

import numpy as np
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.adrio.adrio import (
    ADRIO,
    ADRIOContextError,
    ADRIOProcessingError,
    InspectResult,
    adrio_validate_pipe,
)
from epymorph.adrio.validation import (
    ResultFormat,
    validate_dtype,
    validate_numpy,
    validate_shape_unchecked_arbitrary,
)
from epymorph.data_shape import DataShape
from epymorph.error import MissingContextError
from epymorph.simulation import Context, validate_context_for_shape

SliceOrEllipsis = slice | type(Ellipsis)
"""Type alias: either a `slice` object or an ellipsis (`...`)."""

ArraySlice = SliceOrEllipsis | tuple[SliceOrEllipsis, ...]
"""
Type alias for a numpy array slice.

See Also
--------
It's very convenient to use numpy's `IndexExpression` helpers like [numpy.s\\_][]
to create one of these.
"""


class NumpyFile(ADRIO[np.generic, np.generic]):
    """
    Retrieves an array of data from a .npy or .npz file.

    Parameters
    ----------
    file_path :
        The path to a .npy or .npz file to load.
    shape :
        The expected shape of the array. "Arbitrary" axes lengths will not be
        checked.
    dtype :
        The expected dtype of the array.
    array_name :
        If and only if loading an .npz file, the name of the array to load.
        For .npy files, this must be `None`.
    array_slice :
        The optional array slice to apply to the loaded array. numpy provides a
        helper called `np.s_` which is a convenient way to construct these.
        If `None`, the entire array is returned.
    """

    _file_path: Path
    """The path to a .npy or .npz file to load."""
    _shape: DataShape
    """The expected shape of the array."""
    _dtype: np.dtype
    """The expected dtype of the array."""
    _array_name: str | None
    """The name of the array to load from an npz file."""
    _array_slice: ArraySlice | None
    """The optional array slice to apply to the loaded array."""

    def __init__(
        self,
        *,
        file_path: str | Path,
        shape: DataShape,
        dtype: np.dtype | type[np.generic],
        array_name: str | None = None,
        array_slice: ArraySlice | None = None,
    ) -> None:
        file_path = file_path if isinstance(file_path, Path) else Path(file_path)
        match file_path.suffix:
            case ".npy":
                if array_name is not None:
                    err = "To load an .npy file, do not specify an `array_name`."
                    raise ValueError(err)
            case ".npz":
                if array_name is None:
                    err = "To load an .npz file, specify the `array_name` to load."
                    raise ValueError(err)
            case _:
                err = "This ADRIO supports .npz or .npy files only."
                raise ValueError(err)

        self._file_path = file_path
        self._shape = shape
        self._dtype = dtype if isinstance(dtype, np.dtype) else np.dtype(dtype)
        self._array_slice = array_slice
        self._array_name = array_name

    @property
    @override
    def result_format(self) -> ResultFormat:
        return ResultFormat(shape=self._shape, dtype=self._dtype)

    @override
    def validate_context(self, context: Context) -> None:
        try:
            validate_context_for_shape(context, self._shape)
        except MissingContextError as e:
            raise ADRIOContextError(self, self.context, str(e))

    @override
    def inspect(self) -> InspectResult[np.generic, np.generic]:
        self.validate_context(self.context)
        try:
            data = np.load(self._file_path)
            if isinstance(data, NpzFile):
                if self._array_name is None:
                    err = "To load an .npz file, specify the `array_name` to load."
                    raise ADRIOProcessingError(self, self.context, err)
                result = data[self._array_name]
                data.close()
            elif isinstance(data, np.ndarray):
                result = data
            else:
                err = "File did not contain data as expected."
                raise ADRIOProcessingError(self, self.context, err)

            if self._array_slice is not None:
                result = result[self._array_slice]
        except (OSError, ValueError) as e:
            err = "Error loading file."
            raise ADRIOProcessingError(self, self.context, err) from e
        except IndexError as e:
            err = "Specified array slice is invalid for the shape of this data."
            raise ADRIOProcessingError(self, self.context, err) from e

        self.validate_result(self.context, result)
        return InspectResult(
            adrio=self,
            source=result,
            result=result,
            dtype=self._dtype.type,
            shape=self._shape,
            issues={},
        )

    @override
    def validate_result(self, context: Context, result: NDArray[np.generic]) -> None:
        adrio_validate_pipe(
            self,
            context,
            result,
            validate_numpy(),
            validate_shape_unchecked_arbitrary(self._shape.to_tuple(context.dim)),
            validate_dtype(self._dtype),
        )
