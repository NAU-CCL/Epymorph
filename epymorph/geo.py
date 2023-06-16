from typing import NamedTuple

import numpy as np
from numpy.typing import DTypeLike, NDArray

from epymorph.util import DataDict


def validate_shape(name: str, data: NDArray, shape: tuple[int, ...], dtype: DTypeLike | None = None):
    if not data.shape == shape:
        msg = f"Geo data '{name}' is incorrectly shaped; expected {shape}, loaded {data.shape}"
        raise Exception(msg)

    if dtype is not None:
        exp_dtype = np.dtype(dtype).type
        if data.dtype.type is not exp_dtype:
            msg = f"Geo data '{name}' is not the expected type; expected {exp_dtype.__name__}, loaded {data.dtype.type.__name__}"
            raise Exception(msg)
    return data


class Geo(NamedTuple):
    nodes: int
    labels: list[str]
    data: DataDict
