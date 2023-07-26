from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import DTypeLike, NDArray

from epymorph.util import DataDict


def filter_geo(geo: Geo, selection: NDArray[np.int_]) -> Geo:
    nodes = len(selection)
    labels = (np.array(geo.labels, dtype=str)[selection]).tolist()

    # Handle selections on attribute arrays (NxN arrays need special processing!)
    # TODO: probably need to support TxNxN arrays too,
    # but the relationship between time-series data and geos isn't well-founded yet
    def select(arr: NDArray) -> NDArray:
        if arr.shape == (geo.nodes, geo.nodes):
            return arr[selection[:, np.newaxis], selection]
        else:
            return arr[selection]

    data = {key: select(arr)
            for key, arr in geo.data.items()}
    return Geo(nodes, labels, data)


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
