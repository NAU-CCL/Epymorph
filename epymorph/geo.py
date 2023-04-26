from typing import NamedTuple

from numpy.typing import NDArray

from epymorph.util import DataDict


def validate_shape(name: str, data: NDArray, shape: tuple[int, ...]):
    if not data.shape == shape:
        msg = f"Geo data '{name}' is incorrectly shaped; expected {shape}, loaded {data.shape}"
        raise Exception(msg)
    return data


class Geo(NamedTuple):
    nodes: int
    labels: list[str]
    data: DataDict
