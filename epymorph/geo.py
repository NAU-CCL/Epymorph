from typing import NamedTuple

from epymorph.util import DataDict


class Geo(NamedTuple):
    nodes: int
    labels: list[str]
    data: DataDict
