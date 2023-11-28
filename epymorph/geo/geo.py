"""
A geo represents a simulation's metapopulation model
with all of its attached data attributes.
"""
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from epymorph.geo.spec import GeoSpec

SpecT = TypeVar('SpecT', bound=GeoSpec)


class Geo(Generic[SpecT], ABC):
    """
    Abstract class representing the GEO model.
    Implementations are thus free to vary how they provide the requested data.
    """

    spec: SpecT
    """The specification for this Geo."""

    nodes: int
    """The number of nodes in this Geo."""

    labels: NDArray[np.str_]
    """The labels for every node in this geo."""

    def __init__(self, spec: SpecT, nodes: int):
        self.spec = spec
        self.nodes = nodes

    @abstractmethod
    def __getitem__(self, name: str) -> NDArray:
        pass
