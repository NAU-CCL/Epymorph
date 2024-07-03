"""
A geo represents a simulation's metapopulation model
with all of its attached data attributes.
"""
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from epymorph.geo.spec import GeoSpec
from epymorph.simulation import AttributeArray

SpecT_co = TypeVar('SpecT_co', bound=GeoSpec, covariant=True)


class Geo(Generic[SpecT_co], ABC):
    """
    Abstract class representing the GEO model.
    Implementations are thus free to vary how they provide the requested data.
    """

    spec: SpecT_co
    """The specification for this Geo."""

    nodes: int
    """The number of nodes in this Geo."""

    @property
    @abstractmethod
    def labels(self) -> NDArray[np.str_]:
        """The labels for every node in this geo."""

    def __init__(self, spec: SpecT_co, nodes: int):
        self.spec = spec
        self.nodes = nodes

    # Implement DataSource protocol

    @abstractmethod
    def __getitem__(self, name: str, /) -> AttributeArray:
        pass

    @abstractmethod
    def __contains__(self, name: str, /) -> bool:
        pass
