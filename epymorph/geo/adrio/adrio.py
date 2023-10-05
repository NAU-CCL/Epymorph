from abc import ABC, abstractmethod
from typing import Callable

from numpy.typing import NDArray

from epymorph.geo import AttribDef


class ADRIO:
    attrib: str
    _fetch: Callable[[], NDArray]

    def __init__(self, attrib: str, fetch_data: Callable[[], NDArray]) -> None:
        self.attrib = attrib
        self._fetch = fetch_data

    def get_value(self, **kwargs) -> NDArray:
        # TODO: check for cached value
        return self._fetch()


class ADRIOMaker(ABC):
    """abstract class to serve as an outline for ADRIO makers for specific data sources"""
    attributes: list[AttribDef]

    @abstractmethod
    def make_adrio(self, attrib: AttribDef, granularity: int, nodes: dict[str, list[str]], year: int) -> ADRIO:
        pass

    # @abstractmethod
    # def verify(self):
    #    pass
