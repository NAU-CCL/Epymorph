from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from numpy.typing import NDArray

from epymorph.geo.spec import AttribDef, Geography, TimePeriod


class ADRIO:
    attrib: str
    _fetch: Callable[[], NDArray]
    _cached_value: NDArray | None

    def __init__(self, attrib: str, fetch_data: Callable[[], NDArray]) -> None:
        self.attrib = attrib
        self._fetch = fetch_data
        self._cached_value = None

    def get_value(self, **kwargs) -> NDArray:
        if self._cached_value is None:
            self._cached_value = self._fetch()
        return self._cached_value


class ADRIOMaker(ABC):
    """abstract class to serve as an outline for ADRIO makers for specific data sources"""
    attributes: list[AttribDef]

    @abstractmethod
    def make_adrio(self, attrib: AttribDef, geography: Geography, time_period: TimePeriod) -> ADRIO:
        pass

    # @abstractmethod
    # def verify(self):
    #    pass


ADRIOMakerLibrary = dict[str, type[ADRIOMaker]]
