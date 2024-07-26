"""
ADRIOs enable dynamic geos to fetch data from varied external data sources,
and ADRIOMakers create ADRIOs for a data soruce and specialized for a geo's purposes.
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping

from numpy.typing import NDArray

from epymorph.geo.spec import TimePeriod
from epymorph.geography.scope import GeoScope
from epymorph.simulation import AttributeDef


class ADRIO:
    """Data retrieval class that fetches and stores a data value on demand."""

    attrib: str
    """The name of the attribute to fetch."""

    _fetch: Callable[[], NDArray]
    """The function that carries out data retrieval."""

    _cached_value: NDArray | None
    """The stored value of the attribute once retrieved."""

    def __init__(self, attrib: str, fetch_data: Callable[[], NDArray]) -> None:
        self.attrib = attrib
        self._fetch = fetch_data
        self._cached_value = None

    def get_value(self) -> NDArray:
        """Returns cached data value or retrieves it using callable fetch function if not yet cached."""
        if self._cached_value is None:
            self._cached_value = self._fetch()
        return self._cached_value


class ADRIOMaker(ABC):
    """Abstract class to serve as an outline for ADRIO makers for specific data sources."""
    attributes: list[AttributeDef]

    @staticmethod
    @abstractmethod
    def accepts_source(source: Any) -> bool:
        """Checks whether the ADRIOMaker accepts a given source type and returns the result as a boolean."""

    @abstractmethod
    def make_adrio(self, attrib: AttributeDef, scope: GeoScope, time_period: TimePeriod, source: Any | None = None) -> ADRIO:
        """Creates an ADRIO to fetch the specified attribute for the specified time and place."""


ADRIOMakerLibrary = Mapping[str, type[ADRIOMaker]]
"""ADRIOMaker objects for all supported data sources."""
