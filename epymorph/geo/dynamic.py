"""
A dynamic geo is capable of fetching data from arbitrary external data sources
via the use of ADRIO implementations. It may fetch this data lazily, loading
only the attributes needed by the simulation.
"""
import os
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Self

import numpy as np
from numpy.typing import NDArray

from epymorph.error import AttributeException, GeoValidationException
from epymorph.event import AdrioStart, DynamicGeoEvents, FetchStart
from epymorph.geo.adrio.adrio import ADRIO, ADRIOMaker, ADRIOMakerLibrary
from epymorph.geo.adrio.file.adrio_file import ADRIOMakerFile, FileSpec
from epymorph.geo.geo import Geo
from epymorph.geo.spec import LABEL, DynamicGeoSpec, validate_geo_values
from epymorph.simulation import AttributeArray, geo_attrib
from epymorph.util import Event, MemoDict


def _memoized_adrio_maker_library(lib: ADRIOMakerLibrary) -> MemoDict[str, ADRIOMaker]:
    """
    Memoizes an adrio maker library to avoid constructing the same adrio maker twice.
    Will raise GeoValidationException if asked for an adrio maker that doesn't exist.
    """
    def load_maker(name: str) -> ADRIOMaker:
        maker_cls = lib.get(name)
        if maker_cls is None:
            msg = f"Unknown attribute source: {name}."
            raise GeoValidationException(msg)
        return maker_cls()
    return MemoDict[str, ADRIOMaker](load_maker)


class DynamicGeo(Geo[DynamicGeoSpec], DynamicGeoEvents):
    """A Geo implementation which uses ADRIOs to dynamically fetch data from third-party data sources."""

    @classmethod
    def from_library(cls, spec: DynamicGeoSpec, adrio_maker_library: ADRIOMakerLibrary) -> Self:
        """Given an ADRIOMaker library, construct a DynamicGeo for the given spec."""
        makers = _memoized_adrio_maker_library(adrio_maker_library)

        # loop through attributes and make adrios for each
        adrios = dict[str, ADRIO]()
        for attr in spec.attributes:
            source = spec.source.get(attr.name)
            if source is None:
                msg = f"Missing source for attribute: {attr.name}."
                raise GeoValidationException(msg)

            if isinstance(source, str):
                maker_name = source
                adrio_attrib = attr

                # If source is formatted like "<adrio_maker_name>:<attribute_name>" then
                # the geo wants to use a different name than the one the maker uses;
                # no problem, just provide a modified AttribDef to the maker.
                if ":" in source:
                    maker_name, adrio_attrib_name = source.split(":")[0:2]
                    adrio_attrib = geo_attrib(
                        adrio_attrib_name, attr.dtype, attr.shape)

                # Make and store adrio.
                adrio = makers[maker_name].make_adrio(
                    adrio_attrib,
                    spec.geography,
                    spec.time_period
                )
                adrios[attr.name] = adrio

            else:
                maker = makers['File']
                if isinstance(maker, ADRIOMakerFile) and isinstance(source, FileSpec):
                    adrio = maker.make_adrio(
                        adrio_attrib,
                        spec.geography,
                        spec.time_period,
                        source
                    )
                    adrios[attr.name] = adrio

        return cls(spec, adrios)

    spec: DynamicGeoSpec
    _adrios: dict[str, ADRIO]

    def __init__(self, spec: DynamicGeoSpec, adrios: dict[str, ADRIO]):
        if not LABEL.name in adrios:
            raise ValueError("Geo must contain an attribute called 'label'.")
        self._adrios = adrios
        labels = self._adrios[LABEL.name].get_value()
        super().__init__(spec, labels.size)

        # events
        self.fetch_start = Event()
        self.adrio_start = Event()
        self.fetch_end = Event()

    def __getitem__(self, name: str, /) -> AttributeArray:
        if name not in self._adrios:
            raise AttributeException(f"Attribute not found in geo: '{name}'")
        if self._adrios[name]._cached_value is None:
            self.adrio_start.publish(AdrioStart(name, None, None))
        return self._adrios[name].get_value()

    def __contains__(self, name: str, /) -> bool:
        return name in self._adrios

    @property
    def labels(self) -> NDArray[np.str_]:
        """The labels for every node in this geo."""
        # Since we've already accessed this adrio during construction,
        # the adrio should have already cached this value.
        return self._adrios[LABEL.name].get_value()

    def validate(self) -> None:
        """
        Validate this geo against its specification.
        Raises GeoValidationException for any errors.
        WARNING: this will fetch all data!
        """
        if self.spec.attribute_map.keys() != self._adrios.keys():
            raise GeoValidationException('Geo values do not match the given spec.')
        validate_geo_values(self.spec, self._fetch_all())

    def _fetch_all(self) -> dict[str, NDArray]:
        """For internal purposes: retrieves all Geo attributes using ADRIOs and returns a dict of the values."""
        def fetch(key: str, adrio: ADRIO) -> tuple[str, NDArray]:
            return (key, adrio.get_value())

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(fetch, key, adrio)
                       for key, adrio in self._adrios.items()]
            return dict(result.result() for result in wait(futures).done)

    def fetch_all(self) -> None:
        """Retrieves all Geo attributes from geospec object using ADRIOs"""
        num_adrios = len(self._adrios)
        self.fetch_start.publish(FetchStart(num_adrios))

        def fetch_attribute(adrio: ADRIO, index: int) -> NDArray:
            self.adrio_start.publish(AdrioStart(adrio.attrib, index, num_adrios))
            return adrio.get_value()

        # initialize threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            for index, adrio in enumerate(self._adrios.values()):
                executor.submit(fetch_attribute, adrio, index)

        self.fetch_end.publish(None)


class DynamicGeoFileOps:
    """Helper functions for saving and loading dynamic geos and specs."""

    @staticmethod
    def get_spec_filename(geo_id: str) -> str:
        """Returns the standard filename for a geo spec file."""
        return f"{geo_id}.geo"

    @staticmethod
    def load_from_spec(file: os.PathLike, adrio_maker_library: ADRIOMakerLibrary) -> DynamicGeo:
        """Load a DynamicGeo from its spec file."""
        try:
            with open(file, mode='r', encoding='utf-8') as f:
                spec_json = f.read()
            spec = DynamicGeoSpec.deserialize(spec_json)
            return DynamicGeo.from_library(spec, adrio_maker_library)
        except Exception as e:
            msg = f"Unable to load '{file}' as a geo: {e}"
            raise GeoValidationException(msg) from e
