"""
A dynamic geo is capable of fetching data from arbitrary external data sources
via the use of ADRIO implementations. It may fetch this data lazily, loading
only the attributes needed by the simulation.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
from numpy.typing import NDArray

from epymorph.error import AttributeException, GeoValidationException
from epymorph.geo.adrio.adrio import ADRIO, ADRIOMaker, ADRIOMakerLibrary
from epymorph.geo.geo import Geo
from epymorph.geo.spec import (LABEL, AttribDef, DynamicGeoSpec,
                               validate_geo_values)
from epymorph.util import MemoDict


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


class DynamicGeo(Geo[DynamicGeoSpec]):
    """A Geo implementation which uses ADRIOs to dynamically fetch data from third-party data sources."""

    @staticmethod
    def load(spec_file: os.PathLike, adrio_maker_library: ADRIOMakerLibrary) -> DynamicGeo:
        """Load a DynamicGeo from a geo spec file."""
        return DynamicGeoFileOps.load_from_file(spec_file, adrio_maker_library)

    @classmethod
    def from_library(cls, spec: DynamicGeoSpec, adrio_maker_library: ADRIOMakerLibrary) -> DynamicGeo:
        """Given an ADRIOMaker library, construct a DynamicGeo for the given spec."""
        makers = _memoized_adrio_maker_library(adrio_maker_library)

        # loop through attributes and make adrios for each
        adrios = dict[str, ADRIO]()
        for attrib in spec.attributes:
            source = spec.source.get(attrib.name)
            if source is None:
                msg = f"Missing source for attribute: {attrib.name}."
                raise GeoValidationException(msg)

            # If source is formatted like "<adrio_maker_name>:<attribute_name>" then
            # the geo wants to use a different name than the one the maker uses;
            # no problem, just provide a modified AttribDef to the maker.
            maker_name = source
            adrio_attrib = attrib
            if ":" in source:
                maker_name, adrio_attrib_name = source.split(":")[0:2]
                adrio_attrib = AttribDef(adrio_attrib_name, attrib.dtype, attrib.shape)

            # Make and store adrio.
            adrio = makers[maker_name].make_adrio(
                adrio_attrib,
                spec.geography,
                spec.time_period
            )
            adrios[attrib.name] = adrio

        return cls(spec, adrios)

    spec: DynamicGeoSpec
    _adrios: dict[str, ADRIO]

    def __init__(self, spec: DynamicGeoSpec, adrios: dict[str, ADRIO]):
        self._adrios = adrios
        labels = self._adrios[LABEL.name].get_value()
        super().__init__(spec, len(labels))

    def __getitem__(self, name: str) -> NDArray:
        if name not in self._adrios:
            raise AttributeException(f"Attribute not found in geo: '{name}'")
        return self._adrios[name].get_value()

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
        print('Fetching GEO data from ADRIOs...')

        def fetch_attribute(adrio: ADRIO) -> NDArray:
            print(f'Fetching {adrio.attrib}')
            return adrio.get_value()

        # initialize threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            for adrio in self._adrios.values():
                executor.submit(fetch_attribute, adrio)
        print('...done')


class DynamicGeoFileOps:
    """Helper functions for saving and loading dynamic geos and specs."""

    @staticmethod
    def get_spec_filename(geo_id: str) -> str:
        """Returns the standard filename for a geo spec file."""
        return f"{geo_id}.geo"

    @staticmethod
    def load_from_file(file: os.PathLike, adrio_maker_library: ADRIOMakerLibrary) -> DynamicGeo:
        """Load a DynamicGeo from its spec file."""
        try:
            with open(file, mode='r', encoding='utf-8') as f:
                spec_json = f.read()
            spec = DynamicGeoSpec.deserialize(spec_json)
            return DynamicGeo.from_library(spec, adrio_maker_library)
        except Exception as e:
            raise GeoValidationException(f"Unable to load '{file}' as a geo.") from e
