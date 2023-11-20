from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.typing import NDArray

from epymorph.error import GeoValidationException
from epymorph.geo.adrio.adrio import ADRIO, ADRIOMaker, ADRIOMakerLibrary
from epymorph.geo.geo import Geo
from epymorph.geo.spec import LABEL, AttribDef, DynamicGeoSpec
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
            raise KeyError(f"Attribute not found in geo: '{name}'")
        return self._adrios[name].get_value()

    @property
    def labels(self) -> NDArray[np.str_]:
        # Since we've already accessed this adrio during construction,
        # the adrio should have already cached this value.
        return self._adrios[LABEL.name].get_value()

    # TODO: can we implement a form of validation on dynamic geos short of fetching
    # all of their data? Maybe not... but maybe if ADRIO had an AttribDef, we could at
    # least check to see if the ADRIO *should* produce the expected type/shape.
    # I'll have to think about whether or not this fulfills the purpose of `validate`...

    # def validate(self) -> None:
    #     """
    #     Validate this geo against its specification.
    #     Raises GeoValidationException for any errors.
    #     """

    def fetch_attribute(self, adrio: ADRIO) -> None:
        """Gets a single Geo attribute from an ADRIO asynchronously using threads"""
        print(f'Fetching {adrio.attrib}')
        adrio.get_value()

    def fetch_all(self) -> None:
        """Retrieves all Geo attributes from geospec object using ADRIOs"""

        # TODO: loading from cache is currently disabled
        # load Geo from compressed file if one exists
        # if path.exists(geo_path(self.spec.id)) and not force:
        #     return load_compressed_geo(geo_path(self.spec.id))
        # build Geo using ADRIOs
        # else:

        print('Fetching GEO data from ADRIOs...')

        # initialize threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            for key, adrio in self._adrios.items():
                executor.submit(self.fetch_attribute, adrio)
            # TODO: do we need to explicitly await these futures?

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
