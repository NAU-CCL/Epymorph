from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from os import PathLike
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from epymorph.geo.adrio import adrio_maker_library
from epymorph.geo.adrio.adrio import ADRIO, ADRIOMaker
from epymorph.geo.common import AttribDef
from epymorph.geo.geo import LABEL, Geo, GEOSpec, deserialize
from epymorph.util import MemoDict


class DynamicGeo(Geo):

    @classmethod
    def from_spec(cls, geo_spec: str) -> DynamicGeo:
        """Create a Dynamic Geo from a geo spec text."""
        spec = deserialize(geo_spec)
        return cls(spec)

    _adrios_dict: dict[str, ADRIO]

    def __init__(self, geo_spec: GEOSpec):
        self._adrios_dict = {}
        attributes = MappingProxyType({a.name: a for a in geo_spec.attributes})
        Geo.validate_attributes(attributes.values())

        if geo_spec.source is None:
            msg = "Error: Attribute sources must be specified when creating a Geo dynamically."
            raise ValueError(msg)

        def load_maker(name: str) -> ADRIOMaker:
            maker_cls = adrio_maker_library.get(name)
            if maker_cls is None:
                raise ValueError(f"Unknown attribute source: {source}.")
            return maker_cls()

        maker_dict = MemoDict[str, ADRIOMaker](load_maker)

        # loop through attributes and make adrios for each
        for attrib in attributes.values():
            source = geo_spec.source.get(attrib.name)
            if source is None:
                raise ValueError(f"Missing source for attribute: {attrib.name}.")

            # If source is formatted like "<adrio_maker_name>:<attribute_name>" then
            # the geo wants to use a different name than the one the maker uses;
            # no problem, just provide a modified AttribDef to the maker.
            maker_name = source
            adrio_attrib = attrib
            if ":" in source:
                maker_name, adrio_attrib_name = source.split(":")[0:2]
                adrio_attrib = AttribDef(adrio_attrib_name, attrib.dtype)

            # Make and store adrio.
            adrio = maker_dict[maker_name].make_adrio(
                adrio_attrib,
                geo_spec.granularity,
                geo_spec.nodes,
                geo_spec.year
            )
            self._adrios_dict[attrib.name] = adrio

        # Load required values and validate.
        checked_values = {
            a.name: self._adrios_dict[a.name].get_value()
            for a in Geo.required_attributes
        }
        Geo.validate_values(Geo.required_attributes, checked_values)

        nodes = len(checked_values[LABEL.name])
        super().__init__(attributes, nodes)

    def __getitem__(self, name: str) -> NDArray:
        if name not in self._adrios_dict:
            raise KeyError(f"Attribute not found in geo: '{name}'")
        return self._adrios_dict[name].get_value()

    def fetch_attribute(self, adrio: ADRIO) -> None:
        """Gets a single Geo attribute from an ADRIO asynchronously using threads"""
        # fetch data from adrio
        print(f'Fetching {adrio.attrib}')
        # call adrio fetch method
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
            for key, adrio in self._adrios_dict.items():
                executor.submit(self.fetch_attribute, adrio)
            # TODO: do we need to explicitly await these futures?

        print('...done')

        # build, cache, and return Geo
        # save_compressed_geo(self.spec.id, data)

    def save(self, npz_file: PathLike) -> None:
        values = {}  # type enforce?
        for attrib, adrio in self._adrios_dict.items():
            values[attrib] = adrio.get_value()
        np.savez_compressed(npz_file, **values)
