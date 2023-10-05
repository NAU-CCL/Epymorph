from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import PathLike
from types import MappingProxyType
from typing import Iterable

import jsonpickle
import numpy as np
from attr import dataclass
from numpy.typing import NDArray

from epymorph.geo import AttribDef
from epymorph.geo.adrio import adrio_maker_library
from epymorph.geo.adrio.adrio import ADRIO, ADRIOMaker
from epymorph.util import (NDIndices, NumpyTypeError, check_ndarray,
                           shape_matches)

# There are two attributes required of every geo:
POPULATION = AttribDef('population', np.int64)
LABEL = AttribDef('label', np.str_)


class Geo(ABC):
    """
    Abstract class representing the GEO model.
    Implementations are thus free to vary how they provide the requested data.
    """

    attributes: MappingProxyType[str, AttribDef]
    """The metadata for all attributes provided by this Geo."""

    nodes: int
    """The number of nodes in this Geo."""

    @abstractmethod
    def __getitem__(self, name: str) -> NDArray:
        pass


class StaticGeo(Geo):
    """A Geo implementation which contains all of data pre-fetched and in-memory."""

    @classmethod
    def load(cls, npz_file: PathLike) -> StaticGeo:
        """Load a StaticGeo from its .npz format."""
        with np.load(npz_file) as npz_data:
            values = dict(npz_data)
            return StaticGeo.from_values(values)

    @classmethod
    def from_values(cls, values: dict[str, NDArray]) -> StaticGeo:
        """Create a Geo containing the given values as attributes."""
        return cls(
            # Infer AttribDefs from the given values
            # TODO: this breaks for structural types (like CentroidDType). `.type` comes back as `np.void` which isn't ideal
            attrib_defs=[AttribDef(name, v.dtype.type)
                         for name, v in values.items()],
            attrib_values=values
        )

    _values: dict[str, NDArray]

    def __init__(self, attrib_defs: Iterable[AttribDef], attrib_values: dict[str, NDArray]):
        if LABEL not in attrib_defs:
            raise ValueError("Geo must provide the 'label' attribute.")
        if POPULATION not in attrib_defs:
            raise ValueError("Geo must provide the 'population' attribute.")

        # Check all expected attributes are given and properly typed.
        for a in attrib_defs:
            v = attrib_values.get(a.name)
            if v is None:
                raise ValueError(f"Geo is missing values for attribute '{a.name}'.")
            try:
                check_ndarray(v, dtype=a.dtype)
            except NumpyTypeError as e:
                raise ValueError("Geo attribute '{a.name}' is invalid.") from e

        # Verify that label and population are one-dimensional arrays that match in size.
        labels = attrib_values['label']
        pops = attrib_values['population']
        if len(labels.shape) != 1:
            raise ValueError("Invalid 'label' attribute in Geo.")
        if len(pops.shape) != 1:
            raise ValueError("Invalid 'population' attribute in Geo.")
        if labels.shape != pops.shape:
            msg = "Geo 'population' and 'label' attributes must be the same size."
            raise ValueError(msg)

        # We can now assume values to be properly formed.
        self.attributes = MappingProxyType({a.name: a for a in attrib_defs})
        self.nodes = len(labels)
        self._values = {
            name: values
            for name, values in attrib_values.items()
            if name in self.attributes  # weed out extra values
        }

    def __getitem__(self, name: str) -> NDArray:
        if name not in self._values:
            raise KeyError(f"Attribute not found in geo: '{name}'")
        return self._values[name]

    def filter(self, selection: NDIndices) -> StaticGeo:
        """
        Create a new geo by selecting only certain nodes from another geo.
        Does not alter the original geo.
        """

        n = self.nodes

        def select(arr: NDArray) -> NDArray:
            # Handle selections on attribute arrays
            if shape_matches(arr, (n,)):
                return arr[selection]
            elif shape_matches(arr, (n, n)):
                return arr[selection[:, np.newaxis], selection]
            elif shape_matches(arr, ('?', n)):  # matches TxN
                return arr[:, selection]
            elif shape_matches(arr, (n, '?')):  # matches NxA
                return arr[selection, :]
            else:
                raise Exception(f"Unsupported shape {arr.shape}.")

        filtered_values = {
            attrib_name: select(self._values[attrib_name])
            for attrib_name, attrib in self.attributes.items()
        }
        return StaticGeo(self.attributes.values(), filtered_values)

    def save(self, npz_file: PathLike) -> None:
        """Saves a StaticGeo to .npz format."""
        np.savez_compressed(npz_file, **self._values)


class DynamicGeo(Geo):

    @classmethod
    def from_spec(cls, geo_spec: str) -> DynamicGeo:
        """Create a Dynamic Geo from a geo spec text."""
        spec = deserialize(geo_spec)
        return cls(spec)

    _adrios_dict: dict[str, ADRIO]

    def __init__(self, geo_spec: GEOSpec) -> None:
        if geo_spec.source is None:
            msg = "Error: Attribute sources must be specified when creating a Geo dynamically."
            raise Exception(msg)

        maker_dict: dict[str, ADRIOMaker]
        maker_dict = {}
        self._adrios_dict = {}

        all_attributes = [geo_spec.label] + geo_spec.attributes

        # loop through attributes and make adrios for each
        for attrib in all_attributes:
            source = geo_spec.source.get(attrib.name)
            # make appropriate adrio maker if it does not already exist
            if source not in maker_dict.keys() and source is not None:
                maker = adrio_maker_library.get(source)
                if maker is not None:
                    maker_dict[source] = maker()

            # make adrio
            if source is not None:
                self._adrios_dict[attrib.name] = maker_dict[source].make_adrio(
                    attrib, geo_spec.granularity, geo_spec.nodes, geo_spec.year)

        self._adrios_dict['label'] = self._adrios_dict.pop(geo_spec.label.name)

    def __getitem__(self, name: str) -> NDArray:
        if name not in self._adrios_dict.keys():
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

        print('...done')

        # build, cache, and return Geo
        # save_compressed_geo(self.spec.id, data)

    def save(self, npz_file: PathLike) -> None:
        values = {}  # type enforce?
        for attrib, adrio in self._adrios_dict.items():
            values[attrib] = adrio.get_value()
        np.savez_compressed(npz_file, **values)


@dataclass
class GEOSpec:
    """class to create geo spec files used by the ADRIO system to create geos"""
    id: str
    label: AttribDef
    attributes: list[AttribDef]
    granularity: int
    nodes: dict[str, list[str]]
    year: int
    type: str
    source: dict[str, str] | None = None


def serialize(spec: GEOSpec, file_path: str) -> None:
    """serializes a GEOSpec object to a file at the given path"""
    json_spec = str(jsonpickle.encode(spec, unpicklable=True))
    with open(file_path, 'w') as stream:
        stream.write(json_spec)


def deserialize(spec_enc: str) -> GEOSpec:
    """deserializes a GEOSpec object from a pickled text"""
    spec_dec = jsonpickle.decode(spec_enc)

    # ensure decoded object is of type GEOSpec
    if type(spec_dec) is GEOSpec:
        return spec_dec
    else:
        msg = 'GEO spec does not decode to GEOSpec object; ensure file path is correct and file is correctly formatted'
        raise Exception(msg)
