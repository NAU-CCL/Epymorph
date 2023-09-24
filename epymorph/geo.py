from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import PathLike
from types import MappingProxyType
from typing import Iterable, NamedTuple

import numpy as np
from numpy.typing import NDArray

from epymorph.adrio import uscounties_library
from epymorph.adrio.adrio import ADRIOSpec, GEOSpec, deserialize
from epymorph.util import (DTLike, NDIndices, NumpyTypeError, check_ndarray,
                           shape_matches)

CentroidDType = np.dtype([('longitude', np.float64), ('latitude', np.float64)])


class AttribDef(NamedTuple):
    """Metadata about a Geo attribute."""
    name: str
    dtype: DTLike


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


# Save and Load


class GEOBuilder:

    @classmethod
    def from_spec(cls, geo_spec: str) -> GEOBuilder:
        """Create a GEOBuilder from a geo spec text."""
        spec = deserialize(geo_spec)
        return cls(spec)

    def __init__(self, geo_spec: GEOSpec):
        self.spec = geo_spec

    def get_attribute(self, key: str | None, spec: ADRIOSpec) -> tuple[str, NDArray]:
        """Gets a single Geo attribute from an ADRIO asynchronously using threads"""
        # get adrio class from library dictionary
        adrio_class = uscounties_library.get(spec.class_name)

        # fetch data from adrio
        if adrio_class is None:
            raise Exception(f"Unable to load ADRIO for {spec.class_name}; "
                            "please check that your GEOSpec is valid.")
        else:
            adrio = adrio_class(spec=self.spec)

            print(f'Fetching {adrio.attribute}')
            # call adrio fetch method
            data = adrio.fetch()

            # check for no key
            if key is None:
                # assign key to attribute
                key = adrio.attribute

            # return tuple of key, resulting array
            return (key, data)

    def build(self, force=False) -> Geo:
        """Builds Geo from cached file or geospec object using ADRIOs"""

        # TODO: loading from cache is currently disabled
        # load Geo from compressed file if one exists
        # if path.exists(geo_path(self.spec.id)) and not force:
        #     return load_compressed_geo(geo_path(self.spec.id))
        # build Geo using ADRIOs
        # else:

        data = dict[str, NDArray]()
        print('Fetching GEO data from ADRIOs...')

        # mapping the ADRIOs by key as they will show up in the geo data; we can either declare:
        # - a literal key to use, or
        # - None to use the ADRIO's attribute
        all_adrios = \
            [('label', self.spec.label)] + \
            [(None, x) for x in self.spec.adrios]

        # initialize threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            # call thread on get_attribute
            thread_data = (executor.submit(self.get_attribute, key, spec)
                           for key, spec in all_adrios)

            # loop for threads as completed
            for future in as_completed(thread_data):
                # get result of future
                curr_data = future.result()

                # assign dictionary at attribute to resulting array
                data[curr_data[0]] = curr_data[1]

        print('...done')

        # build, cache, and return Geo
        # save_compressed_geo(self.spec.id, data)
        return StaticGeo.from_values(data)
