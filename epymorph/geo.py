from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from os import PathLike, path
from typing import NamedTuple, TypeVar

import numpy as np
from numpy.typing import DTypeLike, NDArray

from epymorph.adrio import uscounties_library
from epymorph.adrio.adrio import ADRIOSpec, GEOSpec, deserialize
from epymorph.util import NDIndices


class Geo(NamedTuple):
    nodes: int
    labels: list[str]
    data: dict[str, NDArray]


# GEO processing utilities


def filter_geo(geo: Geo, selection: NDIndices) -> Geo:
    nodes = len(selection)
    labels = (np.array(geo.labels, dtype=str)[selection]).tolist()

    # Handle selections on attribute arrays (NxN arrays need special processing!)
    # TODO: probably need to support TxNxN arrays too,
    # but the relationship between time-series data and geos isn't well-founded yet
    def select(arr: NDArray) -> NDArray:
        if arr.shape == (geo.nodes, geo.nodes):
            return arr[selection[:, np.newaxis], selection]
        else:
            return arr[selection]

    data = {key: select(arr)
            for key, arr in geo.data.items()}
    return Geo(nodes, labels, data)


# Schema and Validation


class Attribute(NamedTuple):
    dtype: DTypeLike
    shape: tuple[int, ...]


Schema = dict[str, Attribute]

CentroidDType = np.dtype([('longitude', np.float64), ('latitude', np.float64)])


def validate_schema(schema: Schema, data: dict[str, NDArray]) -> None:
    for name, attr in schema.items():
        attr_data = data.get(name)
        validate_attribute(name, attr, attr_data)


def validate_attribute(name: str, attr: Attribute, data: NDArray | None) -> None:
    validate_shape(name, data, attr.shape, attr.dtype)


T = TypeVar('T', bound=NDArray)


def validate_shape(name: str, data: T | None, shape: tuple[int, ...], dtype: DTypeLike | None = None) -> T:
    if data is None:
        msg = f"Geo data '{name}' is missing."
        raise Exception(msg)

    if not data.shape == shape:
        msg = f"Geo data '{name}' is incorrectly shaped; expected {shape}, loaded {data.shape}"
        raise Exception(msg)

    if dtype is not None:
        exp_dtype = np.dtype(dtype).type
        if data.dtype.type is not exp_dtype:
            msg = f"Geo data '{name}' is not the expected type; expected {exp_dtype.__name__}, loaded {data.dtype.type.__name__}"
            raise Exception(msg)

    return data


# Save and Load


def load_compressed_geo(npz_file: str | PathLike) -> Geo:
    """Read a GEO from its .npz format."""
    with np.load(npz_file) as npz_data:
        data = dict(npz_data)
    labels = data.get('label')
    if labels is None:
        msg = f"Geo {id} is missing a 'label' attribute. Cannot be loaded."
        raise Exception(msg)
    nodes = len(labels)
    return Geo(nodes, labels, data)


def geo_path(id: str) -> str:
    return f"./epymorph/data/geo/{id}_geo.npz"


def save_compressed_geo(id: str, data: dict[str, NDArray]) -> None:
    if not 'label' in data:
        msg = f"Geo {id} must have a 'label' attribute in order to be saved and loaded."
        raise Exception(msg)
    np.savez_compressed(geo_path(id), **data)


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
        # load Geo from compressed file if one exists
        if path.exists(geo_path(self.spec.id)) and not force:
            return load_compressed_geo(geo_path(self.spec.id))
        # build Geo using ADRIOs
        else:
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
            labels = data['label']
            # save_compressed_geo(self.spec.id, data)
            return Geo(
                nodes=len(labels),
                labels=labels.tolist(),
                data=data
            )
