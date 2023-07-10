from typing import NamedTuple, TypeVar

import numpy as np
from numpy.typing import DTypeLike, NDArray

from epymorph.adrio import uscounties_library
from epymorph.adrio.adrio import deserialize
from epymorph.util import DataDict, NumpyIndices


class Geo(NamedTuple):
    nodes: int
    labels: list[str]
    data: dict[str, NDArray]


# GEO processing utilities


def filter_geo(geo: Geo, selection: NumpyIndices) -> Geo:
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

CentroidDType = np.dtype([('longitude', float), ('latitude', float)])


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


def geo_path(id: str) -> str:
    return f"./epymorph/data/geo/{id}_geo.npz"


def save_compressed_geo(id: str, data: dict[str, NDArray]) -> None:
    if not 'label' in data:
        msg = f"Geo {id} must have a 'label' attribute in order to be saved and loaded."
        raise Exception(msg)
    np.savez_compressed(geo_path(id), **data)


def load_compressed_geo(id: str) -> Geo:
    with np.load(geo_path(id)) as npz_data:
        data = dict(npz_data)
    labels = data.get('label')
    if labels is None:
        msg = f"Geo {id} is missing a 'label' attribute. Cannot be loaded."
        raise Exception(msg)
    nodes = len(labels)
    return Geo(nodes, labels, data)


class GEOBuilder:
    def __init__(self, path: str):
        self.spec = deserialize(path)

    def build(self, force=False) -> Geo:
        # create GEOSpec object from file
        data = DataDict()
        print('Fetching GEO data from ADRIOs...')
        # loop for all ADRIOSpecs
        for i in range(len(self.spec.adrios)):
            # get adrio class from library dictionary (library hardcoded for now)
            current = uscounties_library.get(self.spec.adrios[i].class_name)
            # fetch data from adrio
            if current is not None:
                current_obj = current()
                print(f'Fetching {current_obj.attribute}')
                data[current_obj.attribute] = current_obj.fetch(
                    force, nodes=self.spec.nodes)

        print('...done')
        # build and return Geo (what to do for nodes/label?)
        return Geo(
            nodes=len(data['name and state']),
            labels=[name[0] + ', ' + name[1]
                    for name in data['name and state']],
            data=data
        )
