from __future__ import annotations

from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Iterable

import jsonpickle
import numpy as np
from attr import dataclass
from numpy.typing import NDArray

from epymorph.geo.common import AttribDef
from epymorph.util import NumpyTypeError, check_ndarray

# There are two attributes required of every geo:
LABEL = AttribDef('label', np.str_)
POPULATION = AttribDef('population', np.int64)


class Geo(ABC):
    """
    Abstract class representing the GEO model.
    Implementations are thus free to vary how they provide the requested data.
    """

    required_attributes = (LABEL, POPULATION)

    @staticmethod
    def validate_attributes(attrib_defs: Iterable[AttribDef]) -> None:
        """
        Checks if the required set of geo attributes are present.
        Raises ValueError if not.
        """
        missing = set(Geo.required_attributes) - set(attrib_defs)
        names = ", ".join(f"'{a.name}'" for a in missing)
        if len(missing) == 1:
            raise ValueError(f"Geo must provide the {names} attribute.")
        if len(missing) > 1:
            raise ValueError(f"Geo must provide attributes: {names}.")

    @staticmethod
    def validate_values(attrib_defs: Iterable[AttribDef], values: dict[str, NDArray]) -> None:
        """
        Checks if required geo attribute values are in good form.
        Raises ValueError if not.
        """
        for a in attrib_defs:
            v = values.get(a.name)
            if v is None:
                raise ValueError(f"Geo is missing values for attribute '{a.name}'.")
            try:
                dims = 1 if a in (LABEL, POPULATION) else None
                check_ndarray(v, dtype=a.dtype, dimensions=dims)
            except NumpyTypeError as e:
                raise ValueError(f"Geo attribute '{a.name}' is invalid.") from e

        if values[LABEL.name].shape != values[POPULATION.name].shape:
            msg = "Geo attributes 'population' and 'label' must be the same size."
            raise ValueError(msg)

    attributes: MappingProxyType[str, AttribDef]
    """The metadata for all attributes provided by this Geo."""

    nodes: int
    """The number of nodes in this Geo."""

    def __init__(self, attributes: MappingProxyType[str, AttribDef], nodes: int):
        self.attributes = attributes
        self.nodes = nodes

    @abstractmethod
    def __getitem__(self, name: str) -> NDArray:
        pass


@dataclass
class GEOSpec:
    """class to create geo spec files used by the ADRIO system to create geos"""
    id: str
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
