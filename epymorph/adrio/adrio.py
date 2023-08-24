from abc import ABC, abstractmethod

import jsonpickle
from attr import dataclass
from numpy.typing import NDArray


class ADRIO(ABC):
    """abstract class to serve as an outline for individual ADRIO implementations"""
    attribute: str

    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def fetch(self, **kwargs) -> NDArray:
        pass


@dataclass
class ADRIOSpec:
    """class used to reference specific ADRIO implementations"""
    class_name: str


@dataclass
class GEOSpec:
    """class to create geo spec files used by the ADRIO system to create geos"""
    id: str
    granularity: int
    nodes: dict[str, list[str]]
    label: ADRIOSpec
    adrios: list[ADRIOSpec]
    year: int


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
