from abc import ABC, abstractmethod

import jsonpickle
from attr import dataclass
from numpy.typing import NDArray


# abstract class to serve as an outline for individual ADRIO implementations
class ADRIO(ABC):
    attribute: str

    @abstractmethod
    def fetch(self, **kwargs) -> NDArray:
        pass


# class used to reference specific ADRIO implementations"""
@dataclass
class ADRIOSpec:
    class_name: str


# class to create geo spec files used by the ADRIO system to create geos
@dataclass
class GEOSpec:
    id: str
    nodes: list[str]
    adrios: list[ADRIOSpec]


# serializes a GEOSpec object
def serialize(spec: GEOSpec, file_path: str) -> None:
    json_spec = str(jsonpickle.encode(spec, unpicklable=True))
    with open(file_path, 'w') as stream:
        stream.write(json_spec)


# deserializes a GEOSpec object
def deserialize(file_path: str) -> GEOSpec:
    with open(file_path, 'r') as stream:
        spec = stream.readline()
    spec_dec = jsonpickle.decode(spec)

    # ensure decoded opject is of type GEOSpec
    if type(spec_dec) is GEOSpec:
        return spec_dec
    else:
        msg = f'{file_path} does not decode to GEOSpec object; ensure file path is correct and file is correctly formatted'
        raise Exception(msg)
