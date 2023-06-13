import pickle
from abc import ABC, abstractmethod

from numpy.typing import NDArray


# abstract class to serve as an outline for individual ADRIO implementations
class ADRIO(ABC):
    attribute: str

    @abstractmethod
    def fetch(self, **kwargs) -> NDArray:
        pass


# class used to reference specific ADRIO implementations"""
class ADRIOSpec:
    class_name: str

    def __init__(self, class_name):
        self.class_name = class_name


# class to create geo spec files used by the ADRIO system to create geos
class GEOSpec:
    id: str
    nodes: list[str]
    adrios: list[ADRIOSpec]

    def __init__(self, id: str, nodes: list[str], adrios: list[ADRIOSpec]):
        self.id = id
        self.nodes = nodes
        self.adrios = adrios


# serializes a GEOSpec object
def serialize(spec: GEOSpec, file_path: str) -> None:
    with open(file_path, 'wb') as stream:
        pickle.dump(spec, stream)


# deserializes a GEOSpec object
def deserialize(file_path: str) -> GEOSpec:
    with open(file_path, 'rb') as stream:
        return pickle.load(stream)
