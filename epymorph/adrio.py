from abc import ABC, abstractmethod

import jsonpickle
import pandas as pd
from attr import dataclass
from numpy.typing import NDArray


# abstract class to serve as an outline for individual ADRIO implementations
class ADRIO(ABC):
    attribute: str

    @abstractmethod
    def fetch(self, **kwargs) -> NDArray:
        pass

    # formats geo codes to be usable by census library function
    # TODO: move to "census" ADRIO template
    def format_geo_codes(self, args: dict) -> str:
        nodes = args.get('nodes')
        code_string = ''
        if type(nodes) is list:
            for i in range(len(nodes)):
                if i < len(nodes) - 1:
                    code_string += (nodes[i] + ',')
                else:
                    code_string += nodes[i]
            return code_string
        else:
            msg = 'nodes parameter is not formatted correctly; must be a list of strings'
            raise Exception(msg)

    # sort census data by state and county fips codes
    # TODO: move to "census" ADRIO template
    def sort_counties(self, data: list[dict]) -> list[list]:
        dataframe = pd.DataFrame.from_records(data)
        dataframe = dataframe.sort_values(by=['state', 'county'])
        return dataframe.values.tolist()


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
