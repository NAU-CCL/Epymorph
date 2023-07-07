import json
import os
from abc import ABC, abstractmethod

import jsonpickle
from attr import dataclass
from census import Census
from genericpath import isfile
from numpy.typing import NDArray
from pandas import DataFrame, concat, read_csv


class ADRIO(ABC):
    """abstract class to serve as an outline for individual ADRIO implementations"""
    attribute: str
    census: Census
    year: int

    def __init__(self):
        """
        initializer to create Census object
        TODO: move to "census" ADRIO template
        """
        self.census = Census(os.environ['CENSUS_API_KEY'])

    @abstractmethod
    def fetch(self, **kwargs) -> NDArray:
        pass

    def type_check(self, args: dict) -> list[str]:
        """
        type checks the 'nodes' argument to make sure data was passed in correctly
        * functionality now included in cache_fetch - obsolete?
        TODO: move to "census" ADRIO template
        """
        nodes = args.get('nodes')
        if type(nodes) is list:
            return nodes
        else:
            msg = 'nodes parameter is not formatted correctly; must be a list of strings'
            raise Exception(msg)

    def cache_fetch(self, args: dict, attribute: str) -> tuple[list[str], DataFrame]:
        # csv file name components
        nodes = args.get('nodes')
        attr = attribute
        year = str(self.year)

        data = DataFrame()
        num_cached = 0

        if type(nodes) is list:
            uncached = []
            for i in nodes:
                # create csv file path (attribute + node GEOID + year)
                string = attr + i + year
                path = f'epymorph/adrio/.cache/{string}.csv'
                # check for csv file
                if os.path.isfile(path):
                    # retrieve cached data
                    num_cached += 1
                    curr_data = read_csv(path)
                    data = concat([data, curr_data])
                # append node to uncached list
                else:
                    uncached.append(i)

            print(f'{num_cached} items retrieved from cache')
            # return list of uncached GEOIDs and retrieved cached data
            return uncached, data

        else:
            msg = 'nodes parameter is not formatted correctly; must be a list of strings'
            raise Exception(msg)

    def cache_store(self, data: DataFrame, nodes: list[str], attribute: str) -> None:
        attr = attribute
        year = str(self.year)
        # create .cache file if needed
        if not os.path.isdir('epymorph/adrio/.cache'):
            os.mkdir('epymorph/adrio/.cache')

        # loop through nodes and cache data for each
        for i in nodes:
            string = attr + i + year
            data.loc[data['state'] == i].to_csv(
                f'epymorph/adrio/.cache/{string}.csv')


@dataclass
class ADRIOSpec:
    """class used to reference specific ADRIO implementations"""
    class_name: str


@dataclass
class GEOSpec:
    """class to create geo spec files used by the ADRIO system to create geos"""
    id: str
    nodes: list[str]
    adrios: list[ADRIOSpec]


def serialize(spec: GEOSpec, file_path: str) -> None:
    """serializes a GEOSpec object to a file at the given path"""
    json_spec = str(jsonpickle.encode(spec, unpicklable=True))
    with open(file_path, 'w') as stream:
        stream.write(json_spec)


def deserialize(file_path: str) -> GEOSpec:
    """deserializes a GEOSpec object from a file at the given path"""
    with open(file_path, 'r') as stream:
        spec = stream.readline()
    spec_dec = jsonpickle.decode(spec)

    # ensure decoded opject is of type GEOSpec
    if type(spec_dec) is GEOSpec:
        return spec_dec
    else:
        msg = f'{file_path} does not decode to GEOSpec object; ensure file path is correct and file is correctly formatted'
        raise Exception(msg)
