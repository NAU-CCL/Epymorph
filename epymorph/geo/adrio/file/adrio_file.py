import os
from csv import reader

import numpy as np
from attr import dataclass
from numpy.typing import NDArray
from pandas import DataFrame, read_csv

from epymorph.error import DataResourceException
from epymorph.geo.adrio.adrio import ADRIO, ADRIOMaker
from epymorph.geo.spec import AttributeDef, SourceSpec, TimePeriod, Year
from epymorph.geography.scope import GeoScope
from epymorph.geography.us_census import (CountyScope, StateScope,
                                          state_fips_to_code)


@dataclass
class FileSpec(SourceSpec):
    file_path: os.PathLike
    label_key: str | int
    data_key: list[str] | list[int]
    label_type: str
    file_type: str
    header: int | None


@dataclass
class FileSpecTime(FileSpec):
    time_key: str | int


class ADRIOMakerFile(ADRIOMaker):

    def make_adrio(self, attrib: AttributeDef, scope: GeoScope, time_period: TimePeriod, spec: FileSpec) -> ADRIO:
        def fetch() -> NDArray:
            # check if file exists
            if os.path.exists(spec.file_path):
                path = spec.file_path

                sort_key = self.label_sort(spec.label_type, scope)

                time_sort_key = 0
                if isinstance(spec, FileSpecTime):
                    if isinstance(time_period, Year):
                        time_sort_key = time_period.year

                # read value from csv
                if spec.file_type == 'csv':
                    if isinstance(spec.label_key, int) and spec.header is not None:
                        df = read_csv(path, skiprows=spec.header, header=None)
                    else:
                        df = read_csv(path, header=spec.header)

                    if isinstance(spec.label_key, str) and spec.header is None:
                        msg = "Header row is required to get column attributes by name."
                        raise DataResourceException(msg)

                    df = df.loc[df[spec.label_key].isin(sort_key)]
                    if isinstance(spec, FileSpecTime):
                        df = df.loc[df[spec.time_key] == time_sort_key]
                    sort_df = DataFrame(sort_key, columns=[spec.label_key])

                    data_values = df[spec.data_key]
                    df = df.merge(sort_df, how='right')

                    # check for null values (missing data in file)
                    if data_values.isnull().any().any():
                        msg = f"Data for required geographies missing from {attrib.name} attribute file or could not be found."
                        raise DataResourceException(msg)

                    if len(spec.data_key) == 1:
                        return df[spec.data_key[0]].to_numpy(dtype=attrib.dtype)
                    else:
                        return df[spec.data_key].to_numpy(dtype=attrib.dtype)

                # read value from npz
                elif spec.file_type == 'npz':
                    return np.load(path)[attrib.name]

                # raise exception for any other file type
                else:
                    msg = "Invalid file type. Supported file types are .csv and .npz"
                    raise DataResourceException(msg)
            else:
                msg = f"File {spec.file_path} not found"
                raise DataResourceException(msg)

        return ADRIO(attrib.name, fetch)

    def label_sort(self, join: str, scope: GeoScope) -> list:
        """
        Creates sort key according to the type of label specified.
        Returns a list of labels sorted in order of the geo's label attribute.
        """
        if join == "state_abbr":  # ex 'AZ'
            if isinstance(scope, StateScope):
                state_mapping = state_fips_to_code(scope.year)
                states = [state_mapping.get(fips) for fips in scope.get_node_ids()]

                return states

            else:
                msg = "State scope is required to sort by state abbreviation."
                raise DataResourceException(msg)

        elif join == "county_state":  # ex "Maricopa County, Arizona"
            return get_county_from_fips(scope)

        else:
            msg = "Invalid label type specifier."
            raise DataResourceException(msg)


def get_county_from_fips(scope: GeoScope) -> list[str]:
    """
    Converts county fips codes from a census geography filter to county names.
    Returns a list of strings containing the name of each county in county, state format.
    """
    if not isinstance(scope, CountyScope):
        msg = "State or county scope is required to use county, state label format."
        raise DataResourceException(msg)

    with open("./epymorph/data/geo/county_mapping.csv", 'r') as f:
        county_reader = reader(f)
        county_mapping = {county[1]: county[0] for county in county_reader}

    county_list = []
    match scope:
        case CountyScope('state'):
            for state_fips in scope.get_node_ids():
                county_list.append(
                    [county for fips, county in county_mapping.items() if fips.startswith(state_fips)])
        case CountyScope('county'):
            for county_fips in scope.get_node_ids():
                county_list.append(county_mapping.get(county_fips))

    return list(np.concatenate(county_list).flat)


def get_fips_from_county(counties: list[str]) -> list[str]:
    """
    Converts a list of county names in county, state format to county fips codes.
    Returns a list of strings containing the fips code for each county provided.
    """
    with open("./epymorph/data/geo/county_mapping.csv", 'r') as f:
        county_reader = reader(f)
        county_mapping = {county[0]: county[1] for county in county_reader}

    fips_list = []
    for county in counties:
        fips_list.append(county_mapping.get(county))

    return fips_list
