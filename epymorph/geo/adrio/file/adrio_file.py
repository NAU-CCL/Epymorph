import os
from csv import reader

import numpy as np
from attr import dataclass
from numpy.typing import NDArray
from pandas import DataFrame, read_csv
from us import STATES
from us.states import lookup

from epymorph.error import GeoValidationException
from epymorph.geo.adrio.adrio import ADRIO, ADRIOMaker
from epymorph.geo.adrio.census.adrio_census import CensusGeography
from epymorph.geo.spec import (AttributeDef, Geography, SourceSpec, TimePeriod,
                               Year)


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

    def make_adrio(self, attrib: AttributeDef, geography: Geography, time_period: TimePeriod, spec: FileSpec) -> ADRIO:
        def fetch() -> NDArray:
            # check if file exists
            if os.path.exists(spec.file_path):
                path = spec.file_path

                sort_key = self.label_sort(spec.label_type, geography)

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
                        raise GeoValidationException(msg)

                    df = df.loc[df[spec.label_key].isin(sort_key)]
                    if isinstance(spec, FileSpecTime):
                        df = df.loc[df[spec.time_key] == time_sort_key]
                    sort_df = DataFrame(sort_key, columns=[spec.label_key])

                    data_values = df[spec.data_key]
                    df = df.merge(sort_df, how='right')

                    # check for null values (missing data in file)
                    if data_values.isnull().any().any():
                        msg = f"Data for required geographies missing from {attrib.name} attribute file or could not be found."
                        raise GeoValidationException(msg)
                    if len(spec.data_key) == 1:
                        return df[spec.data_key[0]].to_numpy()
                    else:
                        return df[spec.data_key].to_numpy()

                # read value from npz
                elif spec.file_type == 'npz':
                    return np.load(path)[attrib.name]

                # raise exception for any other file type
                else:
                    msg = "Invalid file type. Supported file types are .csv and .npz"
                    raise Exception(msg)
            else:
                msg = f"File {spec.file_path} not found"
                raise Exception(msg)

        return ADRIO(attrib.name, fetch)

    def label_sort(self, join: str, geography: Geography) -> list[str]:
        """
        Creates sort key according to the type of label specified.
        Returns a list of labels sorted in order of the geo's label attribute.
        """
        if join == "state_abbr":  # ex 'AZ'
            if isinstance(geography, CensusGeography):
                states = list()
                state_fips = geography.filter.get('state')
                if state_fips is not None:
                    if state_fips[0] == '*':
                        states = [state.abbr for state in STATES]
                    else:
                        for fip in state_fips:
                            state = lookup(fip)
                            if state is not None:
                                states.append(state.abbr)
                return states

            else:
                msg = "Census geography is required to sort by state abbreviation."
                raise GeoValidationException(msg)

        elif join == "county_state":  # ex "Maricopa County, Arizona"
            if isinstance(geography, CensusGeography):
                return get_county_from_fips(geography)
            else:
                msg = "Census geography is required to sort by county, state pair."
                raise GeoValidationException(msg)

        else:
            msg = "Invalid label type specifier."
            raise GeoValidationException(msg)


def get_county_from_fips(geography: CensusGeography) -> list[str]:
    """
    Converts county fips codes from a census geography filter to county names.
    Returns a list of strings containing the name of each county in county, state format.
    """
    state_filter = geography.filter.get("state")
    county_filter = geography.filter.get("county")
    if state_filter is None or county_filter is None:
        raise Exception("State or county fips missing from geography.")

    county_mapping = {}
    county_list = []
    with open("./epymorph/data/geo/county_mapping.csv", 'r') as f:
        county_reader = reader(f)
        for county in county_reader:
            county_mapping[county[1]] = county[0]

    if state_filter[0] == '*':
        county_list = list(county_mapping.values())
    elif county_filter[0] == '*':
        for state_fips in state_filter:
            county_list.append(
                [county for fips, county in county_mapping.items() if fips.startswith(state_fips)])
    else:
        for state_fips in state_filter:
            for county_fips in county_filter:
                county_list.append(county_mapping.get(f"{state_fips}{county_fips}"))
    return list(np.concatenate(county_list).flat)


def get_fips_from_county(counties: list[str]) -> list[str]:
    """
    Converts a list of county names in county, state format to county fips codes.
    Returns a list of strings containing the fips code for each county provided.
    """
    county_mapping = {}
    fips_list = []
    with open("./epymorph/data/geo/county_mapping.csv", 'r') as f:
        county_reader = reader(f)
        for county in county_reader:
            county_mapping[county[0]] = county[1]

    if county[0] == '*':
        fips_list = list(county_mapping.values())
    else:
        for county in counties:
            fips_list.append(county_mapping.get(county))
    return fips_list
