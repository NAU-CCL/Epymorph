import os
from csv import reader
from datetime import date
from typing import Mapping

from attr import dataclass
from numpy.typing import NDArray
from pandas import DataFrame, Series, read_csv

from epymorph.error import DataResourceException
from epymorph.geo.adrio.adrio import ADRIO, ADRIOMaker
from epymorph.geo.spec import (AttributeDef, SourceSpec, SpecificTimePeriod,
                               TimePeriod, Year)
from epymorph.geography.scope import GeoScope
from epymorph.geography.us_census import (CensusScope, CountyScope, StateScope,
                                          get_census_granularity,
                                          state_code_to_fips)


@dataclass
class FileSpec(SourceSpec):
    file_path: os.PathLike
    key_col: int
    data_col: int
    key_type: str
    header: int | None
    time_col: int | None


@dataclass
class FileSpecMatrix(FileSpec):
    key_col2: int


class ADRIOMakerCSV(ADRIOMaker):
    def make_adrio(self, attrib: AttributeDef, scope: GeoScope, time_period: TimePeriod, spec: FileSpec) -> ADRIO:
        if isinstance(spec, FileSpecMatrix):
            return self._make_matrix_adrio(attrib, scope, time_period, spec)
        else:
            return self._make_single_column_adrio(attrib, scope, time_period, spec)

    def _make_single_column_adrio(self, attrib: AttributeDef, scope: GeoScope, time_period: TimePeriod, spec: FileSpec) -> ADRIO:
        """Makes an ADRIO to fetch data from a single relevant column in a .csv file."""
        def fetch() -> NDArray:
            df = self._load_from_file(spec.file_path, spec, time_period, scope)

            df.rename(columns={spec.key_col: 'key'}, inplace=True)
            df.sort_values(by='key', inplace=True)

            data_values = df[spec.data_col]

            # check for null values (missing data in file)
            if data_values.isnull().any():
                msg = f"Data for required geographies missing from {attrib.name} attribute file or could not be found."
                raise DataResourceException(msg)

            return df[spec.data_col].to_numpy(dtype=attrib.dtype)

        return ADRIO(attrib.name, fetch)

    def _make_matrix_adrio(self, attrib: AttributeDef, scope: GeoScope, time_period: TimePeriod, spec: FileSpecMatrix) -> ADRIO:
        """Makes an ADRIO to fetch data from a single column within a .csv file and convert it to matrix format."""
        def fetch() -> NDArray:
            df = self._load_from_file(spec.file_path, spec, time_period, scope)

            df = df.pivot(index=spec.key_col, columns=spec.key_col2,
                          values=spec.data_col)

            df.sort_index(axis=0, inplace=True)
            df.sort_index(axis=1, inplace=True)

            df.fillna(0, inplace=True)

            return df.to_numpy(dtype=attrib.dtype)

        return ADRIO(attrib.name, fetch)

    def _load_from_file(self, path: os.PathLike, spec: FileSpec, time_period: TimePeriod, scope: GeoScope) -> DataFrame:
        """
        Loads .csv at path location into a pandas DataFrame, filtering out data outside of the specified
        geographic scope and time period.
        Returns a DataFrame with the resulting data.
        """
        if os.path.exists(spec.file_path):
            if spec.header is not None:
                df = read_csv(path, skiprows=spec.header,
                              header=None, dtype={spec.key_col: str})
            else:
                df = read_csv(path, header=None, dtype={spec.key_col: str})

            if spec.time_col is not None:
                df[spec.time_col] = df[spec.time_col].apply(date.fromisoformat)

                if isinstance(time_period, Year):
                    time_sort_key = date.fromisoformat(f'{time_period.year}-01-01')

                    df = df.loc[df[spec.time_col] == time_sort_key]

                elif isinstance(time_period, SpecificTimePeriod):
                    df = df.loc[df[spec.time_col] >
                                time_period.start_date and df[spec.time_col] < time_period.end_date]

            df = self._parse_label(spec.key_type, scope, df, spec.key_col)

            return df

        else:
            msg = f"File {spec.file_path} not found"
            raise DataResourceException(msg)

    def _parse_label(self, key_type: str, scope: GeoScope, df: DataFrame, key_col: int) -> DataFrame:
        """
        Reads labels from dataframe according to key type specified and replaces them 
        with a uniform value to sort by.
        Returns a dataframe with values replaced in the label column.
        """
        match (key_type):
            case "state":
                result = self._parse_abbrev(scope, df, key_col)

            case "county_state":
                result = self._parse_county_state(scope, df, key_col)

            case "geoid":
                result = self._parse_geoid(scope, df, key_col)

            case _:
                raise DataResourceException("Invalid key type specifier.")

        if isinstance(scope, CensusScope):
            self._validate_result(scope, result[key_col])

        return result

    def _parse_abbrev(self, scope: GeoScope, df: DataFrame, key_col: int) -> DataFrame:
        """
        Replaces values in label column containing state abreviations (i.e. AZ) with state
        fips codes and filters out any not in the specified geographic scope.
        """
        if isinstance(scope, StateScope):
            state_mapping = state_code_to_fips(scope.year)
            df[key_col] = [state_mapping.get(x) for x in df[key_col]]
            if df[key_col].isnull().any():
                raise DataResourceException("Invalid state code in key column.")
            df = df.loc[df[key_col].isin(scope.get_node_ids())]
            return df

        else:
            msg = "State scope is required to use state abbreviation label format."
            raise DataResourceException(msg)

    def _parse_county_state(self, scope: GeoScope, df: DataFrame, key_col: int) -> DataFrame:
        """
        Replaces values in label column containing county and state names (i.e. Maricopa County, Arizona)
        with state county fips codes and filters out any not in the specified geographic scope.
        """
        county_mapping = map_fips_to_county()
        df[key_col] = [county_mapping.get(x) for x in df[key_col]]
        if df[key_col].isnull().any():
            raise DataResourceException("Invalid county name in key column.")
        df = df.loc[df[key_col].isin(scope.get_node_ids())]

        return df

    def _parse_geoid(self, scope: GeoScope, df: DataFrame, key_col: int) -> DataFrame:
        """
        Replaces values in label column containing state abreviations (i.e. AZ) 
        with state fips codes and filters out any not in the specified geographic scope.
        """
        if not isinstance(scope, CensusScope):
            raise DataResourceException(
                "Census scope is required to use geoid label format.")

        validation = Series([get_census_granularity(
            scope.granularity).matches(x) for x in df[key_col]])

        if (validation == False).any():
            raise DataResourceException("Invalid geoid in key column.")

        df = df.loc[df[key_col].isin(scope.get_node_ids())]

        return df

    def _validate_result(self, scope: CensusScope, data: Series):
        """Ensures that key column for an attribute contains exactly one entry for every node in the scope."""
        if set(data) != set(scope.get_node_ids()):
            msg = "Key column missing keys for geographies in scope or contains unrecognized keys."
            raise DataResourceException(msg)


def map_county_to_fips(scope: GeoScope) -> Mapping[str, str]:
    """
    Uses local .csv file to read in and map county names to their geoids.
    Returns a string to string mapping of county fips name to fips code for all US counties.
    """
    if not isinstance(scope, CountyScope):
        msg = "County scope is required to use county, state label format."
        raise DataResourceException(msg)

    with open("./epymorph/data/geo/county_mapping.csv", 'r') as f:
        county_reader = reader(f)
        county_mapping = {county[1]: county[0] for county in county_reader}

    return county_mapping


def map_fips_to_county() -> Mapping[str, str]:
    """
    Uses local .csv file to read in and map county geoids to their names.
    Returns a string to string mapping of county fips code to name for all US counties.
    """
    with open("./epymorph/data/geo/county_mapping.csv", 'r') as f:
        county_reader = reader(f)
        county_mapping = {county[0]: county[1] for county in county_reader}

    return county_mapping
