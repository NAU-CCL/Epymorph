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
    """Dataclass to store parameters for CSV ADRIO with data shape N."""
    file_path: os.PathLike
    key_col: int
    data_col: int
    key_type: str
    skiprows: int | None


@dataclass
class FileSpecTime(FileSpec):
    """Dataclass to store parameters for time-series CSV ADRIO with data shape TxN."""
    time_col: int


@dataclass
class FileSpecMatrix(SourceSpec):
    """Dataclass to store parameters for CSV ADRIO with data shape NxN."""
    file_path: os.PathLike
    from_key_col: int
    to_key_col: int
    data_col: int
    key_type: str
    skiprows: int | None


@dataclass
class FileSpecMatrixTime(FileSpecMatrix):
    """Dataclass to store parameters for time-series CSV ADRIO with data shape TxNxN."""

    time_col: int


class ADRIOMakerCSV(ADRIOMaker):
    def make_adrio(self, attrib: AttributeDef, scope: GeoScope, time_period: TimePeriod, spec: FileSpec | FileSpecMatrix) -> ADRIO:
        if isinstance(spec, FileSpecMatrix):
            return self._make_matrix_adrio(attrib, scope, time_period, spec)
        else:
            return self._make_single_column_adrio(attrib, scope, time_period, spec)

    def _make_single_column_adrio(self, attrib: AttributeDef, scope: GeoScope, time_period: TimePeriod, spec: FileSpec) -> ADRIO:
        """Makes an ADRIO to fetch data from a single relevant column in a .csv file."""
        def fetch() -> NDArray:
            if os.path.exists(spec.file_path):
                if spec.skiprows is not None:
                    df = read_csv(spec.file_path, skiprows=spec.skiprows,
                                  header=None, dtype={spec.key_col: str})
                else:
                    df = read_csv(spec.file_path, header=None,
                                  dtype={spec.key_col: str})

                if isinstance(spec, FileSpecTime) or isinstance(spec, FileSpecMatrixTime):
                    df[spec.time_col] = df[spec.time_col].apply(date.fromisoformat)

                    if isinstance(time_period, Year):
                        time_sort_key = date.fromisoformat(f'{time_period.year}-01-01')

                        df = df.loc[df[spec.time_col] == time_sort_key]

                    elif isinstance(time_period, SpecificTimePeriod):
                        df = df.loc[df[spec.time_col] >
                                    time_period.start_date and df[spec.time_col] < time_period.end_date]

                df = self._parse_label(spec.key_type, scope, df, spec.key_col)

                df.rename(columns={spec.key_col: 'key'}, inplace=True)
                df.sort_values(by='key', inplace=True)

                data_values = df[spec.data_col]

                # check for null values (missing data in file)
                if data_values.isnull().any():
                    msg = f"Data for required geographies missing from {attrib.name} attribute file or could not be found."
                    raise DataResourceException(msg)

                return df[spec.data_col].to_numpy(dtype=attrib.dtype)

            else:
                msg = f"File {spec.file_path} not found"
                raise DataResourceException(msg)

        return ADRIO(attrib.name, fetch)

    def _make_matrix_adrio(self, attrib: AttributeDef, scope: GeoScope, time_period: TimePeriod, spec: FileSpecMatrix) -> ADRIO:
        """Makes an ADRIO to fetch data from a single column within a .csv file and converts it to matrix format."""
        def fetch() -> NDArray:
            if os.path.exists(spec.file_path):
                if spec.skiprows is not None:
                    df = read_csv(spec.file_path, skiprows=spec.skiprows,
                                  header=None, dtype={spec.from_key_col: str, spec.to_key_col: str})
                else:
                    df = read_csv(spec.file_path, header=None,
                                  dtype={spec.from_key_col: str, spec.to_key_col: str})

                if isinstance(spec, FileSpecTime) or isinstance(spec, FileSpecMatrixTime):
                    df[spec.time_col] = df[spec.time_col].apply(date.fromisoformat)

                    if isinstance(time_period, Year):
                        time_sort_key = date.fromisoformat(f'{time_period.year}-01-01')

                        df = df.loc[df[spec.time_col] == time_sort_key]

                    elif isinstance(time_period, SpecificTimePeriod):
                        df = df.loc[df[spec.time_col] >
                                    time_period.start_date and df[spec.time_col] < time_period.end_date]

                df = self._parse_label(spec.key_type, scope, df,
                                       spec.from_key_col, spec.to_key_col)

                df = df.pivot(index=spec.from_key_col, columns=spec.to_key_col,
                              values=spec.data_col)

                df.sort_index(axis=0, inplace=True)
                df.sort_index(axis=1, inplace=True)

                df.fillna(0, inplace=True)

                return df.to_numpy(dtype=attrib.dtype)

            else:
                msg = f"File {spec.file_path} not found"
                raise DataResourceException(msg)

        return ADRIO(attrib.name, fetch)

    def _parse_label(self, key_type: str, scope: GeoScope, df: DataFrame, key_col: int, key_col2: int | None = None) -> DataFrame:
        """
        Reads labels from a dataframe according to key type specified and replaces them 
        with a uniform value to sort by.
        Returns dataframe with values replaced in the label column.
        """
        match (key_type):
            case "state_abbrev":
                result = self._parse_abbrev(scope, df, key_col, key_col2)

            case "county_state":
                result = self._parse_county_state(scope, df, key_col, key_col2)

            case "geoid":
                result = self._parse_geoid(scope, df, key_col, key_col2)

            case _:
                raise DataResourceException("Invalid key type specifier.")

        if isinstance(scope, CensusScope):
            self._validate_result(scope, result[key_col])

        return result

    def _parse_abbrev(self, scope: GeoScope, df: DataFrame, key_col: int, key_col2: int | None = None) -> DataFrame:
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
            if key_col2 is not None:
                df = df.loc[df[key_col2].isin(scope.get_node_ids())]
            return df

        else:
            msg = "State scope is required to use state abbreviation label format."
            raise DataResourceException(msg)

    def _parse_county_state(self, scope: GeoScope, df: DataFrame, key_col: int, key_col2: int | None = None) -> DataFrame:
        """
        Replaces values in label column containing county and state names (i.e. Maricopa County, Arizona)
        with state county fips codes and filters out any not in the specified geographic scope.
        """
        county_mapping = map_fips_to_county()
        df[key_col] = [county_mapping.get(x) for x in df[key_col]]
        if df[key_col].isnull().any():
            raise DataResourceException("Invalid county name in key column.")
        df = df.loc[df[key_col].isin(scope.get_node_ids())]
        if key_col2 is not None:
            df = df.loc[df[key_col2].isin(scope.get_node_ids())]

        return df

    def _parse_geoid(self, scope: GeoScope, df: DataFrame, key_col: int, key_col2: int | None = None) -> DataFrame:
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
        if key_col2 is not None:
            df = df.loc[df[key_col2].isin(scope.get_node_ids())]

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

    with open("./epymorph/data/geo/csv/county_mapping.csv", 'r') as f:
        county_reader = reader(f)
        county_mapping = {county[1]: county[0] for county in county_reader}

    return county_mapping


def map_fips_to_county() -> Mapping[str, str]:
    """
    Uses local .csv file to read in and map county geoids to their names.
    Returns a string to string mapping of county fips code to name for all US counties.
    """
    with open("./epymorph/data/geo/csv/county_mapping.csv", 'r') as f:
        county_reader = reader(f)
        county_mapping = {county[0]: county[1] for county in county_reader}

    return county_mapping
