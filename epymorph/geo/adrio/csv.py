import os
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from numpy import dtype
from numpy.typing import NDArray
from pandas import DataFrame, Series, read_csv

from epymorph.error import DataResourceException, GeoValidationException
from epymorph.geo.adrio.adrio2 import Adrio
from epymorph.geo.spec import SpecificTimePeriod, TimePeriod
from epymorph.geography.scope import GeoScope
from epymorph.geography.us_census import (STATE, CensusScope, CountyScope,
                                          StateScope, get_census_granularity,
                                          get_us_counties, get_us_states,
                                          state_code_to_fips)

KeySpecifier = Literal['state_abbrev', 'county_state', 'geoid']


@dataclass
class _BaseCSVSpec():
    file_path: os.PathLike
    key_col: int
    data_col: int
    data_type: dtype
    key_type: KeySpecifier
    skiprows: int | None


@dataclass
class CSVSpec(_BaseCSVSpec):
    """Dataclass to store parameters for CSV ADRIO with data shape N."""


@dataclass
class CSVSpecTime(_BaseCSVSpec):
    """Dataclass to store parameters for time-series CSV ADRIO with data shape TxN."""
    time_period: TimePeriod
    time_col: int


@dataclass
class _BaseCSVSpecMatrix():
    file_path: os.PathLike
    from_key_col: int
    to_key_col: int
    data_col: int
    data_type: dtype
    key_type: KeySpecifier
    skiprows: int | None


@dataclass
class CSVSpecMatrix(_BaseCSVSpecMatrix):
    """Dataclass to store parameters for CSV ADRIO with data shape NxN."""


def _single_column(scope: GeoScope, spec: CSVSpec | CSVSpecTime) -> NDArray[Any]:
    """Makes an ADRIO to fetch data from a single relevant column in a .csv file."""
    if spec.key_col == spec.data_col:
        msg = "Key column and data column must not be the same."
        raise GeoValidationException(msg)

    df = _load_from_file(spec, scope)

    # check for null values (missing data in file)
    if df[spec.data_col].isnull().any():
        msg = f"Data for required geographies missing from CSV file or could not be found."
        raise DataResourceException(msg)

    if isinstance(spec, CSVSpec):
        df.rename(columns={spec.key_col: 'key'}, inplace=True)
        df.sort_values(by='key', inplace=True)
        return df[spec.data_col].to_numpy(dtype=spec.data_type)
    else:
        if not isinstance(spec.time_period, SpecificTimePeriod):
            raise GeoValidationException("Unsupported time period.")

        df[spec.time_col] = df[spec.time_col].apply(date.fromisoformat)

        if any(df[spec.time_col] < spec.time_period.start_date) or any(df[spec.time_col] > spec.time_period.end_date):
            msg = "Found time column value(s) outside of provided date range."
            raise DataResourceException(msg)

        df.rename(columns={spec.key_col: 'key', spec.data_col: 'data',
                           spec.time_col: 'time'}, inplace=True)
        df.sort_values(by=['time', 'key'], inplace=True)
        df = df.pivot(index='time', columns='key', values='data')
        return df.to_numpy(dtype=spec.data_type)


def _matrix(scope: GeoScope, spec: CSVSpecMatrix) -> NDArray[Any]:
    """Makes an ADRIO to fetch data from a single column within a .csv file and converts it to matrix format."""
    if len({spec.from_key_col, spec.to_key_col, spec.data_col}) != 3:
        msg = "From key column, to key column, and data column must all be unique."
        raise GeoValidationException(msg)

    df = _load_from_file(spec, scope)

    df = df.pivot(index=spec.from_key_col, columns=spec.to_key_col,
                  values=spec.data_col)

    df.sort_index(axis=0, inplace=True)
    df.sort_index(axis=1, inplace=True)

    df.fillna(0, inplace=True)

    return df.to_numpy(dtype=spec.data_type)


def _load_from_file(spec: CSVSpec | CSVSpecTime | CSVSpecMatrix, scope: GeoScope) -> DataFrame:
    """
    Loads .csv at path location into a pandas DataFrame, filtering out data outside of the specified
    geographic scope and time period.
    Returns a DataFrame with the resulting data.
    """
    path = spec.file_path
    if os.path.exists(path):
        if isinstance(spec, CSVSpec | CSVSpecTime):
            if spec.skiprows is not None:
                df = read_csv(path, skiprows=spec.skiprows,
                              header=None, dtype={spec.key_col: str})
            else:
                df = read_csv(path, header=None, dtype={spec.key_col: str})
        else:
            if spec.skiprows is not None:
                df = read_csv(path, skiprows=spec.skiprows, header=None, dtype={
                    spec.from_key_col: str, spec.to_key_col: str})
            else:
                df = read_csv(path, header=None, dtype={
                    spec.from_key_col: str, spec.to_key_col: str})

        if isinstance(spec, CSVSpec | CSVSpecTime):
            df = _parse_label(spec.key_type, scope, df, spec.key_col)
        else:
            df = _parse_label(spec.key_type, scope, df,
                              spec.from_key_col, spec.to_key_col)

        return df

    else:
        msg = f"File {spec.file_path} not found"
        raise DataResourceException(msg)


def _parse_label(key_type: KeySpecifier, scope: GeoScope, df: DataFrame, key_col: int, key_col2: int | None = None) -> DataFrame:
    """
    Reads labels from a dataframe according to key type specified and replaces them 
    with a uniform value to sort by.
    Returns dataframe with values replaced in the label column.
    """
    match (key_type):
        case "state_abbrev":
            result = _parse_abbrev(scope, df, key_col, key_col2)

        case "county_state":
            result = _parse_county_state(scope, df, key_col)

        case "geoid":
            result = _parse_geoid(scope, df, key_col, key_col2)

    _validate_result(scope, result[key_col])

    return result


def _parse_abbrev(scope: GeoScope, df: DataFrame, key_col: int, key_col2: int | None = None) -> DataFrame:
    """
    Replaces values in label column containing state abreviations (i.e. AZ) with state
    fips codes and filters out any not in the specified geographic scope.
    """
    if isinstance(scope, StateScope):
        state_mapping = state_code_to_fips(scope.year)
        df[key_col] = [state_mapping.get(x) for x in df[key_col]]
        if df[key_col].isnull().any():
            raise DataResourceException("Invalid state code in key column.")
        df = df[df[key_col].isin(scope.get_node_ids())]
        if key_col2 is not None:
            df = df[df[key_col2].isin(scope.get_node_ids())]
        return df

    else:
        msg = "State scope is required to use state abbreviation key format."
        raise DataResourceException(msg)


def _parse_county_state(scope: GeoScope, df: DataFrame, key_col: int) -> DataFrame:
    """
    Replaces values in label column containing county and state names (i.e. Maricopa, Arizona)
    with state county fips codes and filters out any not in the specified geographic scope.
    """
    if not isinstance(scope, CountyScope):
        msg = "County scope is required to use county, state key format."
        raise DataResourceException(msg)

    # make counties info dataframe
    counties_info = get_us_counties(scope.year)
    counties_info_df = DataFrame(
        {'geoid': counties_info.geoid, 'name': counties_info.name})

    # make states info dataframe
    states_info = get_us_states(scope.year)
    states_info_df = DataFrame(
        {'state_geoid': states_info.geoid, 'state_name': states_info.name})

    # merge dataframes on state geoid
    counties_info_df['state_geoid'] = counties_info_df['geoid'].apply(
        STATE.truncate)
    counties_info_df = counties_info_df.merge(
        states_info_df, how='left', on='state_geoid')

    # concatenate county, state names
    counties_info_df['name'] = counties_info_df['name'] + \
        ", " + counties_info_df['state_name']

    # merge with csv dataframe and set key column to geoid
    df = df.merge(counties_info_df, how='left', left_on=key_col, right_on='name')
    df[key_col] = df['geoid']

    return df


def _parse_geoid(scope: GeoScope, df: DataFrame, key_col: int, key_col2: int | None = None) -> DataFrame:
    """
    Replaces values in label column containing state abreviations (i.e. AZ) 
    with state fips codes and filters out any not in the specified geographic scope.
    """
    if not isinstance(scope, CensusScope):
        msg = "Census scope is required to use geoid key format."
        raise DataResourceException(msg)

    granularity = get_census_granularity(scope.granularity)
    if not all(granularity.matches(x) for x in df[key_col]):
        raise DataResourceException("Invalid geoid in key column.")

    df = df[df[key_col].isin(scope.get_node_ids())]
    if key_col2 is not None:
        df = df[df[key_col2].isin(scope.get_node_ids())]

    return df


def _validate_result(scope: GeoScope, data: Series) -> None:
    """Ensures that key column for an attribute contains exactly one entry for every node in the scope."""
    if set(data) != set(scope.get_node_ids()):
        msg = "Key column missing keys for geographies in scope or contains unrecognized keys."
        raise DataResourceException(msg)


class CSV(Adrio[Any]):

    def __init__(self, csv_spec: CSVSpec | CSVSpecTime | CSVSpecMatrix) -> None:
        self.spec = csv_spec

    def evaluate(self) -> NDArray[Any]:
        if isinstance(self.spec, CSVSpec) or isinstance(self.spec, CSVSpecTime):
            return _single_column(self.scope, self.spec)
        else:
            return _matrix(self.scope, self.spec)
