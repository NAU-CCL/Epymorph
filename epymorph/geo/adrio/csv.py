import os
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
            result = _parse_county_state(scope, df, key_col, key_col2)

        case "geoid":
            result = _parse_geoid(scope, df, key_col, key_col2)

    _validate_result(scope, result[key_col])

    if key_col2 is not None:
        _validate_result(scope, result[key_col2])

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
            df[key_col2] = [state_mapping.get(x) for x in df[key_col2]]
            if df[key_col2].isnull().any():
                raise DataResourceException("Invalid state code in second key column.")
            df = df[df[key_col2].isin(scope.get_node_ids())]

        return df

    else:
        msg = "State scope is required to use state abbreviation key format."
        raise DataResourceException(msg)


def _parse_county_state(scope: GeoScope, df: DataFrame, key_col: int, key_col2: int | None = None) -> DataFrame:
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

    if key_col2 is not None:
        df = df.merge(counties_info_df, how='left', left_on=key_col2, right_on='name')
        df[key_col2] = df['geoid_y']

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
    """Ensures that the key column for an attribute contains at least one entry for every node in the scope."""
    if set(data) != set(scope.get_node_ids()):
        msg = "Key column missing keys for geographies in scope or contains unrecognized keys."
        raise DataResourceException(msg)


class CSV(Adrio[Any]):
    """Retrieves an N-shaped array of any type from a user-provided CSV file."""

    key_type: KeySpecifier

    def __init__(self, file_path: os.PathLike, key_col: int, data_col: int, data_type: dtype, key_type: KeySpecifier, skiprows: int | None):
        self.file_path = file_path
        """The path to the CSV file containing data."""
        self.key_col = key_col
        """Numerical index of the column containing information to identify geographies."""
        self.data_col = data_col
        """Numerical index of the column containing the data of interest."""
        self.data_type = data_type
        """The data type of values in the data column."""
        self.key_type = key_type
        """The type of geographic identifier in the key column."""
        self.skiprows = skiprows
        """Number of header rows in the file to be skipped."""

    def evaluate(self) -> NDArray[Any]:

        if self.key_col == self.data_col:
            msg = "Key column and data column must not be the same."
            raise GeoValidationException(msg)

        path = self.file_path
        # workaround for bad pandas type definitions
        skiprows: int = self.skiprows  # type: ignore
        if os.path.exists(path):
            df = read_csv(path, skiprows=skiprows,
                          header=None, dtype={self.key_col: str})
            df = _parse_label(self.key_type, self.scope, df, self.key_col)

            if df[self.data_col].isnull().any():
                msg = "Data for required geographies missing from CSV file or could not be found."
                raise DataResourceException(msg)

            df.rename(columns={self.key_col: 'key'}, inplace=True)
            df.sort_values(by='key', inplace=True)
            return df[self.data_col].to_numpy(dtype=self.data_type)

        else:
            msg = f"File {self.file_path} not found"
            raise DataResourceException(msg)


class CSVTimeSeries(Adrio[Any]):
    """Retrieves a TxN-shaped array of any type from a user-provided CSV file."""

    key_type: KeySpecifier

    def __init__(self, file_path: os.PathLike, key_col: int, data_col: int, data_type: dtype, key_type: KeySpecifier, skiprows: int | None, time_period: TimePeriod, time_col: int):
        self.file_path = file_path
        """The path to the CSV file containing data."""
        self.key_col = key_col
        """Numerical index of the column containing information to identify geographies."""
        self.data_col = data_col
        """Numerical index of the column containing the data of interest."""
        self.data_type = data_type
        """The data type of values in the data column."""
        self.key_type = key_type
        """The type of geographic identifier in the key column."""
        self.skiprows = skiprows
        """Number of header rows in the file to be skipped."""
        self.time_period = time_period
        """The time period encompassed by data in the file."""
        self.time_col = time_col
        """The numerical index of the column containing time information."""

    def evaluate(self) -> NDArray[Any]:

        if self.key_col == self.data_col:
            msg = "Key column and data column must not be the same."
            raise GeoValidationException(msg)

        path = self.file_path
        skiprows: int = self.skiprows  # type: ignore
        if os.path.exists(path):
            df = read_csv(path, skiprows=skiprows,
                          header=None, dtype={self.key_col: str})
            df = _parse_label(self.key_type, self.scope, df, self.key_col)

            if df[self.data_col].isnull().any():
                msg = "Data for required geographies missing from CSV file or could not be found."
                raise DataResourceException(msg)

            if not isinstance(self.time_period, SpecificTimePeriod):
                raise GeoValidationException("Unsupported time period.")

            df[self.time_col] = df[self.time_col].apply(date.fromisoformat)

            if any(df[self.time_col] < self.time_period.start_date) or any(df[self.time_col] > self.time_period.end_date):
                msg = "Found time column value(s) outside of provided date range."
                raise DataResourceException(msg)

            df.rename(columns={self.key_col: 'key', self.data_col: 'data',
                               self.time_col: 'time'}, inplace=True)
            df.sort_values(by=['time', 'key'], inplace=True)
            df = df.pivot(index='time', columns='key', values='data')
            return df.to_numpy(dtype=self.data_type)

        else:
            msg = f"File {self.file_path} not found"
            raise DataResourceException(msg)


class CSVMatrix(Adrio[Any]):
    """Recieves an NxN-shaped array of any type from a user-provided CSV file."""

    key_type: KeySpecifier

    def __init__(self, file_path: os.PathLike, from_key_col: int, to_key_col: int, data_col: int, data_type: dtype, key_type: KeySpecifier, skiprows: int | None):
        self.file_path = file_path
        """The path to the CSV file containing data."""
        self.from_key_col = from_key_col
        """Numerical index of the column containing information to identify source geographies."""
        self.to_key_col = to_key_col
        """Numerical index of the column containing information to identify destination geographies."""
        self.data_col = data_col
        """Numerical index of the column containing the data of interest."""
        self.data_type = data_type
        """The data type of values in the data column."""
        self.key_type = key_type
        """The type of geographic identifier in the key columns."""
        self.skiprows = skiprows
        """Number of header rows in the file to be skipped."""

    def evaluate(self) -> NDArray[Any]:

        if len({self.from_key_col, self.to_key_col, self.data_col}) != 3:
            msg = "From key column, to key column, and data column must all be unique."
            raise GeoValidationException(msg)

        path = self.file_path
        skiprows: int = self.skiprows  # type: ignore
        if os.path.exists(path):
            df = read_csv(path, skiprows=skiprows, header=None, dtype={
                self.from_key_col: str, self.to_key_col: str})
            df = _parse_label(self.key_type, self.scope, df,
                              self.from_key_col, self.to_key_col)

            df = df.pivot(index=self.from_key_col,
                          columns=self.to_key_col, values=self.data_col)

            df.sort_index(axis=0, inplace=True)
            df.sort_index(axis=1, inplace=True)

            df.fillna(0, inplace=True)

            return df.to_numpy(dtype=self.data_type)

        else:
            msg = f"File {self.file_path} not found"
            raise DataResourceException(msg)
