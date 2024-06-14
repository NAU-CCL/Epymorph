import os
from dataclasses import dataclass
from datetime import date
from typing import Literal

from numpy.typing import NDArray
from pandas import DataFrame, Series, read_csv

from epymorph.error import DataResourceException
from epymorph.geo.adrio.adrio import ADRIO, ADRIOMaker
from epymorph.geo.spec import (AttributeDef, SpecificTimePeriod, TimePeriod,
                               Year)
from epymorph.geography.scope import GeoScope
from epymorph.geography.us_census import (STATE, CensusScope, CountyScope,
                                          StateScope, get_census_granularity,
                                          get_us_counties, get_us_states,
                                          state_code_to_fips)

KeySpecifier = Literal['state_abbrev', 'county_state', 'geoid']


@dataclass
class CSVSpec():
    """Dataclass to store parameters for CSV ADRIO with data shape N."""
    file_path: os.PathLike
    key_col: int
    data_col: int
    key_type: KeySpecifier
    skiprows: int | None


@dataclass
class CSVSpecTime(CSVSpec):
    """Dataclass to store parameters for time-series CSV ADRIO with data shape TxN."""
    time_col: int


@dataclass
class CSVSpecMatrix():
    """Dataclass to store parameters for CSV ADRIO with data shape NxN."""
    file_path: os.PathLike
    from_key_col: int
    to_key_col: int
    data_col: int
    key_type: KeySpecifier
    skiprows: int | None


@dataclass
class CSVSpecMatrixTime(CSVSpecMatrix):
    """Dataclass to store parameters for time-series CSV ADRIO with data shape TxNxN."""

    time_col: int


class ADRIOMakerCSV(ADRIOMaker):
    def make_adrio(self, attrib: AttributeDef, scope: GeoScope, time_period: TimePeriod, spec: CSVSpec | CSVSpecMatrix) -> ADRIO:
        if isinstance(spec, CSVSpecMatrix):
            return self._make_matrix_adrio(attrib, scope, time_period, spec)
        else:
            return self._make_single_column_adrio(attrib, scope, time_period, spec)

    def _make_single_column_adrio(self, attrib: AttributeDef, scope: GeoScope, time_period: TimePeriod, spec: CSVSpec) -> ADRIO:
        """Makes an ADRIO to fetch data from a single relevant column in a .csv file."""
        def fetch() -> NDArray:
            df = self._load_from_file(spec, time_period, scope)

            df.rename(columns={spec.key_col: 'key'}, inplace=True)
            df.sort_values(by='key', inplace=True)

            data_values = df[spec.data_col]

            # check for null values (missing data in file)
            if data_values.isnull().any():
                msg = f"Data for required geographies missing from {attrib.name} attribute file or could not be found."
                raise DataResourceException(msg)

            return df[spec.data_col].to_numpy(dtype=attrib.dtype)

        return ADRIO(attrib.name, fetch)

    def _make_matrix_adrio(self, attrib: AttributeDef, scope: GeoScope, time_period: TimePeriod, spec: CSVSpecMatrix) -> ADRIO:
        """Makes an ADRIO to fetch data from a single column within a .csv file and converts it to matrix format."""
        def fetch() -> NDArray:
            df = self._load_from_file(spec, time_period, scope)

            df = df.pivot(index=spec.from_key_col, columns=spec.to_key_col,
                          values=spec.data_col)

            df.sort_index(axis=0, inplace=True)
            df.sort_index(axis=1, inplace=True)

            df.fillna(0, inplace=True)

            return df.to_numpy(dtype=attrib.dtype)

        return ADRIO(attrib.name, fetch)

    def _load_from_file(self, spec: CSVSpec | CSVSpecMatrix, time_period: TimePeriod, scope: GeoScope) -> DataFrame:
        """
        Loads .csv at path location into a pandas DataFrame, filtering out data outside of the specified
        geographic scope and time period.
        Returns a DataFrame with the resulting data.
        """
        path = spec.file_path
        if os.path.exists(path):
            if isinstance(spec, CSVSpec):
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

            if isinstance(spec, CSVSpecTime) or isinstance(spec, CSVSpecMatrixTime):
                df[spec.time_col] = df[spec.time_col].apply(date.fromisoformat)

                if isinstance(time_period, Year):
                    time_sort_key = date.fromisoformat(f'{time_period.year}-01-01')

                    df = df.loc[df[spec.time_col] == time_sort_key]

                elif isinstance(time_period, SpecificTimePeriod):
                    df = df.loc[df[spec.time_col] >
                                time_period.start_date and df[spec.time_col] < time_period.end_date]

            if isinstance(spec, CSVSpec):
                df = self._parse_label(spec.key_type, scope, df, spec.key_col)
            else:
                df = self._parse_label(spec.key_type, scope, df,
                                       spec.from_key_col, spec.to_key_col)

            return df

        else:
            msg = f"File {spec.file_path} not found"
            raise DataResourceException(msg)

    def _parse_label(self, key_type: KeySpecifier, scope: GeoScope, df: DataFrame, key_col: int, key_col2: int | None = None) -> DataFrame:
        """
        Reads labels from a dataframe according to key type specified and replaces them 
        with a uniform value to sort by.
        Returns dataframe with values replaced in the label column.
        """
        match (key_type):
            case "state_abbrev":
                result = self._parse_abbrev(scope, df, key_col, key_col2)

            case "county_state":
                result = self._parse_county_state(scope, df, key_col)

            case "geoid":
                result = self._parse_geoid(scope, df, key_col, key_col2)

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
            msg = "State scope is required to use state abbreviation key format."
            raise DataResourceException(msg)

    def _parse_county_state(self, scope: GeoScope, df: DataFrame, key_col: int) -> DataFrame:
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

    def _parse_geoid(self, scope: GeoScope, df: DataFrame, key_col: int, key_col2: int | None = None) -> DataFrame:
        """
        Replaces values in label column containing state abreviations (i.e. AZ) 
        with state fips codes and filters out any not in the specified geographic scope.
        """
        if not isinstance(scope, CensusScope):
            raise DataResourceException(
                "Census scope is required to use geoid key format.")

        validation = Series([get_census_granularity(
            scope.granularity).matches(x) for x in df[key_col]])

        if (validation == False).any():
            raise DataResourceException("Invalid geoid in key column.")

        df = df.loc[df[key_col].isin(scope.get_node_ids())]
        if key_col2 is not None:
            df = df.loc[df[key_col2].isin(scope.get_node_ids())]

        return df

    def _validate_result(self, scope: GeoScope, data: Series):
        """Ensures that key column for an attribute contains exactly one entry for every node in the scope."""
        if set(data) != set(scope.get_node_ids()):
            msg = "Key column missing keys for geographies in scope or contains unrecognized keys."
            raise DataResourceException(msg)
