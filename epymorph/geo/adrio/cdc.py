from datetime import date
from typing import NamedTuple
from urllib.parse import quote, urlencode
from warnings import warn

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, concat, read_csv

from epymorph.error import DataResourceException
from epymorph.geo.adrio.adrio2 import Adrio
from epymorph.geo.spec import SpecificTimePeriod
from epymorph.geography.scope import GeoScope
from epymorph.geography.us_census import (STATE, CensusGranularityName,
                                          CensusScope, get_us_states,
                                          state_fips_to_code)


class QueryInfo(NamedTuple):
    url_base: str
    date_col: str
    fips_col: str
    data_col: str
    state_level: bool = False
    """Whether we are querying a dataset reporting state-level data."""


def _fetch_cases(attrib_name: str, scope: CensusScope, time_period: SpecificTimePeriod) -> NDArray[np.float64]:
    """
    Fetches data from HealthData dataset reporting COVID-19 cases per 100k population.
    Available between 2/24/2022 and 5/4/2023 at state and county granularities.
    https://healthdata.gov/dataset/United-States-COVID-19-Community-Levels-by-County/nn5b-j5u9/about_data
    """
    if time_period.start_date <= date(2022, 2, 17) or time_period.end_date >= date(2023, 5, 11):
        msg = "COVID cases data is only available between 2/24/2022 and 5/4/2023."
        raise DataResourceException(msg)

    info = QueryInfo("https://data.cdc.gov/resource/3nnm-4jni.csv?",
                     "date_updated", "county_fips", attrib_name)

    df = _api_query(info, scope.get_node_ids(),
                    time_period, scope.granularity)

    df.rename(columns={'county_fips': 'fips'}, inplace=True)

    if scope.granularity == 'state':
        df['fips'] = [STATE.extract(x) for x in df['fips']]

        df = df.groupby(['date_updated', 'fips']).sum()
        df.reset_index(inplace=True)

    df = df.pivot(index='date_updated', columns='fips', values=info.data_col)

    dates = df.index.to_numpy(dtype='datetime64[D]')

    return np.array([
        list(zip(dates, df[col]))
        for col in df.columns
    ], dtype=[('date', 'datetime64[D]'), ('data', np.float64)]).T


def _fetch_facility_hospitalization(attrib_name: str, scope: CensusScope, time_period: SpecificTimePeriod) -> NDArray[np.float64]:
    """
    Fetches data from HealthData dataset reporting number of people hospitalized for COVID-19 
    and other respiratory illnesses at facility level during manditory reporting period.
    Available between 12/13/2020 and 5/10/2023 at state and county granularities.
    https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/anag-cw7u/about_data
    """
    if time_period.start_date <= date(2020, 12, 6) or time_period.end_date >= date(2023, 5, 17):
        msg = "Facility level hospitalization data is only available between 12/13/2020 and 5/10/2023."
        raise DataResourceException(msg)

    info = QueryInfo("https://healthdata.gov/resource/anag-cw7u.csv?",
                     "collection_week", "fips_code", attrib_name)

    df = _api_query(info, scope.get_node_ids(),
                    time_period, scope.granularity)

    if scope.granularity == 'state':
        df['fips_code'] = [STATE.extract(x) for x in df['fips_code']]

    # if the sentinel value '-999999' appears in the data, ensure aggregated value is also -999999
    df['is_sentinel'] = df[info.data_col] == -999999
    df = df.groupby(['collection_week', 'fips_code']).agg(
        {info.data_col: 'sum', 'is_sentinel': any})
    df.loc[df['is_sentinel'], info.data_col] = -999999

    df.reset_index(inplace=True)
    df = df.pivot(index='collection_week',
                  columns='fips_code', values=info.data_col)

    dates = df.index.to_numpy(dtype='datetime64[D]')

    return np.array([
        list(zip(dates, df[col]))
        for col in df.columns
    ], dtype=[('date', 'datetime64[D]'), ('data', np.float64)]).T


def _fetch_state_hospitalization(attrib_name: str, scope: CensusScope, time_period: SpecificTimePeriod) -> NDArray[np.float64]:
    """
    Fetches data from CDC dataset reporting number of people hospitalized for COVID-19 
    and other respiratory illnesses at state level during manditory and voluntary reporting periods.
    Available from 1/4/2020 to present at state granularity. Data reported voluntarily past 5/1/2024.
    https://data.cdc.gov/Public-Health-Surveillance/Weekly-United-States-Hospitalization-Metrics-by-Ju/aemt-mg7g/about_data
    """
    if scope.granularity != 'state':
        msg = "State level hospitalization data can only be retrieved for state granularity."
        raise DataResourceException(msg)
    if time_period.start_date <= date(2019, 12, 29):
        msg = "State level hospitalization data is only available starting 1/4/2020."
        raise DataResourceException(msg)

    if time_period.end_date >= date(2024, 5, 1):
        warn("State level hospitalization data is voluntary past 5/1/2024.")

    info = QueryInfo("https://data.cdc.gov/resource/aemt-mg7g.csv?",
                     "week_end_date", "jurisdiction", attrib_name, True)

    state_mapping = state_fips_to_code(scope.year)
    fips = scope.get_node_ids()
    state_codes = np.array([state_mapping[x] for x in fips])

    df = _api_query(info, state_codes, time_period, scope.granularity)

    df = df.groupby(['week_end_date', 'jurisdiction']).sum()
    df.reset_index(inplace=True)
    df = df.pivot(index='week_end_date',
                  columns='jurisdiction', values=info.data_col)

    dates = df.index.to_numpy(dtype='datetime64[D]')

    return np.array([
        list(zip(dates, df[col]))
        for col in df.columns
    ], dtype=[('date', 'datetime64[D]'), ('data', np.float64)]).T


def _fetch_vaccination(attrib_name: str, scope: CensusScope, time_period: SpecificTimePeriod) -> NDArray[np.float64]:
    """
    Fetches data from CDC dataset reporting total COVID-19 vaccination numbers.
    Available between 12/13/2020 and 5/10/2024 at state and county granularities.
    https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh/about_data
    """
    if time_period.start_date <= date(2020, 12, 6) or time_period.end_date >= date(2024, 5, 17):
        msg = "Vaccination data is only available between 12/13/2020 and 5/10/2024."
        raise DataResourceException(msg)

    info = QueryInfo("https://data.cdc.gov/resource/8xkx-amqh.csv?",
                     "date", "fips", attrib_name)

    df = _api_query(info, scope.get_node_ids(),
                    time_period, scope.granularity)

    df = df.pivot(index='date', columns='fips', values=info.data_col)

    dates = df.index.to_numpy(dtype='datetime64[D]')

    return np.array([
        list(zip(dates, df[col]))
        for col in df.columns
    ], dtype=[('date', 'datetime64[D]'), ('data', np.float64)]).T


def _fetch_deaths_county(attrib_name: str, scope: CensusScope, time_period: SpecificTimePeriod) -> NDArray[np.float64]:
    """
    Fetches data from CDC dataset reporting number of deaths from COVID-19.
    Available between 1/4/2020 and 4/5/2024 at state and county granularities.
    https://data.cdc.gov/NCHS/AH-COVID-19-Death-Counts-by-County-and-Week-2020-p/ite7-j2w7/about_data
    """
    if time_period.start_date <= date(2019, 12, 28) or time_period.end_date >= date(2024, 4, 12):
        msg = "County level deaths data is only available between 1/4/2020 and 4/5/2024."
        raise DataResourceException(msg)

    if scope.granularity == 'state':
        info = QueryInfo("https://data.cdc.gov/resource/ite7-j2w7.csv?",
                         "week_ending_date", "stfips", attrib_name, True)
    else:
        info = QueryInfo("https://data.cdc.gov/resource/ite7-j2w7.csv?",
                         "week_ending_date", "fips_code", attrib_name)

    df = _api_query(info, scope.get_node_ids(),
                    time_period, scope.granularity)

    if scope.granularity == 'state':
        df = df.groupby(['week_ending_date', info.fips_col]).sum()
        df.reset_index(inplace=True)

    df = df.pivot(index='week_ending_date',
                  columns=info.fips_col, values=info.data_col)

    dates = df.index.to_numpy(dtype='datetime64[D]')

    return np.array([
        list(zip(dates, df[col]))
        for col in df.columns
    ], dtype=[('date', 'datetime64[D]'), ('data', np.float64)]).T


def _fetch_deaths_state(attrib_name: str, scope: CensusScope, time_period: SpecificTimePeriod) -> NDArray[np.float64]:
    """
    Fetches data from CDC dataset reporting number of deaths from COVID-19 and other respiratory illnesses.
    Available from 1/4/2020 to present at state granularity.
    https://data.cdc.gov/NCHS/Provisional-COVID-19-Death-Counts-by-Week-Ending-D/r8kw-7aab/about_data
    """
    if time_period.start_date <= date(2019, 12, 29):
        msg = "State level deaths data is only available starting 1/4/2020."
        raise DataResourceException(msg)

    fips = scope.get_node_ids()
    states = get_us_states(scope.year)
    state_mapping = dict(zip(states.geoid, states.name))
    state_names = np.array([state_mapping[x] for x in fips])

    info = QueryInfo("https://data.cdc.gov/resource/r8kw-7aab.csv?",
                     "end_date", "state", attrib_name, True)

    df = _api_query(info, state_names, time_period, scope.granularity)

    df = df.groupby(['end_date', 'state']).sum()
    df.reset_index(inplace=True)
    df = df.pivot(index='end_date', columns='state', values=info.data_col)

    dates = df.index.to_numpy(dtype='datetime64[D]')

    return np.array([
        list(zip(dates, df[col]))
        for col in df.columns
    ], dtype=[('date', 'datetime64[D]'), ('data', np.float64)]).T


def _api_query(info: QueryInfo, fips: NDArray, time_period: SpecificTimePeriod, granularity: CensusGranularityName) -> DataFrame:
    """
    Composes URLs to query API and sends query requests.
    Limits each query to 10000 rows, combining several query results if this number is exceeded.
    Returns pandas Dataframe containing requested data sorted by date and location fips.
    """
    # query county level data with state fips codes
    if granularity == 'state' and not info.state_level:
        location_clauses = [
            f"starts_with({info.fips_col}, '{state}')"
            for state in fips
        ]
    # query county or state level data with full fips codes for the respective granularity
    else:
        formatted_fips = ",".join(f"'{node}'" for node in fips)
        location_clauses = [
            f"{info.fips_col} in ({formatted_fips})"
        ]

    date_clause = f"{info.date_col} " \
        + f"between '{time_period.start_date}T00:00:00' " \
        + f"and '{time_period.end_date}T00:00:00'"

    df = concat([_query_location(info, loc_clause, date_clause)
                for loc_clause in location_clauses])

    df = df.sort_values(by=[info.date_col, info.fips_col])
    return df


def _query_location(info: QueryInfo, loc_clause: str, date_clause: str) -> DataFrame:
    """
    Helper function for _api_query() that builds and sends queries for individual locations.
    """
    current_return = 10000
    total_returned = 0
    df = DataFrame()
    while current_return == 10000:
        url = info.url_base + urlencode(
            quote_via=quote,
            safe=",()'$:",
            query={
                '$select': f'{info.date_col},{info.fips_col},{info.data_col}',
                '$where': f"{loc_clause} AND {date_clause}",
                '$limit': 10000,
                '$offset': total_returned
            })

        df = concat([df, read_csv(url, dtype={info.fips_col: str})])

        current_return = len(df.index) - total_returned
        total_returned += current_return

    return df


def _validate_scope(scope: GeoScope) -> CensusScope:
    if not isinstance(scope, CensusScope):
        msg = 'Census scope is required for CDC attributes.'
        raise DataResourceException(msg)

    return scope


class CovidCasesPer100k(Adrio[np.float64]):
    """Number of COVID-19 cases per 100k population."""

    def __init__(self, time_period: SpecificTimePeriod):
        self.time_period = time_period
        """The time period the data encompasses."""

    def evaluate(self) -> NDArray[np.float64]:
        scope = self.scope
        scope = _validate_scope(scope)
        return _fetch_cases('covid_cases_per_100k', scope, self.time_period)


class CovidHospitalizationsPer100k(Adrio[np.float64]):
    """Number of COVID-19 hospitalizations per 100k population."""

    def __init__(self, time_period: SpecificTimePeriod):
        self.time_period = time_period
        """The time period the data encompasses."""

    def evaluate(self) -> NDArray[np.float64]:
        scope = self.scope
        scope = _validate_scope(scope)
        return _fetch_cases('covid_hospital_admissions_per_100k', scope, self.time_period)


class CovidHospitalizationAvgFacility(Adrio[np.float64]):
    """Weekly averages of COVID-19 hospitalizations from facility level dataset."""

    def __init__(self, time_period: SpecificTimePeriod):
        self.time_period = time_period
        """The time period the data encompasses."""

    def evaluate(self) -> NDArray[np.float64]:
        scope = self.scope
        scope = _validate_scope(scope)
        return _fetch_facility_hospitalization('total_adult_patients_hospitalized_confirmed_covid_7_day_avg', scope, self.time_period)


class CovidHospitalizationSumFacility(Adrio[np.float64]):
    """Weekly sums of all COVID-19 hospitalizations from facility level dataset."""

    def __init__(self, time_period: SpecificTimePeriod):
        self.time_period = time_period
        """The time period the data encompasses."""

    def evaluate(self) -> NDArray[np.float64]:
        scope = self.scope
        scope = _validate_scope(scope)
        return _fetch_facility_hospitalization('total_adult_patients_hospitalized_confirmed_covid_7_day_sum', scope, self.time_period)


class InfluenzaHosptializationAvgFacility(Adrio[np.float64]):
    """Weekly averages of influenza hospitalizations from facility level dataset."""

    def __init__(self, time_period: SpecificTimePeriod):
        self.time_period = time_period
        """The time period the data encompasses."""

    def evaluate(self) -> NDArray[np.float64]:
        scope = self.scope
        scope = _validate_scope(scope)
        return _fetch_facility_hospitalization('total_patients_hospitalized_confirmed_influenza_7_day_avg', scope, self.time_period)


class InfluenzaHospitalizationSumFacility(Adrio[np.float64]):
    """Weekly sums of influenza hospitalizations from facility level dataset."""

    def __init__(self, time_period: SpecificTimePeriod):
        self.time_period = time_period
        """The time period the data encompasses."""

    def evaluate(self) -> NDArray[np.float64]:
        scope = self.scope
        scope = _validate_scope(scope)
        return _fetch_facility_hospitalization('total_patients_hospitalized_confirmed_influenza_7_day_sum', scope, self.time_period)


class CovidHospitalizationAvgState(Adrio[np.float64]):
    """Weekly averages of COVID-19 hospitalizations from state level dataset."""

    def __init__(self, time_period: SpecificTimePeriod):
        self.time_period = time_period
        """The time period the data encompasses."""

    def evaluate(self) -> NDArray[np.float64]:
        scope = self.scope
        scope = _validate_scope(scope)
        return _fetch_state_hospitalization('avg_admissions_all_covid_confirmed', scope, self.time_period)


class CovidHospitalizationSumState(Adrio[np.float64]):
    """Weekly sums of COVID-19 hospitalizations from state level dataset."""

    def __init__(self, time_period: SpecificTimePeriod):
        self.time_period = time_period
        """The time period the data encompasses."""

    def evaluate(self) -> NDArray[np.float64]:
        scope = self.scope
        scope = _validate_scope(scope)
        return _fetch_state_hospitalization('total_admissions_all_covid_confirmed', scope, self.time_period)


class InfluenzaHospitalizationAvgState(Adrio[np.float64]):
    """Weekly averages of influenza hospitalizations from state level dataset."""

    def __init__(self, time_period: SpecificTimePeriod):
        self.time_period = time_period
        """The time period the data encompasses."""

    def evaluate(self) -> NDArray[np.float64]:
        scope = self.scope
        scope = _validate_scope(scope)
        return _fetch_state_hospitalization('avg_admissions_all_influenza_confirmed', scope, self.time_period)


class InfluenzaHospitalizationSumState(Adrio[np.float64]):
    """Weekly sums of influenza hospitalizations from state level dataset."""

    def __init__(self, time_period: SpecificTimePeriod):
        self.time_period = time_period
        """The time period the data encompasses."""

    def evaluate(self) -> NDArray[np.float64]:
        scope = self.scope
        scope = _validate_scope(scope)
        return _fetch_state_hospitalization('total_admissions_all_influenza_confirmed', scope, self.time_period)


class FullCovidVaccinations(Adrio[np.float64]):
    """Cumulative total number of individuals fully vaccinated for COVID-19."""

    def __init__(self, time_period: SpecificTimePeriod):
        self.time_period = time_period
        """The time period the data encompasses."""

    def evaluate(self) -> NDArray[np.float64]:
        scope = self.scope
        scope = _validate_scope(scope)
        return _fetch_vaccination('series_complete_yes', scope, self.time_period)


class OneDoseCovidVaccinations(Adrio[np.float64]):
    """Cumulative total number of individuals with at least one dose of COVID-19 vaccination."""

    def __init__(self, time_period: SpecificTimePeriod):
        self.time_period = time_period
        """The time period the data encompasses."""

    def evaluate(self) -> NDArray[np.float64]:
        scope = self.scope
        scope = _validate_scope(scope)
        return _fetch_vaccination('administered_dose1_recip', scope, self.time_period)


class CovidBoosterDoses(Adrio[np.float64]):
    """Cumulative total number of COVID-19 booster doses administered."""

    def __init__(self, time_period: SpecificTimePeriod):
        self.time_period = time_period
        """The time period the data encompasses."""

    def evaluate(self) -> NDArray[np.float64]:
        scope = self.scope
        scope = _validate_scope(scope)
        return _fetch_vaccination('booster_doses', scope, self.time_period)


class CovidDeathsCounty(Adrio[np.float64]):
    """Weekly total COVID-19 deaths from county level dataset."""

    def __init__(self, time_period: SpecificTimePeriod):
        self.time_period = time_period
        """The time period the data encompasses."""

    def evaluate(self) -> NDArray[np.float64]:
        scope = self.scope
        scope = _validate_scope(scope)
        return _fetch_deaths_county('covid_19_deaths', scope, self.time_period)


class CovidDeathsState(Adrio[np.float64]):
    """Weekly total COVID-19 deaths from state level dataset."""

    def __init__(self, time_period: SpecificTimePeriod):
        self.time_period = time_period
        """The time period the data encompasses."""

    def evaluate(self) -> NDArray[np.float64]:
        scope = self.scope
        scope = _validate_scope(scope)
        return _fetch_deaths_state('covid_19_deaths', scope, self.time_period)


class InfluenzaDeathsState(Adrio[np.float64]):
    """Weekly total influenza deaths from state level dataset."""

    def __init__(self, time_period: SpecificTimePeriod):
        self.time_period = time_period
        """The time period the data encompasses."""

    def evaluate(self) -> NDArray[np.float64]:
        scope = self.scope
        scope = _validate_scope(scope)
        return _fetch_deaths_state('influenza_deaths', scope, self.time_period)
