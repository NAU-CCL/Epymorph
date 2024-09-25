"""ADRIOs that access data.cdc.gov website for various health data."""

from datetime import date, timedelta
from typing import NamedTuple
from urllib.parse import quote, urlencode
from warnings import warn

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, concat, read_csv
from typing_extensions import override

from epymorph.adrio.adrio import Adrio
from epymorph.error import DataResourceException
from epymorph.geography.scope import GeoScope
from epymorph.geography.us_census import (
    STATE,
    CensusGranularityName,
    CensusScope,
    get_us_states,
    state_fips_to_code,
)
from epymorph.simulation import TimeFrame


class QueryInfo(NamedTuple):
    url_base: str
    date_col: str
    fips_col: str
    data_col: str
    state_level: bool = False
    """Whether we are querying a dataset reporting state-level data."""


def _fetch_cases(
    attrib_name: str, scope: CensusScope, time_frame: TimeFrame
) -> NDArray[np.float64]:
    """
    Fetches data from HealthData dataset reporting COVID-19 cases per 100k population.
    Available between 2/24/2022 and 5/4/2023 at state and county granularities.
    https://healthdata.gov/dataset/United-States-COVID-19-Community-Levels-by-County/nn5b-j5u9/about_data
    """
    if time_frame.start_date < date(2022, 2, 24) or time_frame.end_date > date(
        2023, 5, 4
    ):
        msg = "COVID cases data is only available between 2/24/2022 and 5/4/2023."
        raise DataResourceException(msg)

    info = QueryInfo(
        "https://data.cdc.gov/resource/3nnm-4jni.csv?",
        "date_updated",
        "county_fips",
        attrib_name,
    )

    cdc_df = _api_query(info, scope.get_node_ids(), time_frame, scope.granularity)

    cdc_df = cdc_df.rename(columns={"county_fips": "fips"})

    if scope.granularity == "state":
        cdc_df["fips"] = [STATE.extract(x) for x in cdc_df["fips"]]

        cdc_df = cdc_df.groupby(["date_updated", "fips"]).sum()
        cdc_df = cdc_df.reset_index()

    cdc_df = cdc_df.pivot_table(
        index="date_updated", columns="fips", values=info.data_col
    )

    dates = cdc_df.index.to_numpy(dtype="datetime64[D]")

    return np.array(
        [list(zip(dates, cdc_df[col])) for col in cdc_df.columns],
        dtype=[("date", "datetime64[D]"), ("data", np.float64)],
    ).T


def _fetch_facility_hospitalization(
    attrib_name: str, scope: CensusScope, time_frame: TimeFrame, replace_sentinel: int
) -> NDArray[np.float64]:
    """
    Fetches data from HealthData dataset reporting number of people hospitalized for
    COVID-19 and other respiratory illnesses at facility level during manditory
    reporting period.
    Available between 12/13/2020 and 5/10/2023 at state and county granularities.
    https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/anag-cw7u/about_data
    """
    if time_frame.start_date < date(2020, 12, 13) or time_frame.end_date > date(
        2023, 5, 10
    ):
        msg = (
            "Facility level hospitalization data is only available between 12/13/2020 "
            "and 5/10/2023."
        )
        raise DataResourceException(msg)

    info = QueryInfo(
        "https://healthdata.gov/resource/anag-cw7u.csv?",
        "collection_week",
        "fips_code",
        attrib_name,
    )

    cdc_df = _api_query(info, scope.get_node_ids(), time_frame, scope.granularity)

    if scope.granularity == "state":
        cdc_df["fips_code"] = [STATE.extract(x) for x in cdc_df["fips_code"]]

    # replace sentinel values with the integer provided
    cdc_df["is_sentinel"] = cdc_df[info.data_col] == -999999
    num_sentinel = cdc_df["is_sentinel"].sum()
    cdc_df = cdc_df.replace(-999999, replace_sentinel)
    cdc_df = cdc_df.groupby(["collection_week", "fips_code"]).agg(
        {info.data_col: "sum", "is_sentinel": any}
    )
    if num_sentinel > 0:
        warn(
            f"{num_sentinel} values < 4 were replaced with {replace_sentinel} "
            "in returned data."
        )

    cdc_df = cdc_df.reset_index()
    cdc_df = cdc_df.pivot_table(
        index="collection_week", columns="fips_code", values=info.data_col
    )

    dates = cdc_df.index.to_numpy(dtype="datetime64[D]")

    return np.array(
        [list(zip(dates, cdc_df[col])) for col in cdc_df.columns],
        dtype=[("date", "datetime64[D]"), ("data", np.float64)],
    ).T


def _fetch_state_hospitalization(
    attrib_name: str, scope: CensusScope, time_frame: TimeFrame
) -> NDArray[np.float64]:
    """
    Fetches data from CDC dataset reporting number of people hospitalized for COVID-19
    and other respiratory illnesses at state level during manditory and voluntary
    reporting periods.
    Available from 1/4/2020 to present at state granularity.
    Data reported voluntarily past 5/1/2024.
    https://data.cdc.gov/Public-Health-Surveillance/Weekly-United-States-Hospitalization-Metrics-by-Ju/aemt-mg7g/about_data
    """
    if scope.granularity != "state":
        msg = (
            "State level hospitalization data can only be retrieved for state "
            "granularity."
        )
        raise DataResourceException(msg)
    if time_frame.start_date < date(2020, 1, 4):
        msg = "State level hospitalization data is only available starting 1/4/2020."
        raise DataResourceException(msg)
    if time_frame.end_date > date(2024, 5, 1):
        warn("State level hospitalization data is voluntary past 5/1/2024.")

    info = QueryInfo(
        "https://data.cdc.gov/resource/aemt-mg7g.csv?",
        "week_end_date",
        "jurisdiction",
        attrib_name,
        True,
    )

    state_mapping = state_fips_to_code(scope.year)
    fips = scope.get_node_ids()
    state_codes = np.array([state_mapping[x] for x in fips])

    cdc_df = _api_query(info, state_codes, time_frame, scope.granularity)

    cdc_df = cdc_df.groupby(["week_end_date", "jurisdiction"]).sum()
    cdc_df = cdc_df.reset_index()
    cdc_df = cdc_df.pivot_table(
        index="week_end_date", columns="jurisdiction", values=info.data_col
    )

    dates = cdc_df.index.to_numpy(dtype="datetime64[D]")

    return np.array(
        [list(zip(dates, cdc_df[col])) for col in cdc_df.columns],
        dtype=[("date", "datetime64[D]"), ("data", np.float64)],
    ).T


def _fetch_vaccination(
    attrib_name: str, scope: CensusScope, time_frame: TimeFrame
) -> NDArray[np.float64]:
    """
    Fetches data from CDC dataset reporting total COVID-19 vaccination numbers.
    Available between 12/13/2020 and 5/10/2024 at state and county granularities.
    https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh/about_data
    """
    if time_frame.start_date < date(2020, 12, 13) or time_frame.end_date > date(
        2024, 5, 10
    ):
        msg = "Vaccination data is only available between 12/13/2020 and 5/10/2024."
        raise DataResourceException(msg)

    info = QueryInfo(
        "https://data.cdc.gov/resource/8xkx-amqh.csv?", "date", "fips", attrib_name
    )

    cdc_df = _api_query(info, scope.get_node_ids(), time_frame, scope.granularity)

    cdc_df = cdc_df.pivot_table(index="date", columns="fips", values=info.data_col)

    dates = cdc_df.index.to_numpy(dtype="datetime64[D]")

    return np.array(
        [list(zip(dates, cdc_df[col])) for col in cdc_df.columns],
        dtype=[("date", "datetime64[D]"), ("data", np.float64)],
    ).T


def _fetch_deaths_county(
    attrib_name: str, scope: CensusScope, time_frame: TimeFrame
) -> NDArray[np.float64]:
    """
    Fetches data from CDC dataset reporting number of deaths from COVID-19.
    Available between 1/4/2020 and 4/5/2024 at state and county granularities.
    https://data.cdc.gov/NCHS/AH-COVID-19-Death-Counts-by-County-and-Week-2020-p/ite7-j2w7/about_data
    """
    if time_frame.start_date < date(2020, 1, 4) or time_frame.end_date > date(
        2024, 4, 5
    ):
        msg = (
            "County level deaths data is only available between 1/4/2020 and 4/5/2024."
        )
        raise DataResourceException(msg)

    if scope.granularity == "state":
        info = QueryInfo(
            "https://data.cdc.gov/resource/ite7-j2w7.csv?",
            "week_ending_date",
            "stfips",
            attrib_name,
            True,
        )
    else:
        info = QueryInfo(
            "https://data.cdc.gov/resource/ite7-j2w7.csv?",
            "week_ending_date",
            "fips_code",
            attrib_name,
        )

    cdc_df = _api_query(info, scope.get_node_ids(), time_frame, scope.granularity)

    if scope.granularity == "state":
        cdc_df = cdc_df.groupby(["week_ending_date", info.fips_col]).sum()
        cdc_df = cdc_df.reset_index()

    cdc_df = cdc_df.pivot_table(
        index="week_ending_date", columns=info.fips_col, values=info.data_col
    )

    dates = cdc_df.index.to_numpy(dtype="datetime64[D]")

    return np.array(
        [list(zip(dates, cdc_df[col])) for col in cdc_df.columns],
        dtype=[("date", "datetime64[D]"), ("data", np.float64)],
    ).T


def _fetch_deaths_state(
    attrib_name: str, scope: CensusScope, time_frame: TimeFrame
) -> NDArray[np.float64]:
    """
    Fetches data from CDC dataset reporting number of deaths from COVID-19 and other
    respiratory illnesses.
    Available from 1/4/2020 to present at state granularity.
    https://data.cdc.gov/NCHS/Provisional-COVID-19-Death-Counts-by-Week-Ending-D/r8kw-7aab/about_data
    """
    if time_frame.start_date < date(2020, 1, 4):
        msg = "State level deaths data is only available starting 1/4/2020."
        raise DataResourceException(msg)

    fips = scope.get_node_ids()
    states = get_us_states(scope.year)
    state_mapping = dict(zip(states.geoid, states.name))
    state_names = np.array([state_mapping[x] for x in fips])

    info = QueryInfo(
        "https://data.cdc.gov/resource/r8kw-7aab.csv?",
        "end_date",
        "state",
        attrib_name,
        True,
    )

    cdc_df = _api_query(info, state_names, time_frame, scope.granularity)

    cdc_df = cdc_df.groupby(["end_date", "state"]).sum()
    cdc_df = cdc_df.reset_index()
    cdc_df = cdc_df.pivot_table(index="end_date", columns="state", values=info.data_col)

    dates = cdc_df.index.to_numpy(dtype="datetime64[D]")

    return np.array(
        [list(zip(dates, cdc_df[col])) for col in cdc_df.columns],
        dtype=[("date", "datetime64[D]"), ("data", np.float64)],
    ).T


def _api_query(
    info: QueryInfo,
    fips: NDArray,
    time_frame: TimeFrame,
    granularity: CensusGranularityName,
) -> DataFrame:
    """
    Composes URLs to query API and sends query requests.
    Limits each query to 10000 rows, combining several query results if this number
    is exceeded.
    Returns pandas Dataframe containing requested data sorted by date and location fips.
    """
    # query county level data with state fips codes
    if granularity == "state" and not info.state_level:
        location_clauses = [
            f"starts_with({info.fips_col}, '{state}')" for state in fips
        ]
    # query county or state level data with full fips codes for respective granularity
    else:
        formatted_fips = ",".join(f"'{node}'" for node in fips)
        location_clauses = [f"{info.fips_col} in ({formatted_fips})"]

    date_clause = (
        f"{info.date_col} "
        f"between '{time_frame.start_date}T00:00:00' "
        f"and '{time_frame.end_date + timedelta(days=1)}T00:00:00'"
    )

    cdc_df = concat(
        [
            _query_location(info, loc_clause, date_clause)
            for loc_clause in location_clauses
        ]
    )

    cdc_df = cdc_df.sort_values(by=[info.date_col, info.fips_col])
    return cdc_df


def _query_location(info: QueryInfo, loc_clause: str, date_clause: str) -> DataFrame:
    """
    Helper function for _api_query() that builds and sends queries for
    individual locations.
    """
    current_return = 10000
    total_returned = 0
    cdc_df = DataFrame()
    while current_return == 10000:
        url = info.url_base + urlencode(
            quote_via=quote,
            safe=",()'$:",
            query={
                "$select": f"{info.date_col},{info.fips_col},{info.data_col}",
                "$where": f"{loc_clause} AND {date_clause}",
                "$limit": 10000,
                "$offset": total_returned,
            },
        )

        cdc_df = concat([cdc_df, read_csv(url, dtype={info.fips_col: str})])

        current_return = len(cdc_df.index) - total_returned
        total_returned += current_return

    return cdc_df


def _validate_scope(scope: GeoScope) -> CensusScope:
    if not isinstance(scope, CensusScope):
        msg = "Census scope is required for CDC attributes."
        raise DataResourceException(msg)
    return scope


class CovidCasesPer100k(Adrio[np.float64]):
    """Number of COVID-19 cases per 100k population."""

    time_frame: TimeFrame
    """The time period the data encompasses."""

    def __init__(self, time_frame: TimeFrame):
        self.time_frame = time_frame

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_cases("covid_cases_per_100k", scope, self.time_frame)


class CovidHospitalizationsPer100k(Adrio[np.float64]):
    """Number of COVID-19 hospitalizations per 100k population."""

    time_frame: TimeFrame
    """The time period the data encompasses."""

    def __init__(self, time_frame: TimeFrame):
        self.time_frame = time_frame

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_cases(
            "covid_hospital_admissions_per_100k", scope, self.time_frame
        )


class CovidHospitalizationAvgFacility(Adrio[np.float64]):
    """Weekly averages of COVID-19 hospitalizations from facility level dataset."""

    time_frame: TimeFrame
    """The time period the data encompasses."""
    replace_sentinel: int
    """The integer value in range 0-3 to replace sentinel values with."""

    def __init__(self, time_frame: TimeFrame, replace_sentinel: int):
        self.time_frame = time_frame
        if replace_sentinel not in range(4):
            msg = "Sentinel substitute value must be in range 0-3."
            raise DataResourceException(msg)
        self.replace_sentinel = replace_sentinel

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_facility_hospitalization(
            "total_adult_patients_hospitalized_confirmed_covid_7_day_avg",
            scope,
            self.time_frame,
            self.replace_sentinel,
        )


class CovidHospitalizationSumFacility(Adrio[np.float64]):
    """Weekly sums of all COVID-19 hospitalizations from facility level dataset."""

    time_frame: TimeFrame
    """The time period the data encompasses."""
    replace_sentinel: int
    """The integer value in range 0-3 to replace sentinel values with."""

    def __init__(self, time_frame: TimeFrame, replace_sentinel: int):
        self.time_frame = time_frame
        if replace_sentinel not in range(4):
            msg = "Sentinel substitute value must be in range 0-3."
            raise DataResourceException(msg)
        self.replace_sentinel = replace_sentinel

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_facility_hospitalization(
            "total_adult_patients_hospitalized_confirmed_covid_7_day_sum",
            scope,
            self.time_frame,
            self.replace_sentinel,
        )


class InfluenzaHosptializationAvgFacility(Adrio[np.float64]):
    """Weekly averages of influenza hospitalizations from facility level dataset."""

    time_frame: TimeFrame
    """The time period the data encompasses."""
    replace_sentinel: int
    """The integer value in range 0-3 to replace sentinel values with."""

    def __init__(self, time_frame: TimeFrame, replace_sentinel: int):
        self.time_frame = time_frame
        if replace_sentinel not in range(4):
            msg = "Sentinel substitute value must be in range 0-3."
            raise DataResourceException(msg)
        self.replace_sentinel = replace_sentinel

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_facility_hospitalization(
            "total_patients_hospitalized_confirmed_influenza_7_day_avg",
            scope,
            self.time_frame,
            self.replace_sentinel,
        )


class InfluenzaHospitalizationSumFacility(Adrio[np.float64]):
    """Weekly sums of influenza hospitalizations from facility level dataset."""

    time_frame: TimeFrame
    """The time period the data encompasses."""
    replace_sentinel: int
    """The integer value in range 0-3 to replace sentinel values with."""

    def __init__(self, time_frame: TimeFrame, replace_sentinel: int):
        self.time_frame = time_frame
        if replace_sentinel not in range(4):
            msg = "Sentinel substitute value must be in range 0-3."
            raise DataResourceException(msg)
        self.replace_sentinel = replace_sentinel

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_facility_hospitalization(
            "total_patients_hospitalized_confirmed_influenza_7_day_sum",
            scope,
            self.time_frame,
            self.replace_sentinel,
        )


class CovidHospitalizationAvgState(Adrio[np.float64]):
    """Weekly averages of COVID-19 hospitalizations from state level dataset."""

    time_frame: TimeFrame
    """The time period the data encompasses."""

    def __init__(self, time_frame: TimeFrame):
        self.time_frame = time_frame

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_state_hospitalization(
            "avg_admissions_all_covid_confirmed", scope, self.time_frame
        )


class CovidHospitalizationSumState(Adrio[np.float64]):
    """Weekly sums of COVID-19 hospitalizations from state level dataset."""

    time_frame: TimeFrame
    """The time period the data encompasses."""

    def __init__(self, time_frame: TimeFrame):
        self.time_frame = time_frame

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_state_hospitalization(
            "total_admissions_all_covid_confirmed", scope, self.time_frame
        )


class InfluenzaHospitalizationAvgState(Adrio[np.float64]):
    """Weekly averages of influenza hospitalizations from state level dataset."""

    time_frame: TimeFrame
    """The time period the data encompasses."""

    def __init__(self, time_frame: TimeFrame):
        self.time_frame = time_frame

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_state_hospitalization(
            "avg_admissions_all_influenza_confirmed", scope, self.time_frame
        )


class InfluenzaHospitalizationSumState(Adrio[np.float64]):
    """Weekly sums of influenza hospitalizations from state level dataset."""

    time_frame: TimeFrame
    """The time period the data encompasses."""

    def __init__(self, time_frame: TimeFrame):
        self.time_frame = time_frame

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_state_hospitalization(
            "total_admissions_all_influenza_confirmed", scope, self.time_frame
        )


class FullCovidVaccinations(Adrio[np.float64]):
    """Cumulative total number of individuals fully vaccinated for COVID-19."""

    time_frame: TimeFrame
    """The time period the data encompasses."""

    def __init__(self, time_frame: TimeFrame):
        self.time_frame = time_frame

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_vaccination("series_complete_yes", scope, self.time_frame)


class OneDoseCovidVaccinations(Adrio[np.float64]):
    """
    Cumulative total number of individuals with at least one dose of
    COVID-19 vaccination.
    """

    time_frame: TimeFrame
    """The time period the data encompasses."""

    def __init__(self, time_frame: TimeFrame):
        self.time_frame = time_frame

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_vaccination("administered_dose1_recip", scope, self.time_frame)


class CovidBoosterDoses(Adrio[np.float64]):
    """Cumulative total number of COVID-19 booster doses administered."""

    time_frame: TimeFrame
    """The time period the data encompasses."""

    def __init__(self, time_frame: TimeFrame):
        self.time_frame = time_frame

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_vaccination("booster_doses", scope, self.time_frame)


class CovidDeathsCounty(Adrio[np.float64]):
    """Weekly total COVID-19 deaths from county level dataset."""

    time_frame: TimeFrame
    """The time period the data encompasses."""

    def __init__(self, time_frame: TimeFrame):
        self.time_frame = time_frame

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_deaths_county("covid_19_deaths", scope, self.time_frame)


class CovidDeathsState(Adrio[np.float64]):
    """Weekly total COVID-19 deaths from state level dataset."""

    time_frame: TimeFrame
    """The time period the data encompasses."""

    def __init__(self, time_frame: TimeFrame):
        self.time_frame = time_frame

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_deaths_state("covid_19_deaths", scope, self.time_frame)


class InfluenzaDeathsState(Adrio[np.float64]):
    """Weekly total influenza deaths from state level dataset."""

    time_frame: TimeFrame
    """The time period the data encompasses."""

    def __init__(self, time_frame: TimeFrame):
        self.time_frame = time_frame

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_deaths_state("influenza_deaths", scope, self.time_frame)
