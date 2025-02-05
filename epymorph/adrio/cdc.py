"""ADRIOs that access data.cdc.gov website for various health data."""

from dataclasses import dataclass
from datetime import date
from typing import Mapping
from urllib.parse import quote, urlencode
from warnings import warn

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, concat, read_csv
from typing_extensions import override

from epymorph.adrio.adrio import Adrio, ProgressCallback, adrio_cache
from epymorph.error import DataResourceError
from epymorph.geography.scope import GeoScope
from epymorph.geography.us_census import CensusScope
from epymorph.geography.us_geography import STATE, CensusGranularityName
from epymorph.geography.us_tiger import get_states
from epymorph.time import TimeFrame


@dataclass(frozen=True)
class DataSource:
    url_base: str
    date_col: str
    fips_col: str
    data_col: str
    granularity: CensusGranularityName
    """The geographic granularity of the source data."""
    replace_sentinel: float | None
    """If None, ignore sentinel values (-999999); otherwise, replace them with
    the given value."""
    map_geo_ids: Mapping[str, str] | None = None
    """If None, use the scope node IDs as they are, otherwise use this mapping
    to map them."""


_SENTINEL = -999999
"""A common sentinel value which represents values which have been redacted
for privacy because there were less than 4 individuals in that data point."""


def _api_query(
    source: DataSource,
    scope: CensusScope,
    time_frame: TimeFrame,
    progress: ProgressCallback,
) -> NDArray[np.float64]:
    """
    Composes URLs to query API and sends query requests. Limits each query to
    10000 rows, combining several query results if this number is exceeded.
    Returns Dataframe containing requested data sorted by date and location fips.
    """
    node_ids = (
        [source.map_geo_ids[x] for x in scope.node_ids]
        if source.map_geo_ids is not None
        else scope.node_ids
    )
    if scope.granularity == "state" and source.granularity != "state":
        # query county level data with state fips codes
        location_clauses = [f"starts_with({source.fips_col}, '{x}')" for x in node_ids]
    else:
        # query with full fips codes
        formatted_fips = ",".join(f"'{node}'" for node in node_ids)
        location_clauses = [f"{source.fips_col} in ({formatted_fips})"]

    date_clause = (
        f"{source.date_col} "
        f"between '{time_frame.start_date}T00:00:00' "
        f"and '{time_frame.end_date}T00:00:00'"
    )

    processing_steps = len(location_clauses) + 1

    def query_step(index, loc_clause) -> DataFrame:
        step_result = _query_location(source, loc_clause, date_clause)
        progress((index + 1) / processing_steps, None)
        return step_result

    cdc_df = concat(
        [query_step(i, loc_clause) for i, loc_clause in enumerate(location_clauses)]
    )

    if source.replace_sentinel is not None:
        num_sentinel = (cdc_df[source.data_col] == _SENTINEL).sum()
        if num_sentinel > 0:
            cdc_df = cdc_df.replace(_SENTINEL, source.replace_sentinel)
            warn(
                f"{num_sentinel} values < 4 were replaced with "
                f"{source.replace_sentinel} in returned data."
            )

    if scope.granularity == "state" and source.granularity != "state":
        # aggregate county data to state level
        cdc_df[source.fips_col] = cdc_df[source.fips_col].map(STATE.truncate)
        cdc_df = cdc_df.groupby([source.fips_col, source.date_col]).sum().reset_index()

    return _as_numpy(
        cdc_df.sort_values(by=[source.date_col, source.fips_col]).pivot_table(
            index=source.date_col,
            columns=source.fips_col,
            values=source.data_col,
        )
    )


def _as_numpy(data_df: DataFrame) -> NDArray[np.float64]:
    """Convert a DataFrame to a time-series by node numpy array where each value is a
    tuple of date and data value. Note: this time-series is not necessarily the same
    length as simulation T, because not all ADRIOs produce a daily value."""
    dates = data_df.index.to_numpy(dtype="datetime64[D]")
    return np.array(
        [list(zip(dates, data_df[col], strict=True)) for col in data_df.columns],
        dtype=[("date", "datetime64[D]"), ("data", np.float64)],
    ).T


def _fetch_cases(
    attrib_name: str,
    scope: CensusScope,
    time_frame: TimeFrame,
    progress: ProgressCallback,
) -> NDArray[np.float64]:
    """
    Fetches data from HealthData dataset reporting COVID-19 cases per 100k population.
    Available between 2/24/2022 and 5/4/2023 at state and county granularities.
    https://healthdata.gov/dataset/United-States-COVID-19-Community-Levels-by-County/nn5b-j5u9/about_data
    """
    if (
        time_frame.start_date < date(2022, 2, 24)  #
        or time_frame.end_date > date(2023, 5, 4)
    ):
        msg = "COVID cases data is only available between 2/24/2022 and 5/4/2023."
        raise DataResourceError(msg)

    source = DataSource(
        url_base="https://data.cdc.gov/resource/3nnm-4jni.csv?",
        date_col="date_updated",
        fips_col="county_fips",
        data_col=attrib_name,
        granularity="county",
        replace_sentinel=None,
    )

    return _api_query(source, scope, time_frame, progress)


def _fetch_facility_hospitalization(
    attrib_name: str,
    scope: CensusScope,
    time_frame: TimeFrame,
    replace_sentinel: int,
    progress: ProgressCallback,
) -> NDArray[np.float64]:
    """
    Fetches data from HealthData dataset reporting number of people hospitalized for
    COVID-19 and other respiratory illnesses at facility level during manditory
    reporting period.
    Available between 12/13/2020 and 5/10/2023 at state and county granularities.
    https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/anag-cw7u/about_data
    """
    if (
        time_frame.start_date < date(2020, 12, 13)  #
        or time_frame.end_date > date(2023, 5, 10)
    ):
        msg = (
            "Facility level hospitalization data is only available between 12/13/2020 "
            "and 5/10/2023."
        )
        raise DataResourceError(msg)

    source = DataSource(
        url_base="https://healthdata.gov/resource/anag-cw7u.csv?",
        date_col="collection_week",
        fips_col="fips_code",
        data_col=attrib_name,
        granularity="county",
        replace_sentinel=replace_sentinel,
    )

    return _api_query(source, scope, time_frame, progress)


def _fetch_state_hospitalization(
    attrib_name: str,
    scope: CensusScope,
    time_frame: TimeFrame,
    progress: ProgressCallback,
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
        raise DataResourceError(msg)
    if time_frame.start_date < date(2020, 1, 4):
        msg = "State level hospitalization data is only available starting 1/4/2020."
        raise DataResourceError(msg)
    if time_frame.end_date > date(2024, 5, 1):
        warn("State level hospitalization data is voluntary past 5/1/2024.")

    source = DataSource(
        url_base="https://data.cdc.gov/resource/aemt-mg7g.csv?",
        date_col="week_end_date",
        fips_col="jurisdiction",
        data_col=attrib_name,
        granularity="state",
        replace_sentinel=None,
        map_geo_ids=get_states(scope.year).state_fips_to_code,
    )

    return _api_query(source, scope, time_frame, progress)


def _fetch_vaccination(
    attrib_name: str,
    scope: CensusScope,
    time_frame: TimeFrame,
    progress: ProgressCallback,
) -> NDArray[np.float64]:
    """
    Fetches data from CDC dataset reporting total COVID-19 vaccination numbers.
    Available between 12/13/2020 and 5/10/2024 at state and county granularities.
    https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh/about_data
    """
    if (
        time_frame.start_date < date(2020, 12, 13)  #
        or time_frame.end_date > date(2024, 5, 10)
    ):
        msg = "Vaccination data is only available between 12/13/2020 and 5/10/2024."
        raise DataResourceError(msg)

    source = DataSource(
        url_base="https://data.cdc.gov/resource/8xkx-amqh.csv?",
        date_col="date",
        fips_col="fips",
        data_col=attrib_name,
        granularity="county",
        replace_sentinel=None,
    )

    return _api_query(source, scope, time_frame, progress)


def _fetch_deaths_county(
    attrib_name: str,
    scope: CensusScope,
    time_frame: TimeFrame,
    progress: ProgressCallback,
) -> NDArray[np.float64]:
    """
    Fetches data from CDC dataset reporting number of deaths from COVID-19.
    Available between 1/4/2020 and 4/5/2024 at state and county granularities.
    https://data.cdc.gov/NCHS/AH-COVID-19-Death-Counts-by-County-and-Week-2020-p/ite7-j2w7/about_data
    """
    if (
        time_frame.start_date < date(2020, 1, 4)  #
        or time_frame.end_date > date(2024, 4, 5)
    ):
        msg = (
            "County level deaths data is only available between 1/4/2020 and 4/5/2024."
        )
        raise DataResourceError(msg)

    if scope.granularity == "state":
        source = DataSource(
            url_base="https://data.cdc.gov/resource/ite7-j2w7.csv?",
            date_col="week_ending_date",
            fips_col="stfips",
            data_col=attrib_name,
            granularity="state",
            replace_sentinel=None,
        )
    else:
        source = DataSource(
            url_base="https://data.cdc.gov/resource/ite7-j2w7.csv?",
            date_col="week_ending_date",
            fips_col="fips_code",
            data_col=attrib_name,
            granularity="county",
            replace_sentinel=None,
        )

    return _api_query(source, scope, time_frame, progress)


def _fetch_deaths_state(
    attrib_name: str,
    scope: CensusScope,
    time_frame: TimeFrame,
    progress: ProgressCallback,
) -> NDArray[np.float64]:
    """
    Fetches data from CDC dataset reporting number of deaths from COVID-19 and other
    respiratory illnesses.
    Available from 1/4/2020 to present at state granularity.
    https://data.cdc.gov/NCHS/Provisional-COVID-19-Death-Counts-by-Week-Ending-D/r8kw-7aab/about_data
    """
    if time_frame.start_date < date(2020, 1, 4):
        msg = "State level deaths data is only available starting 1/4/2020."
        raise DataResourceError(msg)

    states = get_states(scope.year)
    state_mapping = dict(zip(states.geoid, states.name, strict=True))

    source = DataSource(
        url_base="https://data.cdc.gov/resource/r8kw-7aab.csv?",
        date_col="end_date",
        fips_col="state",
        data_col=attrib_name,
        granularity="state",
        replace_sentinel=None,
        map_geo_ids=state_mapping,
    )

    return _api_query(source, scope, time_frame, progress)


def _query_location(info: DataSource, loc_clause: str, date_clause: str) -> DataFrame:
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
        raise DataResourceError(msg)
    return scope


@adrio_cache
class CovidCasesPer100k(Adrio[np.float64]):
    """
    Creates a TxN matrix of tuples, containing a date and a float representing the
    number of COVID-19 cases per 100k population.
    """

    override_time_frame: TimeFrame | None
    """The time period the data encompasses."""

    def __init__(self, time_frame: TimeFrame | None = None):
        self.override_time_frame = time_frame

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_cases(
            "covid_cases_per_100k",
            scope,
            self.override_time_frame or self.time_frame,
            self.progress,
        )


@adrio_cache
class CovidHospitalizationsPer100k(Adrio[np.float64]):
    """
    Creates a TxN matrix of tuples, containing a date and a float representing the
    number of COVID-19 hospitalizations per 100k population.
    """

    override_time_frame: TimeFrame | None
    """The time period the data encompasses."""

    def __init__(self, time_frame: TimeFrame | None = None):
        self.override_time_frame = time_frame

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_cases(
            "covid_hospital_admissions_per_100k",
            scope,
            self.override_time_frame or self.time_frame,
            self.progress,
        )


@adrio_cache
class CovidHospitalizationAvgFacility(Adrio[np.float64]):
    """
    Creates a TxN matrix of tuples, containing a date and a float representing the
    weekly averages of COVID-19 hospitalizations from the facility level dataset.
    """

    override_time_frame: TimeFrame | None
    """The time period the data encompasses."""
    replace_sentinel: int
    """The integer value in range 0-3 to replace sentinel values with."""

    def __init__(self, replace_sentinel: int, time_frame: TimeFrame | None = None):
        if replace_sentinel not in range(4):
            msg = "Sentinel substitute value must be in range 0-3."
            raise DataResourceError(msg)
        self.replace_sentinel = replace_sentinel
        self.override_time_frame = time_frame

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_facility_hospitalization(
            "total_adult_patients_hospitalized_confirmed_covid_7_day_avg",
            scope,
            self.override_time_frame or self.time_frame,
            self.replace_sentinel,
            self.progress,
        )


@adrio_cache
class CovidHospitalizationSumFacility(Adrio[np.float64]):
    """
    Creates a TxN matrix of tuples, containing a date and a float representing the
    weekly sums of all COVID-19 hospitalizations from the facility level dataset.
    """

    override_time_frame: TimeFrame | None
    """The time period the data encompasses."""
    replace_sentinel: int
    """The integer value in range 0-3 to replace sentinel values with."""

    def __init__(self, replace_sentinel: int, time_frame: TimeFrame | None = None):
        if replace_sentinel not in range(4):
            msg = "Sentinel substitute value must be in range 0-3."
            raise DataResourceError(msg)
        self.replace_sentinel = replace_sentinel
        self.override_time_frame = time_frame

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_facility_hospitalization(
            "total_adult_patients_hospitalized_confirmed_covid_7_day_sum",
            scope,
            self.override_time_frame or self.time_frame,
            self.replace_sentinel,
            self.progress,
        )


@adrio_cache
class InfluenzaHosptializationAvgFacility(Adrio[np.float64]):
    """
    Creates a TxN matrix of tuples, containing a date and a float representing the
    weekly averages of influenza hospitalizations from the facility level dataset.
    """

    override_time_frame: TimeFrame | None
    """The time period the data encompasses."""
    replace_sentinel: int
    """The integer value in range 0-3 to replace sentinel values with."""

    def __init__(self, replace_sentinel: int, time_frame: TimeFrame | None = None):
        if replace_sentinel not in range(4):
            msg = "Sentinel substitute value must be in range 0-3."
            raise DataResourceError(msg)
        self.replace_sentinel = replace_sentinel
        self.override_time_frame = time_frame

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_facility_hospitalization(
            "total_patients_hospitalized_confirmed_influenza_7_day_avg",
            scope,
            self.override_time_frame or self.time_frame,
            self.replace_sentinel,
            self.progress,
        )


@adrio_cache
class InfluenzaHospitalizationSumFacility(Adrio[np.float64]):
    """
    Creates a TxN matrix of tuples, containing a date and a float representing the
    weekly sums of influenza hospitalizations from the facility level dataset.
    """

    override_time_frame: TimeFrame | None
    """The time period the data encompasses."""
    replace_sentinel: int
    """The integer value in range 0-3 to replace sentinel values with."""

    def __init__(self, replace_sentinel: int, time_frame: TimeFrame | None = None):
        if replace_sentinel not in range(4):
            msg = "Sentinel substitute value must be in range 0-3."
            raise DataResourceError(msg)
        self.replace_sentinel = replace_sentinel
        self.override_time_frame = time_frame

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_facility_hospitalization(
            "total_patients_hospitalized_confirmed_influenza_7_day_sum",
            scope,
            self.override_time_frame or self.time_frame,
            self.replace_sentinel,
            self.progress,
        )


@adrio_cache
class CovidHospitalizationAvgState(Adrio[np.float64]):
    """
    Creates a TxN matrix of tuples, containing a date and a float representing the
    weekly averages of COVID-19 hospitalizations from the state level dataset.
    """

    override_time_frame: TimeFrame | None
    """The time period the data encompasses."""

    def __init__(self, time_frame: TimeFrame | None = None):
        self.override_time_frame = time_frame

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_state_hospitalization(
            "avg_admissions_all_covid_confirmed",
            scope,
            self.override_time_frame or self.time_frame,
            self.progress,
        )


@adrio_cache
class CovidHospitalizationSumState(Adrio[np.float64]):
    """
    Creates a TxN matrix of tuples, containing a date and a float representing the
    weekly sums of COVID-19 hospitalizations from the state level dataset.
    """

    override_time_frame: TimeFrame | None
    """The time period the data encompasses."""

    def __init__(self, time_frame: TimeFrame | None = None):
        self.override_time_frame = time_frame

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_state_hospitalization(
            "total_admissions_all_covid_confirmed",
            scope,
            self.override_time_frame or self.time_frame,
            self.progress,
        )


@adrio_cache
class InfluenzaHospitalizationAvgState(Adrio[np.float64]):
    """
    Creates a TxN matrix of tuples, containing a date and a float representing the
    weekly averages of influenza hospitalizations from the state level dataset.
    """

    override_time_frame: TimeFrame | None
    """The time period the data encompasses."""

    def __init__(self, time_frame: TimeFrame | None = None):
        self.override_time_frame = time_frame

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_state_hospitalization(
            "avg_admissions_all_influenza_confirmed",
            scope,
            self.override_time_frame or self.time_frame,
            self.progress,
        )


@adrio_cache
class InfluenzaHospitalizationSumState(Adrio[np.float64]):
    """
    Creates a TxN matrix of tuples, containing a date and a float representing the
    weekly sums of influenza hospitalizations from the state level dataset.
    """

    override_time_frame: TimeFrame | None
    """The time period the data encompasses."""

    def __init__(self, time_frame: TimeFrame | None = None):
        self.override_time_frame = time_frame

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_state_hospitalization(
            "total_admissions_all_influenza_confirmed",
            scope,
            self.override_time_frame or self.time_frame,
            self.progress,
        )


@adrio_cache
class FullCovidVaccinations(Adrio[np.float64]):
    """
    Creates a TxN matrix of tuples, containing a date and a float representing the
    cumulative total number of individuals fully vaccinated for COVID-19.
    """

    override_time_frame: TimeFrame | None
    """The time period the data encompasses."""

    def __init__(self, time_frame: TimeFrame | None = None):
        self.override_time_frame = time_frame

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_vaccination(
            "series_complete_yes",
            scope,
            self.override_time_frame or self.time_frame,
            self.progress,
        )


@adrio_cache
class OneDoseCovidVaccinations(Adrio[np.float64]):
    """
    Creates a TxN matrix of tuples, containing a date and a float representing the
    cumulative total number of individuals with at least one dose of COVID-19
    vaccination.
    """

    override_time_frame: TimeFrame | None
    """The time period the data encompasses."""

    def __init__(self, time_frame: TimeFrame | None = None):
        self.override_time_frame = time_frame

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_vaccination(
            "administered_dose1_recip",
            scope,
            self.override_time_frame or self.time_frame,
            self.progress,
        )


@adrio_cache
class CovidBoosterDoses(Adrio[np.float64]):
    """
    Creates a TxN matrix of tuples, containing a date and a float representing the
    cumulative total number of COVID-19 booster doses administered.
    """

    override_time_frame: TimeFrame | None
    """The time period the data encompasses."""

    def __init__(self, time_frame: TimeFrame | None = None):
        self.override_time_frame = time_frame

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_vaccination(
            "booster_doses",
            scope,
            self.override_time_frame or self.time_frame,
            self.progress,
        )


@adrio_cache
class CovidDeathsCounty(Adrio[np.float64]):
    """
    Creates a TxN matrix of tuples, containing a date and a float representing the
    weekly total of COVID-19 deaths from the county level dataset.
    """

    override_time_frame: TimeFrame | None
    """The time period the data encompasses."""

    def __init__(self, time_frame: TimeFrame | None = None):
        self.override_time_frame = time_frame

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_deaths_county(
            "covid_19_deaths",
            scope,
            self.override_time_frame or self.time_frame,
            self.progress,
        )


@adrio_cache
class CovidDeathsState(Adrio[np.float64]):
    """
    Creates a TxN matrix of tuples, containing a date and a float representing the
    weekly total of COVID-19 deaths from the state level dataset.
    """

    override_time_frame: TimeFrame | None
    """The time period the data encompasses."""

    def __init__(self, time_frame: TimeFrame | None = None):
        self.override_time_frame = time_frame

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_deaths_state(
            "covid_19_deaths",
            scope,
            self.override_time_frame or self.time_frame,
            self.progress,
        )


@adrio_cache
class InfluenzaDeathsState(Adrio[np.float64]):
    """
    Creates a TxN matrix of tuples, containing a date and a float representing the
    weekly total of influenza deaths from the state level dataset.
    """

    override_time_frame: TimeFrame | None
    """The time period the data encompasses."""

    def __init__(self, time_frame: TimeFrame | None = None):
        self.override_time_frame = time_frame

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        scope = _validate_scope(self.scope)
        return _fetch_deaths_state(
            "influenza_deaths",
            scope,
            self.override_time_frame or self.time_frame,
            self.progress,
        )
