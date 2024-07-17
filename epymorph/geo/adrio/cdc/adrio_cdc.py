from datetime import date
from typing import Any, NamedTuple
from urllib.parse import quote, urlencode
from warnings import warn

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, concat, read_csv

from epymorph.data_shape import Shapes
from epymorph.error import DataResourceException, GeoValidationException
from epymorph.geo.adrio.adrio import ADRIO, ADRIOMaker
from epymorph.geo.spec import SpecificTimePeriod, TimePeriod
from epymorph.geography.scope import GeoScope
from epymorph.geography.us_census import (STATE, CensusGranularityName,
                                          CensusScope, CountyScope, StateScope,
                                          StateScopeAll, get_us_states,
                                          state_fips_to_code)
from epymorph.simulation import AttributeDef


class QueryInfo(NamedTuple):
    url_base: str
    date_col: str
    fips_col: str
    data_col: str
    state_level: bool = False
    """Whether we are querying a dataset reporting state-level data."""


class ADRIOMakerCDC(ADRIOMaker):
    """
    CDC ADRIO template to serve as a parent class for ADRIOs that fetch data from various 
    HealthData and CDC datasets.
    """

    attributes = [
        AttributeDef("covid_cases_per_100k", int, Shapes.TxN,
                     comment='Number of COVID-19 cases per 100k population.'),
        AttributeDef("covid_hospitalizations_per_100k", int, Shapes.TxN,
                     comment='Number of COVID-19 hospitalizations per 100k population.'),
        AttributeDef("covid_hospitalization_avg_facility", float, Shapes.TxN,
                     comment='Weekly averages of COVID-19 hospitalizations from facility level dataset.'),
        AttributeDef("covid_hospitalization_sum_facility", int, Shapes.TxN,
                     comment='Weekly sums of all COVID-19 hospitalizations from facility level dataset.'),
        AttributeDef("influenza_hospitalization_avg_facility", float, Shapes.TxN,
                     comment='Weekly averages of influenza hospitalizations from facility level dataset.'),
        AttributeDef("influenza_hospitalization_sum_facility", int, Shapes.TxN,
                     comment='Weekly sums of influenza hospitalizations from facility level dataset.'),
        AttributeDef("covid_hospitalization_avg_state", float, Shapes.TxN,
                     comment='Weekly averages of COVID-19 hospitalizations from state level dataset.'),
        AttributeDef("covid_hospitalization_sum_state", int, Shapes.TxN,
                     comment='Weekly sums of COVID-19 hospitalizations from state level dataset.'),
        AttributeDef("influenza_hospitalization_avg_state", float, Shapes.TxN,
                     comment='Weekly averages of influenza hospitalizations from state level dataset.'),
        AttributeDef("influenza_hospitalization_sum_state", int, Shapes.TxN,
                     comment='Weekly sums of influenza hospitalizations from state level dataset.'),
        AttributeDef("full_covid_vaccinations", int, Shapes.TxN,
                     comment='Cumulative total number of individuals fully vaccinated for COVID-19.'),
        AttributeDef("one_dose_covid_vaccinations", int, Shapes.TxN,
                     comment='Cumulative total number of individuals with at least one dose of COVID-19 vaccination.'),
        AttributeDef("covid_booster_doses", int, Shapes.TxN,
                     comment='Cumulative total number of COVID-19 booster doses administered.'),
        AttributeDef("covid_deaths_county", int, Shapes.TxN,
                     comment='Weekly total COVID-19 deaths from county level dataset.'),
        AttributeDef("covid_deaths_state", int, Shapes.TxN,
                     comment='Weekly total COVID-19 deaths from state level dataset.'),
        AttributeDef("influenza_deaths", int, Shapes.TxN,
                     comment='Weekly total influenza deaths from state level dataset.')
    ]

    attribute_cols = {
        "covid_cases_per_100k": "covid_cases_per_100k",
        "covid_hospitalizations_per_100k": "covid_hospital_admissions_per_100k",
        "covid_hospitalization_avg_facility": "total_adult_patients_hospitalized_confirmed_covid_7_day_avg",
        "covid_hospitalization_sum_facility": "total_adult_patients_hospitalized_confirmed_covid_7_day_sum",
        "influenza_hospitalization_avg_facility": "total_patients_hospitalized_confirmed_influenza_7_day_avg",
        "influenza_hospitalization_sum_facility": "total_patients_hospitalized_confirmed_influenza_7_day_sum",
        "covid_hospitalization_avg_state": "avg_admissions_all_covid_confirmed",
        "covid_hospitalization_sum_state": "total_admissions_all_covid_confirmed",
        "influenza_hospitalization_avg_state": "avg_admissions_all_influenza_confirmed",
        "influenza_hospitalization_sum_state": "total_admissions_all_influenza_confirmed",
        "full_covid_vaccinations": "series_complete_yes",
        "one_dose_covid_vaccinations": "administered_dose1_recip",
        "covid_booster_doses": "booster_doses",
        "covid_deaths_county": "covid_19_deaths",
        "covid_deaths_state": "covid_19_deaths",
        "influenza_deaths": "influenza_deaths"
    }

    @staticmethod
    def accepts_source(source: Any) -> bool:
        return False

    def make_adrio(self, attrib: AttributeDef, scope: GeoScope, time_period: TimePeriod, source: Any | None = None) -> ADRIO:
        if attrib not in self.attributes:
            msg = f"{attrib.name} is not supported for the CDC data source."
            raise GeoValidationException(msg)
        if not isinstance(scope, StateScope | StateScopeAll | CountyScope):
            msg = "CDC data requires a CensusScope object and can only be retrieved for state and county granularities."
            raise GeoValidationException(msg)
        if not isinstance(time_period, SpecificTimePeriod):
            msg = "CDC data requires a specific time period."
            raise GeoValidationException(msg)

        if attrib.name in ["covid_cases_per_100k", "covid_hospitalizations_per_100k"]:
            return self._make_cases_adrio(attrib, scope, time_period)
        elif attrib.name in ["covid_hospitalization_avg_facility", "covid_hospitalization_sum_facility", "influenza_hospitalization_avg_facility", "influenza_hospitalization_sum_facility"]:
            return self._make_facility_hospitalization_adrio(attrib, scope, time_period)
        elif attrib.name in ["covid_hospitalization_avg_state", "covid_hospitalization_sum_state", "influenza_hospitalization_avg_state", "influenza_hospitalization_sum_state"]:
            return self._make_state_hospitalization_adrio(attrib, scope, time_period)
        elif attrib.name in ["full_covid_vaccinations", "one_dose_covid_vaccinations", "covid_booster_doses"]:
            return self._make_vaccination_adrio(attrib, scope, time_period)
        elif attrib.name == "covid_deaths_county":
            return self._make_deaths_adrio_county(attrib, scope, time_period)
        elif attrib.name in ["covid_deaths_state", "influenza_deaths"]:
            return self._make_deaths_adrio_state(attrib, scope, time_period)
        else:
            raise GeoValidationException(f"Invalid attribute: {attrib.name}.")

    def _make_cases_adrio(self, attrib: AttributeDef, scope: CensusScope, time_period: SpecificTimePeriod) -> ADRIO:
        """
        Makes ADRIOs for HealthData dataset reporting COVID-19 cases per 100k population.
        Available between 2/24/2022 and 5/4/2023 at state and county granularities.
        https://healthdata.gov/dataset/United-States-COVID-19-Community-Levels-by-County/nn5b-j5u9/about_data
        """
        if time_period.start_date <= date(2022, 2, 17) or time_period.end_date >= date(2023, 5, 11):
            msg = "COVID cases data is only available between 2/24/2022 and 5/4/2023."
            raise DataResourceException(msg)

        def fetch() -> NDArray:
            info = QueryInfo("https://data.cdc.gov/resource/3nnm-4jni.csv?",
                             "date_updated", "county_fips", self.attribute_cols[attrib.name])

            df = self._api_query(info, scope.get_node_ids(),
                                 time_period, scope.granularity)

            df.rename(columns={'county_fips': 'fips'}, inplace=True)

            if scope.granularity == 'state':
                df['fips'] = [STATE.extract(x) for x in df['fips']]

                df = df.groupby(['date_updated', 'fips']).sum()
                df.reset_index(inplace=True)

            df = df.pivot(index='date_updated', columns='fips', values=info.data_col)

            array = np.array(list(zip(df.index.values, df.to_numpy(dtype=attrib.dtype))), dtype=[
                             ('date', object), ('data', object)])

            return np.array(
                [[(tick[0], node) for node in tick[1]] for tick in array],
                dtype=[('date', object), ('data', attrib.dtype)]
            )

        return ADRIO(attrib.name, fetch)

    def _make_facility_hospitalization_adrio(self, attrib: AttributeDef, scope: CensusScope, time_period: SpecificTimePeriod) -> ADRIO:
        """
        Makes ADRIOs for HealthData dataset reporting number of people hospitalized for COVID-19 
        and other respiratory illnesses at facility level during manditory reporting period.
        Available between 12/13/2020 and 5/10/2023 at state and county granularities.
        https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/anag-cw7u/about_data
        """
        if time_period.start_date <= date(2020, 12, 6) or time_period.end_date >= date(2023, 5, 17):
            msg = "Facility level hospitalization data is only available between 12/13/2020 and 5/10/2023."
            raise DataResourceException(msg)

        def fetch() -> NDArray:
            info = QueryInfo("https://healthdata.gov/resource/anag-cw7u.csv?",
                             "collection_week", "fips_code", self.attribute_cols[attrib.name])

            df = self._api_query(info, scope.get_node_ids(),
                                 time_period, scope.granularity)

            if scope.granularity == 'state':
                df['fips_code'] = [STATE.extract(x) for x in df['fips_code']]

            df = df.groupby(['collection_week', 'fips_code']).sum()
            df.reset_index(inplace=True)
            df = df.pivot(index='collection_week',
                          columns='fips_code', values=info.data_col)

            df[df < 0] = -999999
            df.fillna(0, inplace=True)

            array = np.array(list(zip(df.index.values, df.to_numpy(dtype=attrib.dtype))), dtype=[
                             ('date', object), ('data', object)])

            return np.array(
                [[(tick[0], node) for node in tick[1]] for tick in array],
                dtype=[('date', object), ('data', attrib.dtype)]
            )

        return ADRIO(attrib.name, fetch)

    def _make_state_hospitalization_adrio(self, attrib: AttributeDef, scope: CensusScope, time_period: SpecificTimePeriod) -> ADRIO:
        """
        Makes ADRIOs for CDC dataset reporting number of people hospitalized for COVID-19 
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

        def fetch() -> NDArray:
            if time_period.end_date >= date(2024, 5, 1):
                warn("State level hospitalization data is voluntary past 5/1/2024.")

            info = QueryInfo("https://data.cdc.gov/resource/aemt-mg7g.csv?",
                             "week_end_date", "jurisdiction", self.attribute_cols[attrib.name], True)

            state_mapping = state_fips_to_code(scope.year)
            fips = scope.get_node_ids()
            state_codes = np.array([state_mapping[x] for x in fips])

            df = self._api_query(info, state_codes, time_period, scope.granularity)

            df = df.groupby(['week_end_date', 'jurisdiction']).sum()
            df.reset_index(inplace=True)
            df = df.pivot(index='week_end_date',
                          columns='jurisdiction', values=info.data_col)

            array = np.array(list(zip(df.index.values, df.to_numpy(dtype=attrib.dtype))), dtype=[
                             ('date', object), ('data', object)])

            return np.array(
                [[(tick[0], node) for node in tick[1]] for tick in array],
                dtype=[('date', object), ('data', attrib.dtype)]
            )

        return ADRIO(attrib.name, fetch)

    def _make_vaccination_adrio(self, attrib: AttributeDef, scope: CensusScope, time_period: SpecificTimePeriod) -> ADRIO:
        """
        Makes ADRIOs for CDC dataset reporting total COVID-19 vaccination numbers.
        Available between 12/13/2020 and 5/10/2024 at state and county granularities.
        https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh/about_data
        """
        if time_period.start_date <= date(2020, 12, 6) or time_period.end_date >= date(2024, 5, 17):
            msg = "Vaccination data is only available between 12/13/2020 and 5/10/2024."
            raise DataResourceException(msg)

        def fetch() -> NDArray:
            info = QueryInfo("https://data.cdc.gov/resource/8xkx-amqh.csv?",
                             "date", "fips", self.attribute_cols[attrib.name])

            df = self._api_query(info, scope.get_node_ids(),
                                 time_period, scope.granularity)

            df.fillna(0, inplace=True)

            df = df.pivot(index='date', columns='fips', values=info.data_col)

            array = np.array(list(zip(df.index.values, df.to_numpy(dtype=attrib.dtype))), dtype=[
                             ('date', object), ('data', object)])

            return np.array(
                [[(tick[0], node) for node in tick[1]] for tick in array],
                dtype=[('date', object), ('data', attrib.dtype)]
            )

        return ADRIO(attrib.name, fetch)

    def _make_deaths_adrio_county(self, attrib: AttributeDef, scope: CensusScope, time_period: SpecificTimePeriod) -> ADRIO:
        """
        Makes ADRIOs for CDC dataset reporting number of deaths from COVID-19.
        Available between 1/4/2020 and 4/5/2024 at state and county granularities.
        https://data.cdc.gov/NCHS/AH-COVID-19-Death-Counts-by-County-and-Week-2020-p/ite7-j2w7/about_data
        """
        if time_period.start_date <= date(2019, 12, 28) or time_period.end_date >= date(2024, 4, 12):
            msg = "County level deaths data is only available between 1/4/2020 and 4/5/2024."
            raise DataResourceException(msg)

        def fetch() -> NDArray:
            if scope.granularity == 'state':
                fips_col = 'stfips'
            else:
                fips_col = 'fips_code'

            info = QueryInfo("https://data.cdc.gov/resource/ite7-j2w7.csv?",
                             "week_ending_date", fips_col, self.attribute_cols[attrib.name])

            df = self._api_query(info, scope.get_node_ids(),
                                 time_period, scope.granularity)

            df.fillna(0, inplace=True)

            if scope.granularity == 'state':
                df = df.groupby(['week_ending_date', fips_col]).sum()
                df.reset_index(inplace=True)

            df = df.pivot(index='week_ending_date',
                          columns=fips_col, values=info.data_col)

            array = np.array(list(zip(df.index.values, df.to_numpy(dtype=attrib.dtype))), dtype=[
                             ('date', object), ('data', object)])

            return np.array(
                [[(tick[0], node) for node in tick[1]] for tick in array],
                dtype=[('date', object), ('data', attrib.dtype)]
            )

        return ADRIO(attrib.name, fetch)

    def _make_deaths_adrio_state(self, attrib: AttributeDef, scope: CensusScope, time_period: SpecificTimePeriod) -> ADRIO:
        """
        Makes ADRIOs for CDC dataset reporting number of deaths from COVID-19 and other respiratory illnesses.
        Available from 1/4/2020 to present at state granularity.
        https://data.cdc.gov/NCHS/Provisional-COVID-19-Death-Counts-by-Week-Ending-D/r8kw-7aab/about_data
        """
        if time_period.start_date <= date(2019, 12, 29):
            msg = "State level deaths data is only available starting 1/4/2020."
            raise DataResourceException(msg)

        def fetch() -> NDArray:
            fips = scope.get_node_ids()
            states = get_us_states(scope.year)
            state_mapping = dict(zip(states.geoid, states.name))
            state_names = np.array([state_mapping[x] for x in fips])

            info = QueryInfo("https://data.cdc.gov/resource/r8kw-7aab.csv?",
                             "end_date", "state", self.attribute_cols[attrib.name], True)

            df = self._api_query(info, state_names, time_period, scope.granularity)

            df = df.groupby(['end_date', 'state']).sum()
            df.reset_index(inplace=True)
            df = df.pivot(index='end_date', columns='state', values=info.data_col)

            array = np.array(list(zip(df.index.values, df.to_numpy(dtype=attrib.dtype))), dtype=[
                             ('date', object), ('data', object)])

            return np.array(
                [[(tick[0], node) for node in tick[1]] for tick in array],
                dtype=[('date', object), ('data', attrib.dtype)]
            )

        return ADRIO(attrib.name, fetch)

    def _api_query(self, info: QueryInfo, fips: NDArray, time_period: SpecificTimePeriod, granularity: CensusGranularityName) -> DataFrame:
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

        current_return = 10000
        total_returned = 0
        df = DataFrame()
        while current_return == 10000:
            urls = [
                info.url_base + urlencode(
                    quote_via=quote,
                    safe=",()'$:",
                    query={
                        '$select': f'{info.date_col},{info.fips_col},{info.data_col}',
                        '$where': f"{loc_clause} AND {date_clause}",
                        '$limit': 10000,
                        '$offset': total_returned
                    })
                for loc_clause in location_clauses
            ]

            df = concat([df] + [
                read_csv(url, dtype={info.fips_col: str})
                for url in urls]
            )

            current_return = len(df.index) - total_returned
            total_returned += current_return

        df[info.date_col] = df[info.date_col].apply(lambda x: str(x)[:10])
        df = df.sort_values(by=[info.date_col, info.fips_col])
        return df
