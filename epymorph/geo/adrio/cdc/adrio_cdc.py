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
from epymorph.simulation import AttributeDef, geo_attrib


class QueryInfo(NamedTuple):
    url_base: str
    date_col: str
    fips_col: str
    data_col: str
    state_level: bool = False


class ADRIOMakerCDC(ADRIOMaker):
    """
    CDC ADRIO template to serve as a parent class for ADRIOs that fetch data from various 
    HealthData and CDC datasets.
    """

    attributes = [
        geo_attrib("covid_cases_per_100k", int, Shapes.TxN,
                   comment='Number of COVID-19 cases per 100k population.'),
        geo_attrib("covid_hospitalizations_per_100k", int, Shapes.TxN,
                   comment='Number of COVID-19 hospitalizations per 100k population.'),
        geo_attrib("covid_hospitalization_avg_facility", float, Shapes.TxN,
                   comment='Weekly averages of COVID-19 hospitalizations from facility level dataset.'),
        geo_attrib("covid_hospitalization_sum_facility", int, Shapes.TxN,
                   comment='Weekly sums of all COVID-19 hospitalizations from facility level dataset.'),
        geo_attrib("influenza_hospitalization_avg_facility", float, Shapes.TxN,
                   comment='Weekly averages of influenza hospitalizations from facility level dataset.'),
        geo_attrib("influenza_hospitalization_sum_facility", int, Shapes.TxN,
                   comment='Weekly sums of influenza hospitalizations from facility level dataset.'),
        geo_attrib("covid_hospitalization_avg_state", float, Shapes.TxN,
                   comment='Weekly averages of COVID-19 hospitalizations from state level dataset.'),
        geo_attrib("covid_hospitalization_sum_state", int, Shapes.TxN,
                   comment='Weekly sums of COVID-19 hospitalizations from state level dataset.'),
        geo_attrib("influenza_hospitalization_avg_state", float, Shapes.TxN,
                   comment='Weekly averages of influenza hospitalizations from state level dataset.'),
        geo_attrib("influenza_hospitalization_sum_state", int, Shapes.TxN,
                   comment='Weekly sums of influenza hospitalizations from state level dataset.'),
        geo_attrib("full_covid_vaccinations", int, Shapes.TxN,
                   comment='Cumulative total number of individuals fully vaccinated for COVID-19.'),
        geo_attrib("one_dose_covid_vaccinations", int, Shapes.TxN,
                   comment='Cumulative total number of individuals with at least one dose of COVID-19 vaccination.'),
        geo_attrib("covid_booster_doses", int, Shapes.TxN,
                   comment='Cumulative total number of COVID-19 booster doses administered.'),
        geo_attrib("covid_deaths_county", int, Shapes.TxN,
                   comment='Weekly total COVID-19 deaths from county level dataset.'),
        geo_attrib("covid_deaths_state", int, Shapes.TxN,
                   comment='Weekly total COVID-19 deaths from state level dataset.'),
        geo_attrib("influenza_deaths", int, Shapes.TxN,
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

    def make_adrio(self, attrib: AttributeDef, scope: GeoScope, time_period: TimePeriod) -> ADRIO:
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
            # if scope.granularity == 'state':
            #     fips = scope.get_node_ids()
            #     urls = [f"https://data.cdc.gov/resource/3nnm-4jni.csv?$select=date_updated,county_fips,{col_name}&$where=starts_with(county_fips,'{state}')%20AND%20date_updated%20between%20'{time_period.start_date}T00:00:00'%20and%20'{time_period.end_date}T00:00:00'&$limit=206334"
            #             for state in fips]
            #     df = concat(read_csv(url, dtype={'county_fips': str}) for url in urls)
            # else:
            #     fips = "'" + "','".join(scope.get_node_ids()) + "'"
            #     params = urlencode({'$select': f'date_updated,county_fips,{col_name}', '$where': self._compress_where(
            #         fips, time_period), '$limit': 206334}, quote_via=quote, safe=",()'$:")
            #     url = "https://data.cdc.gov/resource/3nnm-4jni.csv?" + params
            #     # url = f"https://data.cdc.gov/resource/3nnm-4jni.csv?$select=date_updated,county_fips,{col_name}&$where=county_fips%20in({fips})%20AND%20date_updated%20between%20'{time_period.start_date}T00:00:00'%20and%20'{time_period.end_date}T00:00:00'&$limit=206334")
            #     df = read_csv(url, dtype={'county_fips': str})

            df.rename(columns={'county_fips': 'fips'}, inplace=True)

            if scope.granularity == 'state':
                df['fips'] = [STATE.extract(x) for x in df['fips']]

                df = df.groupby(['date_updated', 'fips']).sum()
                df.reset_index(inplace=True)

            df = df.pivot(index='date_updated', columns='fips', values=info.data_col)

            return df.to_numpy(dtype=attrib.dtype)

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
            # col_name = self.attribute_cols[attrib.name]

            # if scope.granularity == 'state':
            #     fips = scope.get_node_ids()
            #     urls = [f"https://healthdata.gov/resource/anag-cw7u.csv?$select=collection_week,fips_code,{col_name}&$where=starts_with(fips_code,'{state}')%20AND%20collection_week%20between%20'{time_period.start_date}T00:00:00'%20and%20'{time_period.end_date}T00:00:00'&$limit=1045406"
            #             for state in fips]
            #     df = concat(read_csv(url, dtype={'fips_code': str}) for url in urls)
            # else:
            #     fips = "'" + "','".join(scope.get_node_ids()) + "'"
            #     url = f"https://healthdata.gov/resource/anag-cw7u.csv?$select=collection_week,fips_code,{col_name}&$where=fips_code%20in({fips})%20AND%20collection_week%20between%20'{time_period.start_date}T00:00:00'%20and%20'{time_period.end_date}T00:00:00'&$limit=1045406"
            #     df = read_csv(url, dtype={'fips_code': str})

            if scope.granularity == 'state':
                df['fips_code'] = [STATE.extract(x) for x in df['fips_code']]

            df = df.groupby(['collection_week', 'fips_code']).sum()
            df.reset_index(inplace=True)
            df = df.pivot(index='collection_week',
                          columns='fips_code', values=info.data_col)

            df[df < 0] = -999999
            df.fillna(0, inplace=True)

            return df.to_numpy(dtype=attrib.dtype)

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
            # state_codes = ",".join(f"'{state_mapping[x]}'" for x in fips)
            state_codes = np.array([state_mapping[x] for x in fips])
            # col_name = self.attribute_cols[attrib.name]

            df = self._api_query(info, state_codes, time_period, scope.granularity)

            # url = f"https://data.cdc.gov/resource/aemt-mg7g.csv?$select=week_end_date,jurisdiction,{col_name}&$where=jurisdiction%20in({state_codes})%20AND%20week_end_date%20between%20'{time_period.start_date}T00:00:00'%20and%20'{time_period.end_date}T00:00:00'&$limit=11514"
            # df = read_csv(url)

            df = df.groupby(['week_end_date', 'jurisdiction']).sum()
            df.reset_index(inplace=True)
            df = df.pivot(index='week_end_date',
                          columns='jurisdiction', values=info.data_col)

            return df.to_numpy(dtype=attrib.dtype)

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
            # col_name = self.attribute_cols[attrib.name]

            info = QueryInfo("https://data.cdc.gov/resource/8xkx-amqh.csv?",
                             "date", "fips", self.attribute_cols[attrib.name])

            df = self._api_query(info, scope.get_node_ids(),
                                 time_period, scope.granularity)

            # if scope.granularity == 'state':
            #     fips = scope.get_node_ids()
            #     urls = [f"https://data.cdc.gov/resource/8xkx-amqh.csv?$select=date,fips,{col_name}&$where=starts_with(fips,'{state}')%20AND%20date%20between%20'{time_period.start_date}T00:00:00'%20and%20'{time_period.end_date}T00:00:00'&$limit=1962781"
            #             for state in fips]
            #     df = concat(read_csv(url, dtype={'fips': str}) for url in urls)
            #     df['fips'] = [STATE.extract(x) for x in df['fips']]
            #     df = df.groupby(['date', 'fips']).sum()
            #     df.reset_index(inplace=True)
            # else:
            #     fips = "'" + "','".join(scope.get_node_ids()) + "'"
            #     url = f"https://data.cdc.gov/resource/8xkx-amqh.csv?$select=date,fips,{col_name}&$where=fips%20in({fips})%20AND%20date%20between%20'{time_period.start_date}T00:00:00'%20and%20'{time_period.end_date}T00:00:00'&$limit=1962781"
            #     df = read_csv(url, dtype={'fips': str})

            df.fillna(0, inplace=True)

            df = df.pivot(index='date', columns='fips', values=info.data_col)

            return df.to_numpy(dtype=attrib.dtype)

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
            # fips = ','.join(scope.get_node_ids())

            if scope.granularity == 'state':
                fips_col = 'stfips'
            else:
                fips_col = 'fips_code'

            info = QueryInfo("https://data.cdc.gov/resource/ite7-j2w7.csv?",
                             "week_ending_date", fips_col, self.attribute_cols[attrib.name])

            df = self._api_query(info, scope.get_node_ids(),
                                 time_period, scope.granularity)
            # data_col = self.attribute_cols[attrib.name]

            # url = f"https://data.cdc.gov/resource/ite7-j2w7.csv?$select=week_ending_date,{fips_col},{data_col}&$where={fips_col}%20in({fips})%20AND%20week_ending_date%20between%20'{time_period.start_date}T00:00:00'%20and%20'{time_period.end_date}T00:00:00'&$limit=534140"

            # df = read_csv(url)

            df.fillna(0, inplace=True)

            if scope.granularity == 'state':
                df = df.groupby(['week_ending_date', fips_col]).sum()
                df.reset_index(inplace=True)

            df = df.pivot(index='week_ending_date',
                          columns=fips_col, values=info.data_col)

            return df.to_numpy(attrib.dtype)

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
            # state_names = ",".join(f"'{state_mapping[x]}'" for x in fips)
            # col_name = self.attribute_cols[attrib.name]

            info = QueryInfo("https://data.cdc.gov/resource/r8kw-7aab.csv?",
                             "end_date", "state", self.attribute_cols[attrib.name], True)

            df = self._api_query(info, state_names, time_period, scope.granularity)

            # url = f"https://data.cdc.gov/resource/r8kw-7aab.csv?$select=end_date,state,{col_name}&$where=state%20in({state_names})%20AND%20end_date%20between%20'{time_period.start_date}T00:00:00'%20and%20'{time_period.end_date}T00:00:00'&$limit=15822"

            # df = read_csv(url)

            df = df.groupby(['end_date', 'state']).sum()
            df.reset_index(inplace=True)
            df = df.pivot(index='end_date', columns='state', values=info.data_col)

            return df.to_numpy(dtype=attrib.dtype)

        return ADRIO(attrib.name, fetch)

    def _api_query(self, info: QueryInfo, fips: NDArray, time_period: SpecificTimePeriod, granularity: CensusGranularityName) -> DataFrame:
        if granularity == 'state' and not info.state_level:
            where = [f"starts_with({info.fips_col}, '{state}') AND {info.date_col} between " +
                     f"'{time_period.start_date}T00:00:00' and '{time_period.end_date}T00:00:00'" for state in fips]
            params = [urlencode({'$select': f'{info.date_col},{info.fips_col},{info.data_col}',
                                '$where': state, '$limit': 206334}, quote_via=quote, safe=",()'$:") for state in where]
            urls = [info.url_base + state_params for state_params in params]
            return concat(read_csv(url, dtype={f'{info.fips_col}': str}) for url in urls)
        else:
            formatted_fips = "'" + "','".join(fips) + "'"
            where = f"{info.fips_col} in ({formatted_fips}) AND {info.date_col} between " + \
                    f"'{time_period.start_date}T00:00:00' and '{time_period.end_date}T00:00:00'"
            params = urlencode({'$select': f'{info.date_col},{info.fips_col},{info.data_col}',
                               '$where': where, '$limit': 206334}, quote_via=quote, safe=",()'$:")
            url = info.url_base + params
            return read_csv(url, dtype={f'{info.fips_col}': str})

    # def _compress_where(self, info: QueryInfo, fips: str, time_period: SpecificTimePeriod, granularity: Literal['state', 'county']) -> str:
    #     if granularity == 'state':
    #         return f"starts_with({info.fips_col},'{fips}') AND {info.date_col} between '{time_period.start_date}T00:00:00' and '{time_period.end_date}T00:00:00'"
    #     else:
    #         return f"{info.fips_col} in({fips}) AND {info.date_col} between '{time_period.start_date}T00:00:00' and '{time_period.end_date}T00:00:00'"
