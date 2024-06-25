from datetime import datetime
from typing import Any

from numpy.typing import NDArray
from pandas import concat, read_csv

from epymorph.data_shape import Shapes
from epymorph.error import GeoValidationException
from epymorph.geo.adrio.adrio import ADRIO, ADRIOMaker
from epymorph.geo.spec import SpecificTimePeriod, TimePeriod
from epymorph.geography.scope import GeoScope
from epymorph.geography.us_census import (STATE, CensusScope, CountyScope,
                                          StateScope, StateScopeAll)
from epymorph.simulation import AttributeDef, geo_attrib


class ADRIOMakerCDC(ADRIOMaker):
    """
    CDC ADRIO template to serve as a parent class for ADRIOs that fetch data from various 
    HealthData and CDC datasets.
    """

    attributes = [
        geo_attrib("covid_cases_per_100k", int, Shapes.N),
        geo_attrib("covid_hospitalizations_per_100k", int, Shapes.N),
        geo_attrib("covid_hospitalization_avg", float, Shapes.N),
        geo_attrib("covid_hospitalization_sum", int, Shapes.N),
        geo_attrib("influenza_hospitalization_avg", float, Shapes.N),
        geo_attrib("influenza_hospitalization_sum", int, Shapes.N),
        geo_attrib("full_covid_vaccinations", int, Shapes.N),
        geo_attrib("one_dose_covid_vaccinations", int, Shapes.N),
        geo_attrib("covid_booster_doses", int, Shapes.N),
        geo_attrib("covid_deaths", int, Shapes.N)
    ]

    attribute_cols = {
        "covid_cases_per_100k": "covid_cases_per_100k",
        "covid_hospitalizations_per_100k": "covid_hospital_addmissions_per_100k",
        "covid_hospitalization_avg": "total_adult_patients_hospitalized_confirmed_covid_7_day_avg",
        "covid_hospitalization_sum": "total_adult_patients_hospitalized_confirmed_covid_7_day_sum",
        "influenza_hospitalization_avg": "total_patients_hospitalized_confirmed_influenza_7_day_avg",
        "influenza_hospitalization_sum": "total_patients_hospitalized_confirmed_influenza_7_day_sum",
        "full_covid_vaccinations": "series_complete_yes",
        "one_dose_covid_vaccinations": "administered_dose1_recip",
        "covid_booster_doses": "booster_doses",
        "covid_deaths": "covid_19_deaths"
    }

    def accepts_source(self, source: Any) -> bool:
        return False

    def make_adrio(self, attrib: AttributeDef, scope: GeoScope, time_period: TimePeriod) -> ADRIO:
        if not isinstance(scope, StateScope | StateScopeAll | CountyScope):
            msg = "CDC data requires a CensusScope object and can only be retrieved for state and county granularities."
            raise GeoValidationException(msg)

        if attrib.name in ["covid_cases_per_100k", "covid_hospitalizations_per_100k"]:
            return self._make_cases_adrio(attrib, scope, time_period)
        elif attrib.name in ["covid_hospitalization_avg", "covid_hospitalization_sum", "influenza_hospitalization_avg", "influenza_hospitalization_sum"]:
            return self._make_hospitalization_adrio(attrib, scope, time_period)
        elif attrib.name in ["full_covid_vaccinations", "one_dose_covid_vaccinations", "covid_booster_doses"]:
            return self._make_vaccination_adrio(attrib, scope, time_period)
        elif attrib.name == "covid_deaths":
            return self._make_deaths_adrio(attrib, scope, time_period)
        else:
            raise GeoValidationException(f"Invalid attribute: {attrib.name}.")

    def _make_cases_adrio(self, attrib: AttributeDef, scope: CensusScope, time_period: TimePeriod) -> ADRIO:
        """
        Makes ADRIOs for HealthData dataset reporting COVID-19 cases per 100k population.
        https://healthdata.gov/dataset/United-States-COVID-19-Community-Levels-by-County/nn5b-j5u9/about_data
        """
        def fetch() -> NDArray:
            col_name = self.attribute_cols[attrib.name]
            if scope.granularity == 'state':
                fips = scope.get_node_ids()
                urls = [f"https://data.cdc.gov/resource/3nnm-4jni.csv?$select=date_updated,county_fips,{col_name}&$where=starts_with(county_fips,\'{state}\')&$limit=206334"
                        for state in fips]
                df = concat(read_csv(url, dtype={'county_fips': str}) for url in urls)
            else:
                fips = '\'' + '\',\''.join(scope.get_node_ids()) + '\''
                url = f"https://data.cdc.gov/resource/3nnm-4jni.csv?$select=date_updated,county_fips,{col_name}&$where=county_fips%20in({fips})&$limit=206334"
                df = read_csv(url, dtype={'county_fips': str})

            df['date_updated'] = [datetime.fromisoformat(
                week).date() for week in df['date_updated']]

            df.rename(columns={'county_fips': 'fips'}, inplace=True)

            if isinstance(time_period, SpecificTimePeriod):
                df = df[df['date_updated'] >= time_period.start_date]
                df = df[df['date_updated'] < time_period.end_date]

            if scope.granularity == 'state':
                df['fips'] = [STATE.extract(x) for x in df['fips']]

            df = df.groupby('fips')
            df = df.agg({col_name: 'sum'})
            df.reset_index(inplace=True)

            return df[col_name].to_numpy()

        return ADRIO(attrib.name, fetch)

    def _make_hospitalization_adrio(self, attrib: AttributeDef, scope: CensusScope, time_period: TimePeriod) -> ADRIO:
        """
        Makes ADRIOs for HealthData dataset reporting number of people hospitalized for COVID-19 
        and other respiratory illnesses.
        https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/anag-cw7u/about_data
        """
        def fetch() -> NDArray:
            col_name = self.attribute_cols[attrib.name]

            if scope.granularity == 'state':
                fips = scope.get_node_ids()
                urls = [f"https://healthdata.gov/resource/anag-cw7u.csv?$select=collection_week,fips_code,{col_name}&$where=starts_with(fips_code,\'{state}\')&$limit=1045406"
                        for state in fips]
                df = concat(read_csv(url, dtype={'fips_code': str}) for url in urls)
            else:
                fips = '\'' + '\',\''.join(scope.get_node_ids()) + '\''
                url = f"https://healthdata.gov/resource/anag-cw7u.csv?$select=collection_week,fips_code,{col_name}&$where=fips_code%20in({fips})&$limit=1045406"
                df = read_csv(url, dtype={'fips_code': str})

            df['collection_week'] = [datetime.fromisoformat(
                week.replace('/', '-')).date() for week in df['collection_week']]

            if isinstance(time_period, SpecificTimePeriod):
                df = df[df['collection_week'] >= time_period.start_date]
                df = df[df['collection_week'] < time_period.end_date]

            if scope.granularity == 'state':
                df['fips_code'] = [STATE.extract(x) for x in df['fips_code']]

            df.replace(-999999, 0, inplace=True)

            df = df.groupby('fips_code')
            df = df.agg({col_name: 'sum'})
            df.reset_index(inplace=True)

            return df[col_name].to_numpy(dtype=attrib.dtype)

        return ADRIO(attrib.name, fetch)

    def _make_vaccination_adrio(self, attrib: AttributeDef, scope: CensusScope, time_period: TimePeriod) -> ADRIO:
        """
        Makes ADRIOs for CDC dataset reporting total COVID-19 vaccination numbers.
        https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh/about_data
        """
        def fetch() -> NDArray:
            col_name = self.attribute_cols[attrib.name]

            if scope.granularity == 'state':
                fips = scope.get_node_ids()
                urls = [f"https://data.cdc.gov/resource/8xkx-amqh.csv?$select=date,fips,{col_name}&$where=starts_with(fips,\'{state}\')&$limit=1962781"
                        for state in fips]
                df = concat(read_csv(url, dtype={'fips': str}) for url in urls)
                df['fips'] = [STATE.extract(x) for x in df['fips']]
                df = df.groupby(['date', 'fips'])
                df = df.agg({col_name: 'sum'})
                df.reset_index(inplace=True)
            else:
                fips = '\'' + '\',\''.join(scope.get_node_ids()) + '\''
                url = f"https://data.cdc.gov/resource/8xkx-amqh.csv?$select=date,fips,{col_name}&$where=fips%20in({fips})&$limit=1962781"
                df = read_csv(url, dtype={'fips': str})

            df['date'] = [datetime.fromisoformat(
                week.replace('/', '-')).date() for week in df['date']]

            if isinstance(time_period, SpecificTimePeriod):
                df = df[df['date'] >= time_period.start_date]
                df = df[df['date'] < time_period.end_date]

            return df[col_name].to_numpy(dtype=attrib.dtype)

        return ADRIO(attrib.name, fetch)

    def _make_deaths_adrio(self, attrib: AttributeDef, scope: CensusScope, time_period: TimePeriod) -> ADRIO:
        """
        Makes ADRIOs for CDC dataset reporting number of deaths from COVID-19.
        https://data.cdc.gov/NCHS/AH-COVID-19-Death-Counts-by-County-and-Week-2020-p/ite7-j2w7/about_data
        """
        if not isinstance(scope, StateScope | StateScopeAll | CountyScope):
            msg = "Deaths data requires a CensusScope object and can only be retrieved for state and county granularities."
            raise GeoValidationException(msg)

        def fetch() -> NDArray:
            fips = ','.join(scope.get_node_ids())

            if scope.granularity == 'state':
                fips_col = 'stfips'
            else:
                fips_col = 'fips_code'

            data_col = self.attribute_cols[attrib.name]

            url = f"https://data.cdc.gov/resource/ite7-j2w7.csv?$select=week_ending_date,{fips_col},{data_col}&$where={fips_col}%20in({fips})&$limit=534140"

            df = read_csv(url)

            df['week_ending_date'] = [datetime.fromisoformat(
                week.replace('/', '-')).date() for week in df['week_ending_date']]

            if isinstance(time_period, SpecificTimePeriod):
                df = df[df['week_ending_date'] >= time_period.start_date]
                df = df[df['week_ending_date'] < time_period.end_date]

            df.fillna(0, inplace=True)

            df = df.groupby(fips_col)
            df = df.agg({data_col: 'sum'})
            df.reset_index(inplace=True)

            return df[data_col].to_numpy(attrib.dtype)

        return ADRIO(attrib.name, fetch)
