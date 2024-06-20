from typing import Any

from numpy.typing import NDArray
from pandas import read_csv

from epymorph.data_shape import Shapes
from epymorph.error import GeoValidationException
from epymorph.geo.adrio.adrio import ADRIO, ADRIOMaker
from epymorph.geo.spec import TimePeriod
from epymorph.geography.scope import GeoScope
from epymorph.geography.us_census import (STATE, CountyScope, StateScope,
                                          StateScopeAll)
from epymorph.simulation import AttributeDef, geo_attrib


class ADRIOMakerHHS(ADRIOMaker):

    attributes = [
        geo_attrib("covid_hospitalization_avg", float, Shapes.TxN)
    ]

    attribute_cols = {
        "covid_hospitalization_avg": "inpatient_beds_used_covid_7_day_avg"
    }

    def accepts_source(self, source: Any) -> bool:
        return False

    def make_adrio(self, attrib: AttributeDef, scope: GeoScope, time_period: TimePeriod) -> ADRIO:
        if not isinstance(scope, StateScope | StateScopeAll | CountyScope):
            msg = "Hospitalization data requires a CensusScope and can only be retrieved for state and county granularities"
            raise GeoValidationException(msg)

        def fetch() -> NDArray:
            fips = '\'' + '\',\''.join(scope.get_node_ids()) + '\''
            col_name = self.attribute_cols[attrib.name]

            url = f"https://healthdata.gov/resource/anag-cw7u.csv?$select=fips_code,{col_name}&$where=fips_code%20in({fips})&$limit=1045406"

            df = read_csv(url, dtype={'fips_code': str})

            if isinstance(scope, StateScopeAll) or scope.includes_granularity == 'state':
                df['fips_code'] = [STATE.extract(x) for x in df['fips_code']]

            df = df.groupby('fips_code')
            df = df.agg({col_name: 'sum'})
            df.reset_index(inplace=True)

            return df[col_name].to_numpy(dtype=float)

        return ADRIO(attrib.name, fetch)
