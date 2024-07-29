import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, read_excel

from epymorph.error import DataResourceException
from epymorph.geo.adrio.adrio2 import Adrio
from epymorph.geography.us_census import (BlockGroupScope, CensusScope,
                                          StateScope, StateScopeAll,
                                          TractScope)


def _fetch_commuters(scope: CensusScope, year: int) -> DataFrame:
    """Utility function to fetch commuting data from .xslx format filtered down to requested regions."""
    # check for invalid granularity
    if isinstance(scope, TractScope | BlockGroupScope):
        msg = "Commuting data cannot be retrieved for tract or block group granularities"
        raise DataResourceException(msg)

    # check for valid year
    if year not in [2010, 2015, 2020]:
        # if invalid year is close to a valid year, fetch valid data and notify user
        passed_year = year
        if year in range(2008, 2012):
            year = 2010
        elif year in range(2013, 2017):
            year = 2015
        elif year in range(2018, 2022):
            year = 2020
        else:
            msg = "Invalid year. Communting data is only available for 2008-2022"
            raise DataResourceException(msg)

        print(
            f"Commuting data cannot be retrieved for {passed_year}, fetching {year} data instead.")

    if year != 2010:
        url = f'https://www2.census.gov/programs-surveys/demo/tables/metro-micro/{year}/commuting-flows-{year}/table1.xlsx'

        # organize dataframe column names
        group_fields = ['state_code',
                        'county_code',
                        'state',
                        'county']

        all_fields = ['res_' + field for field in group_fields] + \
            ['wrk_' + field for field in group_fields] + \
            ['workers', 'moe']

        header_num = 7

    else:
        url = 'https://www2.census.gov/programs-surveys/demo/tables/metro-micro/2010/commuting-employment-2010/table1.xlsx'

        all_fields = ['res_state_code', 'res_county_code', 'wrk_state_code', 'wrk_county_code',
                      'workers', 'moe', 'res_state', 'res_county', 'wrk_state', 'wrk_county']

        header_num = 4

    # download communter data spreadsheet as a pandas dataframe
    data = read_excel(url, header=header_num, names=all_fields, dtype={
        'res_state_code': str, 'wrk_state_code': str, 'res_county_code': str, 'wrk_county_code': str})

    node_ids = scope.get_node_ids()
    match scope.granularity:
        case 'state':
            data.rename(columns={'res_state_code': 'res_geoid',
                        'wrk_state_code': 'wrk_geoid'}, inplace=True)

        case 'county':
            data['res_geoid'] = data['res_state_code'] + \
                data['res_county_code']
            data['wrk_geoid'] = data['wrk_state_code'] + \
                data['wrk_county_code']

        case _:
            raise DataResourceException("Unsupported query.")

    data = data[data['res_geoid'].isin(node_ids)]
    data = data[data['wrk_geoid'].isin(['0' + x for x in node_ids])]

    return data


class Commuters(Adrio[np.int64]):
    """Makes an ADRIO to retrieve ACS commuting flow data."""

    def evaluate(self) -> NDArray[np.int64]:
        scope = self.scope

        if not isinstance(scope, CensusScope):
            msg = "Census scope is required for commuting flows data."
            raise DataResourceException(msg)

        df = _fetch_commuters(scope, scope.year)

        if isinstance(scope, StateScope | StateScopeAll):
            # group and aggregate data
            data_group = df.groupby(['res_geoid', 'wrk_geoid'])
            df = data_group.agg({'workers': 'sum'})
            df.reset_index(inplace=True)

        df = df.pivot(index='res_geoid', columns='wrk_geoid', values='workers')
        df.fillna(0, inplace=True)

        return df.to_numpy(dtype=np.int64)
