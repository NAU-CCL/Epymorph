import os
from collections import defaultdict

import numpy as np
from census import Census
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from pandas import DataFrame, concat, read_excel
from pygris import block_groups, counties, states, tracts

from epymorph.data_shape import Shapes
from epymorph.data_type import CentroidDType
from epymorph.error import GeoValidationException
from epymorph.geo.adrio.adrio import ADRIO, ADRIOMaker
from epymorph.geo.spec import AttributeDef, TimePeriod, Year
from epymorph.geography.us_census import (BLOCK_GROUP, COUNTY, STATE, TRACT,
                                          BlockGroupScope, CensusScope,
                                          CountyScope, StateScope,
                                          StateScopeAll, TractScope)
from epymorph.simulation import AttributeDef, geo_attrib

CENSUS_GRANULARITY_CODE = {'state': 0, 'county': 1, 'tract': 2, 'block group': 3}


class ADRIOMakerCensus(ADRIOMaker):
    """
    Census ADRIO template to serve as parent class and provide utility functions for Census-based ADRIOS.
    """

    population_query = ['B01001_003E',  # population 0-19
                        'B01001_004E',
                        'B01001_005E',
                        'B01001_006E',
                        'B01001_007E',
                        'B01001_027E',  # women
                        'B01001_028E',
                        'B01001_029E',
                        'B01001_030E',
                        'B01001_031E',
                        'B01001_008E',  # population 20-34
                        'B01001_009E',
                        'B01001_010E',
                        'B01001_011E',
                        'B01001_012E',
                        'B01001_032E',  # women
                        'B01001_033E',
                        'B01001_034E',
                        'B01001_035E',
                        'B01001_036E',
                        'B01001_013E',  # population 35-54
                        'B01001_014E',
                        'B01001_015E',
                        'B01001_016E',
                        'B01001_037E',  # women
                        'B01001_038E',
                        'B01001_039E',
                        'B01001_040E',
                        'B01001_017E',  # population 55-64
                        'B01001_018E',
                        'B01001_019E',
                        'B01001_041E',  # women
                        'B01001_042E',
                        'B01001_043E',
                        'B01001_020E',  # population 65-74
                        'B01001_021E',
                        'B01001_022E',
                        'B01001_044E',  # women
                        'B01001_045E',
                        'B01001_046E',
                        'B01001_023E',  # population 75+
                        'B01001_024E',
                        'B01001_025E',
                        'B01001_047E',  # women
                        'B01001_048E',
                        'B01001_049E']

    attributes = [
        geo_attrib('name', dtype=str, shape=Shapes.N,
                   comment='The proper name of the place.'),
        geo_attrib('population', dtype=int, shape=Shapes.N,
                   comment='The number of residents of the place.'),
        geo_attrib('population_by_age', dtype=int, shape=Shapes.NxA(3),
                   comment='The number of residents, divided into three age categories: 0-19, 20-64, 65+'),
        geo_attrib('population_by_age_x6', dtype=int, shape=Shapes.NxA(6),
                   comment='The number of residents, divided into six age categories: 0-19, 20-34, 35-54, 55-64, 65-75, 75+'),
        geo_attrib('centroid', dtype=CentroidDType, shape=Shapes.N,
                   comment='A geographic centroid for the place, in longitude/latitude.'),
        geo_attrib('geoid', dtype=str, shape=Shapes.N,
                   comment='The GEOID (in many cases synonymous with FIPS code) for the place.'),
        geo_attrib('average_household_size', dtype=int, shape=Shapes.N,
                   comment='Average household size within the place.'),
        geo_attrib('dissimilarity_index', dtype=float, shape=Shapes.N,
                   comment='An index describing the amount of racial segregation in the place, from 0 to 1.'),
        geo_attrib('commuters', dtype=int, shape=Shapes.NxN,
                   comment='The number of commuters between places, as reported by the ACS Commuting Flows data.'),
        geo_attrib('gini_index', dtype=float, shape=Shapes.N,
                   comment='An index describing wealth inequality in the place, from 0 to 1.'),
        geo_attrib('median_age', dtype=int, shape=Shapes.N,
                   comment='The median age of residents in the place.'),
        geo_attrib('median_income', dtype=int, shape=Shapes.N,
                   comment='The median income of residents in the place.'),
        geo_attrib('tract_median_income', dtype=int, shape=Shapes.N,
                   comment='The median income according to the Census Tract which encloses this place.'
                   'This attribute is only valid if the geo granularity is below tract.'),
        geo_attrib('pop_density_km2', dtype=float, shape=Shapes.N,
                   comment='The population density of this place by square kilometer.'),
    ]

    attrib_vars = {
        'name': ['NAME'],
        'geoid': ['NAME'],
        'population': ['B01001_001E'],
        'population_by_age': population_query,
        'population_by_age_x6': population_query,
        'median_income': ['B19013_001E'],
        'median_age': ['B01002_001E'],
        'tract_median_income': ['B19013_001E'],
        'dissimilarity_index': ['B03002_003E',  # white population
                                'B03002_013E',
                                'B03002_004E',  # minority population
                                'B03002_014E'],
        'average_household_size': ['B25010_001E'],
        'gini_index': ['B19083_001E'],
        'pop_density_km2': ['B01003_001E'],
    }

    census: Census
    """Census API interface object."""

    def __init__(self) -> None:
        """Initializer to create Census object."""
        api_key = os.environ.get('CENSUS_API_KEY')
        if api_key is None:
            msg = "Census API key not found. Please ensure you have an API key and have assigned it to an environment variable named 'CENSUS_API_KEY'"
            raise Exception(msg)
        self.census = Census(api_key)

    def make_adrio(self, attrib: AttributeDef, scope: CensusScope, time_period: TimePeriod) -> ADRIO:
        if attrib not in self.attributes:
            msg = f"{attrib.name} is not supported for the Census data source."
            raise GeoValidationException(msg)
        if not isinstance(time_period, Year):
            msg = f"Census ADRIO requires Year (TimePeriod), given {type(time_period)}."
            raise GeoValidationException(msg)

        year = time_period.year

        if attrib.name == 'geoid':
            return self._make_geoid_adrio(scope)
        elif attrib.name == 'population_by_age':
            return self._make_population_adrio(scope, 3)
        elif attrib.name == 'dissimilarity_index':
            return self._make_dissimilarity_index_adrio(scope)
        elif attrib.name == 'gini_index':
            return self._make_gini_index_adrio(scope)
        elif attrib.name == 'pop_density_km2':
            return self._make_pop_density_adrio(scope)
        elif attrib.name == 'centroid':
            return self._make_centroid_adrio(scope)
        elif attrib.name == 'tract_median_income':
            return self._make_tract_med_income_adrio(scope)
        elif attrib.name == 'commuters':
            return self._make_commuter_adrio(scope, year)
        else:
            return self._make_simple_adrios(attrib, scope)

    def fetch_acs5(self, variables: list[str], scope: CensusScope, granularity: int | None = None) -> DataFrame:
        """Utility function to fetch Census data by building queries from ADRIO data."""
        queries = []
        match scope:
            case StateScopeAll():
                queries = [{"for": "state:*"}]

            case StateScope('state', includes):
                queries = [{"for": f"state:{','.join(includes)}"}]

            case CountyScope('state', includes):
                queries = [{
                    "for": "county:*",
                    "in": f"state:{','.join(includes)}",
                }]

            case CountyScope('county', includes):
                # NOTE: this is a case where our scope results in multiple queries!
                counties_by_state: dict[str, list[str]] = defaultdict(list)
                for state, county in map(COUNTY.decompose, includes):
                    counties_by_state[state].append(county)
                queries = [
                    {"for": f"county:{','.join(cs)}", "in": f"state:{s}"}
                    for s, cs in counties_by_state.items()
                ]

            case TractScope('state', includes):
                queries = [{
                    "for": "tract:*",
                    "in": f"state:{','.join(includes)} county:*",
                }]

            case TractScope('county', includes):
                counties_by_state: dict[str, list[str]] = defaultdict(list)
                for state, county in map(COUNTY.decompose, includes):
                    counties_by_state[state].append(county)
                queries = [
                    {"for": "tract:*",
                        "in": f"state:{s} county:{','.join(cs)}"}
                    for s, cs in counties_by_state.items()
                ]

            case TractScope('tract', includes):
                counties_by_state: dict[str, list[str]] = defaultdict(list)
                tracts_by_county: dict[str, list[str]] = defaultdict(list)

                for state, county, tract in map(TRACT.decompose, includes):
                    if state not in counties_by_state:
                        counties_by_state[state].append(county)
                    tracts_by_county[county].append(tract)

                queries = []
                for state, counties in counties_by_state.items():
                    for cs in counties:
                        queries.append(
                            {"for": f"tract:{','.join(tracts_by_county[cs])}",
                             "in": f"state:{state} county:{cs}"}
                        )

            case BlockGroupScope('state', includes):
                # This wouldn't normally need to be multiple queries,
                # but Census API won't let you fetch CBGs for multiple states.
                states = {STATE.extract(x) for x in includes}
                queries = [
                    {"for": "block group:*", "in": f"state:{s} county:* tract:*"}
                    for s in states
                ]

            case BlockGroupScope('county', includes):
                counties_by_state: dict[str, list[str]] = defaultdict(list)
                for state, county in map(COUNTY.decompose, includes):
                    counties_by_state[state].append(county)
                queries = [
                    {"for": "block group:*",
                        "in": f"state:{s} county:{','.join(cs)} tract:*"}
                    for s, cs in counties_by_state.items()
                ]

            case BlockGroupScope('tract', includes):
                counties_by_state: dict[str, list[str]] = defaultdict(list)
                tracts_by_county: dict[str, list[str]] = defaultdict(list)

                for state, county, tract in map(TRACT.decompose, includes):
                    if state not in counties_by_state or county not in counties_by_state[state]:
                        counties_by_state[state].append(county)
                    tracts_by_county[county].append(tract)

                queries = []
                for state, counties in counties_by_state.items():
                    for cs in counties:
                        queries.append(
                            {"for": f"block group:*",
                                "in": f"state:{state} county:{cs} tract:{','.join(tracts_by_county[cs])}"}
                        )

            case BlockGroupScope('block group', includes):
                counties_by_state: dict[str, list[str]] = defaultdict(list)
                tracts_by_county: dict[str, list[str]] = defaultdict(list)
                block_groups_by_tract: dict[str, list[str]] = defaultdict(list)

                for state, county, tract, block_group in map(BLOCK_GROUP.decompose, includes):
                    if state not in counties_by_state or county not in counties_by_state[state]:
                        counties_by_state[state].append(county)
                    if county not in tracts_by_county or tract not in tracts_by_county[county]:
                        tracts_by_county[county].append(tract)
                    block_groups_by_tract[tract].append(block_group)

                queries = []
                for state, counties in counties_by_state.items():
                    for cs, tracts in tracts_by_county.items():
                        for tcs in tracts:
                            if cs in counties_by_state[state]:
                                queries.append(
                                    {"for": f"block group:{','.join(block_groups_by_tract[tcs])}",
                                     "in": f"state:{state} county:{cs} tract:{tcs}"}
                                )

            case _:
                raise Exception("Unsupported query.")

        # check if granularity other than scope granularity requested
        if granularity is None:
            granularity = CENSUS_GRANULARITY_CODE[scope.granularity]

        # fetch and combine all queries
        results_df = concat([
            DataFrame.from_records(
                self.census.acs5.get(
                    variables, granularity=granularity, geo=query, year=scope.year)
            )
            for query in queries
        ])

        columns: list[str] = {
            'state': ['state'],
            'county': ['state', 'county'],
            'tract': ['state', 'county', 'tract'],
            'block group': ['state', 'county', 'tract', 'block group'],
        }[scope.granularity]
        df = results_df.loc[:, variables]
        df['geoid'] = results_df[columns].apply(lambda xs: ''.join(xs), axis=1)
        return df

    def fetch_sf(self, scope: CensusScope) -> GeoDataFrame:
        """Utility function to fetch shape files from Census for specified regions."""

        # call appropriate pygris function based on granularity and sort result
        match scope:
            case StateScopeAll() | StateScope():
                data_df = states(year=scope.year)
                data_df = data_df.rename(columns={'STATEFP': 'state'})
                if isinstance(scope, StateScope):
                    data_df = data_df.loc[data_df['state'].isin(scope.includes)]

                sort_param = ['state']

            case CountyScope('state', includes):
                data_df = counties(state=includes, year=scope.year)
                data_df = data_df.rename(
                    columns={'STATEFP': 'state', 'COUNTYFP': 'county'})

                sort_param = ['state', 'county']

            case CountyScope('county', includes):
                state = list({STATE.extract(x) for x in includes})
                data_df = counties(state=state, year=scope.year)
                data_df = data_df.rename(
                    columns={'STATEFP': 'state', 'COUNTYFP': 'county'})
                data_df['county_full'] = data_df['state'] + data_df['county']
                data_df = data_df.loc[data_df['county_full'].isin(includes)]

                sort_param = ['state', 'county']

            case TractScope('state', includes):
                data_df = concat(tracts(state=s, year=scope.year) for s in includes)
                data_df = data_df.rename(
                    columns={'STATEFP': 'state', 'COUNTYFP': 'county', 'TRACTCE': 'tract'})

                sort_param = ['state', 'county', 'tract']

            case TractScope('county', includes):
                data_df = concat(tracts(state=s, county=c)
                                 for s, c in map(COUNTY.decompose, includes))

                data_df = data_df.rename(
                    columns={'STATEFP': 'state', 'COUNTYFP': 'county', 'TRACTCE': 'tract'})

                sort_param = ['state', 'county', 'tract']

            case TractScope('tract', includes):
                data_df = concat(tracts(state=s, county=c)
                                 for s, c, t in map(TRACT.decompose, includes))

                data_df = data_df.rename(
                    columns={'STATEFP': 'state', 'COUNTYFP': 'county', 'TRACTCE': 'tract'})
                data_df['tract_full'] = data_df['state'] + \
                    data_df['county'] + data_df['tract']
                data_df = data_df.loc[data_df['tract_full'].isin(includes)]

                sort_param = ['state', 'county', 'tract']

            case BlockGroupScope('state', includes):
                data_df = concat(block_groups(state=s, year=scope.year)
                                 for s in includes)
                data_df = data_df.rename(columns={
                                         'STATEFP': 'state', 'COUNTYFP': 'county', 'TRACTCE': 'tract', 'BLKGRPCE': 'block group'})

                sort_param = ['state', 'county', 'tract', 'block group']

            case BlockGroupScope('county', includes):
                data_df = concat(block_groups(state=s, county=c)
                                 for s, c in map(COUNTY.decompose, includes))

                data_df = data_df.rename(columns={
                                         'STATEFP': 'state', 'COUNTYFP': 'county', 'TRACTCE': 'tract', 'BLKGRPCE': 'block group'})

                sort_param = ['state', 'county', 'tract', 'block group']

            case BlockGroupScope('tract', includes):
                data_df = concat(block_groups(state=s, county=c)
                                 for s, c, t in map(TRACT.decompose, includes))

                data_df = data_df.rename(columns={
                                         'STATEFP': 'state', 'COUNTYFP': 'county', 'TRACTCE': 'tract', 'BLKGRPCE': 'block_group'})
                data_df['tract_full'] = data_df['state'] + \
                    data_df['county'] + data_df['tract']
                data_df = data_df.loc[data_df['tract_full'].isin(includes)]

                sort_param = ['state', 'county', 'tract', 'block_group']

            case BlockGroupScope('block group', includes):
                data_df = concat(block_groups(state=s, county=c)
                                 for s, c, t, bg in map(BLOCK_GROUP.decompose, includes))

                data_df = data_df.rename(columns={
                                         'STATEFP': 'state', 'COUNTYFP': 'county', 'TRACTCE': 'tract', 'BLKGRPCE': 'block_group'})
                data_df['bg_full'] = data_df['state'] + data_df['county'] + \
                    data_df['tract'] + data_df['block_group']
                data_df = data_df.loc[data_df['bg_full'].isin(includes)]

                sort_param = ['state', 'county', 'tract', 'block_group']

            case _:
                raise Exception("Unsupported query.")

        data_df = GeoDataFrame(data_df.sort_values(by=sort_param))
        data_df.reset_index(drop=True, inplace=True)

        return data_df

    def fetch_commuters(self, scope: CensusScope, year: int) -> DataFrame:
        """
        Utility function to fetch commuting data from .xslx format filtered down to requested regions.
        """
        # check for invalid granularity
        if isinstance(scope, TractScope) or isinstance(scope, BlockGroupScope):
            msg = "Commuting data cannot be retrieved for tract or block group granularities"
            raise Exception(msg)

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
                raise Exception(msg)

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

            header_num = 3

        # download communter data spreadsheet as a pandas dataframe
        data = read_excel(url, header=header_num, names=all_fields, dtype={
                          'res_state_code': str, 'wrk_state_code': str, 'res_county_code': str, 'wrk_county_code': str})

        match scope:
            case StateScopeAll():
                data = data.loc[data['res_state_code'] < '57']
                data = data.loc[data['res_state_code'] != '11']
                data = data.loc[data['wrk_state_code'] < '057']
                data = data.loc[data['wrk_state_code'] != '011']

            case StateScope(includes) | CountyScope('state', includes):
                states = list(includes)
                data = data.loc[data['res_state_code'].isin(states)]

                for i in range(len(states)):
                    states[i] = states[i].zfill(3)
                data = data.loc[data['wrk_state_code'].isin(states)]

            case CountyScope('county', includes):
                data['res_county_full'] = data['res_state_code'] + \
                    data['res_county_code']
                data['wrk_county_full'] = data['wrk_state_code'] + \
                    data['wrk_county_code']
                data = data.loc[data['res_county_full'].isin(includes)]
                data = data.loc[data['wrk_county_full'].isin(
                    ['0' + x for x in includes])]

            case _:
                raise Exception("Unsupported query.")

        return data

    def _make_geoid_adrio(self, scope: CensusScope) -> ADRIO:
        """Makes an ADRIO to retrieve GEOID."""
        def fetch() -> NDArray:
            data_df = self.fetch_acs5(self.attrib_vars['geoid'], scope)
            # strange interaction here - name field is fetched only because a field is required
            data_df = data_df.drop(columns='NAME')

            return np.array(data_df, dtype=np.str_)
        return ADRIO('geoid', fetch)

    def _make_population_adrio(self, scope: CensusScope, num_groups: int) -> ADRIO:
        """Makes an ADRIO to retrieve population data split into 3 or 6 age groups."""
        def fetch() -> NDArray:
            data_df = self.fetch_acs5(self.population_query, scope)
            # calculate population of each age bracket and enter into a numpy array to return
            output = np.zeros((len(data_df.index), num_groups), dtype=np.int64)
            pop = [0, 0, 0, 0, 0, 0]
            for i in range(len(data_df.index)):
                for j in range(len(data_df.iloc[i].index)):
                    if j >= 0 and j < 10:
                        pop[0] += data_df.iloc[i].iloc[j]
                    elif j >= 10 and j < 20:
                        pop[1] += data_df.iloc[i].iloc[j]
                    elif j >= 20 and j < 28:
                        pop[2] += data_df.iloc[i].iloc[j]
                    elif j >= 28 and j < 34:
                        pop[3] += data_df.iloc[i].iloc[j]
                    elif j >= 34 and j < 40:
                        pop[4] += data_df.iloc[i].iloc[j]
                    elif j < 47:
                        pop[5] += data_df.iloc[i].iloc[j]

                if num_groups == 3:
                    output[i] = [pop[0], pop[1] + pop[2] + pop[3], pop[4] + pop[5]]
                else:
                    output[i] = pop

            return output

        if num_groups == 3:
            return ADRIO('population_by_age', fetch)
        else:
            return ADRIO('population_by_age_x6', fetch)

    def _make_dissimilarity_index_adrio(self, scope: CensusScope) -> ADRIO:
        """Makes an ADRIO to retrieve dissimilarity index."""
        def fetch() -> NDArray:
            data_df = self.fetch_acs5(
                self.attrib_vars['dissimilarity_index'], scope)
            data_df2 = self.fetch_acs5(
                self.attrib_vars['dissimilarity_index'], scope, CENSUS_GRANULARITY_CODE[scope.granularity] + 1)

            output = np.zeros(len(data_df2.index), dtype=np.float64)

            # loop for counties
            j = 0
            for i in range(len(data_df2.index)):
                # assign county fip to variable
                county_fip = data_df2.iloc[i][scope.granularity]
                # loop for all tracts in county (while fip == variable)
                sum = 0.0
                while data_df.iloc[j][scope.granularity] == county_fip and j < len(data_df.index) - 1:
                    # preliminary calculations
                    tract_minority = data_df.iloc[j]['B03002_004E'] + \
                        data_df.iloc[j]['B03002_014E']
                    county_minority = data_df2.iloc[i]['B03002_004E'] + \
                        data_df2.iloc[i]['B03002_014E']
                    tract_majority = data_df.iloc[j]['B03002_003E'] + \
                        data_df.iloc[j]['B03002_013E']
                    county_majority = data_df2.iloc[i]['B03002_003E'] + \
                        data_df2.iloc[i]['B03002_013E']

                    # run calculation sum += ( |minority(tract) / minority(county) - majority(tract) / majority(county)| )
                    if county_minority != 0 and county_majority != 0:
                        sum = sum + abs(tract_minority / county_minority -
                                        tract_majority / county_majority)
                    j += 1

                sum *= .5
                if sum == 0.:
                    sum = 0.5

                # assign current output element to sum
                output[i] = sum

            return output
        return ADRIO('dissimilarity_index', fetch)

    def _make_gini_index_adrio(self, scope: CensusScope) -> ADRIO:
        """Makes an ADRIO to retrieve gini index."""
        def fetch() -> NDArray:
            data_df = self.fetch_acs5(
                self.attrib_vars['gini_index'], scope)
            data_df2 = None
            data_df['B19083_001E'] = data_df['B19083_001E'].astype(
                np.float64).fillna(0.5).replace(-666666666, 0.5)

            # set cbg data to that of the parent tract if geo granularity = cbg
            if isinstance(scope, BlockGroupScope):
                print(
                    'Gini Index cannot be retrieved for block group level, fetching tract level data instead.')
                data_df2 = self.fetch_acs5(
                    self.attrib_vars['gini_index'], scope, CENSUS_GRANULARITY_CODE[scope.granularity] - 1)
                j = 0
                for i in range(len(data_df.index)):
                    tract_fip = data_df.loc[i, 'tract']
                    while data_df2.loc[j, 'tract'] == tract_fip and j < len(data_df2.index) - 1:
                        data_df2.loc[j, 'B01001_001E'] = data_df.loc[i, 'B19083_001E']
                        j += 1
                data_df = data_df2
            return data_df[self.attrib_vars['gini_index']].to_numpy(dtype=np.float64).squeeze()
        return ADRIO('gini_index', fetch)

    def _make_pop_density_adrio(self, scope: CensusScope) -> ADRIO:
        """Makes an ADRIO to retrieve population density per km2."""
        def fetch() -> NDArray:
            data_df = self.fetch_acs5(
                self.attrib_vars['pop_density_km2'], scope)
            geo_df = self.fetch_sf(scope)
            # merge census data with shapefile data
            if isinstance(scope, StateScope):
                geo_df = geo_df.merge(data_df, on=['state'])
            elif isinstance(scope, CountyScope):
                geo_df = geo_df.merge(data_df, on=['state', 'county'])
            elif isinstance(scope, TractScope):
                geo_df = geo_df.merge(data_df, on=['state', 'county', 'tract'])
            else:
                geo_df = geo_df.merge(
                    data_df, on=['state', 'county', 'tract', 'block group'])

            # calculate population density, storing it in a numpy array to return
            output = np.zeros(len(geo_df.index), dtype=np.float64)
            for i in range(len(geo_df.index)):
                output[i] = round(int(geo_df.iloc[i]['B01003_001E']) /
                                  (geo_df.iloc[i]['ALAND'] / 1e6))
            return output
        return ADRIO('pop_density_km2', fetch)

    def _make_centroid_adrio(self, scope: CensusScope):
        """Makes an ADRIO to retrieve geographic centroid coordinates."""
        def fetch() -> NDArray:
            data_df = self.fetch_sf(scope)
            # map node's name to its centroid in a numpy array and return
            output = np.zeros(len(data_df.index), dtype=CentroidDType)
            for i in range(len(data_df.index)):
                output[i] = data_df.iloc[i]['geometry'].centroid.coords[0]

            return output
        return ADRIO('centroid', fetch)

    def _make_tract_med_income_adrio(self, scope: CensusScope) -> ADRIO:
        """Makes an ADRIO to retrieve median income at the Census tract level."""
        def fetch() -> NDArray:
            data_df = self.fetch_acs5(
                self.attrib_vars['median_income'], scope, CENSUS_GRANULARITY_CODE['tract'])
            data_df2 = self.fetch_acs5(
                self.attrib_vars['median_income'], scope, CENSUS_GRANULARITY_CODE['block group'])
            data_df = data_df.fillna(0).replace(-666666666, 0)
            # set cbg data to that of the parent tract
            j = 0
            for i in range(len(data_df.index)):
                tract_fip = data_df.loc[i, 'tract']
                while data_df2.loc[j, 'tract'] == tract_fip and j < len(data_df2.index) - 1:
                    data_df2.loc[j, 'B19013_001E'] = data_df.loc[i, 'B19013_001E']
                    j += 1
            data_df = data_df2
            data_df = data_df.fillna(0).replace(-666666666, 0)
            return data_df[self.attrib_vars['median_income']].to_numpy(dtype=np.int64).squeeze()
        return ADRIO('tract_median_income', fetch)

    def _make_commuter_adrio(self, scope: CensusScope, year: int) -> ADRIO:
        """Makes an ADRIO to retrieve ACS commuting flow data."""
        def fetch() -> NDArray:
            data_df = self.fetch_commuters(scope, year)
            # state level
            if isinstance(scope, StateScope):
                # get unique state identifier
                unique_states = ('0' + data_df['res_state_code']).unique()
                state_len = np.count_nonzero(unique_states)

                # create dictionary to be used as array indices
                states_enum = enumerate(unique_states)
                states_dict = dict((j, i) for i, j in states_enum)

                # group and aggregate data
                data_group = data_df.groupby(['res_state_code', 'wrk_state_code'])
                data_df = data_group.agg({'workers': 'sum'})

                # create and return array for each state
                output = np.zeros((state_len, state_len), dtype=np.int64)

                # fill array with commuting data
                for i, row in data_df.iterrows():
                    if type(i) is tuple:
                        x = states_dict.get('0' + i[0])
                        y = states_dict.get(i[1])

                        output[x][y] = row['workers']

            # county level
            else:
                # get unique identifier for each county
                geoid_df = DataFrame()
                geoid_df['geoid'] = '0' + data_df['res_state_code'] + \
                    data_df['res_county_code']
                unique_counties = geoid_df['geoid'].unique()

                # create empty output array
                county_len = np.count_nonzero(unique_counties)
                output = np.zeros((county_len, county_len), dtype=np.int64)

                # create dictionary to be used as array indices
                counties_enum = enumerate(unique_counties)
                counties_dict = dict((j, i) for i, j in counties_enum)

                data_df.reset_index(drop=True, inplace=True)

                # fill array with commuting data
                for i in range(len(data_df.index)):
                    x = counties_dict.get('0' +
                                          data_df.iloc[i]['res_state_code'] + data_df.iloc[i]['res_county_code'])
                    y = counties_dict.get(
                        data_df.iloc[i]['wrk_state_code'] + data_df.iloc[i]['wrk_county_code'])

                    output[x][y] = data_df.iloc[i]['workers']

            return output
        return ADRIO('commuters', fetch)

    def _make_simple_adrios(self, attrib: AttributeDef, scope: CensusScope) -> ADRIO:
        """Makes ADRIOs for simple attributes that require no additional postprocessing."""
        def fetch() -> NDArray:
            data_df = self.fetch_acs5(
                self.attrib_vars[attrib.name], scope)
            if attrib.name == 'median_income' or attrib.name == 'median_age':
                data_df = data_df.fillna(0).replace(-666666666, 0)

            return data_df[self.attrib_vars[attrib.name]].to_numpy(dtype=attrib.dtype).squeeze()
        return ADRIO(attrib.name, fetch)
