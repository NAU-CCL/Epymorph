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
from epymorph.geo.spec import TimePeriod, Year
from epymorph.geography.us_census import (BLOCK_GROUP, COUNTY, STATE, TRACT,
                                          BlockGroupScope,
                                          CensusGranularityName, CensusScope,
                                          CountyScope, StateScope,
                                          StateScopeAll, TractScope)
from epymorph.simulation import AttributeDef, geo_attrib


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
            return self._make_geoid_adrio(scope, year)
        elif attrib.name == 'population_by_age':
            return self._make_population_adrio(scope, year, 3)
        elif attrib.name == 'population_by_age_x6':
            return self._make_population_adrio(scope, year, 6)
        elif attrib.name == 'dissimilarity_index':
            return self._make_dissimilarity_index_adrio(scope, year)
        elif attrib.name == 'gini_index':
            return self._make_gini_index_adrio(scope, year)
        elif attrib.name == 'pop_density_km2':
            return self._make_pop_density_adrio(scope, year)
        elif attrib.name == 'centroid':
            return self._make_centroid_adrio(scope)
        elif attrib.name == 'tract_median_income':
            return self._make_tract_med_income_adrio(scope, year)
        elif attrib.name == 'commuters':
            return self._make_commuter_adrio(scope, year)
        else:
            return self._make_simple_adrios(attrib, scope, year)

    def make_acs5_queries(self, scope: CensusScope) -> list[dict[str, str]]:
        """Utility function to format scope geography information into dictionaries usable in census queries."""
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
                tracts_by_county: dict[str, list[str]] = defaultdict(list)

                for state, county, tract in map(TRACT.decompose, includes):
                    tracts_by_county[state + county].append(tract)

                queries = [
                    {"for": f"tract:{','.join(tracts_by_county[state + county])}",
                     "in": f"state:{state} county:{county}"}
                    for state, county in [COUNTY.decompose(c) for c in tracts_by_county.keys()]
                ]

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
                tracts_by_county: dict[str, list[str]] = defaultdict(list)

                for state, county, tract in map(TRACT.decompose, includes):
                    tracts_by_county[state + county].append(tract)

                queries = [
                    {"for": f"block group:*",
                     "in": f"state:{state} county:{county} tract:{','.join(tracts_by_county[state + county])}"}
                    for state, county in [COUNTY.decompose(c) for c in tracts_by_county.keys()]
                ]

            case BlockGroupScope('block group', includes):
                block_groups_by_tract: dict[str, list[str]] = defaultdict(list)

                for state, county, tract, block_group in map(BLOCK_GROUP.decompose, includes):
                    block_groups_by_tract[state + county + tract].append(block_group)

                queries = [
                    {"for": f"block group:{'.'.join(block_groups_by_tract[state + county + tract])}",
                     "in": f"state:{state} county:{county} tract:{tract}"}
                    for state, county, tract in [TRACT.decompose(t) for t in block_groups_by_tract.keys()]
                ]

            case _:
                raise Exception("Unsupported query.")

        return queries

    def raise_scope_granularity(self, scope: CensusScope) -> CensusScope:
        """
        Utility function to create and return a CensusScope object at one granularity higher than
        the scope provided.
        """
        match scope:
            case StateScope():
                raise Exception("No granularity higher than state.")

            case CountyScope():
                raised_granularity = scope.includes_granularity
                raised_includes = scope.includes
                if raised_granularity == 'county':
                    raised_granularity = 'state'
                    raised_includes = [fips[:-3] for fips in raised_includes]
                raised_scope = StateScope(
                    raised_granularity, raised_includes, scope.year)

            case TractScope():
                raised_granularity = scope.includes_granularity
                raised_includes = scope.includes
                if raised_granularity == 'tract':
                    raised_granularity = 'county'
                    raised_includes = [fips[:-6] for fips in raised_includes]
                raised_scope = CountyScope(
                    raised_granularity, raised_includes, scope.year)

            case BlockGroupScope():
                raised_granularity = scope.includes_granularity
                raised_includes = scope.includes
                if raised_granularity == 'block group':
                    raised_granularity = 'tract'
                    raised_includes = [fips[:-1] for fips in raised_includes]
                raised_scope = TractScope(
                    raised_granularity, raised_includes, scope.year)

            case _:
                raise Exception("Invalid scope.")

        return raised_scope

    def lower_scope_granularity(self, scope: CensusScope) -> CensusScope:
        """
        Utility function to create and return a CensusScope object at one granularity lower than
        the scope provided.
        """
        match scope:
            case StateScope():
                lowered_scope = CountyScope(
                    scope.includes_granularity, scope.includes, scope.year)

            case CountyScope():
                lowered_scope = TractScope(
                    scope.includes_granularity, scope.includes, scope.year)

            case TractScope():
                lowered_scope = BlockGroupScope(
                    scope.includes_granularity, scope.includes, scope.year)

            case BlockGroupScope():
                raise Exception("No valid granularity lower than block group.")

            case _:
                raise Exception("Invalid Scope.")

        return lowered_scope

    def concatenate_fips(self, df: DataFrame, granularity: CensusGranularityName) -> DataFrame:
        """
        Adds column to dataframe resulting from an acs5 query that is a concatination
        of all component fips codes up to the specified granularity.
        Returns dataframe with new column named 'fips full'
        """
        match granularity:
            case 'state':
                df['fips full'] = df['state']

            case 'county':
                df['fips full'] = df['state'] + df['county']

            case 'tract':
                df['fips full'] = df['state'] + df['county'] + df['tract']

            case 'block group':
                df['fips full'] = df['state'] + df['county'] + \
                    df['tract'] + df['block group']

        return df

    def fetch_acs5(self, variables: list[str], scope: CensusScope, year: int) -> DataFrame:
        """Utility function to fetch Census data by building queries from ADRIO data."""
        queries = self.make_acs5_queries(scope)

        # fetch and combine all queries
        df = concat([
            DataFrame.from_records(
                self.census.acs5.get(variables, geo=query, year=year)
            )
            for query in queries
        ])
        return df

    def fetch_sf(self, scope: CensusScope) -> GeoDataFrame:
        """Utility function to fetch shape files from Census for specified regions."""

        # call appropriate pygris function based on granularity and sort result
        match scope:
            case StateScopeAll() | StateScope():
                data_df = states(year=scope.year)
                data_df = data_df.rename(columns={'STATEFP': 'state'})

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
                data_df = concat(tracts(state=s, county=c, year=scope.year)
                                 for s, c in map(COUNTY.decompose, includes))

                data_df = data_df.rename(
                    columns={'STATEFP': 'state', 'COUNTYFP': 'county', 'TRACTCE': 'tract'})

                sort_param = ['state', 'county', 'tract']

            case TractScope('tract', includes):
                data_df = concat(tracts(state=s, county=c, year=scope.year)
                                 for s, c, t in map(TRACT.decompose, includes))

                data_df = data_df.rename(
                    columns={'STATEFP': 'state', 'COUNTYFP': 'county', 'TRACTCE': 'tract'})

                sort_param = ['state', 'county', 'tract']

            case BlockGroupScope('state', includes):
                data_df = concat(block_groups(state=s, year=scope.year)
                                 for s in includes)
                data_df = data_df.rename(columns={
                                         'STATEFP': 'state', 'COUNTYFP': 'county', 'TRACTCE': 'tract', 'BLKGRPCE': 'block group'})

                sort_param = ['state', 'county', 'tract', 'block group']

            case BlockGroupScope('county', includes):
                data_df = concat(block_groups(state=s, county=c, year=scope.year)
                                 for s, c in map(COUNTY.decompose, includes))

                data_df = data_df.rename(columns={
                                         'STATEFP': 'state', 'COUNTYFP': 'county', 'TRACTCE': 'tract', 'BLKGRPCE': 'block group'})

                sort_param = ['state', 'county', 'tract', 'block group']

            case BlockGroupScope('tract', includes):
                data_df = concat(block_groups(state=s, county=c, year=scope.year)
                                 for s, c, t in map(TRACT.decompose, includes))

                data_df = data_df.rename(columns={
                                         'STATEFP': 'state', 'COUNTYFP': 'county', 'TRACTCE': 'tract', 'BLKGRPCE': 'block_group'})

                sort_param = ['state', 'county', 'tract', 'block_group']

            case BlockGroupScope('block group', includes):
                data_df = concat(block_groups(state=s, county=c, year=scope.year)
                                 for s, c, t, bg in map(BLOCK_GROUP.decompose, includes))

                data_df = data_df.rename(columns={
                                         'STATEFP': 'state', 'COUNTYFP': 'county', 'TRACTCE': 'tract', 'BLKGRPCE': 'block_group'})

                sort_param = ['state', 'county', 'tract', 'block_group']

            case _:
                raise Exception("Unsupported query.")

        if not isinstance(scope, StateScopeAll):
            data_df = self.concatenate_fips(data_df, scope.includes_granularity)
            data_df = data_df.loc[data_df['fips full'].isin(scope.includes)]

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

            header_num = 4

        # download communter data spreadsheet as a pandas dataframe
        data = read_excel(url, header=header_num, names=all_fields, dtype={
                          'res_state_code': str, 'wrk_state_code': str, 'res_county_code': str, 'wrk_county_code': str})

        match scope:
            case StateScopeAll():
                data = data.loc[data['res_state_code'] < '57']
                data = data.loc[data['res_state_code'] != '11']
                data = data.loc[data['wrk_state_code'] < '057']
                data = data.loc[data['wrk_state_code'] != '011']

            case StateScope('state', includes) | CountyScope('state', includes):
                states = list(includes)
                data = data.loc[data['res_state_code'].isin(states)]

                for state in range(len(states)):
                    states[state] = states[state].zfill(3)
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

    def _make_geoid_adrio(self, scope: CensusScope, year: int) -> ADRIO:
        """Makes an ADRIO to retrieve GEOID."""
        def fetch() -> NDArray:
            data_df = self.fetch_acs5(self.attrib_vars['geoid'], scope, year)
            data_df = self.concatenate_fips(data_df, scope.granularity)

            return data_df['fips full'].to_numpy(dtype=str)
        return ADRIO('geoid', fetch)

    def _make_population_adrio(self, scope: CensusScope, year: int, num_groups: int) -> ADRIO:
        """Makes an ADRIO to retrieve population data split into 3 or 6 age groups."""
        def fetch() -> NDArray:
            data_df = self.fetch_acs5(self.population_query, scope, year)
            data_df = data_df.loc[:, self.population_query]
            # calculate population of each age bracket and enter into a numpy array to return
            output = np.zeros((len(data_df.index), num_groups), dtype=int)
            pop = [0, 0, 0, 0, 0, 0]
            for node in range(len(data_df.index)):
                for age_group in range(len(data_df.iloc[node].index)):
                    if age_group >= 0 and age_group < 10:
                        pop[0] += data_df.iloc[node][age_group]
                    elif age_group >= 10 and age_group < 20:
                        pop[1] += data_df.iloc[node][age_group]
                    elif age_group >= 20 and age_group < 28:
                        pop[2] += data_df.iloc[node][age_group]
                    elif age_group >= 28 and age_group < 34:
                        pop[3] += data_df.iloc[node][age_group]
                    elif age_group >= 34 and age_group < 40:
                        pop[4] += data_df.iloc[node][age_group]
                    elif age_group < 47:
                        pop[5] += data_df.iloc[node][age_group]

                if num_groups == 3:
                    output[node] = [pop[0], pop[1] + pop[2] + pop[3], pop[4] + pop[5]]
                else:
                    output[node] = pop

            return output

        if num_groups == 3:
            return ADRIO('population_by_age', fetch)
        else:
            return ADRIO('population_by_age_x6', fetch)

    def _make_dissimilarity_index_adrio(self, scope: CensusScope, year: int) -> ADRIO:
        """Makes an ADRIO to retrieve dissimilarity index."""
        if isinstance(scope, BlockGroupScope):
            msg = "Dissimilarity index cannot be retreived for block group scope."
            raise Exception(msg)

        def fetch() -> NDArray:
            data_df = self.fetch_acs5(
                self.attrib_vars['dissimilarity_index'], scope, year)
            data_df2 = self.fetch_acs5(
                self.attrib_vars['dissimilarity_index'], self.lower_scope_granularity(scope), year)
            output = np.zeros(len(data_df2.index), dtype=float)

            data_df = self.concatenate_fips(data_df, scope.granularity)
            data_df2 = self.concatenate_fips(data_df2, scope.granularity)
            # loop for scope granularity
            low_index = 0
            for high_index in range(len(data_df2.index)):
                # assign county fip to variable
                county_fip = data_df2.iloc[high_index]['fips full']
                # loop for lower granularity
                sum = 0.0
                while data_df.iloc[low_index]['fips full'] == county_fip and low_index < len(data_df.index) - 1:
                    # preliminary calculations
                    tract_minority = data_df.iloc[low_index]['B03002_004E'] + \
                        data_df.iloc[low_index]['B03002_014E']
                    county_minority = data_df2.iloc[high_index]['B03002_004E'] + \
                        data_df2.iloc[high_index]['B03002_014E']
                    tract_majority = data_df.iloc[low_index]['B03002_003E'] + \
                        data_df.iloc[low_index]['B03002_013E']
                    county_majority = data_df2.iloc[high_index]['B03002_003E'] + \
                        data_df2.iloc[high_index]['B03002_013E']

                    # run calculation sum += ( |minority(tract) / minority(county) - majority(tract) / majority(county)| )
                    if county_minority != 0 and county_majority != 0:
                        sum = sum + abs(tract_minority / county_minority -
                                        tract_majority / county_majority)
                    low_index += 1

                sum *= .5
                if sum == 0.:
                    sum = 0.5

                # assign current output element to sum
                output[high_index] = sum

            return output
        return ADRIO('dissimilarity_index', fetch)

    def _make_gini_index_adrio(self, scope: CensusScope, year: int) -> ADRIO:
        """Makes an ADRIO to retrieve gini index."""
        def fetch() -> NDArray:
            data_df = self.fetch_acs5(
                self.attrib_vars['gini_index'], scope, year)
            data_df2 = None
            data_df['B19083_001E'] = data_df['B19083_001E'].astype(
                np.float64).fillna(0.5).replace(-666666666, 0.5)

            # set cbg data to that of the parent tract if geo granularity = cbg
            if isinstance(scope, BlockGroupScope):
                print(
                    'Gini Index cannot be retrieved for block group level, fetching tract level data instead.')
                data_df2 = self.fetch_acs5(
                    self.attrib_vars['gini_index'], self.raise_scope_granularity(scope), scope.year)

                data_df = self.concatenate_fips(data_df, 'tract')
                data_df2 = self.concatenate_fips(data_df2, 'tract')

                output = np.zeros(len(data_df.index), dtype=float)
                bg = 0
                for tract in range(len(data_df2.index)):
                    tract_fip = data_df2.iloc[tract]['fips full']
                    while data_df.iloc[bg]['fips full'] == tract_fip:
                        output[bg] = data_df2.iloc[tract]['B19083_001E']
                        bg += 1
                return output

            return data_df[self.attrib_vars['gini_index']].to_numpy(dtype=float).squeeze()
        return ADRIO('gini_index', fetch)

    def _make_pop_density_adrio(self, scope: CensusScope, year: int) -> ADRIO:
        """Makes an ADRIO to retrieve population density per km2."""
        def fetch() -> NDArray:
            data_df = self.fetch_acs5(
                self.attrib_vars['pop_density_km2'], scope, year)
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
            output = np.zeros(len(geo_df.index), dtype=float)
            for node in range(len(geo_df.index)):
                output[node] = round(geo_df.iloc[node]['B01003_001E'] /
                                     (geo_df.iloc[node]['ALAND'] / 1e6), 4)
            return output
        return ADRIO('pop_density_km2', fetch)

    def _make_centroid_adrio(self, scope: CensusScope):
        """Makes an ADRIO to retrieve geographic centroid coordinates."""
        def fetch() -> NDArray:
            data_df = self.fetch_sf(scope)
            # map node's name to its centroid in a numpy array and return
            output = np.zeros(len(data_df.index), dtype=CentroidDType)
            for node in range(len(data_df.index)):
                output[node] = data_df.iloc[node]['geometry'].centroid.coords[0]

            return output
        return ADRIO('centroid', fetch)

    def _make_tract_med_income_adrio(self, scope: CensusScope, year: int) -> ADRIO:
        """Makes an ADRIO to retrieve median income at the Census tract level."""
        def fetch() -> NDArray:
            if isinstance(scope, BlockGroupScope):
                # query median income at cbg and tract level
                data_df = self.fetch_acs5(
                    self.attrib_vars['median_income'], self.raise_scope_granularity(scope), year)
                data_df2 = self.fetch_acs5(
                    self.attrib_vars['median_income'], BlockGroupScope(scope.includes_granularity, scope.includes, scope.year), year)
                data_df = data_df.fillna(0).replace(-666666666, 0)

                # set cbg data to that of the parent tract
                data_df = self.concatenate_fips(data_df, 'tract')
                data_df2 = self.concatenate_fips(data_df2, 'tract')
                bg = 0
                for tract in range(len(data_df.index)):
                    tract_fip = data_df.loc['fips full']
                    while data_df2['fips full'] == tract_fip and bg < len(data_df2.index) - 1:
                        data_df2.loc[bg,
                                     'B19013_001E'] = data_df.loc[tract, 'B19013_001E']
                        bg += 1
                data_df = data_df2
                data_df = data_df.fillna(0).replace(-666666666, 0)

            else:
                msg = "Tract median income can only be retrieved for block group scope."
                raise Exception(msg)

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
                output = np.zeros((state_len, state_len), dtype=int)

                # fill array with commuting data
                for index, row in data_df.iterrows():
                    if type(index) is tuple:
                        x = states_dict.get('0' + index[0])
                        y = states_dict.get(index[1])

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
                output = np.zeros((county_len, county_len), dtype=int)

                # create dictionary to be used as array indices
                counties_enum = enumerate(unique_counties)
                counties_dict = dict((fips, index) for index, fips in counties_enum)

                data_df.reset_index(drop=True, inplace=True)

                # fill array with commuting data
                for county in range(len(data_df.index)):
                    x = counties_dict.get('0' +
                                          data_df.iloc[county]['res_state_code'] + data_df.iloc[county]['res_county_code'])
                    y = counties_dict.get(
                        data_df.iloc[county]['wrk_state_code'] + data_df.iloc[county]['wrk_county_code'])

                    output[x][y] = data_df.iloc[county]['workers']

            return output
        return ADRIO('commuters', fetch)

    def _make_simple_adrios(self, attrib: AttributeDef, scope: CensusScope, year: int) -> ADRIO:
        """Makes ADRIOs for simple attributes that require no additional postprocessing."""
        def fetch() -> NDArray:
            data_df = self.fetch_acs5(
                self.attrib_vars[attrib.name], scope, year)
            if attrib.name == 'median_income' or attrib.name == 'median_age':
                data_df = data_df.fillna(0).replace(-666666666, 0)

            return data_df[self.attrib_vars[attrib.name]].to_numpy(dtype=attrib.dtype).squeeze()
        return ADRIO(attrib.name, fetch)
