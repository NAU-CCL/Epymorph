import os
from collections import defaultdict
from functools import partial

import numpy as np
from census import Census
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from pandas import DataFrame, Series, concat, read_excel
from shapely import area

from epymorph.data_shape import Shapes
from epymorph.data_type import CentroidDType
from epymorph.error import DataResourceException, GeoValidationException
from epymorph.geo.adrio.adrio import ADRIO, ADRIOMaker
from epymorph.geo.spec import TimePeriod, Year
from epymorph.geography.us_census import (BLOCK_GROUP, COUNTY, STATE, TRACT,
                                          BlockGroupScope,
                                          CensusGranularityName, CensusScope,
                                          CountyScope, StateScope,
                                          StateScopeAll, TractScope)
from epymorph.geography.us_tiger import (get_block_groups_geo,
                                         get_counties_geo, get_states_geo,
                                         get_tracts_geo)
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

    def fetch_acs5(self, variables: list[str], scope: CensusScope, year: int) -> DataFrame:
        """Utility function to fetch Census data by building queries from ADRIO data and sorting the result."""
        queries = self.make_acs5_queries(scope)

        # fetch and combine all queries
        df = concat([
            DataFrame.from_records(
                self.census.acs5.get(variables, geo=query, year=year)
            )
            for query in queries
        ])

        if df.empty:
            msg = "ACS5 query returned empty. Ensure all geographies included in your scope are supported and try again."
            raise DataResourceException(msg)

        df = self.concatenate_fips(df, scope.granularity)

        return df

    def fetch_sf(self, scope: CensusScope) -> GeoDataFrame:
        """Utility function to fetch shape files from Census for specified regions."""

        # call appropriate pygris function based on granularity and sort result
        match scope:
            case StateScopeAll() | StateScope():
                df = get_states_geo(year=scope.year)

            case CountyScope():
                df = get_counties_geo(year=scope.year)

            case TractScope():
                df = get_tracts_geo(year=scope.year)

            case BlockGroupScope():
                df = get_block_groups_geo(year=scope.year)

            case _:
                raise DataResourceException("Unsupported query.")

        df = df.rename(columns={'GEOID': 'geoid'})

        df = df.loc[df['geoid'].isin(scope.get_node_ids())]

        return GeoDataFrame(df)

    def fetch_commuters(self, scope: CensusScope, year: int) -> DataFrame:
        """
        Utility function to fetch commuting data from .xslx format filtered down to requested regions.
        """
        # check for invalid granularity
        if isinstance(scope, TractScope) or isinstance(scope, BlockGroupScope):
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

        match scope:
            case StateScopeAll():
                # remove nodes not in acs5 data for all states case
                data = data.loc[data['res_state_code'].isin(scope.get_node_ids())]
                data = data.loc[data['wrk_state_code'].isin(
                    '0' + x for x in scope.get_node_ids())]

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
                raise DataResourceException("Unsupported query.")

        return data

    def make_acs5_queries(self, scope: CensusScope) -> list[dict[str, str]]:
        """Formats scope geography information into dictionaries usable in census queries."""
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
                raise DataResourceException("Unsupported query.")

        return queries

    def concatenate_fips(self, df: DataFrame, granularity: CensusGranularityName) -> DataFrame:
        """
        Adds column to dataframe resulting from an acs5 query that is a concatination
        of all component fips codes up to the specified granularity.
        Returns dataframe sorted by new column named 'geoid'.
        """
        columns: list[str] = {
            'state': ['state'],
            'county': ['state', 'county'],
            'tract': ['state', 'county', 'tract'],
            'block group': ['state', 'county', 'tract', 'block group'],
        }[granularity]
        df['geoid'] = df[columns].apply(lambda xs: ''.join(xs), axis=1)
        df = df.sort_values(by='geoid')

        return df

    def _validate_result(self, scope: CensusScope, data: Series):
        """Ensures that data produced for an attribute contains exactly one entry for every node in the scope."""
        if set(data) != set(scope.get_node_ids()):
            msg = "Attribute result missing data for geographies in scope or contains data for geographies not supported by ACS5."
            raise DataResourceException(msg)

    def _make_geoid_adrio(self, scope: CensusScope, year: int) -> ADRIO:
        """Makes an ADRIO to retrieve GEOID."""
        def fetch() -> NDArray:
            df = self.fetch_acs5(self.attrib_vars['geoid'], scope, year)

            self._validate_result(scope, df['geoid'])

            return df['geoid'].to_numpy(dtype=str)
        return ADRIO('geoid', fetch)

    def _make_population_adrio(self, scope: CensusScope, year: int, num_groups: int) -> ADRIO:
        """Makes an ADRIO to retrieve population data split into 3 or 6 age groups."""
        def fetch() -> NDArray:
            def group_cols(first: int, last: int, source: DataFrame) -> Series:
                result = source[f"B01001_{first:03d}E"]
                for line in range(first + 1, last + 1):
                    result = result + source[f"B01001_{line:03d}E"]
                return result

            df = self.fetch_acs5(self.population_query, scope, year)

            self._validate_result(scope, df['geoid'])

            group = partial(group_cols, source=df)

            if num_groups == 3:
                output = DataFrame({'pop_0-19': group(3, 7) + group(27, 31),
                                    'pop_20-64': group(8, 19) + group(32, 43),
                                    'pop_65+': group(20, 25) + group(44, 49)})
            else:
                output = DataFrame({'pop_0-19': group(3, 7) + group(27, 31),
                                    'pop_20-34': group(8, 12) + group(32, 36),
                                    'pop_35-54': group(13, 16) + group(37, 40),
                                    'pop_55-64': group(17, 19) + group(41, 43),
                                    'pop_65-75': group(20, 22) + group(44, 46),
                                    'pop_75+': group(23, 25) + group(47, 49)})

            return output.to_numpy(dtype=int)

        if num_groups == 3:
            return ADRIO('population_by_age', fetch)
        else:
            return ADRIO('population_by_age_x6', fetch)

    def _make_dissimilarity_index_adrio(self, scope: CensusScope, year: int) -> ADRIO:
        """Makes an ADRIO to retrieve dissimilarity index."""
        if isinstance(scope, BlockGroupScope):
            msg = "Dissimilarity index cannot be retreived for block group scope."
            raise DataResourceException(msg)

        def fetch() -> NDArray:
            vars = self.attrib_vars['dissimilarity_index']
            df = self.fetch_acs5(vars, scope, year)
            df2 = self.fetch_acs5(vars, scope.lower_granularity(), year)
            df2 = self.concatenate_fips(df2, scope.granularity)

            df['high_majority'] = df[vars[0]] + df[vars[1]]
            df2['low_majority'] = df2[vars[0]] + df2[vars[1]]
            df['high_minority'] = df[vars[2]] + df[vars[3]]
            df2['low_minority'] = df2[vars[2]] + df2[vars[3]]

            df3 = df.merge(df2, on='geoid')

            self._validate_result(scope, df3['geoid'])

            df3['score'] = abs(df3['low_minority'] / df3['high_minority'] -
                               df3['low_majority'] / df3['high_majority'])
            df3 = df3.groupby('geoid').sum()
            df3['score'] *= .5
            df3['score'].replace(0., 0.5)

            return df3['score'].to_numpy(dtype=float)
        return ADRIO('dissimilarity_index', fetch)

    def _make_gini_index_adrio(self, scope: CensusScope, year: int) -> ADRIO:
        """Makes an ADRIO to retrieve gini index."""
        def fetch() -> NDArray:
            var = self.attrib_vars['gini_index']
            df = self.fetch_acs5(var, scope, year)
            df[var] = df[var].astype(np.float64).fillna(0.5).replace(-666666666, 0.5)

            self._validate_result(scope, df['geoid'])

            # set cbg data to that of the parent tract if geo granularity = cbg
            if isinstance(scope, BlockGroupScope):
                print(
                    "Gini Index cannot be retrieved for block group level, fetching tract level data instead.")
                df2 = self.fetch_acs5(var, scope.raise_granularity(), scope.year)
                df['geoid'] = df['geoid'].apply(lambda x: x[:-1])
                df = df.drop(columns=var)

                df = df.merge(df2, on='geoid')

            return df[var].to_numpy(dtype=float).squeeze()
        return ADRIO('gini_index', fetch)

    def _make_pop_density_adrio(self, scope: CensusScope, year: int) -> ADRIO:
        """Makes an ADRIO to retrieve population density per km2."""
        def fetch() -> NDArray:
            df = self.fetch_acs5(
                self.attrib_vars['pop_density_km2'], scope, year)
            geo_df = self.fetch_sf(scope)

            geo_df = geo_df.merge(df, on='geoid')

            self._validate_result(scope, geo_df['geoid'])

            # calculate population density
            output = geo_df['B01003_001E'] / (area(geo_df['geometry']) / 1e6)

            return output.to_numpy(dtype=float)
        return ADRIO('pop_density_km2', fetch)

    def _make_centroid_adrio(self, scope: CensusScope) -> ADRIO:
        """Makes an ADRIO to retrieve geographic centroid coordinates."""
        def fetch() -> NDArray:
            df = self.fetch_sf(scope)

            output = DataFrame(
                {'geoid': df['geoid'], 'centroid': df['geometry'].apply(lambda x: x.centroid.coords[0])})

            self._validate_result(scope, output['geoid'])

            return output['centroid'].to_numpy(dtype=CentroidDType)
        return ADRIO('centroid', fetch)

    def _make_tract_med_income_adrio(self, scope: CensusScope, year: int) -> ADRIO:
        """Makes an ADRIO to retrieve median income at the Census tract level."""
        def fetch() -> NDArray:
            if isinstance(scope, BlockGroupScope):
                var = self.attrib_vars['tract_median_income']
                # query median income at cbg and tract level
                df = self.fetch_acs5(['NAME'], scope, year)
                df2 = self.fetch_acs5(var, scope.raise_granularity(), year)
                df2 = df2.fillna(0).replace(-666666666, 0)

                self._validate_result(scope, df['geoid'])

                df['geoid'] = df['geoid'].apply(lambda x: x[:-1])
                df = df.merge(df2, on='geoid')

                return df[var].to_numpy(dtype=int).squeeze()

            else:
                msg = "Tract median income can only be retrieved for block group scope."
                raise DataResourceException(msg)

        return ADRIO('tract_median_income', fetch)

    def _make_commuter_adrio(self, scope: CensusScope, year: int) -> ADRIO:
        """Makes an ADRIO to retrieve ACS commuting flow data."""
        def fetch() -> NDArray:
            df = self.fetch_commuters(scope, year)
            # state level
            if isinstance(scope, StateScope) or isinstance(scope, StateScopeAll):
                # get unique state identifier
                unique_states = ('0' + df['res_state_code']).unique()
                state_len = np.count_nonzero(unique_states)

                # create dictionary to be used as array indices
                states_enum = enumerate(unique_states)
                states_dict = dict((y, x) for x, y in states_enum)

                # group and aggregate data
                data_group = df.groupby(['res_state_code', 'wrk_state_code'])
                df = data_group.agg({'workers': 'sum'})

                # create and return array for each state
                output = np.zeros((state_len, state_len), dtype=int)

                # fill array with commuting data
                for index, row in df.iterrows():
                    if type(index) is tuple:
                        x = states_dict.get('0' + index[0])
                        y = states_dict.get(index[1])

                        output[x][y] = row['workers']

            # county level
            else:
                # get unique identifier for each county
                geoid_df = DataFrame()
                geoid_df['geoid'] = '0' + df['res_state_code'] + df['res_county_code']
                unique_counties = geoid_df['geoid'].unique()

                # create empty output array
                county_len = np.count_nonzero(unique_counties)
                output = np.zeros((county_len, county_len), dtype=int)

                # create dictionary to be used as array indices
                counties_enum = enumerate(unique_counties)
                counties_dict = dict((fips, index) for index, fips in counties_enum)

                df.reset_index(drop=True, inplace=True)

                # fill array with commuting data
                for county in range(len(df.index)):
                    x = counties_dict.get('0' +
                                          df.iloc[county]['res_state_code'] + df.iloc[county]['res_county_code'])
                    y = counties_dict.get(
                        df.iloc[county]['wrk_state_code'] + df.iloc[county]['wrk_county_code'])

                    output[x][y] = df.iloc[county]['workers']

            return output
        return ADRIO('commuters', fetch)

    def _make_simple_adrios(self, attrib: AttributeDef, scope: CensusScope, year: int) -> ADRIO:
        """Makes ADRIOs for simple attributes that require no additional postprocessing."""
        def fetch() -> NDArray:
            df = self.fetch_acs5(
                self.attrib_vars[attrib.name], scope, year)
            df = df.fillna(0).replace(-666666666, 0)

            self._validate_result(scope, df['geoid'])

            return df[self.attrib_vars[attrib.name]].to_numpy(dtype=attrib.dtype).squeeze()
        return ADRIO(attrib.name, fetch)
