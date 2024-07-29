import json
import os
import re
from collections import defaultdict
from functools import cache
from typing import Literal, NamedTuple, TypeGuard
from urllib.request import urlopen

import numpy as np
import pandas as pd
from census import Census
from numpy.typing import NDArray
from pandas import DataFrame

from epymorph.data_shape import Shapes
from epymorph.error import DataResourceException
from epymorph.geo.adrio.adrio2 import Adrio
from epymorph.geography.us_census import (BLOCK_GROUP, COUNTY, STATE, TRACT,
                                          BlockGroupScope, CensusScope,
                                          CountyScope, StateScope,
                                          StateScopeAll, TractScope,
                                          get_census_granularity)
from epymorph.simulation import AttributeDef
from epymorph.util import filter_with_mask


@cache
def _get_api() -> Census:
    api_key = os.environ.get('CENSUS_API_KEY')
    if api_key is None:
        msg = "Census API key not found. " \
            "Please ensure you have set the environment variable 'CENSUS_API_KEY'"
        raise DataResourceException(msg)
    return Census(api_key)


@cache
def _get_vars(year: int) -> dict[str, dict]:
    # TODO:
    # - check year is good
    # - disk cache vars files
    vars_url = f"https://api.census.gov/data/{year}/acs/acs5/variables.json"
    with urlopen(vars_url) as f:
        return json.load(f)['variables']


@cache
def _get_group_vars(year: int, group: str) -> list[tuple[str, dict]]:
    return sorted((
        (name, attrs)
        for name, attrs in _get_vars(year).items()
        if attrs['group'] == group
    ), key=lambda x: x[0])


def _make_acs5_queries(scope: CensusScope) -> list[dict[str, str]]:
    """Formats scope geography information into dictionaries usable in census queries."""
    match scope:
        case StateScopeAll():
            return [{"for": "state:*"}]

        case StateScope(includes_granularity='state', includes=includes):
            return [{"for": f"state:{','.join(includes)}"}]

        case CountyScope(includes_granularity='state', includes=includes):
            return [{
                "for": "county:*",
                "in": f"state:{','.join(includes)}",
            }]

        case CountyScope(includes_granularity='county', includes=includes):
            counties_by_state: dict[str, list[str]] = defaultdict(list)
            for state, county in map(COUNTY.decompose, includes):
                counties_by_state[state].append(county)
            return [
                {"for": f"county:{','.join(cs)}", "in": f"state:{s}"}
                for s, cs in counties_by_state.items()
            ]

        case TractScope(includes_granularity='state', includes=includes):
            return [{
                "for": "tract:*",
                "in": f"state:{','.join(includes)} county:*",
            }]

        case TractScope(includes_granularity='county', includes=includes):
            counties_by_state: dict[str, list[str]] = defaultdict(list)
            for state, county in map(COUNTY.decompose, includes):
                counties_by_state[state].append(county)
            return [
                {"for": "tract:*",
                    "in": f"state:{s} county:{','.join(cs)}"}
                for s, cs in counties_by_state.items()
            ]

        case TractScope(includes_granularity='tract', includes=includes):
            tracts_by_county: dict[str, list[str]] = defaultdict(list)

            for state, county, tract in map(TRACT.decompose, includes):
                tracts_by_county[state + county].append(tract)

            return [
                {"for": f"tract:{','.join(tracts_by_county[state + county])}",
                    "in": f"state:{state} county:{county}"}
                for state, county in [COUNTY.decompose(c) for c in tracts_by_county.keys()]
            ]

        case BlockGroupScope(includes_granularity='state', includes=includes):
            # This wouldn't normally need to be multiple queries,
            # but Census API won't let you fetch CBGs for multiple states.
            states = {STATE.extract(x) for x in includes}
            return [
                {"for": "block group:*", "in": f"state:{s} county:* tract:*"}
                for s in states
            ]

        case BlockGroupScope(includes_granularity='county', includes=includes):
            counties_by_state: dict[str, list[str]] = defaultdict(list)
            for state, county in map(COUNTY.decompose, includes):
                counties_by_state[state].append(county)
            return [
                {"for": "block group:*",
                    "in": f"state:{s} county:{','.join(cs)} tract:*"}
                for s, cs in counties_by_state.items()
            ]

        case BlockGroupScope(includes_granularity='tract', includes=includes):
            tracts_by_county: dict[str, list[str]] = defaultdict(list)

            for state, county, tract in map(TRACT.decompose, includes):
                tracts_by_county[state + county].append(tract)

            return [
                {"for": "block group:*",
                    "in": f"state:{state} county:{county} tract:{','.join(tracts_by_county[state + county])}"}
                for state, county in [COUNTY.decompose(c) for c in tracts_by_county.keys()]
            ]

        case BlockGroupScope(includes_granularity='block group', includes=includes):
            block_groups_by_tract: dict[str, list[str]] = defaultdict(list)

            for state, county, tract, block_group in map(BLOCK_GROUP.decompose, includes):
                block_groups_by_tract[state + county + tract].append(block_group)

            return [
                {"for": f"block group:{'.'.join(block_groups_by_tract[state + county + tract])}",
                    "in": f"state:{state} county:{county} tract:{tract}"}
                for state, county, tract in [TRACT.decompose(t) for t in block_groups_by_tract.keys()]
            ]

        case _:
            raise DataResourceException("Unsupported query.")


def _fetch_acs5(variables: list[str], scope: CensusScope) -> DataFrame:
    census = _get_api()
    queries = _make_acs5_queries(scope)

    # fetch all queries and combine results
    df = pd.concat(
        pd.DataFrame.from_records(
            census.acs5.get(variables, geo=query, year=scope.year)
        ) for query in queries
    )
    if df.empty:
        msg = "ACS5 query returned empty. Ensure all geographies included in your scope are supported and try again."
        raise DataResourceException(msg)

    # concatenate geoid components to create 'geoid' column
    columns: list[str] = {
        'state': ['state'],
        'county': ['state', 'county'],
        'tract': ['state', 'county', 'tract'],
        'block group': ['state', 'county', 'tract', 'block group'],
    }[scope.granularity]
    df['geoid'] = df[columns].apply(''.join, axis=1)

    # check and sort results for 1:1 match with scope
    try:
        return pd.DataFrame({'geoid': scope.get_node_ids()})\
            .merge(df, on='geoid', how='left', validate="1:1")
    except pd.errors.MergeError:
        msg = "Fetched data was not an exact match for the scope's geographies."
        raise DataResourceException(msg) from None


class Population(Adrio[np.int64]):

    def evaluate(self) -> NDArray[np.int64]:
        scope = self.scope
        if not isinstance(scope, CensusScope):
            msg = "Census scope is required for Census population attribute."
            raise DataResourceException(msg)

        df = _fetch_acs5(['B01001_001E'], scope)
        return df['B01001_001E'].to_numpy(dtype=np.int64)


class PopulationByAgeTable(Adrio[np.int64]):

    def evaluate(self) -> NDArray[np.int64]:
        scope = self.scope
        if not isinstance(scope, CensusScope):
            msg = "Census scope is required to construct population by age table."
            raise DataResourceException(msg)

        # NOTE: asking acs5 explicitly for the [B01001_001E, ...] vars
        # seems to be about twice as fast as asking for group(B01001)
        age_vars = [var for var, _
                    in _get_group_vars(scope.year, 'B01001')]
        age_vars.sort()
        df = _fetch_acs5(age_vars, scope)
        return df[age_vars].to_numpy(dtype=np.int64)


_exact_pattern = re.compile(r"^(\d+) years$")
_under_pattern = re.compile(r"^Under (\d+) years$")
_range_pattern = re.compile(r"^(\d+) (?:to|and) (\d+) years")
_over_pattern = re.compile(r"^(\d+) years and over")


class AgeRange(NamedTuple):
    start: int
    end: int | None

    def contains(self, other: 'AgeRange') -> bool:
        if self.start > other.start:
            return False
        if self.end is None:
            return True
        if other.end is None:
            return False
        return self.end >= other.end

    @staticmethod
    def parse(label: str) -> 'AgeRange | None':
        parts = label.split("!!")
        if len(parts) != 4:
            return None
        bracket = parts[-1]
        if (m := _exact_pattern.match(bracket)) is not None:
            start = int(m.group(1))
            end = start
        elif (m := _under_pattern.match(bracket)) is not None:
            start = 0
            end = int(m.group(1))
        elif (m := _range_pattern.match(bracket)) is not None:
            start = int(m.group(1))
            end = int(m.group(2))
        elif (m := _over_pattern.match(bracket)) is not None:
            start = int(m.group(1))
            end = None
        else:
            raise DataResourceException(f"No match for {label}")
        return AgeRange(start, end)


class PopulationByAge(Adrio[np.int64]):
    POP_BY_AGE_TABLE = AttributeDef('population_by_age_table', int, Shapes.NxA)

    attributes = [POP_BY_AGE_TABLE]

    _age_range: AgeRange

    def __init__(self, age_range_start: int, age_range_end: int | None):
        self._age_range = AgeRange(age_range_start, age_range_end)

    def evaluate(self) -> NDArray[np.int64]:
        scope = self.scope
        if not isinstance(scope, CensusScope):
            msg = "Census scope is required for Census population by age attribute."
            raise DataResourceException(msg)

        age_ranges = [
            AgeRange.parse(attrs['label'])
            for var, attrs
            in _get_group_vars(scope.year, 'B01001')
        ]

        adrio_range = self._age_range

        def is_included(x: AgeRange | None) -> TypeGuard[AgeRange]:
            return x is not None and adrio_range.contains(x)

        included, col_mask = filter_with_mask(age_ranges, is_included)

        # At least one var must have its start equal to the ADRIO range
        if not any((x.start == adrio_range.start for x in included)):
            raise DataResourceException(f"bad start {adrio_range}")
        # At least one var must have its end equal to the ADRIO range
        if not any((x.end == adrio_range.end for x in included)):
            raise DataResourceException(f"bad end {adrio_range}")

        table = self.data(self.POP_BY_AGE_TABLE)
        return table[:, col_mask].sum(axis=1)


class AverageHouseholdSize(Adrio[np.int64]):

    def evaluate(self) -> NDArray[np.int64]:
        scope = self.scope
        if not isinstance(scope, CensusScope):
            msg = "Census scope is required for Census average household size attribute."
            raise DataResourceException(msg)

        df = _fetch_acs5(['B25010_001E'], scope)
        return df['B25010_001E'].to_numpy(dtype=np.int64)


class DissimilarityIndex(Adrio[np.float64]):

    RACE_CATEGORY = Literal['White', 'Black',
                            'Native', 'Asian',
                            'Pacific Islander', 'Other']

    race_variables = {'White': 'B02001_002E',
                      'Black': 'B02001_003E',
                      'Native': 'B02001_004E',
                      'Asian': 'B02001_005E',
                      'Pacific Islander': 'B02001_006E',
                      'Other': 'B02001_007E'}

    def __init__(self, majority_pop: RACE_CATEGORY, minority_pop: RACE_CATEGORY):
        self.majority_var = self.race_variables[majority_pop]
        self.minority_var = self.race_variables[minority_pop]

    def evaluate(self) -> NDArray[np.float64]:
        scope = self.scope
        if not isinstance(scope, CensusScope):
            msg = "Census scope is required for Census dissimilarity index attribute."
            raise DataResourceException(msg)

        if isinstance(scope, BlockGroupScope):
            msg = "Dissimilarity index cannot be retreived for block group scope."
            raise DataResourceException(msg)

        df = _fetch_acs5([self.majority_var, self.minority_var], scope)
        df2 = _fetch_acs5([self.majority_var, self.minority_var],
                          scope.lower_granularity())
        df2['geoid'] = df2['geoid'].apply(
            get_census_granularity(scope.granularity).extract)

        df.rename(columns={self.majority_var: 'high_majority',
                  self.minority_var: 'high_minority'}, inplace=True)
        df2.rename(columns={self.majority_var: 'low_majority',
                   self.minority_var: 'low_minority'}, inplace=True)

        df3 = df.merge(df2, on='geoid')

        df3['score'] = abs(df3['low_minority'] / df3['high_minority'] -
                           df3['low_majority'] / df3['high_majority'])
        df3 = df3.groupby('geoid').sum()
        df3['score'] *= .5
        df3['score'] = df3['score'].replace(0., 0.5)
        df3 = df3.reset_index()

        return df3['score'].to_numpy(dtype=np.float64)


class GiniIndex(Adrio[np.float64]):

    def evaluate(self) -> NDArray[np.float64]:
        scope = self.scope
        if not isinstance(scope, CensusScope):
            msg = "Census scope is required for Census gini index attribute."
            raise DataResourceException(msg)

        df = _fetch_acs5(['B19083_001E'], scope)
        df['B19083_001E'] = df['B19083_001E'].astype(
            np.float64).fillna(0.5).replace(-666666666, 0.5)

        # set cbg data to that of the parent tract if geo granularity = cbg
        if isinstance(scope, BlockGroupScope):
            print(
                "Gini Index cannot be retrieved for block group level, fetching tract level data instead.")
            df2 = _fetch_acs5(['B19083_001E'], scope.raise_granularity())
            df['merge_geoid'] = df['geoid'].apply(lambda x: x[:-1])
            df = df.drop(columns='B19083_001E')

            df = df.merge(df2, left_on='merge_geoid',
                          right_on='geoid', suffixes=(None, '_y'))

        return df['B19083_001E'].to_numpy(dtype=np.float64)


class MedianAge(Adrio[np.float64]):

    def evaluate(self) -> NDArray[np.float64]:
        scope = self.scope
        if not isinstance(scope, CensusScope):
            msg = "Census scope is required for Census median age attribute."
            raise DataResourceException(msg)

        df = _fetch_acs5(['B01002_001E'], scope)
        return df['B01002_001E'].to_numpy(dtype=np.int64)


class MedianIncome(Adrio[np.float64]):

    def evaluate(self) -> NDArray[np.float64]:
        scope = self.scope
        if not isinstance(scope, CensusScope):
            msg = "Census scope is required for Census median income attribute."
            raise DataResourceException(msg)

        df = _fetch_acs5(['B19013_001E'], scope)
        return df['B19013_001E'].to_numpy(dtype=np.int64)


class TractMedianIncome(Adrio[np.int64]):

    def evaluate(self) -> NDArray[np.int64]:
        if isinstance(self.scope, BlockGroupScope):
            # query median income at cbg and tract level
            df = _fetch_acs5(['NAME'], self.scope)
            df2 = _fetch_acs5(['B19013_001E'], self.scope.raise_granularity())
            df2 = df2.fillna(0).replace(-666666666, 0)

            df['tract_geoid'] = df['geoid'].apply(lambda x: x[:-1])
            df = df.merge(df2, left_on='tract_geoid',
                          right_on='geoid', suffixes=(None, '_y'))

            return df['B19013_001E'].to_numpy(dtype=np.int64)

        else:
            msg = "Tract median income can only be retrieved for block group scope."
            raise DataResourceException(msg)
