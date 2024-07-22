import numpy as np
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from pandas import DataFrame, to_numeric

from epymorph.data_type import CentroidDType, StructDType
from epymorph.geo.adrio.adrio2 import Adrio
from epymorph.geography.scope import GeoScope
from epymorph.geography.us_census import CensusScope
from epymorph.geography.us_tiger import (get_block_groups_geo,
                                         get_block_groups_info,
                                         get_counties_geo, get_counties_info,
                                         get_states_geo, get_states_info,
                                         get_tracts_geo, get_tracts_info,
                                         is_tiger_year)


def _get_geo(scope: GeoScope) -> GeoDataFrame:
    if not isinstance(scope, CensusScope):
        raise Exception("booo")
    year = scope.year
    if not is_tiger_year(year):
        raise Exception("booo2")

    match scope.granularity:
        case 'state':
            gdf = get_states_geo(year)
        case 'county':
            gdf = get_counties_geo(year)
        case 'tract':
            gdf = get_tracts_geo(year)
        case 'block group':
            gdf = get_block_groups_geo(year)
        case _:
            raise Exception("booo3")

    df = DataFrame({'GEOID': scope.get_node_ids()})
    return GeoDataFrame(df.merge(gdf, on='GEOID', how='left', sort=True))


def _get_info(scope: GeoScope) -> DataFrame:
    if not isinstance(scope, CensusScope):
        raise Exception("booo")
    year = scope.year
    if not is_tiger_year(year):
        raise Exception("booo2")

    match scope.granularity:
        case 'state':
            gdf = get_states_info(year)
        case 'county':
            gdf = get_counties_info(year)
        case 'tract':
            gdf = get_tracts_info(year)
        case 'block group':
            gdf = get_block_groups_info(year)
        case _:
            raise Exception("booo3")

    df = DataFrame({'GEOID': scope.get_node_ids()})
    return df.merge(gdf, on='GEOID', how='left', sort=True)


class GeometricCentroid(Adrio[StructDType]):

    def evaluate(self) -> NDArray:
        return _get_geo(self.scope)['geometry']\
            .apply(lambda x: x.centroid.coords[0])\
            .to_numpy(dtype=CentroidDType)


class InternalPoint(Adrio[StructDType]):

    def evaluate(self) -> NDArray:
        df = _get_info(self.scope)
        return np.array([x for x in zip(
            to_numeric(df['INTPTLON']),
            to_numeric(df['INTPTLAT'])
        )], dtype=CentroidDType)


class Name(Adrio[np.str_]):

    def evaluate(self) -> NDArray:
        scope = self.scope
        if not isinstance(scope, CensusScope):
            raise Exception("booo")

        if scope.granularity in ('state', 'county'):
            return _get_info(scope)['NAME'].to_numpy(dtype=np.str_)
        else:
            # There aren't good names for Tracts or CBGs, just use GEOID
            return scope.get_node_ids()


class PostalCode(Adrio[np.str_]):

    def evaluate(self) -> NDArray[np.str_]:
        scope = self.scope
        if not isinstance(scope, CensusScope):
            raise Exception("booo")
        if scope.granularity != 'state':
            raise Exception("booo3")
        return _get_info(scope)['STUSPS'].to_numpy(dtype=np.str_)


class LandAreaM2(Adrio[np.float64]):

    def evaluate(self) -> NDArray[np.float64]:
        return _get_info(self.scope)['ALAND'].to_numpy(dtype=np.float64)
