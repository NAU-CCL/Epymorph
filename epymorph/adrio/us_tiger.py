"""ADRIOs that access the US Census TIGER geography files."""

from abc import ABC
from typing import TypeVar

import numpy as np
from geopandas import GeoDataFrame
from pandas import DataFrame, to_numeric
from typing_extensions import override

from epymorph.adrio.adrio import Adrio, adrio_cache
from epymorph.data_type import CentroidDType, StructDType
from epymorph.data_usage import DataEstimate
from epymorph.error import DataResourceException
from epymorph.geography.scope import GeoScope
from epymorph.geography.us_census import STATE, CensusScope
from epymorph.geography.us_tiger import (
    TigerYear,
    check_cache_block_groups,
    check_cache_counties,
    check_cache_states,
    check_cache_tracts,
    get_block_groups_geo,
    get_block_groups_info,
    get_counties_geo,
    get_counties_info,
    get_states_geo,
    get_states_info,
    get_tracts_geo,
    get_tracts_info,
    is_tiger_year,
)


def _validate_scope(scope: GeoScope) -> CensusScope:
    if not isinstance(scope, CensusScope):
        raise DataResourceException("Census scope is required for us_tiger attributes.")
    return scope


def _validate_year(scope: CensusScope) -> TigerYear:
    year = scope.year
    if not is_tiger_year(year):
        raise DataResourceException(
            f"{year} is not a supported year for us_tiger attributes."
        )
    return year


def _get_geo(scope: CensusScope) -> GeoDataFrame:
    year = _validate_year(scope)
    match scope.granularity:
        case "state":
            gdf = get_states_geo(year)
        case "county":
            gdf = get_counties_geo(year)
        case "tract":
            gdf = get_tracts_geo(
                year, list({STATE.extract(x) for x in scope.get_node_ids()})
            )
        case "block group":
            gdf = get_block_groups_geo(
                year, list({STATE.extract(x) for x in scope.get_node_ids()})
            )
        case x:
            raise DataResourceException(
                f"{x} is not a supported granularity for us_tiger attributes."
            )
    geoid_df = DataFrame({"GEOID": scope.get_node_ids()})
    return GeoDataFrame(geoid_df.merge(gdf, on="GEOID", how="left", sort=True))


def _get_info(scope: CensusScope) -> DataFrame:
    year = _validate_year(scope)
    match scope.granularity:
        case "state":
            gdf = get_states_info(year)
        case "county":
            gdf = get_counties_info(year)
        case "tract":
            gdf = get_tracts_info(
                year, list({STATE.extract(x) for x in scope.get_node_ids()})
            )
        case "block group":
            gdf = get_block_groups_info(
                year, list({STATE.extract(x) for x in scope.get_node_ids()})
            )
        case x:
            raise DataResourceException(
                f"{x} is not a supported granularity for us_tiger attributes."
            )
    geoid_df = DataFrame({"GEOID": scope.get_node_ids()})
    return geoid_df.merge(gdf, on="GEOID", how="left", sort=True)


T_co = TypeVar("T_co", bound=np.generic)


class _UsTigerAdrio(Adrio[T_co], ABC):
    """Abstract class for shared functionality in US Tiger ADRIOs."""

    def estimate_data(self) -> DataEstimate:
        scope = _validate_scope(self.scope)
        year = _validate_year(scope)
        match scope.granularity:
            case "state":
                est = check_cache_states(year)
            case "county":
                est = check_cache_counties(year)
            case "tract":
                est = check_cache_tracts(
                    year, list({STATE.extract(x) for x in scope.get_node_ids()})
                )
            case "block group":
                est = check_cache_block_groups(
                    year, list({STATE.extract(x) for x in scope.get_node_ids()})
                )
            case x:
                raise DataResourceException(
                    f"{x} is not a supported granularity for us_tiger attributes."
                )
        key = f"us_tiger:{scope.granularity}:{year}"
        return DataEstimate(
            name=self.full_name,
            cache_key=key,
            new_network_bytes=est.missing_cache_size,
            new_cache_bytes=est.missing_cache_size,
            total_cache_bytes=est.total_cache_size,
            max_bandwidth=None,
        )


@adrio_cache
class GeometricCentroid(_UsTigerAdrio[StructDType]):
    """The centroid of the geographic polygons."""

    @override
    def evaluate(self):
        scope = _validate_scope(self.scope)
        return (
            _get_geo(scope)["geometry"]
            .apply(lambda x: x.centroid.coords[0])
            .to_numpy(dtype=CentroidDType)
        )


@adrio_cache
class InternalPoint(_UsTigerAdrio[StructDType]):
    """
    The internal point provided by TIGER data. These points are selected by
    Census workers so as to be guaranteed to be within the geographic polygons,
    while geometric centroids have no such guarantee.
    """

    @override
    def evaluate(self):
        scope = _validate_scope(self.scope)
        info_df = _get_info(scope)
        centroids = zip(
            to_numeric(info_df["INTPTLON"]),
            to_numeric(info_df["INTPTLAT"]),
        )
        return np.array(list(centroids), dtype=CentroidDType)


@adrio_cache
class Name(_UsTigerAdrio[np.str_]):
    """For states and counties, the proper name of the location; otherwise its GEOID."""

    @override
    def evaluate(self):
        scope = _validate_scope(self.scope)
        if scope.granularity in ("state", "county"):
            return _get_info(scope)["NAME"].to_numpy(dtype=np.str_)
        else:
            # There aren't good names for Tracts or CBGs, just use GEOID
            return scope.get_node_ids()


@adrio_cache
class PostalCode(_UsTigerAdrio[np.str_]):
    """
    For states only, the postal code abbreviation for the state
    ("AZ" for Arizona, and so on).
    """

    @override
    def evaluate(self):
        scope = _validate_scope(self.scope)
        if scope.granularity != "state":
            raise DataResourceException(
                "PostalCode is only available at state granularity."
            )
        return _get_info(scope)["STUSPS"].to_numpy(dtype=np.str_)


@adrio_cache
class LandAreaM2(_UsTigerAdrio[np.float64]):
    """
    The land area of the geo node in meters-squared. This is the 'ALAND' attribute
    from the TIGER data files.
    """

    @override
    def evaluate(self):
        scope = _validate_scope(self.scope)
        return _get_info(scope)["ALAND"].to_numpy(dtype=np.float64)
