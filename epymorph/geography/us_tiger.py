"""
Functions for fetching TIGER files for common US Census geographic delineations.
This is designed to return information for the entire United States, including territories,
and handles quirks and differences between the supported census years.
"""
from io import BytesIO
from pathlib import Path
from typing import Literal, Sequence, TypeGuard

from geopandas import GeoDataFrame
from geopandas import read_file as gp_read_file
from pandas import DataFrame
from pandas import concat as pd_concat

from epymorph.cache import load_or_fetch_url, module_cache_path
from epymorph.error import GeographyError

# A fair question is why did we implement our own TIGER files loader instead of using pygris?
# The short answer is for efficiently and to correct inconsistencies that matter for our use-case.
# For one, pygris always loads geography but we only want the geography sometimes. By loading it ourselves,
# we can tell Geopandas to skip it, which is a lot faster.
# Second, asking pygris for counties in 2020 returns all territories, while 2010 and 2000 do not.
# This *is* consistent with the TIGER files themselves, but not ideal for us.
# (You can compare the following two files to see for yourself:)
# https://www2.census.gov/geo/tiger/TIGER2020/COUNTY/tl_2020_us_county.zip
# https://www2.census.gov/geo/tiger/TIGER2010/COUNTY/2010/tl_2010_us_county10.zip
# Lastly, pygris has a bug which is patched but not available in a release version at this time:
# https://github.com/walkerke/pygris/commit/9ad16208b5b1e67909ff2dfdea26333ddd4a2e17

# NOTE on which states/territories are included in our results --
# We have chosen to filter results to include only the 50 states, District of Columbia, and Puerto Rico.
# This is not the entire set of data provided by TIGER files, but does align with the
# data that ACS5 provides. Since that is our primary data source at the moment, we felt
# that this was an acceptable simplification. Either we make the two sets match
# (as we've done here, by removing 4 territories) OR we have a special "all states for the ACS5" scope.
# We chose this solution as the less-bad option, but this may be revised in future.
# Below there are some commented-code remnants which demonstrate what it takes to support the additional
# territories, in case we ever want to reverse this choice.

# NOTE: TIGER files express areas in meters-squared.

TigerYear = Literal[2000, 2009, 2010, 2011, 2012, 2013, 2014,
                    2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
"""A supported TIGER file year."""

TIGER_YEARS: Sequence[TigerYear] = (2000, 2009, 2010, 2011, 2012, 2013, 2014,
                                    2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023)
"""All supported TIGER file years."""

_TIGER_URL = "https://www2.census.gov/geo/tiger"

_TIGER_CACHE_PATH = module_cache_path(__name__)

# _SUPPORTED_STATE_FILES = ['us', '60', '66', '69', '78']
_SUPPORTED_STATE_FILES = ['us']
"""
The IDs of TIGER files that are included in our set of supported states.
In some TIGER years, data for the 4 territories were given in separate files.
"""

_SUPPORTED_STATES = ['01', '02', '04', '05', '06', '08', '09', '10', '11', '12', '13',
                     '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25',
                     '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36',
                     '37', '38', '39', '40', '41', '42', '44', '45', '46', '47', '48',
                     '49', '50', '51', '53', '54', '55', '56', '72']
#  '60', '66', '69', '78']
"""
The FIPS IDs of states which are included in our set of supported states.
Not needed if we didn't have to filter out 4 territories.
"""


def _load_urls(urls: list[str]) -> list[BytesIO]:
    """
    Attempt to load the list of URLs from disk cache, or failing that, from the network.
    If the files are not cached, they will be saved to our TIGER cache path.
    """
    try:
        return [
            load_or_fetch_url(u, _TIGER_CACHE_PATH / Path(u).name)
            for u in urls
        ]
    except Exception as e:
        msg = "Unable to retrieve TIGER files for US Census geography."
        raise GeographyError(msg) from e


def _get_geo(cols: list[str], urls: list[str], result_cols: list[str]) -> GeoDataFrame:
    """Universal logic for loading a data set with its geography."""
    files = _load_urls(urls)
    gdf = GeoDataFrame(
        pd_concat([
            gp_read_file(f, engine="fiona", ignore_geometry=False, include_fields=cols)
            for f in files
        ], ignore_index=True)
    )
    gdf.rename(columns=dict(zip(cols, result_cols)), inplace=True)
    return GeoDataFrame(gdf[gdf['GEOID'].apply(lambda x: x[0:2]).isin(_SUPPORTED_STATES)])


def _get_info(cols: list[str], urls: list[str], result_cols: list[str]) -> DataFrame:
    """Universal logic for loading a data set without its geography."""
    files = _load_urls(urls)
    df = pd_concat([
        gp_read_file(f, engine="fiona", ignore_geometry=True, include_fields=cols)
        for f in files
    ], ignore_index=True)
    df.rename(columns=dict(zip(cols, result_cols)), inplace=True)
    df.drop_duplicates(inplace=True)
    return df[df['GEOID'].apply(lambda x: x[0:2]).isin(_SUPPORTED_STATES)]


def is_tiger_year(year: int) -> TypeGuard[TigerYear]:
    """A type-guard function to ensure a year is a supported TIGER year."""
    return year in TIGER_YEARS


##########
# STATES #
##########


def _get_states_config(year: TigerYear) -> tuple[list[str], list[str], list[str]]:
    """Produce the args for _get_info or _get_geo (states)."""
    match year:
        case year if year in range(2011, 2024):
            cols = ["GEOID", "NAME", "STUSPS", "ALAND", "INTPTLAT", "INTPTLON"]
            urls = [
                f"{_TIGER_URL}/TIGER{year}/STATE/tl_{year}_us_state.zip"
            ]
        case 2010:
            cols = ["GEOID10", "NAME10", "STUSPS10",
                    "ALAND10", "INTPTLAT10", "INTPTLON10"]
            urls = [
                f"{_TIGER_URL}/TIGER2010/STATE/2010/tl_2010_{xx}_state10.zip"
                for xx in _SUPPORTED_STATE_FILES
            ]
        case 2009:
            cols = ["STATEFP00", "NAME00", "STUSPS00",
                    "ALAND00", "INTPTLAT00", "INTPTLON00"]
            urls = [
                f"{_TIGER_URL}/TIGER2009/tl_2009_us_state00.zip"
            ]
        case 2000:
            cols = ["STATEFP00", "NAME00", "STUSPS00",
                    "ALAND00", "INTPTLAT00", "INTPTLON00"]
            urls = [
                f"{_TIGER_URL}/TIGER2010/STATE/2000/tl_2010_{xx}_state00.zip"
                for xx in _SUPPORTED_STATE_FILES
            ]
        case _:
            raise GeographyError(f"Unsupported year: {year}")
    return cols, urls, ["GEOID", "NAME", "STUSPS", "ALAND", "INTPTLAT", "INTPTLON"]


def get_states_geo(year: TigerYear) -> GeoDataFrame:
    """Get all US states and territories for the given census year, with geography."""
    return _get_geo(*_get_states_config(year))


def get_states_info(year: TigerYear) -> DataFrame:
    """Get all US states and territories for the given census year, without geography."""
    return _get_info(*_get_states_config(year))


############
# COUNTIES #
############


def _get_counties_config(year: TigerYear) -> tuple[list[str], list[str], list[str]]:
    """Produce the args for _get_info or _get_geo (counties)."""
    match year:
        case year if year in range(2011, 2024):
            cols = ["GEOID", "NAME", "ALAND", "INTPTLAT", "INTPTLON"]
            urls = [
                f"{_TIGER_URL}/TIGER{year}/COUNTY/tl_{year}_us_county.zip"
            ]
        case 2010:
            cols = ["GEOID10", "NAME10", "ALAND10", "INTPTLAT10", "INTPTLON10"]
            urls = [
                f"{_TIGER_URL}/TIGER2010/COUNTY/2010/tl_2010_{xx}_county10.zip"
                for xx in _SUPPORTED_STATE_FILES
            ]
        case 2009:
            cols = ["CNTYIDFP00", "NAME00", "ALAND00",
                    "AWATER00", "INTPTLAT00", "INTPTLON00"]
            urls = [
                f"{_TIGER_URL}/TIGER2009/tl_2009_us_county00.zip"
            ]
        case 2000:
            cols = ["CNTYIDFP00", "NAME00", "ALAND00", "INTPTLAT00", "INTPTLON00"]
            urls = [
                f"{_TIGER_URL}/TIGER2010/COUNTY/2000/tl_2010_{xx}_county00.zip"
                for xx in _SUPPORTED_STATE_FILES
            ]
        case _:
            raise GeographyError(f"Unsupported year: {year}")
    return cols, urls, ["GEOID", "NAME", "ALAND", "INTPTLAT", "INTPTLON"]


def get_counties_geo(year: TigerYear) -> GeoDataFrame:
    """Get all US counties and county-equivalents for the given census year, with geography."""
    return _get_geo(*_get_counties_config(year))


def get_counties_info(year: TigerYear) -> DataFrame:
    """Get all US counties and county-equivalents for the given census year, without geography."""
    return _get_info(*_get_counties_config(year))


##########
# TRACTS #
##########


def _get_tracts_config(year: TigerYear, state_id: Sequence[str] | None = None) -> tuple[list[str], list[str], list[str]]:
    """Produce the args for _get_info or _get_geo (tracts)."""
    states = get_states_info(year)
    if state_id is not None:
        states = states[states["GEOID"].isin(state_id)]

    match year:
        case year if year in range(2011, 2024):
            cols = ["GEOID", "ALAND", "INTPTLAT", "INTPTLON"]
            urls = [
                f"{_TIGER_URL}/TIGER{year}/TRACT/tl_{year}_{xx}_tract.zip"
                for xx in states["GEOID"]
            ]
        case 2010:
            cols = ["GEOID10", "ALAND10", "INTPTLAT10", "INTPTLON10"]
            urls = [
                f"{_TIGER_URL}/TIGER2010/TRACT/2010/tl_2010_{xx}_tract10.zip"
                for xx in states["GEOID"]
            ]
        case 2009:
            def state_folder(fips, name):
                return f"{fips}_{name.upper().replace(' ', '_')}"

            cols = ["CTIDFP00", "ALAND00", "INTPTLAT00", "INTPTLON00"]
            urls = [
                f"{_TIGER_URL}/TIGER2009/{state_folder(xx, name)}/tl_2009_{xx}_tract00.zip"
                for xx, name in zip(states["GEOID"], states["NAME"])
            ]
        case 2000:
            cols = ["CTIDFP00", "ALAND00", "INTPTLAT00", "INTPTLON00"]
            urls = [
                f"{_TIGER_URL}/TIGER2010/TRACT/2000/tl_2010_{xx}_tract00.zip"
                for xx in states["GEOID"]
            ]
        case _:
            raise GeographyError(f"Unsupported year: {year}")
    return cols, urls, ["GEOID", "ALAND", "INTPTLAT", "INTPTLON"]


def get_tracts_geo(year: TigerYear, state_id: Sequence[str] | None = None) -> GeoDataFrame:
    """Get all US census tracts for the given census year, with geography."""
    return _get_geo(*_get_tracts_config(year, state_id))


def get_tracts_info(year: TigerYear, state_id: Sequence[str] | None = None) -> DataFrame:
    """Get all US census tracts for the given census year, without geography."""
    return _get_info(*_get_tracts_config(year, state_id))


################
# BLOCK GROUPS #
################


def _get_block_groups_config(year: TigerYear, state_id: Sequence[str] | None = None) -> tuple[list[str], list[str], list[str]]:
    """Produce the args for _get_info or _get_geo (block groups)."""
    states = get_states_info(year)
    if state_id is not None:
        states = states[states["GEOID"].isin(state_id)]

    match year:
        case year if year in range(2011, 2024):
            cols = ["GEOID", "ALAND", "INTPTLAT", "INTPTLON"]
            urls = [
                f"{_TIGER_URL}/TIGER{year}/BG/tl_{year}_{xx}_bg.zip"
                for xx in states["GEOID"]
            ]
        case 2010:
            cols = ["GEOID10", "ALAND10", "INTPTLAT10", "INTPTLON10"]
            urls = [
                f"{_TIGER_URL}/TIGER2010/BG/2010/tl_2010_{xx}_bg10.zip"
                for xx in states["GEOID"]
            ]
        case 2009:
            def state_folder(fips, name):
                return f"{fips}_{name.upper().replace(' ', '_')}"

            cols = ["BKGPIDFP00", "ALAND00", "INTPTLAT00", "INTPTLON00"]
            urls = [
                f"{_TIGER_URL}/TIGER2009/{state_folder(xx, name)}/tl_2009_{xx}_bg00.zip"
                for xx, name in zip(states["GEOID"], states["NAME"])
            ]
        case 2000:
            cols = ["BKGPIDFP00", "ALAND00", "INTPTLAT00", "INTPTLON00"]
            urls = [
                f"{_TIGER_URL}/TIGER2010/BG/2000/tl_2010_{xx}_bg00.zip"
                for xx in states["GEOID"]
            ]
        case _:
            raise GeographyError(f"Unsupported year: {year}")
    return cols, urls, ["GEOID", "ALAND", "INTPTLAT", "INTPTLON"]


def get_block_groups_geo(year: TigerYear, state_id: Sequence[str] | None = None) -> GeoDataFrame:
    """Get all US census block groups for the given census year, with geography."""
    return _get_geo(*_get_block_groups_config(year, state_id))


def get_block_groups_info(year: TigerYear, state_id: Sequence[str] | None = None) -> DataFrame:
    """Get all US census block groups for the given census year, without geography."""
    return _get_info(*_get_block_groups_config(year, state_id))
