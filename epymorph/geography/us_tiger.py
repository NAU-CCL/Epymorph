"""
Functions for fetching TIGER files for common US Census geographic delineations.
This is designed to return information for the entire United States, including territories,
and handles quirks and differences between the supported census years.
"""
from io import BytesIO
from pathlib import Path
from typing import Literal
from urllib.request import urlopen
from warnings import warn

from geopandas import GeoDataFrame
from geopandas import read_file as gp_read_file
from pandas import DataFrame
from pandas import concat as pd_concat

from epymorph.cache import (CacheMiss, CacheWarning, load_file_from_cache,
                            save_file_to_cache)
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


TigerYear = Literal[2000, 2010, 2020]
"""A supported TIGER file year."""

_TIGER_URL = "https://www2.census.gov/geo/tiger"

_TIGER_CACHE_PATH = Path("geography/tiger")


def _fetch_url(url: str) -> BytesIO:
    """Reads a file from a URL as a BytesIO."""
    with urlopen(url) as f:
        file_buffer = BytesIO()
        file_buffer.write(f.read())
        file_buffer.seek(0)
        return file_buffer


def _load_urls(urls: list[str]) -> list[BytesIO]:
    """
    Attempt to load the list of URLs from disk cache, or failing that, from the network.
    If the files are not cached, they will be saved to our TIGER cache path.
    """
    try:
        cached_paths = [
            _TIGER_CACHE_PATH / Path(u).name
            for u in urls
        ]
        try:
            # Try to load files from cache.
            files = [
                load_file_from_cache(path)
                for path in cached_paths
            ]
        except CacheMiss:
            # On cache miss, fetch info from the URLs.
            files = [_fetch_url(u) for u in urls]

            # Attempt to save the files to the cache for next time.
            try:
                for f, path in zip(files, cached_paths):
                    save_file_to_cache(path, f)
            except Exception as e:
                # Failure to save the files to the cache is not worth stopping the program.
                # Issue a warning.
                msg = "We were unable to save TIGER files to the cache.\n" \
                    f"Cause: {e}"
                warn(msg, CacheWarning)

    except Exception as e:
        msg = "Unable to retrieve TIGER files for US Census geography."
        raise GeographyError(msg) from e
    return files


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
    return gdf


def _get_info(cols: list[str], urls: list[str], result_cols: list[str]) -> DataFrame:
    """Universal logic for loading a data set without its geography."""
    files = _load_urls(urls)
    df = pd_concat([
        gp_read_file(f, engine="fiona", ignore_geometry=True, include_fields=cols)
        for f in files
    ], ignore_index=True)
    df.rename(columns=dict(zip(cols, result_cols)), inplace=True)
    df.drop_duplicates(inplace=True)
    return df


##########
# STATES #
##########


def _get_states_config(year: TigerYear) -> tuple[list[str], list[str], list[str]]:
    """Produce the args for _get_info or _get_geo (states)."""
    match year:
        case 2020:
            cols = ["GEOID", "NAME", "STUSPS"]
            urls = [
                f"{_TIGER_URL}/TIGER2020/STATE/tl_2020_us_state.zip"
            ]
        case 2010:
            cols = ["GEOID10", "NAME10", "STUSPS10"]
            urls = [
                f"{_TIGER_URL}/TIGER2010/STATE/2010/tl_2010_{xx}_state10.zip"
                for xx in ['us', '60', '66', '69', '78']
            ]
        case 2000:
            cols = ["STATEFP00", "NAME00", "STUSPS00"]
            urls = [
                f"{_TIGER_URL}/TIGER2010/STATE/2000/tl_2010_{xx}_state00.zip"
                for xx in ['us', '60', '66', '69', '78']
            ]
        case _:
            raise GeographyError(f"Unsupported year: {year}")
    return cols, urls, ["GEOID", "NAME", "STUSPS"]


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
        case 2020:
            cols = ["GEOID", "NAME"]
            urls = [
                f"{_TIGER_URL}/TIGER2020/COUNTY/tl_2020_us_county.zip"
            ]
        case 2010:
            cols = ["GEOID10", "NAME10"]
            urls = [
                f"{_TIGER_URL}/TIGER2010/COUNTY/2010/tl_2010_{xx}_county10.zip"
                for xx in ['us', '60', '66', '69', '78']
            ]
        case 2000:
            cols = ["CNTYIDFP00", "NAME00"]
            urls = [
                f"{_TIGER_URL}/TIGER2010/COUNTY/2000/tl_2010_{xx}_county00.zip"
                for xx in ['us', '60', '66', '69', '78']
            ]
        case _:
            raise GeographyError(f"Unsupported year: {year}")
    return cols, urls, ["GEOID", "NAME"]


def get_counties_geo(year: TigerYear) -> GeoDataFrame:
    """Get all US counties and county-equivalents for the given census year, with geography."""
    return _get_geo(*_get_counties_config(year))


def get_counties_info(year: TigerYear) -> DataFrame:
    """Get all US counties and county-equivalents for the given census year, without geography."""
    return _get_info(*_get_counties_config(year))


##########
# TRACTS #
##########


def _get_tracts_config(year: TigerYear) -> tuple[list[str], list[str], list[str]]:
    """Produce the args for _get_info or _get_geo (tracts)."""
    states = get_states_info(year)["GEOID"]
    match year:
        case 2020:
            cols = ["GEOID"]
            urls = [
                f"{_TIGER_URL}/TIGER2020/TRACT/tl_2020_{xx}_tract.zip"
                for xx in states
            ]
        case 2010:
            cols = ["GEOID10"]
            urls = [
                f"{_TIGER_URL}/TIGER2010/TRACT/2010/tl_2010_{xx}_tract10.zip"
                for xx in states
            ]
        case 2000:
            cols = ["CTIDFP00"]
            urls = [
                f"{_TIGER_URL}/TIGER2010/TRACT/2000/tl_2010_{xx}_tract00.zip"
                for xx in states
            ]
        case _:
            raise GeographyError(f"Unsupported year: {year}")
    return cols, urls, ["GEOID"]


def get_tracts_geo(year: TigerYear) -> GeoDataFrame:
    """Get all US census tracts for the given census year, with geography."""
    return _get_geo(*_get_tracts_config(year))


def get_tracts_info(year: TigerYear) -> DataFrame:
    """Get all US census tracts for the given census year, without geography."""
    return _get_info(*_get_tracts_config(year))


################
# BLOCK GROUPS #
################


def _get_block_groups_config(year: TigerYear) -> tuple[list[str], list[str], list[str]]:
    """Produce the args for _get_info or _get_geo (block groups)."""
    states = get_states_info(year)["GEOID"]
    match year:
        case 2020:
            cols = ["GEOID"]
            urls = [
                f"{_TIGER_URL}/TIGER2020/BG/tl_2020_{xx}_bg.zip"
                for xx in states
            ]
        case 2010:
            cols = ["GEOID10"]
            urls = [
                f"{_TIGER_URL}/TIGER2010/BG/2010/tl_2010_{xx}_bg10.zip"
                for xx in states
            ]
        case 2000:
            cols = ["BKGPIDFP00"]
            urls = [
                f"{_TIGER_URL}/TIGER2010/BG/2000/tl_2010_{xx}_bg00.zip"
                for xx in states
            ]
        case _:
            raise GeographyError(f"Unsupported year: {year}")
    return cols, urls, ["GEOID"]


def get_block_groups_geo(year: TigerYear) -> GeoDataFrame:
    """Get all US census block groups for the given census year, with geography."""
    return _get_geo(*_get_block_groups_config(year))


def get_block_groups_info(year: TigerYear) -> DataFrame:
    """Get all US census block groups for the given census year, without geography."""
    return _get_info(*_get_block_groups_config(year))
