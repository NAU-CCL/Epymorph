"""
Functions for fetching TIGER files for common US Census geographic delineations.
This is designed to return information for the entire United States,
including territories, and handles quirks and differences between the supported
census years.
"""

import re
from abc import ABC
from dataclasses import asdict, dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Callable, Literal, Mapping, NamedTuple, Sequence, TypeGuard, TypeVar

import numpy as np
from geopandas import GeoDataFrame
from geopandas import read_file as gp_read_file
from pandas import DataFrame
from pandas import concat as pd_concat
from typing_extensions import override

from epymorph.adrio.adrio import ProgressCallback
from epymorph.cache import (
    CacheMiss,
    check_file_in_cache,
    load_bundle_from_cache,
    load_or_fetch_url,
    module_cache_path,
    save_bundle_to_cache,
)
from epymorph.error import GeographyError
from epymorph.geography.us_geography import STATE, CensusGranularityName
from epymorph.util import cache_transparent, normalize_list, normalize_str, zip_list

# A fair question is why did we implement our own TIGER files loader instead of using
# pygris? The short answer is for efficiently and to correct inconsistencies that matter
# for our use-case. For one, pygris always loads geography but we only want the
# geography sometimes. By loading it ourselves, we can tell Geopandas to skip it,
# which is a lot faster. Second, asking pygris for counties in 2020 returns all
# territories, while 2010 and 2000 do not.
# This *is* consistent with the TIGER files themselves, but not ideal for us.
# (You can compare the following two files to see for yourself:)
# https://www2.census.gov/geo/tiger/TIGER2020/COUNTY/tl_2020_us_county.zip
# https://www2.census.gov/geo/tiger/TIGER2010/COUNTY/2010/tl_2010_us_county10.zip
# Lastly, pygris has a bug which is patched but not available in a release version
# at this time:
# https://github.com/walkerke/pygris/commit/9ad16208b5b1e67909ff2dfdea26333ddd4a2e17

# NOTE on which states/territories are included in our results --
# We have chosen to filter results to include only the 50 states, District of Columbia,
# and Puerto Rico. This is not the entire set of data provided by TIGER files, but does
# align with the data that ACS5 provides. Since that is our primary data source at the
# moment, we felt that this was an acceptable simplification. Either we make the two
# sets match (as we've done here, by removing 4 territories) OR we have a special
# "all states for the ACS5" scope. We chose this solution as the less-bad option,
# but this may be revised in future. Below there are some commented-code remnants which
# demonstrate what it takes to support the additional territories, in case we ever want
# to reverse this choice.

# NOTE: TIGER files express areas in meters-squared.

# fmt: off
TigerYear = Literal[
    2000, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, # noqa
]
"""A supported TIGER file year."""

TIGER_YEARS: Sequence[TigerYear] = (
    2000, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, #noqa
)
"""All supported TIGER file years."""

_SUPPORTED_STATES = [
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12",
    "13", "15", "16", "17", "18", "19", "20", "21", "22", "23",
    "24", "25", "26", "27", "28", "29", "30", "31", "32", "33",
    "34", "35", "36", "37", "38", "39", "40", "41", "42", "44",
    "45", "46", "47", "48", "49", "50", "51", "53", "54", "55",
    "56", "72", # '60', '66', '69', '78'
]
"""
The FIPS IDs of states which are included in our set of supported states.
Not needed if we didn't have to filter out 4 territories.
"""
# fmt: on

_TIGER_URL = "https://www2.census.gov/geo/tiger"

_TIGER_CACHE_PATH = module_cache_path(__name__)

# _SUPPORTED_STATE_FILES = ['us', '60', '66', '69', '78']
_SUPPORTED_STATE_FILES = ["us"]
"""
The IDs of TIGER files that are included in our set of supported states.
In some TIGER years, data for the 4 territories were given in separate files.
"""


def is_tiger_year(year: int) -> TypeGuard[TigerYear]:
    """A type-guard function to ensure a year is a supported TIGER year."""
    return year in TIGER_YEARS


def _url_to_cache_path(url: str) -> Path:
    return _TIGER_CACHE_PATH / Path(url).name


class _DataConfig(NamedTuple):
    urls: list[str]
    """URLs for all of the required data files."""
    columns: list[tuple[str, str]]
    """Map each column's name in the source file to its final name in the result."""


def _load_urls(
    config: _DataConfig,
    *,
    ignore_geometry: bool,
    progress: ProgressCallback | None,
) -> DataFrame:
    """
    Load TIGER files either from disk cache or the network.
    The result is processed and returned as one large DataFrame.
    """
    urls, columns = config
    processing_steps = len(urls) + 1  # add one to account for the post-processing
    try:
        # Fetch the contents of each file and read them as a DataFrame.
        dfs = list[DataFrame]()
        for i, u in enumerate(urls):
            u_df = gp_read_file(
                load_or_fetch_url(u, _url_to_cache_path(u)),
                engine="fiona",
                ignore_geometry=ignore_geometry,
                include_fields=[c for c, _ in columns],
            )
            dfs.append(u_df)
            if progress is not None:
                progress((i + 1) / processing_steps, None)

        # Concat the DataFrames, fix column names, and data quality checks.
        combined_df = (
            pd_concat(dfs, ignore_index=True)
            .rename(columns=dict(columns))
            .drop_duplicates()
        )
        # Drop records that aren't in our supported set of states.
        selection = combined_df["GEOID"].apply(STATE.truncate).isin(_SUPPORTED_STATES)
        return combined_df[selection]
    except Exception as e:
        msg = "Unable to retrieve TIGER files for US Census geography."
        raise GeographyError(msg) from e


def _get_geo(
    config: _DataConfig,
    progress: ProgressCallback | None,
) -> GeoDataFrame:
    """Universal logic for loading a data set with its geography."""
    return GeoDataFrame(_load_urls(config, ignore_geometry=False, progress=progress))


def _get_info(
    config: _DataConfig,
    progress: ProgressCallback | None,
) -> DataFrame:
    """Universal logic for loading a data set without its geography."""
    return _load_urls(config, ignore_geometry=True, progress=progress)


class CacheEstimate(NamedTuple):
    """Estimates related to data needed to fulfill TIGER requests."""

    total_cache_size: int
    """An estimate of the size of the files that we need to have cached to fulfill
    a request."""
    missing_cache_size: int
    """An estimate of the size of the files that are not currently cached that we
    would need to fulfill a request. Zero if we have all of the files already."""


@dataclass(frozen=True)
class GranularitySummary(ABC):
    geoid: list[str]

    def interpret(self, identifiers: Sequence[str]) -> list[str]:
        """Permissively interprets the given set of identifiers as describing nodes,
        and converts them to a sorted list of GEOIDs."""
        # The base case is that the identifiers are literal GEOIDs.
        return self._to_geoid(identifiers, self.geoid, "FIPS code")

    def _to_geoid(
        self,
        identifiers: Sequence[str],
        source: list[str],
        description: str,
    ) -> list[str]:
        results = list[str]()
        for x in identifiers:
            try:
                i = source.index(normalize_str(x))
                results.append(self.geoid[i])
            except ValueError:
                err = f"{x} is not a valid {description}."
                raise GeographyError(err) from None
        results.sort()
        return results


_SummaryT = TypeVar("_SummaryT", bound=GranularitySummary)


def _load_summary_from_cache(
    relpath: str,
    on_miss: Callable[[], _SummaryT],  # load ModelT from another source
    on_hit: Callable[..., _SummaryT],  # load ModelT from cache (constructor)
) -> _SummaryT:
    # NOTE: this would be more natural as a decorator,
    # but Pylance seems to have problems tracking the return type properly
    # with that implementation
    CACHE_VERSION = 1
    path = _TIGER_CACHE_PATH.joinpath(relpath)
    try:
        content = load_bundle_from_cache(path, CACHE_VERSION)
        with np.load(content["data.npz"]) as data_npz:
            return on_hit(**{k: v.tolist() for k, v in data_npz.items()})
    except CacheMiss:
        data = on_miss()
        data_bytes = BytesIO()
        # NOTE: Python doesn't include a type for dataclass instances;
        # you can import DataclassInstance from _typeshed, but that seems
        # to break test discovery. Oh well; just ignore this one.
        model_dict = asdict(data)  # type: ignore
        np.savez_compressed(data_bytes, **model_dict)
        save_bundle_to_cache(path, CACHE_VERSION, {"data.npz": data_bytes})
        return data


##########
# STATES #
##########


def _get_states_config(year: TigerYear) -> _DataConfig:
    """Produce the args for _get_info or _get_geo (states)."""
    match year:
        case year if year in range(2011, 2024):
            cols = ["GEOID", "NAME", "STUSPS", "ALAND", "INTPTLAT", "INTPTLON"]
            urls = [f"{_TIGER_URL}/TIGER{year}/STATE/tl_{year}_us_state.zip"]
        case 2010:
            cols = [
                "GEOID10",
                "NAME10",
                "STUSPS10",
                "ALAND10",
                "INTPTLAT10",
                "INTPTLON10",
            ]
            urls = [
                f"{_TIGER_URL}/TIGER2010/STATE/2010/tl_2010_{xx}_state10.zip"
                for xx in _SUPPORTED_STATE_FILES
            ]
        case 2009:
            cols = [
                "STATEFP00",
                "NAME00",
                "STUSPS00",
                "ALAND00",
                "INTPTLAT00",
                "INTPTLON00",
            ]
            urls = [f"{_TIGER_URL}/TIGER2009/tl_2009_us_state00.zip"]
        case 2000:
            cols = [
                "STATEFP00",
                "NAME00",
                "STUSPS00",
                "ALAND00",
                "INTPTLAT00",
                "INTPTLON00",
            ]
            urls = [
                f"{_TIGER_URL}/TIGER2010/STATE/2000/tl_2010_{xx}_state00.zip"
                for xx in _SUPPORTED_STATE_FILES
            ]
        case _:
            raise GeographyError(f"Unsupported year: {year}")
    columns = zip_list(
        cols, ["GEOID", "NAME", "STUSPS", "ALAND", "INTPTLAT", "INTPTLON"]
    )
    return _DataConfig(urls, columns)


def get_states_geo(
    year: TigerYear,
    progress: ProgressCallback | None = None,
) -> GeoDataFrame:
    """Get all US states and territories for the given census year, with geography."""
    return _get_geo(_get_states_config(year), progress)


def get_states_info(
    year: TigerYear,
    progress: ProgressCallback | None = None,
) -> DataFrame:
    """
    Get all US states and territories for the given census year, without geography.
    """
    return _get_info(_get_states_config(year), progress)


def check_cache_states(year: TigerYear) -> CacheEstimate:
    """Check the cache status for a US states and territories query."""
    urls, _ = _get_states_config(year)
    est_file_size = 9_000_000  # each states file is approx 9MB
    total_files = len(urls)
    missing_files = total_files - sum(
        1 for u in urls if check_file_in_cache(_url_to_cache_path(u))
    )
    return CacheEstimate(
        total_cache_size=total_files * est_file_size,
        missing_cache_size=missing_files * est_file_size,
    )


@dataclass(frozen=True)
class StatesSummary(GranularitySummary):
    """Information about US states (and state equivalents)."""

    geoid: list[str]
    """The GEOID (aka FIPS code) of the state."""
    name: list[str]
    """The typical name for the state."""
    code: list[str]
    """The US postal code for the state."""

    @cached_property
    def state_code_to_fips(self) -> Mapping[str, str]:
        """Mapping from state postal code to FIPS code."""
        return dict(zip(self.code, self.geoid, strict=True))

    @cached_property
    def state_fips_to_code(self) -> Mapping[str, str]:
        """Mapping from state FIPS code to postal code."""
        return dict(zip(self.geoid, self.code, strict=True))

    @override
    def interpret(self, identifiers: Sequence[str]) -> list[str]:
        """Permissively interprets the given set of identifiers as describing nodes,
        and converts them to a sorted list of GEOIDs.

        Identifiers can be given in any of the acceptable forms, but all of the
        identifiers must use the same form. Forms are: GEOID/FIPS code, full name, or
        postal code. Raises GeographyError if invalid or identifiers are given."""
        first_val = identifiers[0]
        if re.fullmatch(r"\d{2}", first_val) is not None:
            return super().interpret(identifiers)
        elif re.fullmatch(r"[A-Z]{2}", first_val, flags=re.IGNORECASE) is not None:
            return self._to_geoid(identifiers, normalize_list(self.code), "postal code")
        else:
            return self._to_geoid(identifiers, normalize_list(self.name), "state name")


@cache_transparent
def get_states(year: int) -> StatesSummary:
    """Loads US States information (assumed to be invariant for all supported years)."""
    if not is_tiger_year(year):
        raise GeographyError(f"Unsupported year: {year}")

    def _get_us_states() -> StatesSummary:
        states_df = get_states_info(year).sort_values("GEOID")
        return StatesSummary(
            geoid=states_df["GEOID"].to_list(),
            name=states_df["NAME"].to_list(),
            code=states_df["STUSPS"].to_list(),
        )

    return _load_summary_from_cache("us_states_all.tgz", _get_us_states, StatesSummary)


############
# COUNTIES #
############


def _get_counties_config(year: TigerYear) -> _DataConfig:
    """Produce the args for _get_info or _get_geo (counties)."""
    match year:
        case year if year in range(2011, 2024):
            cols = ["GEOID", "NAME", "ALAND", "INTPTLAT", "INTPTLON"]
            urls = [f"{_TIGER_URL}/TIGER{year}/COUNTY/tl_{year}_us_county.zip"]
        case 2010:
            cols = ["GEOID10", "NAME10", "ALAND10", "INTPTLAT10", "INTPTLON10"]
            urls = [
                f"{_TIGER_URL}/TIGER2010/COUNTY/2010/tl_2010_{xx}_county10.zip"
                for xx in _SUPPORTED_STATE_FILES
            ]
        case 2009:
            cols = [
                "CNTYIDFP00",
                "NAME00",
                "ALAND00",
                "INTPTLAT00",
                "INTPTLON00",
            ]
            urls = [f"{_TIGER_URL}/TIGER2009/tl_2009_us_county00.zip"]
        case 2000:
            cols = ["CNTYIDFP00", "NAME00", "ALAND00", "INTPTLAT00", "INTPTLON00"]
            urls = [
                f"{_TIGER_URL}/TIGER2010/COUNTY/2000/tl_2010_{xx}_county00.zip"
                for xx in _SUPPORTED_STATE_FILES
            ]
        case _:
            raise GeographyError(f"Unsupported year: {year}")
    columns = zip_list(cols, ["GEOID", "NAME", "ALAND", "INTPTLAT", "INTPTLON"])
    return _DataConfig(urls, columns)


def get_counties_geo(
    year: TigerYear,
    progress: ProgressCallback | None = None,
) -> GeoDataFrame:
    """
    Get all US counties and county-equivalents for the given census year,
    with geography.
    """
    return _get_geo(_get_counties_config(year), progress)


def get_counties_info(
    year: TigerYear,
    progress: ProgressCallback | None = None,
) -> DataFrame:
    """
    Get all US counties and county-equivalents for the given census year,
    without geography.
    """
    return _get_info(_get_counties_config(year), progress)


def check_cache_counties(year: TigerYear) -> CacheEstimate:
    """Check the cache status for a US counties query."""
    urls, _ = _get_counties_config(year)
    est_file_size = 75_000_000  # each county file is approx 75MB
    total_files = len(urls)
    missing_files = total_files - sum(
        1 for u in urls if check_file_in_cache(_url_to_cache_path(u))
    )
    return CacheEstimate(
        total_cache_size=total_files * est_file_size,
        missing_cache_size=missing_files * est_file_size,
    )


@dataclass(frozen=True)
class CountiesSummary(GranularitySummary):
    """Information about US counties (and county equivalents.)"""

    geoid: list[str]
    """The GEOID (aka FIPS code) of the county."""
    name: list[str]
    """The typical name of the county (does not include state)."""
    name_with_state: list[str]
    """The name of the county and state, e.g., `Coconino, AZ`"""

    @cached_property
    def county_fips_to_name(self) -> Mapping[str, str]:
        """Mapping from county FIPS code to name with state."""
        return dict(zip(self.geoid, self.name_with_state, strict=True))

    @override
    def interpret(self, identifiers: Sequence[str]) -> list[str]:
        """Permissively interprets the given set of identifiers as describing nodes,
        and converts them to a sorted list of GEOIDs.

        Identifiers can be given in any of the acceptable forms, but all of the
        identifiers must use the same form. Forms are: GEOID/FIPS code, or
        the name of the county and its state postal code separated by a comma,
        e.g., `Coconino, AZ`. Raises GeographyError if invalid or identifiers are
        given."""
        first_val = identifiers[0]
        if re.fullmatch(r"\d{5}", first_val) is not None:
            return super().interpret(identifiers)
        else:
            return self._to_geoid(
                identifiers,
                normalize_list(self.name_with_state),
                "county name",
            )


@cache_transparent
def get_counties(year: int) -> CountiesSummary:
    """Loads US Counties information for the given year."""
    if not is_tiger_year(year):
        raise GeographyError(f"Unsupported year: {year}")

    def _get_us_counties() -> CountiesSummary:
        counties_df = get_counties_info(year, None).sort_values("GEOID")
        code_map = get_states(year).state_fips_to_code
        counties_df["POSTAL_CODE"] = (
            counties_df["GEOID"].str.slice(0, 2).apply(lambda x: code_map[x])
        )
        counties_df["NAME_WITH_STATE"] = (
            counties_df["NAME"] + ", " + counties_df["POSTAL_CODE"]
        )
        return CountiesSummary(
            geoid=counties_df["GEOID"].to_list(),
            name=counties_df["NAME"].to_list(),
            name_with_state=counties_df["NAME_WITH_STATE"].to_list(),
        )

    return _load_summary_from_cache(
        f"us_counties_{year}.tgz", _get_us_counties, CountiesSummary
    )


##########
# TRACTS #
##########


def _get_tracts_config(
    year: TigerYear,
    state_id: Sequence[str] | None = None,
) -> _DataConfig:
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
                f"{_TIGER_URL}/TIGER2009/{state_folder(xx, name)}/tl_2009_{xx}_tract00.zip"  # noqa: E501
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
    columns = zip_list(cols, ["GEOID", "ALAND", "INTPTLAT", "INTPTLON"])
    return _DataConfig(urls, columns)


def get_tracts_geo(
    year: TigerYear,
    state_id: Sequence[str] | None = None,
    progress: ProgressCallback | None = None,
) -> GeoDataFrame:
    """Get all US census tracts for the given census year, with geography."""
    return _get_geo(_get_tracts_config(year, state_id), progress=progress)


def get_tracts_info(
    year: TigerYear,
    state_id: Sequence[str] | None = None,
    progress: ProgressCallback | None = None,
) -> DataFrame:
    """Get all US census tracts for the given census year, without geography."""
    return _get_info(_get_tracts_config(year, state_id), progress=progress)


def check_cache_tracts(
    year: TigerYear,
    state_id: Sequence[str] | None = None,
) -> CacheEstimate:
    """Check the cache status for a US census tracts query."""
    urls, _ = _get_tracts_config(year, state_id)
    est_file_size = 7_000_000  # each tracts file is approx 7MB
    total_files = len(urls)
    missing_files = total_files - sum(
        1 for u in urls if check_file_in_cache(_url_to_cache_path(u))
    )
    return CacheEstimate(
        total_cache_size=total_files * est_file_size,
        missing_cache_size=missing_files * est_file_size,
    )


@dataclass(frozen=True)
class TractsSummary(GranularitySummary):
    """Information about US census tracts."""

    geoid: list[str]
    """The GEOID (aka FIPS code) of the tract."""


@cache_transparent
def get_tracts(year: int) -> TractsSummary:
    """Loads US Census Tracts information for the given year."""
    if not is_tiger_year(year):
        raise GeographyError(f"Unsupported year: {year}")

    def _get_us_tracts() -> TractsSummary:
        tracts_df = get_tracts_info(year).sort_values("GEOID")
        return TractsSummary(
            geoid=tracts_df["GEOID"].to_list(),
        )

    return _load_summary_from_cache(
        f"us_tracts_{year}.tgz", _get_us_tracts, TractsSummary
    )


################
# BLOCK GROUPS #
################


def _get_block_groups_config(
    year: TigerYear,
    state_id: Sequence[str] | None = None,
) -> _DataConfig:
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
    columns = zip_list(cols, ["GEOID", "ALAND", "INTPTLAT", "INTPTLON"])
    return _DataConfig(urls, columns)


def get_block_groups_geo(
    year: TigerYear,
    state_id: Sequence[str] | None = None,
    progress: ProgressCallback | None = None,
) -> GeoDataFrame:
    """Get all US census block groups for the given census year, with geography."""
    return _get_geo(_get_block_groups_config(year, state_id), progress=progress)


def get_block_groups_info(
    year: TigerYear,
    state_id: Sequence[str] | None = None,
    progress: ProgressCallback | None = None,
) -> DataFrame:
    """Get all US census block groups for the given census year, without geography."""
    return _get_info(_get_block_groups_config(year, state_id), progress=progress)


def check_cache_block_groups(
    year: TigerYear,
    state_id: Sequence[str] | None = None,
) -> CacheEstimate:
    """Check the cache status for a US census block groups query."""
    urls, _ = _get_block_groups_config(year, state_id)
    est_file_size = 1_250_000  # each block groups file is approx 1.25MB
    total_files = len(urls)
    missing_files = total_files - sum(
        1 for u in urls if check_file_in_cache(_url_to_cache_path(u))
    )
    return CacheEstimate(
        total_cache_size=total_files * est_file_size,
        missing_cache_size=missing_files * est_file_size,
    )


@dataclass(frozen=True)
class BlockGroupsSummary(GranularitySummary):
    """Information about US census block groups."""

    geoid: list[str]
    """The GEOID (aka FIPS code) of the block group."""


@cache_transparent
def get_block_groups(year: int) -> BlockGroupsSummary:
    """Loads US Census Block Group information for the given year."""
    if not is_tiger_year(year):
        raise GeographyError(f"Unsupported year: {year}")

    def _get_us_cbgs() -> BlockGroupsSummary:
        cbgs_df = get_block_groups_info(year).sort_values("GEOID")
        return BlockGroupsSummary(
            geoid=cbgs_df["GEOID"].to_list(),
        )

    return _load_summary_from_cache(
        f"us_block_groups_{year}.tgz",
        _get_us_cbgs,
        BlockGroupsSummary,
    )


################
# GENERAL UTIL #
################


def get_summary_of(granularity: CensusGranularityName, year: int) -> GranularitySummary:
    match granularity:
        case "state":
            return get_states(year)
        case "county":
            return get_counties(year)
        case "tract":
            return get_tracts(year)
        case "block group":
            return get_block_groups(year)
        case _:
            raise GeographyError(f"Unsupported granularity: {granularity}")
