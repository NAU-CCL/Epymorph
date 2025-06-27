"""
Functions for fetching information from TIGER files for common US Census geographic
delineations. This is designed to return information for a common selection of the
United States and territories, and handles quirks and differences between the supported
census years.
"""

import re
from abc import ABC
from dataclasses import asdict, dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import (
    Callable,
    Literal,
    Mapping,
    NamedTuple,
    Sequence,
    TypeGuard,
    TypeVar,
    overload,
)

import numpy as np
from geopandas import GeoDataFrame
from geopandas import read_file as gp_read_file
from pandas import DataFrame
from pandas import concat as pd_concat
from typing_extensions import override

from epymorph.adrio.adrio import ProgressCallback
from epymorph.cache import (
    CacheMissError,
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
TigerYear = Literal[2000, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023] # noqa
"""A supported TIGER file year. (2000 and 2009-2023)"""

TIGER_YEARS: Sequence[TigerYear] = (2000, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023) #noqa
"""All supported TIGER file years. (2000 and 2009-2023)"""

_SUPPORTED_STATES = [
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12",
    "13", "15", "16", "17", "18", "19", "20", "21", "22", "23",
    "24", "25", "26", "27", "28", "29", "30", "31", "32", "33",
    "34", "35", "36", "37", "38", "39", "40", "41", "42", "44",
    "45", "46", "47", "48", "49", "50", "51", "53", "54", "55",
    "56", "72",
]
"""
The FIPS IDs of states which are included in our set of supported states.
Not needed if we didn't have to filter out 4 territories. (60, 66, 69, 78)
"""
# fmt: on

_TIGER_URL = "https://www2.census.gov/geo/tiger"

_TIGER_CACHE_PATH = module_cache_path(__name__)

_CACHE_VERSION = 1

_SUPPORTED_STATE_FILES = ["us"]
"""
The IDs of TIGER files that are included in our set of supported states.
In some TIGER years, data for the 4 territories were given in separate files.
"""


def is_tiger_year(year: int) -> TypeGuard[TigerYear]:
    """
    Check that a year is a supported TIGER year, as a type guard.

    Parameters
    ----------
    year :
        The year to check.

    Returns
    -------
    :
        True (as a type guard) if the year is in the set of supported TIGER years.
    """
    return year in TIGER_YEARS


def _url_to_cache_path(url: str) -> Path:
    """Given the URL of a TIGER file, return the cache path we should use for it."""
    return _TIGER_CACHE_PATH / Path(url).name


class _DataConfig(NamedTuple):
    """
    Encodes configuration details needed to access a particular granularity and year
    of TIGER files.
    """

    granularity: CensusGranularityName
    """The granularity these details are for."""
    urls: list[str]
    """URLs for all of the required data files."""
    columns: list[tuple[str, str]]
    """Map each column's name in the source file to its final name in the result."""
    estimated_file_size: int
    """The approximate size of individual TIGER files at this granularity."""


@overload
def _load_tiger_data(
    config: _DataConfig,
    *,
    ignore_geometry: Literal[True],
    progress: ProgressCallback | None = None,
) -> DataFrame: ...


@overload
def _load_tiger_data(
    config: _DataConfig,
    *,
    ignore_geometry: Literal[False],
    progress: ProgressCallback | None = None,
) -> GeoDataFrame: ...


def _load_tiger_data(
    config: _DataConfig,
    *,
    ignore_geometry: bool,
    progress: ProgressCallback | None = None,
) -> DataFrame | GeoDataFrame:
    """
    Load TIGER files either from disk cache or the network.
    The result is processed and returned as one large DataFrame.
    """
    _, urls, columns, _ = config
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
        selected_df = combined_df[selection]
        return selected_df if ignore_geometry else GeoDataFrame(selected_df)
    except Exception as e:
        msg = "Unable to retrieve TIGER files for US Census geography."
        raise GeographyError(msg) from e


@dataclass(frozen=True)
class GranularitySummary(ABC):
    """
    Contains information about the geography at the given level of
    granularity for a specific Census year. The information available may differ
    between the implementations for different granularities, but at a minimum
    each provides the full list of GEOIDs in that granularity.

    Concrete child classes exist for the various Census granularity levels.

    Parameters
    ----------
    geoid :
        The GEOIDs (sometimes called FIPS codes) of all nodes in this granularity.
    """

    geoid: list[str]
    """The GEOIDs (sometimes called FIPS codes) of all nodes in this granularity."""

    def interpret(self, identifiers: Sequence[str]) -> list[str]:
        """
        Permissively interprets the given set of identifiers as describing nodes,
        and converts them to a sorted list of GEOIDs.

        Parameters
        ----------
        identifiers :
            A list of identifiers. Which kind of identifiers are allowed depends
            on the granularity.

        Returns
        -------
        :
            The list of GEOIDs in canonical sort order.

        Raises
        ------
        GeographyError
            If the identifiers cannot be interpreted.
        """
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
    path = _TIGER_CACHE_PATH.joinpath(relpath)
    try:
        content = load_bundle_from_cache(path, _CACHE_VERSION)
        with np.load(content["data.npz"]) as data_npz:
            return on_hit(**{k: v.tolist() for k, v in data_npz.items()})
    except CacheMissError:
        data = on_miss()
        data_bytes = BytesIO()
        # NOTE: Python doesn't include a type for dataclass instances;
        # you can import DataclassInstance from _typeshed, but that seems
        # to break test discovery. Oh well; just ignore this one.
        model_dict = asdict(data)  # type: ignore
        np.savez_compressed(data_bytes, **model_dict)
        save_bundle_to_cache(path, _CACHE_VERSION, {"data.npz": data_bytes})
        return data


##########
# STATES #
##########


def _get_states_config(year: int) -> _DataConfig:
    """Produce the args for _get_info or _get_geo (states)."""
    if not is_tiger_year(year):
        raise GeographyError(f"Unsupported year: {year}")
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
    # each states file is approx 9MB
    return _DataConfig("state", urls, columns, estimated_file_size=9_000_000)


def get_states_geo(
    year: int,
    progress: ProgressCallback | None = None,
) -> GeoDataFrame:
    """
    Get all supported US states and territories for the given census year, with
    geography.

    Parameters
    ----------
    year :
        The geography year.
    progress :
        A optional callback for reporting the progress of downloading TIGER files.

    Returns
    -------
    :
        The TIGER file info with geography.
    """
    config = _get_states_config(year)
    return _load_tiger_data(config, ignore_geometry=False, progress=progress)


def get_states_info(
    year: int,
    progress: ProgressCallback | None = None,
) -> DataFrame:
    """
    Get all US states and territories for the given census year, without geography.

    Parameters
    ----------
    year :
        The geography year.
    progress :
        A optional callback for reporting the progress of downloading TIGER files.

    Returns
    -------
    :
        The TIGER file info without geography.
    """
    config = _get_states_config(year)
    return _load_tiger_data(config, ignore_geometry=True, progress=progress)


@dataclass(frozen=True)
class StatesSummary(GranularitySummary):
    """
    Information about US states (and state equivalents). Typically you will use
    `get_states` to obtain an instance of this class for a particular year.

    Parameters
    ----------
    geoid :
        The GEOIDs (aka FIPS codes) of all states.
    name :
        The typical names for the states.
    code :
        The US postal codes for the states.
    """

    geoid: list[str]
    """The GEOIDs (aka FIPS codes) of all states."""
    name: list[str]
    """The typical names for the states."""
    code: list[str]
    """The US postal codes for the states."""

    @cached_property
    def state_code_to_fips(self) -> Mapping[str, str]:
        """Mapping from state postal code to FIPS code."""
        return dict(zip(self.code, self.geoid, strict=True))

    @cached_property
    def state_fips_to_code(self) -> Mapping[str, str]:
        """Mapping from state FIPS code to postal code."""
        return dict(zip(self.geoid, self.code, strict=True))

    @cached_property
    def state_fips_to_name(self) -> Mapping[str, str]:
        """Mapping from state FIPS code to full name."""
        return dict(zip(self.geoid, self.name, strict=True))

    @override
    def interpret(self, identifiers: Sequence[str]) -> list[str]:
        """
        Permissively interprets the given set of identifiers as describing nodes,
        and converts them to a sorted list of GEOIDs.

        Parameters
        ----------
        identifiers :
            A list of identifiers. Identifiers can be given in any of the acceptable
            forms, but all of the identifiers must use the same form. Forms are:
            GEOID/FIPS code, full name, or postal code.

        Returns
        -------
        :
            The list of GEOIDs in canonical sort order.

        Raises
        ------
        GeographyError
            If invalid identifiers are given.
        """
        first_val = identifiers[0]
        if re.fullmatch(r"\d{2}", first_val) is not None:
            return super().interpret(identifiers)
        elif re.fullmatch(r"[A-Z]{2}", first_val, flags=re.IGNORECASE) is not None:
            return self._to_geoid(identifiers, normalize_list(self.code), "postal code")
        else:
            return self._to_geoid(identifiers, normalize_list(self.name), "state name")


@cache_transparent
def get_states(year: int) -> StatesSummary:
    """
    Load US States information (assumed to be invariant for all supported years).

    Parameters
    ----------
    year :
        The geography year.

    Returns
    -------
    :
        The summary.
    """
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


def _get_counties_config(year: int) -> _DataConfig:
    """Produce the args for _get_info or _get_geo (counties)."""
    if not is_tiger_year(year):
        raise GeographyError(f"Unsupported year: {year}")
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
    # each county file is approx 75MB
    return _DataConfig("county", urls, columns, 75_000_000)


def get_counties_geo(
    year: int,
    progress: ProgressCallback | None = None,
) -> GeoDataFrame:
    """
    Get all supported US counties and county-equivalents for the given census year,
    with geography.

    Parameters
    ----------
    year :
        The geography year.
    progress :
        A optional callback for reporting the progress of downloading TIGER files.

    Returns
    -------
    :
        The TIGER file info with geography.
    """
    config = _get_counties_config(year)
    return _load_tiger_data(config, ignore_geometry=False, progress=progress)


def get_counties_info(
    year: int,
    progress: ProgressCallback | None = None,
) -> DataFrame:
    """
    Get all US counties and county-equivalents for the given census year,
    without geography.

    Parameters
    ----------
    year :
        The geography year.
    progress :
        A optional callback for reporting the progress of downloading TIGER files.

    Returns
    -------
    :
        The TIGER file info without geography.
    """
    config = _get_counties_config(year)
    return _load_tiger_data(config, ignore_geometry=True, progress=progress)


@dataclass(frozen=True)
class CountiesSummary(GranularitySummary):
    """
    Information about US counties (and county equivalents.) Typically you will use
    `get_counties` to obtain an instance of this class for a particular year.

    Parameters
    ----------
    geoid :
        The GEOIDs (aka FIPS codes) of all counties.
    name :
        The typical names of the counties (does not include state).
    """

    geoid: list[str]
    """The GEOIDs (aka FIPS codes) of all counties."""
    name: list[str]
    """
    The typical names of the counties (does not include state).
    Note: county names are not unique across the whole US.
    """
    name_with_state: list[str]
    """The typical names including county and state, e.g., `Coconino, AZ`"""

    @cached_property
    def county_fips_to_name(self) -> Mapping[str, str]:
        """Mapping from county FIPS code to name with state."""
        return dict(zip(self.geoid, self.name_with_state, strict=True))

    @override
    def interpret(self, identifiers: Sequence[str]) -> list[str]:
        """
        Permissively interprets the given set of identifiers as describing nodes,
        and converts them to a sorted list of GEOIDs.

        Parameters
        ----------
        identifiers :
            A list of identifiers. Identifiers can be given in any of the acceptable
            forms, but all of the identifiers must use the same form. Forms are:
            GEOID/FIPS code, or the name of the county and its state postal code
            separated by a comma, e.g., `Coconino, AZ`.

        Returns
        -------
        :
            The list of GEOIDs in canonical sort order.

        Raises
        ------
        GeographyError
            If invalid identifiers are given.
        """
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
    """
    Load US Counties information for the given year.

    Parameters
    ----------
    year :
        The geography year.

    Returns
    -------
    :
        The summary.
    """
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
    year: int,
    state_ids: Sequence[str] | None = None,
) -> _DataConfig:
    """Produce the args for _get_info or _get_geo (tracts)."""
    if not is_tiger_year(year):
        raise GeographyError(f"Unsupported year: {year}")
    states = get_states_info(year)
    if state_ids is not None:
        states = states[states["GEOID"].isin(state_ids)]

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
    # each tracts file is approx 7MB
    return _DataConfig("tract", urls, columns, 7_000_000)


def get_tracts_geo(
    year: int,
    state_id: Sequence[str] | None = None,
    progress: ProgressCallback | None = None,
) -> GeoDataFrame:
    """
    Get all supported US census tracts for the given census year, with geography.

    Parameters
    ----------
    year :
        The geography year.
    state_id :
        If provided, return only the tracts in the given list of states (by GEOID).
    progress :
        A optional callback for reporting the progress of downloading TIGER files.

    Returns
    -------
    :
        The TIGER file info with geography.
    """
    config = _get_tracts_config(year, state_id)
    return _load_tiger_data(config, ignore_geometry=False, progress=progress)


def get_tracts_info(
    year: int,
    state_id: Sequence[str] | None = None,
    progress: ProgressCallback | None = None,
) -> DataFrame:
    """
    Get all US census tracts for the given census year, without geography.

    Parameters
    ----------
    year :
        The geography year.
    state_id :
        If provided, return only the tracts in the given list of states (by GEOID).
    progress :
        A optional callback for reporting the progress of downloading TIGER files.

    Returns
    -------
    :
        The TIGER file info without geography.
    """
    config = _get_tracts_config(year, state_id)
    return _load_tiger_data(config, ignore_geometry=True, progress=progress)


@dataclass(frozen=True)
class TractsSummary(GranularitySummary):
    """
    Information about US Census tracts. Typically you will use
    `get_tracts` to obtain an instance of this class for a particular year.

    Parameters
    ----------
    geoid :
        The GEOIDs (aka FIPS codes) of all tracts.
    """

    geoid: list[str]
    """The GEOIDs (aka FIPS codes) of all tracts."""


@cache_transparent
def get_tracts(year: int) -> TractsSummary:
    """
    Load US Census Tracts information for the given year.

    Parameters
    ----------
    year :
        The geography year.

    Returns
    -------
    :
        The summary.
    """
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
    year: int,
    state_ids: Sequence[str] | None = None,
) -> _DataConfig:
    """Produce the args for _get_info or _get_geo (block groups)."""
    if not is_tiger_year(year):
        raise GeographyError(f"Unsupported year: {year}")
    states = get_states_info(year)
    if state_ids is not None:
        states = states[states["GEOID"].isin(state_ids)]

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
    # each block groups file is approx 1.25MB
    return _DataConfig("block group", urls, columns, 1_250_000)


def get_block_groups_geo(
    year: int,
    state_id: Sequence[str] | None = None,
    progress: ProgressCallback | None = None,
) -> GeoDataFrame:
    """
    Get all supported US census block groups for the given census year, with geography.

    Parameters
    ----------
    year :
        The geography year.
    state_id :
        If provided, return only the block groups in the given list of states
        (by GEOID).
    progress :
        A optional callback for reporting the progress of downloading TIGER files.

    Returns
    -------
    :
        The TIGER file info with geography.
    """
    config = _get_block_groups_config(year, state_id)
    return _load_tiger_data(config, ignore_geometry=False, progress=progress)


def get_block_groups_info(
    year: int,
    state_id: Sequence[str] | None = None,
    progress: ProgressCallback | None = None,
) -> DataFrame:
    """
    Get all US census block groups for the given census year, without geography.

    Parameters
    ----------
    year :
        The geography year.
    state_id :
        If provided, return only the block groups in the given list of states
        (by GEOID).
    progress :
        A optional callback for reporting the progress of downloading TIGER files.

    Returns
    -------
    :
        The TIGER file info without geography.
    """
    config = _get_block_groups_config(year, state_id)
    return _load_tiger_data(config, ignore_geometry=True, progress=progress)


@dataclass(frozen=True)
class BlockGroupsSummary(GranularitySummary):
    """
    Information about US Census block groups.  Typically you will use
    `get_block_groups` to obtain an instance of this class for a particular year.

    Parameters
    ----------
    geoid :
        The GEOIDs (aka FIPS codes) of all block groups.
    """

    geoid: list[str]
    """The GEOIDs (aka FIPS codes) of all block groups."""


@cache_transparent
def get_block_groups(year: int) -> BlockGroupsSummary:
    """
    Load US Census Block Group information for the given year.

    Parameters
    ----------
    year :
        The geography year.

    Returns
    -------
    :
        The summary.
    """
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
    """
    Retrieve a `GranularitySummary` for the given granularity and year.

    Parameters
    ----------
    granularity :
        The granularity.
    year :
        The geography year.

    Returns
    -------
    :
        The summary.

    Raises
    ------
    GeographyError
        If no summary can be retrieved.
    """
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
            err = f"Unsupported granularity: {granularity}"
            raise GeographyError(err)


class CacheEstimate(NamedTuple):
    """Estimates related to data needed to fulfill TIGER requests."""

    total_cache_size: int
    """An estimate of the size of the files that we need to have cached to fulfill
    a request."""
    missing_cache_size: int
    """An estimate of the size of the files that are not currently cached that we
    would need to fulfill a request. Zero if we have all of the files already."""


def check_cache(
    granularity: CensusGranularityName,
    year: int,
    *,
    state_ids: Sequence[str] | None = None,
) -> CacheEstimate:
    """
    Check the status of the cache for a specified TIGER granularity and year.

    Parameters
    ----------
    granularity :
        The Census granularity.
    year :
        The geography year.
    state_ids :
        If specified, only consider places in this set of states.
        Must be in state GEOID (FIPS code) format.

    Returns
    -------
    :
        The estimate of the total size of the cached files for the given granularity
        and year, as well as how much is not currently cached.
    """
    match granularity:
        case "state":
            config = _get_states_config(year)
        case "county":
            config = _get_counties_config(year)
        case "tract":
            config = _get_tracts_config(year, state_ids)
        case "block group":
            config = _get_block_groups_config(year, state_ids)
        case _:
            err = f"Unsupported granularity: {granularity}"
            raise GeographyError(err)

    cache_files = [_url_to_cache_path(u) for u in config.urls]
    missing_files = sum(1 for f in cache_files if not check_file_in_cache(f))
    return CacheEstimate(
        total_cache_size=len(cache_files) * config.estimated_file_size,
        missing_cache_size=missing_files * config.estimated_file_size,
    )
