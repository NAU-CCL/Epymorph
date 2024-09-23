"""
Encodes the geographic system made up of US Census delineations.
This system comprises a set of perfectly-nested granularities,
and a structured ID system for labeling all delineations
(sometimes loosely called FIPS codes or GEOIDs).
"""

import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import cache
from io import BytesIO
from typing import (
    Callable,
    Iterable,
    Literal,
    Mapping,
    NamedTuple,
    ParamSpec,
    Sequence,
    TypeVar,
)

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

import epymorph.geography.us_tiger as us_tiger
from epymorph.cache import (
    CacheMiss,
    load_bundle_from_cache,
    module_cache_path,
    save_bundle_to_cache,
)
from epymorph.error import GeographyError
from epymorph.geography.scope import GeoScope
from epymorph.util import filter_unique, prefix

CensusGranularityName = Literal["state", "county", "tract", "block group", "block"]
"""The name of a supported Census granularity."""

CENSUS_HIERARCHY = ("state", "county", "tract", "block group", "block")
"""The granularities in hierarchy order (largest to smallest)."""


class CensusGranularity(ABC):
    """
    Each CensusGranularity instance defines a set of utility functions for working with
    GEOIDs of that granularity, as well as inspecting and manipulating the granularity
    hierarchy itself.
    """

    _name: CensusGranularityName
    _length: int
    _match_pattern: re.Pattern[str]
    """The pattern used for matching GEOIDs of this granularity."""
    _extract_pattern: re.Pattern[str]
    """The pattern used for extracting GEOIDs of this granularity or smaller."""
    _decompose_pattern: re.Pattern[str]
    """The pattern used for decomposing GEOIDs of this granularity."""

    def __init__(
        self,
        name: CensusGranularityName,
        length: int,
        match_pattern: str,
        extract_pattern: str,
        decompose_pattern: str,
    ):
        self._name = name
        self._length = length
        self._match_pattern = re.compile(match_pattern)
        self._extract_pattern = re.compile(extract_pattern)
        self._decompose_pattern = re.compile(decompose_pattern)

    @property
    def name(self) -> CensusGranularityName:
        """The name of the granularity this class models."""
        return self._name

    @property
    def length(self) -> int:
        """The number of digits in a GEOID of this granularity."""
        return self._length

    def is_nested(self, outer: CensusGranularityName) -> bool:
        """
        Test whether this granularity is nested inside (or equal to)
        the given granularity.
        """
        return CENSUS_HIERARCHY.index(outer) <= CENSUS_HIERARCHY.index(self.name)

    def matches(self, geoid: str) -> bool:
        """Test whether the given GEOID matches this granularity."""
        return self._match_pattern.match(geoid) is not None

    def extract(self, geoid: str) -> str:
        """
        Extracts this level of granularity's GEOID segment, if the given GEOID is of
        this granularity or smaller. Raises a GeographyError if the GEOID is unsuitable
        or poorly formatted.
        """
        if (m := self._extract_pattern.match(geoid)) is not None:
            return m[1]
        else:
            msg = f"Unable to extract {self._name} info from ID {id}; check its format."
            raise GeographyError(msg)

    def truncate(self, geoid: str) -> str:
        """
        Truncates the given GEOID to this level of granularity.
        If the given GEOID is for a granularity larger than this level,
        the GEOID will be returned unchanged.
        """
        return geoid[: self.length]

    def truncate_list(self, geoids: Iterable[str]) -> list[str]:
        """
        Truncates a list of GEOIDs to this level of granularity, returning only
        unique entries without changing the ordering of entries.
        """
        n = self.length
        results = OrderedDict()
        for x in geoids:
            results[x[:n]] = True
        return list(results)

    def _decompose(self, geoid: str) -> re.Match[str]:
        """
        Internal method to decompose a GEOID as a regex match.
        Raises GeographyError if the match fails.
        """
        match = self._decompose_pattern.match(geoid)
        if match is None:
            msg = (
                f"Unable to decompose {self.name} info from ID {id}; check its format."
            )
            raise GeographyError(msg)
        return match

    @abstractmethod
    def decompose(self, geoid: str) -> tuple[str, ...]:
        """
        Decompose a GEOID into a tuple containing all of its granularity component IDs.
        The GEOID must match this granularity exactly,
        or else GeographyError will be raised.
        """

    def grouped(self, sorted_geoids: NDArray[np.str_]) -> dict[str, NDArray[np.str_]]:
        """
        Group a list of GEOIDs by this level of granularity.
        WARNING: Requires that the GEOID array has been sorted!
        """
        group_prefix = prefix(self.length)(sorted_geoids)
        uniques, splits = np.unique(group_prefix, return_index=True)
        grouped = np.split(sorted_geoids, splits[1:])
        return dict(zip(uniques, grouped))


class State(CensusGranularity):
    """State-level utility functions."""

    def __init__(self):
        super().__init__(
            name="state",
            length=2,
            match_pattern=r"^\d{2}$",
            extract_pattern=r"^(\d{2})\d*$",
            decompose_pattern=r"^(\d{2})$",
        )

    def decompose(self, geoid: str) -> tuple[str]:
        m = self._decompose(geoid)
        return (m[1],)


class County(CensusGranularity):
    """County-level utility functions."""

    def __init__(self):
        super().__init__(
            name="county",
            length=5,
            match_pattern=r"^\d{5}$",
            extract_pattern=r"^\d{2}(\d{3})\d*$",
            decompose_pattern=r"^(\d{2})(\d{3})$",
        )

    def decompose(self, geoid: str) -> tuple[str, str]:
        m = self._decompose(geoid)
        return (m[1], m[2])


class Tract(CensusGranularity):
    """Census-tract-level utility functions."""

    def __init__(self):
        super().__init__(
            name="tract",
            length=11,
            match_pattern=r"^\d{11}$",
            extract_pattern=r"^\d{5}(\d{6})\d*$",
            decompose_pattern=r"^(\d{2})(\d{3})(\d{6})$",
        )

    def decompose(self, geoid: str) -> tuple[str, str, str]:
        m = self._decompose(geoid)
        return (m[1], m[2], m[3])


class BlockGroup(CensusGranularity):
    """Block-group-level utility functions."""

    def __init__(self):
        super().__init__(
            name="block group",
            length=12,
            match_pattern=r"^\d{12}$",
            extract_pattern=r"^\d{11}(\d)\d*$",
            decompose_pattern=r"^(\d{2})(\d{3})(\d{6})(\d)$",
        )

    def decompose(self, geoid: str) -> tuple[str, str, str, str]:
        m = self._decompose(geoid)
        return (m[1], m[2], m[3], m[4])


class Block(CensusGranularity):
    """Block-level utility functions."""

    def __init__(self):
        super().__init__(
            name="block",
            length=15,
            match_pattern=r"^\d{15}$",
            extract_pattern=r"^\d{11}(\d{4})$",
            decompose_pattern=r"^(\d{2})(\d{3})(\d{6})(\d{4})$",
        )

    def decompose(self, geoid: str) -> tuple[str, str, str, str, str]:
        # The block group ID is the first digit of the block ID,
        # but the block ID also includes this digit.
        m = self._decompose(geoid)
        return (m[1], m[2], m[3], m[4][0], m[4])


# Singletons for the CensusGranularity classes.
STATE = State()
COUNTY = County()
TRACT = Tract()
BLOCK_GROUP = BlockGroup()
BLOCK = Block()

CENSUS_GRANULARITY = (STATE, COUNTY, TRACT, BLOCK_GROUP, BLOCK)
"""CensusGranularity singletons in hierarchy order."""


def get_census_granularity(name: CensusGranularityName) -> CensusGranularity:
    """Get a CensusGranularity instance by name."""
    match name:
        case "state":
            return STATE
        case "county":
            return COUNTY
        case "tract":
            return TRACT
        case "block group":
            return BLOCK_GROUP
        case "block":
            return BLOCK


# Census data loading and caching


DEFAULT_YEAR = 2020

_USCENSUS_CACHE_PATH = module_cache_path(__name__)

_CACHE_VERSION = 2

ModelT = TypeVar("ModelT", bound=NamedTuple)


def _load_cached(
    relpath: str, on_miss: Callable[[], ModelT], on_hit: Callable[..., ModelT]
) -> ModelT:
    # NOTE: this would be more natural as a decorator,
    # but Pylance seems to have problems tracking the return type properly
    # with that implementation
    path = _USCENSUS_CACHE_PATH.joinpath(relpath)
    try:
        content = load_bundle_from_cache(path, _CACHE_VERSION)
        with np.load(content["data.npz"]) as data_npz:
            return on_hit(**data_npz)
    except CacheMiss:
        data = on_miss()
        data_bytes = BytesIO()
        np.savez_compressed(data_bytes, **data._asdict())
        save_bundle_to_cache(path, _CACHE_VERSION, {"data.npz": data_bytes})
        return data


class StatesInfo(NamedTuple):
    """Information about US states (and state equivalents)."""

    geoid: NDArray[np.str_]
    """The GEOID (aka FIPS code) of the state."""
    name: NDArray[np.str_]
    """The typical name for the state."""
    code: NDArray[np.str_]
    """The US postal code for the state."""

    def as_dataframe(self) -> DataFrame:
        return DataFrame(
            {
                "geoid": self.geoid,
                "name": self.name,
                "code": self.code,
            }
        )


def get_us_states(year: int) -> StatesInfo:
    """Loads US States information (assumed to be invariant for all supported years)."""
    if not us_tiger.is_tiger_year(year):
        raise GeographyError(f"Unsupported year: {year}")

    def _get_us_states() -> StatesInfo:
        states_df = us_tiger.get_states_info(year).sort_values("GEOID")
        return StatesInfo(
            geoid=states_df["GEOID"].to_numpy(np.str_),
            name=states_df["NAME"].to_numpy(np.str_),
            code=states_df["STUSPS"].to_numpy(np.str_),
        )

    return _load_cached("us_states_all.tgz", _get_us_states, StatesInfo)


class CountiesInfo(NamedTuple):
    """Information about US counties (and county equivalents.)"""

    geoid: NDArray[np.str_]
    """The GEOID (aka FIPS code) of the county."""
    name: NDArray[np.str_]
    """The typical name of the county (does not include state)."""

    def as_dataframe(self) -> DataFrame:
        return DataFrame(
            {
                "geoid": self.geoid,
                "name": self.name,
            }
        )


def get_us_counties(year: int) -> CountiesInfo:
    """Loads US Counties information for the given year."""
    if not us_tiger.is_tiger_year(year):
        raise GeographyError(f"Unsupported year: {year}")

    def _get_us_counties() -> CountiesInfo:
        counties_df = us_tiger.get_counties_info(year).sort_values("GEOID")
        return CountiesInfo(
            geoid=counties_df["GEOID"].to_numpy(np.str_),
            name=counties_df["NAME"].to_numpy(np.str_),
        )

    return _load_cached(f"us_counties_{year}.tgz", _get_us_counties, CountiesInfo)


class TractsInfo(NamedTuple):
    """Information about US census tracts."""

    geoid: NDArray[np.str_]
    """The GEOID (aka FIPS code) of the tract."""

    def as_dataframe(self) -> DataFrame:
        return DataFrame(
            {
                "geoid": self.geoid,
            }
        )


def get_us_tracts(year: int) -> TractsInfo:
    """Loads US Census Tracts information for the given year."""
    if not us_tiger.is_tiger_year(year):
        raise GeographyError(f"Unsupported year: {year}")

    def _get_us_tracts() -> TractsInfo:
        tracts_df = us_tiger.get_tracts_info(year).sort_values("GEOID")
        return TractsInfo(
            geoid=tracts_df["GEOID"].to_numpy(np.str_),
        )

    return _load_cached(f"us_tracts_{year}.tgz", _get_us_tracts, TractsInfo)


class BlockGroupsInfo(NamedTuple):
    """Information about US census block groups."""

    geoid: NDArray[np.str_]
    """The GEOID (aka FIPS code) of the block group."""

    def as_dataframe(self) -> DataFrame:
        return DataFrame(
            {
                "geoid": self.geoid,
            }
        )


def get_us_block_groups(year: int) -> BlockGroupsInfo:
    """Loads US Census Block Group information for the given year."""
    if not us_tiger.is_tiger_year(year):
        raise GeographyError(f"Unsupported year: {year}")

    def _get_us_cbgs() -> BlockGroupsInfo:
        cbgs_df = us_tiger.get_block_groups_info(year).sort_values("GEOID")
        return BlockGroupsInfo(
            geoid=cbgs_df["GEOID"].to_numpy(np.str_),
        )

    return _load_cached(f"us_block_groups_{year}.tgz", _get_us_cbgs, BlockGroupsInfo)


# Census utility functions


T = TypeVar("T")
P = ParamSpec("P")


def validate_fips(
    granularity: CensusGranularityName, year: int, fips: Sequence[str]
) -> Sequence[str]:
    """
    Validates a list of FIPS codes are valid for the given granularity and year and
    returns them as a sorted list of FIPS codes.
    If any FIPS code is found to be invalid, raises GeographyError.
    """
    match granularity:
        case "state":
            valid_nodes = get_us_states(year).geoid
        case "county":
            valid_nodes = get_us_counties(year).geoid
        case "tract":
            valid_nodes = get_us_tracts(year).geoid
        case "block group":
            valid_nodes = get_us_block_groups(year).geoid
        case _:
            raise GeographyError(f"Unsupported granularity: {granularity}")

    if not all((curr := x) in valid_nodes for x in fips):
        msg = (
            f"Not all given {granularity} fips codes are valid for {year} "
            f"(for example: {curr})."
        )
        raise GeographyError(msg)
    return tuple(sorted(fips))


# We use the set of 56 two-letter abbreviations and FIPS codes returned by TIGRIS.
# This set should be constant since 1970 when the FIPS standard was introduced.
# It doesn't cover *every* such code, but it's sufficient for Census usage, which
# doesn't provide data for state-or-state-equivalents outside of this set.
# https://www.census.gov/library/reference/code-lists/ansi.html#states


@cache
def state_code_to_fips(year: int) -> Mapping[str, str]:
    """Mapping from state postal code to FIPS code."""
    states = get_us_states(year)
    return dict(zip(states.code.tolist(), states.geoid.tolist()))


@cache
def state_fips_to_code(year: int) -> Mapping[str, str]:
    """Mapping from state FIPS code to postal code."""
    states = get_us_states(year)
    return dict(zip(states.geoid.tolist(), states.code.tolist()))


def validate_state_codes_as_fips(year: int, codes: Sequence[str]) -> Sequence[str]:
    """
    Validates a list of US state postal codes (two-letter abbreviations) and
    returns them as a sorted list of FIPS codes.
    If any postal code is found to be invalid, raises GeographyError.
    """
    curr = ""  # capturing the last tried code lets us report which one is invalid
    try:
        mapping = state_code_to_fips(year)
        fips = [mapping[(curr := x)] for x in codes]
    except KeyError:
        msg = f"Unknown state postal code abbreviation: {curr}"
        raise GeographyError(msg) from None
    return tuple(sorted(fips))


# Census GeoScopes


@dataclass(frozen=True)
class CensusScope(ABC, GeoScope):
    """A GeoScope using US Census delineations."""

    year: int
    """
    The Census delineation year.
    With every decennial census, the Census Department can (and does) define
    new delineations, especially at the smaller granularities. Hence, you must
    know the delineation year in order to absolutely identify any
    particular granularity.
    """

    granularity: CensusGranularityName
    """Which granularity are the nodes in this scope?"""

    @abstractmethod
    def get_node_ids(self) -> NDArray[np.str_]: ...

    @abstractmethod
    def raise_granularity(self) -> "CensusScope":
        """
        Return a scope with granularity one level higher than this.
        This may have the effect of widening the scope; for example,
        raising a TractScope which is filtered to a set of tracts will
        result in a CountyScope containing the counties that contained our
        tracts. Raises GeographyError if the granularity cannot be raised.
        """

    @abstractmethod
    def lower_granularity(self) -> "CensusScope":
        """
        Return a scope with granularity one level lower than this.
        Raises GeographyError if the granularity cannot be lowered.
        """


@dataclass(frozen=True)
class StateScopeAll(CensusScope):
    """GeoScope including all US states and state-equivalents."""

    # NOTE: for the Census API, we need to handle "all states" as a special case.
    year: int
    granularity: Literal["state"] = field(init=False, default="state")

    def get_node_ids(self) -> NDArray[np.str_]:
        return get_us_states(self.year).geoid

    def raise_granularity(self) -> CensusScope:
        raise GeographyError("No granularity higher than state.")

    def lower_granularity(self) -> "CountyScope":
        """Create and return CountyScope object with identical properties."""
        return CountyScope(self.year, "state", self.get_node_ids().tolist())


@dataclass(frozen=True)
class StateScope(CensusScope):
    """GeoScope at the State granularity."""

    includes_granularity: Literal["state"]
    includes: Sequence[str]
    year: int
    granularity: Literal["state"] = field(init=False, default="state")

    @staticmethod
    def all(year: int = DEFAULT_YEAR) -> StateScopeAll:
        """Create a scope including all US states and state-equivalents."""
        return StateScopeAll(year)

    @staticmethod
    def in_states(states_fips: Sequence[str], year: int = DEFAULT_YEAR) -> "StateScope":
        """
        Create a scope including a set of US states/state-equivalents, by FIPS code.
        Raise GeographyError if any FIPS code is invalid.
        """
        states_fips = validate_fips("state", year, states_fips)
        return StateScope(includes_granularity="state", includes=states_fips, year=year)

    @staticmethod
    def in_states_by_code(
        states_code: Sequence[str], year: int = DEFAULT_YEAR
    ) -> "StateScope":
        """
        Create a scope including a set of US states/state-equivalents,
        by postal code (two-letter abbreviation).
        Raise GeographyError if any postal code is invalid.
        """
        states_fips = validate_state_codes_as_fips(year, states_code)
        return StateScope(includes_granularity="state", includes=states_fips, year=year)

    def get_node_ids(self) -> NDArray[np.str_]:
        # As long as we enforce that fips codes will be checked and sorted on init,
        # we can use the value of 'includes'.
        return np.array(self.includes, dtype=np.str_)

    def raise_granularity(self) -> CensusScope:
        raise GeographyError("No granularity higher than state.")

    def lower_granularity(self) -> "CountyScope":
        """Create and return CountyScope object with identical properties."""
        return CountyScope(self.year, self.includes_granularity, self.includes)


@dataclass(frozen=True)
class CountyScope(CensusScope):
    """GeoScope at the County granularity."""

    includes_granularity: Literal["state", "county"]
    includes: Sequence[str]
    year: int
    granularity: Literal["county"] = field(init=False, default="county")

    @staticmethod
    def in_states(
        states_fips: Sequence[str], year: int = DEFAULT_YEAR
    ) -> "CountyScope":
        """
        Create a scope including all counties in a set of US states/state-equivalents.
        Raise GeographyError if any FIPS code is invalid.
        """
        states_fips = validate_fips("state", year, states_fips)
        return CountyScope(
            includes_granularity="state", includes=states_fips, year=year
        )

    @staticmethod
    def in_states_by_code(
        states_code: Sequence[str], year: int = DEFAULT_YEAR
    ) -> "CountyScope":
        """
        Create a scope including all counties in a set of US states/state-equivalents,
        by postal code (two-letter abbreviation).
        Raise GeographyError if any FIPS code is invalid.
        """
        states_fips = validate_state_codes_as_fips(year, states_code)
        return CountyScope(
            includes_granularity="state", includes=states_fips, year=year
        )

    @staticmethod
    def in_counties(
        counties_fips: Sequence[str], year: int = DEFAULT_YEAR
    ) -> "CountyScope":
        """
        Create a scope including a set of US counties, by FIPS code.
        Raise GeographyError if any FIPS code is invalid.
        """
        counties_fips = validate_fips("county", year, counties_fips)
        return CountyScope(
            includes_granularity="county", includes=counties_fips, year=year
        )

    def get_node_ids(self) -> NDArray[np.str_]:
        match self.includes_granularity:
            case "state":
                # return all counties in a list of states
                def is_in_states(geoid: str) -> bool:
                    return STATE.truncate(geoid) in self.includes

                counties = get_us_counties(self.year).geoid
                return counties[[is_in_states(x) for x in counties]]

            case "county":
                # return all counties in a list of counties
                return np.array(self.includes, dtype=np.str_)

    def raise_granularity(self) -> StateScope:
        """Create and return StateScope object with identical properties."""
        raised_granularity = self.includes_granularity
        raised_includes = self.includes
        if raised_granularity == "county":
            raised_granularity = "state"
            raised_includes = filter_unique(fips[:-3] for fips in raised_includes)
        return StateScope(self.year, raised_granularity, raised_includes)

    def lower_granularity(self) -> "TractScope":
        """Create and return TractScope object with identical properties."""
        return TractScope(self.year, self.includes_granularity, self.includes)


@dataclass(frozen=True)
class TractScope(CensusScope):
    """GeoScope at the Tract granularity."""

    includes_granularity: Literal["state", "county", "tract"]
    includes: Sequence[str]
    year: int
    granularity: Literal["tract"] = field(init=False, default="tract")

    @staticmethod
    def in_states(states_fips: Sequence[str], year: int = DEFAULT_YEAR) -> "TractScope":
        """
        Create a scope including all tracts in a set of US states/state-equivalents.
        Raise GeographyError if any FIPS code is invalid.
        """
        states_fips = validate_fips("state", year, states_fips)
        return TractScope(includes_granularity="state", includes=states_fips, year=year)

    @staticmethod
    def in_states_by_code(
        states_code: Sequence[str], year: int = DEFAULT_YEAR
    ) -> "TractScope":
        """
        Create a scope including all tracts in a set of US states/state-equivalents,
        by postal code (two-letter abbreviation).
        Raise GeographyError if any FIPS code is invalid.
        """
        states_fips = validate_state_codes_as_fips(year, states_code)
        return TractScope(includes_granularity="state", includes=states_fips, year=year)

    @staticmethod
    def in_counties(
        counties_fips: Sequence[str], year: int = DEFAULT_YEAR
    ) -> "TractScope":
        """
        Create a scope including all tracts in a set of US counties, by FIPS code.
        Raise GeographyError if any FIPS code is invalid.
        """
        counties_fips = validate_fips("county", year, counties_fips)
        return TractScope(
            includes_granularity="county", includes=counties_fips, year=year
        )

    @staticmethod
    def in_tracts(tract_fips: Sequence[str], year: int = DEFAULT_YEAR) -> "TractScope":
        """
        Create a scope including a set of US tracts, by FIPS code.
        Raise GeographyError if any FIPS code is invalid.
        """
        tract_fips = validate_fips("tract", year, tract_fips)
        return TractScope(includes_granularity="tract", includes=tract_fips, year=year)

    def get_node_ids(self) -> NDArray[np.str_]:
        match self.includes_granularity:
            case "state":
                # return all tracts in a list of states
                def is_in_states(geoid: str) -> bool:
                    return STATE.truncate(geoid) in self.includes

                tracts = get_us_tracts(self.year).geoid
                return tracts[[is_in_states(x) for x in tracts]]

            case "county":
                # return all tracts in a list of counties
                def is_in_counties(geoid: str) -> bool:
                    return COUNTY.truncate(geoid) in self.includes

                tracts = get_us_tracts(self.year).geoid
                return tracts[[is_in_counties(x) for x in tracts]]

            case "tract":
                # return all tracts in a list of tracts
                return np.array(self.includes, dtype=np.str_)

    def raise_granularity(self) -> CountyScope:
        """Create and return CountyScope object with identical properties."""
        raised_granularity = self.includes_granularity
        raised_includes = self.includes
        if raised_granularity == "tract":
            raised_granularity = "county"
            raised_includes = filter_unique(fips[:-6] for fips in raised_includes)
        return CountyScope(self.year, raised_granularity, raised_includes)

    def lower_granularity(self) -> "BlockGroupScope":
        """Create and return BlockGroupScope object with identical properties."""
        return BlockGroupScope(self.year, self.includes_granularity, self.includes)


@dataclass(frozen=True)
class BlockGroupScope(CensusScope):
    """GeoScope at the Block Group granularity."""

    includes_granularity: Literal["state", "county", "tract", "block group"]
    includes: Sequence[str]
    year: int
    granularity: Literal["block group"] = field(init=False, default="block group")

    @staticmethod
    def in_states(
        states_fips: Sequence[str], year: int = DEFAULT_YEAR
    ) -> "BlockGroupScope":
        """
        Create a scope including all block groups in a set of
        US states/state-equivalents.
        Raise GeographyError if any FIPS code is invalid.
        """
        states_fips = validate_fips("state", year, states_fips)
        return BlockGroupScope(
            includes_granularity="state", includes=states_fips, year=year
        )

    @staticmethod
    def in_states_by_code(
        states_code: Sequence[str], year: int = DEFAULT_YEAR
    ) -> "BlockGroupScope":
        """
        Create a scope including all block groups in a set of
        US states/state-equivalents, by postal code (two-letter abbreviation).
        Raise GeographyError if any FIPS code is invalid.
        """
        states_fips = validate_state_codes_as_fips(year, states_code)
        return BlockGroupScope(
            includes_granularity="state", includes=states_fips, year=year
        )

    @staticmethod
    def in_counties(
        counties_fips: Sequence[str], year: int = DEFAULT_YEAR
    ) -> "BlockGroupScope":
        """
        Create a scope including all block groups in a set of US counties, by FIPS code.
        Raise GeographyError if any FIPS code is invalid.
        """
        counties_fips = validate_fips("county", year, counties_fips)
        return BlockGroupScope(
            includes_granularity="county", includes=counties_fips, year=year
        )

    @staticmethod
    def in_tracts(
        tract_fips: Sequence[str], year: int = DEFAULT_YEAR
    ) -> "BlockGroupScope":
        """
        Create a scope including all block gropus in a set of US tracts, by FIPS code.
        Raise GeographyError if any FIPS code is invalid.
        """
        tract_fips = validate_fips("tract", year, tract_fips)
        return BlockGroupScope(
            includes_granularity="tract", includes=tract_fips, year=year
        )

    @staticmethod
    def in_block_groups(
        block_group_fips: Sequence[str], year: int = DEFAULT_YEAR
    ) -> "BlockGroupScope":
        """
        Create a scope including a set of US block groups, by FIPS code.
        Raise GeographyError if any FIPS code is invalid.
        """
        block_group_fips = validate_fips("block group", year, block_group_fips)
        return BlockGroupScope(
            includes_granularity="block group", includes=block_group_fips, year=year
        )

    def get_node_ids(self) -> NDArray[np.str_]:
        match self.includes_granularity:
            case "state":
                # return all block groups in a list of states
                def is_in_states(geoid: str) -> bool:
                    return STATE.truncate(geoid) in self.includes

                block_groups = get_us_block_groups(self.year).geoid
                return block_groups[[is_in_states(x) for x in block_groups]]

            case "county":
                # return all block groups in a list of counties
                def is_in_counties(geoid: str) -> bool:
                    return COUNTY.truncate(geoid) in self.includes

                block_groups = get_us_block_groups(self.year).geoid
                return block_groups[[is_in_counties(x) for x in block_groups]]

            case "tract":
                # return all block groups in a list of tracts
                def is_in_tracts(geoid: str) -> bool:
                    return TRACT.truncate(geoid) in self.includes

                block_groups = get_us_block_groups(self.year).geoid
                return block_groups[[is_in_tracts(x) for x in block_groups]]

            case "block group":
                # return all block groups in a list of block groups
                return np.array(self.includes, dtype=np.str_)

    def raise_granularity(self) -> TractScope:
        """Create and return TractScope object with identical properties."""
        raised_granularity = self.includes_granularity
        raised_includes = self.includes
        if raised_granularity == "block group":
            raised_granularity = "tract"
            raised_includes = filter_unique(fips[:-1] for fips in raised_includes)
        return TractScope(self.year, raised_granularity, raised_includes)

    def lower_granularity(self) -> CensusScope:
        raise GeographyError("No valid granularity lower than block group.")
