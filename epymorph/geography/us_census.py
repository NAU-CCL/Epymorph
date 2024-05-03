"""
Encodes the geographic system made up of US Census delineations.
This system comprises a set of perfectly-nested granularities,
and a structured ID system for labeling all delineations
(sometimes loosely called FIPS codes or GEOIDs).
"""
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cache, wraps
from io import BytesIO
from pathlib import Path
from typing import (Callable, Concatenate, Literal, Mapping, NamedTuple,
                    ParamSpec, Sequence, TypeVar)

import numpy as np
from numpy.typing import NDArray

import epymorph.geography.us_tiger as us_tiger
from epymorph.cache import (CacheMiss, load_bundle_from_cache,
                            save_bundle_to_cache)
from epymorph.error import GeographyError
from epymorph.geography.scope import GeoScope
from epymorph.util import prefix

CensusGranularityName = Literal['state', 'county', 'tract', 'block group', 'block']
"""The name of a supported Census granularity."""

_CENSUS_HIERARCHY = ('state', 'county', 'tract', 'block group', 'block')
"""The granularities in hierarchy order (largest to smallest)."""


class CensusGranularity(ABC):
    """
    Each CensusGranularity instance defines a set of utility functions for working with GEOIDs of that granularity,
    as well as inspecting and manipulating the granularity hierarchy itself.
    """

    _name: CensusGranularityName
    _length: int
    _match_pattern: re.Pattern[str]
    """The pattern used for matching GEOIDs of this granularity."""
    _extract_pattern: re.Pattern[str]
    """The pattern used for extracting GEOIDs of this granularity or smaller."""
    _decompose_pattern: re.Pattern[str]
    """The pattern used for decomposing GEOIDs of this granularity."""

    def __init__(self,
                 name: CensusGranularityName,
                 length: int,
                 match_pattern: str,
                 extract_pattern: str,
                 decompose_pattern: str):
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
        """Test whether this granularity is nested inside (or equal to) the given granularity."""
        return _CENSUS_HIERARCHY.index(outer) <= _CENSUS_HIERARCHY.index(self.name)

    def matches(self, geoid: str) -> bool:
        """Test whether the given GEOID matches this granularity."""
        return self._match_pattern.match(geoid) is not None

    def extract(self, geoid: str) -> str:
        """
        Extracts this level of granularity's GEOID segment, if the given GEOID is of this granularity or smaller.
        Raises a GeographyError if the GEOID is unsuitable or poorly formatted.
        """
        if (m := self._extract_pattern.match(geoid)) is not None:
            return m[1]
        else:
            msg = f"Unable to extract {self._name} info from ID {id}; check its format."
            raise GeographyError(msg)

    def truncate(self, geoid: str) -> str:
        """
        Truncates the given GEOID to this level of granularity.
        If the given GEOID is for a granularity larger than this level, the GEOID will be returned unchanged.
        """
        return geoid[:self.length]

    def _decompose(self, geoid: str) -> re.Match[str]:
        """
        Internal method to decompose a GEOID as a regex match.
        Raises GeographyError if the match fails.
        """
        match = self._decompose_pattern.match(geoid)
        if match is None:
            msg = f"Unable to decompose {self.name} info from ID {id}; check its format."
            raise GeographyError(msg)
        return match

    @abstractmethod
    def decompose(self, geoid: str) -> tuple[str, ...]:
        """
        Decompose a GEOID into a tuple containing all of its granularity component IDs.
        The GEOID must match this granularity exactly, or else GeographyError will be raised.
        """

    def grouped(self, geoids: NDArray[np.str_]) -> dict[str, NDArray[np.str_]]:
        """
        Group a list of GEOIDs by this level of granularity.
        WARNING: Requires that the GEOID array has been sorted!
        """
        group_prefix = prefix(self.length)(geoids)
        uniques, splits = np.unique(group_prefix, return_index=True)
        grouped = np.split(geoids, splits[1:])
        return dict(zip(uniques, grouped))


class State(CensusGranularity):
    """State-level utility functions."""

    def __init__(self):
        super().__init__(
            name='state',
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
            name='county',
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
            name='tract',
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
            name='block group',
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
            name='block',
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

_CENSUS_GRANULARITY = (STATE, COUNTY, TRACT, BLOCK_GROUP, BLOCK)
"""CensusGranularity singletons in hierarchy order."""


def get_census_granularity(name: CensusGranularityName) -> CensusGranularity:
    """Get a CensusGranularity instance by name."""
    match name:
        case 'state':
            return STATE
        case 'county':
            return COUNTY
        case 'tract':
            return TRACT
        case 'block group':
            return BLOCK_GROUP
        case 'block':
            return BLOCK


# Census data loading and caching


T = TypeVar('T')
P = ParamSpec('P')

CensusYear = Literal[2000, 2010, 2020]
"""A supported Census delineation year."""

DEFAULT_YEAR = 2020

GEOGRAPHY_CACHE_PATH = Path("geography")

ModelT = TypeVar('ModelT', bound=NamedTuple)


def _load_cached(relpath: str, on_miss: Callable[[], ModelT], on_hit: Callable[..., ModelT]) -> ModelT:
    # NOTE: this would be more natural as a decorator,
    # but Pylance seems to have problems tracking the return type properly with that implementation
    path = GEOGRAPHY_CACHE_PATH.joinpath(relpath)
    try:
        content = load_bundle_from_cache(path, 1)
        with np.load(content['data.npz']) as data_npz:
            return on_hit(**data_npz)
    except CacheMiss:
        data = on_miss()
        data_bytes = BytesIO()
        np.savez_compressed(data_bytes, **data._asdict())
        save_bundle_to_cache(path, 1, {'data.npz': data_bytes})
        return data


class StatesInfo(NamedTuple):
    """Information about US states (and state equivalents)."""
    geoid: NDArray[np.str_]
    """The GEOID (aka FIPS code) of the state."""
    name: NDArray[np.str_]
    """The typical name for the state."""
    code: NDArray[np.str_]
    """The US postal code for the state."""


def get_us_states() -> StatesInfo:
    """Loads US States information (assumed to be invariant for all supported years)."""
    def _get_us_states() -> StatesInfo:
        df = us_tiger.get_states_info(2020)  # 2020 works for all supported years
        df.sort_values("GEOID", inplace=True)
        return StatesInfo(
            geoid=df["GEOID"].to_numpy(np.str_),
            name=df["NAME"].to_numpy(np.str_),
            code=df["STUSPS"].to_numpy(np.str_),
        )
    return _load_cached("us_states_all.tgz", _get_us_states, StatesInfo)


class CountiesInfo(NamedTuple):
    """Information about US counties (and county equivalents.)"""
    geoid: NDArray[np.str_]
    """The GEOID (aka FIPS code) of the county."""
    name: NDArray[np.str_]
    """The typical name of the county (does not include state)."""


def get_us_counties(year: CensusYear) -> CountiesInfo:
    """Loads US Counties information for the given year."""
    def _get_us_counties() -> CountiesInfo:
        df = us_tiger.get_counties_info(year)
        df.sort_values("GEOID", inplace=True)
        return CountiesInfo(
            geoid=df["GEOID"].to_numpy(np.str_),
            name=df["NAME"].to_numpy(np.str_),
        )
    return _load_cached(f"us_counties_{year}.tgz", _get_us_counties, CountiesInfo)


class TractsInfo(NamedTuple):
    """Information about US census tracts."""
    geoid: NDArray[np.str_]
    """The GEOID (aka FIPS code) of the tract."""


def get_us_tracts(year: CensusYear) -> TractsInfo:
    """Loads US Census Tracts information for the given year."""
    def _get_us_tracts() -> TractsInfo:
        df = us_tiger.get_tracts_info(year)
        df.sort_values("GEOID", inplace=True)
        return TractsInfo(
            geoid=df["GEOID"].to_numpy(np.str_),
        )
    return _load_cached(f"us_tracts_{year}.tgz", _get_us_tracts, TractsInfo)


class BlockGroupsInfo(NamedTuple):
    """Information about US census block groups."""
    geoid: NDArray[np.str_]
    """The GEOID (aka FIPS code) of the block group."""


def get_us_block_groups(year: CensusYear) -> BlockGroupsInfo:
    """Loads US Census Block Group information for the given year."""
    def _get_us_cbgs() -> BlockGroupsInfo:
        df = us_tiger.get_block_groups_info(year)
        df.sort_values("GEOID", inplace=True)
        return BlockGroupsInfo(
            geoid=df["GEOID"].to_numpy(np.str_),
        )
    return _load_cached(f"us_block_groups_{year}.tgz", _get_us_cbgs, BlockGroupsInfo)


# Census utility functions


def verify_fips(granularity: CensusGranularityName):
    """
    Decorates a function in order to validate the list of FIPS codes provided as the first argument.
    If any FIPS codes are found to be invalid, raises GeographyError.
    """
    g = get_census_granularity(granularity)

    def decorator(
        f: Callable[Concatenate[Sequence[str], CensusYear, P], T]
    ) -> Callable[Concatenate[Sequence[str], CensusYear, P], T]:
        @wraps(f)
        def decorated(fips: Sequence[str], year: CensusYear, *args: P.args, **kwargs: P.kwargs) -> T:
            # TODO: if we have a cache of all IDs -- maybe we should check if the codes really exist,
            # not just that they're the correct format.
            # (Or maybe we can at least do this for states/counties, and fall back to format-checking others.)
            match granularity:
                case 'state':
                    states = get_us_states().geoid
                    if not all((last := x) in states for x in fips):
                        msg = f"Not all given state fips codes are valid (see: {last})."
                        raise GeographyError(msg)
                case 'county':
                    counties = get_us_counties(year).geoid
                    if not all((last := x) in counties for x in fips):
                        msg = f"Not all given county fips codes are valid for the given year (see: {last})."
                        raise GeographyError(msg)
                case _:
                    if not all((g.matches(x) for x in fips)):
                        msg = f"Not all fips codes match the expected granularity ({granularity})."
                        raise GeographyError(msg)

            return f(sorted(fips), year, *args, **kwargs)
        return decorated
    return decorator

# We use the set of 56 two-letter abbreviations and FIPS codes returned by TIGRIS.
# This set should be constant since 1970 when the FIPS standard was introduced.
# It doesn't cover *every* such code, but it's sufficient for Census usage, which doesn't provide
# data for state-or-state-equivalents outside of this set.
# https://www.census.gov/library/reference/code-lists/ansi.html#states


@cache
def state_code_to_fips() -> Mapping[str, str]:
    """Mapping from state postal code to FIPS code."""
    states = get_us_states()
    return dict(zip(states.code, states.geoid))


@cache
def state_fips_to_code() -> Mapping[str, str]:
    """Mapping from state FIPS code to postal code."""
    states = get_us_states()
    return dict(zip(states.geoid, states.code))


def convert_state_code_to_fips(
    f: Callable[Concatenate[Sequence[str], P], T]
) -> Callable[Concatenate[Sequence[str], P], T]:
    """
    Decorates a function in order to validate the list of state postal codes provided as the first argument.
    If any postal codes are found to be invalid, raises GeographyError.
    """
    @wraps(f)
    def decorated(codes: Sequence[str], *args: P.args, **kwargs: P.kwargs) -> T:
        code = ""  # capturing the last tried code lets us report which one is invalid
        try:
            mapping = state_code_to_fips()
            fips = [mapping[(code := x)] for x in codes]
        except KeyError:
            msg = f"Unknown state postal code abbreviation: {code}"
            raise GeographyError(msg) from None
        return f(sorted(fips), *args, **kwargs)
    return decorated


# Census GeoScopes


class CensusScope(GeoScope):
    """A GeoScope using US Census delineations."""

    year: CensusYear
    """
    The Census delineation year.
    With every decennial census, the Census Department can (and does) define
    new delineations, especially at the smaller granularities. Hence, you must
    know the delineation year in order to absolutely identify any particular granularity.
    """

    granularity: CensusGranularityName
    """Which granularity are the nodes in this scope?"""

    @abstractmethod
    def get_node_ids(self) -> NDArray[np.str_]: ...


@dataclass(frozen=True)
class StateScopeAll(CensusScope):
    """GeoScope including all US states and state-equivalents."""
    # NOTE: for the Census API, we need to handle "all states" as a special case.
    year: CensusYear
    granularity: Literal['state'] = field(init=False, default='state')

    def get_node_ids(self) -> NDArray[np.str_]:
        return get_us_states().geoid


@dataclass(frozen=True)
class StateScope(CensusScope):
    """GeoScope at the State granularity."""
    includes_granularity: Literal['state']
    includes: Sequence[str]
    year: CensusYear
    granularity: Literal['state'] = field(init=False, default='state')

    @staticmethod
    def all(year: CensusYear = DEFAULT_YEAR) -> StateScopeAll:
        """Create a scope including all US states and state-equivalents."""
        return StateScopeAll(year)

    @staticmethod
    @verify_fips('state')
    def in_states(states_fips: Sequence[str], year: CensusYear = DEFAULT_YEAR) -> 'StateScope':
        """
        Create a scope including a set of US states/state-equivalents, by FIPS code.
        Raise GeographyError if any FIPS code is invalid.
        """
        # NOTE: the verify_fips decorator automatically validates FIPS codes.
        return StateScope('state', states_fips, year)

    @staticmethod
    @convert_state_code_to_fips
    def in_states_by_code(states_code: Sequence[str], year: CensusYear = DEFAULT_YEAR) -> 'StateScope':
        """
        Create a scope including a set of US states/state-equivalents, by postal code (two-letter abbreviation).
        Raise GeographyError if any postal code is invalid.
        """
        # NOTE: the converter decorator automatically converts codes to FIPS
        # (so the states_code argument is actually FIPS codes by the time we're inside this function).
        # It also raises GeographyError on invalid codes, so at this point we can trust they're all valid.
        return StateScope('state', states_code, year)

    def get_node_ids(self) -> NDArray[np.str_]:
        # As long as we enforce that fips codes will be checked and sorted on construction,
        # we can use the value of 'includes'.
        return np.array(self.includes, dtype=np.str_)


@dataclass(frozen=True)
class CountyScope(CensusScope):
    """GeoScope at the County granularity."""
    includes_granularity: Literal['state', 'county']
    includes: Sequence[str]
    year: CensusYear
    granularity: Literal['county'] = field(init=False, default='county')

    @staticmethod
    @verify_fips('state')
    def in_states(states_fips: Sequence[str], year: CensusYear = DEFAULT_YEAR) -> 'CountyScope':
        """
        Create a scope including all counties in a set of US states/state-equivalents.
        Raise GeographyError if any FIPS code is invalid.
        """
        return CountyScope('state', states_fips, year)

    @staticmethod
    @convert_state_code_to_fips
    def in_states_by_code(states_code: Sequence[str], year: CensusYear = DEFAULT_YEAR) -> 'CountyScope':
        """
        Create a scope including all counties in a set of US states/state-equivalents,
        by postal code (two-letter abbreviation).
        Raise GeographyError if any FIPS code is invalid.
        """
        return CountyScope('state', states_code, year)

    @staticmethod
    @verify_fips('county')
    def in_counties(counties_fips: Sequence[str], year: CensusYear = DEFAULT_YEAR) -> 'CountyScope':
        """
        Create a scope including a set of US counties, by FIPS code.
        Raise GeographyError if any FIPS code is invalid.
        """
        return CountyScope('county', counties_fips, year)

    def get_node_ids(self) -> NDArray[np.str_]:
        match self.includes_granularity:
            case 'state':
                def is_in_states(geoid: str) -> bool:
                    return geoid[0:2] in self.includes
                counties = get_us_counties(self.year).geoid
                return counties[[is_in_states(x) for x in counties]]

            case 'county':
                return np.array(self.includes, dtype=np.str_)


@dataclass(frozen=True)
class TractScope(CensusScope):
    """GeoScope at the Tract granularity."""
    includes_granularity: Literal['state', 'county', 'tract']
    includes: Sequence[str]
    year: CensusYear
    granularity: Literal['tract'] = field(init=False, default='tract')

    @staticmethod
    @verify_fips('state')
    def in_states(states_fips: Sequence[str], year: CensusYear = DEFAULT_YEAR) -> 'TractScope':
        """
        Create a scope including all tracts in a set of US states/state-equivalents.
        Raise GeographyError if any FIPS code is invalid.
        """
        return TractScope('state', states_fips, year)

    @staticmethod
    @convert_state_code_to_fips
    def in_states_by_code(states_code: Sequence[str], year: CensusYear = DEFAULT_YEAR) -> 'TractScope':
        """
        Create a scope including all tracts in a set of US states/state-equivalents,
        by postal code (two-letter abbreviation).
        Raise GeographyError if any FIPS code is invalid.
        """
        return TractScope('state', states_code, year)

    @staticmethod
    @verify_fips('county')
    def in_counties(counties_fips: Sequence[str], year: CensusYear = DEFAULT_YEAR) -> 'TractScope':
        """
        Create a scope including all tracts in a set of US counties, by FIPS code.
        Raise GeographyError if any FIPS code is invalid.
        """
        return TractScope('county', counties_fips, year)

    @staticmethod
    @verify_fips('tract')
    def in_tracts(counties_fips: Sequence[str], year: CensusYear = DEFAULT_YEAR) -> 'TractScope':
        """
        Create a scope including a set of US tracts, by FIPS code.
        Raise GeographyError if any FIPS code is invalid.
        """
        return TractScope('tract', counties_fips, year)

    def get_node_ids(self) -> NDArray[np.str_]:
        match self.includes_granularity:
            case 'state':
                def is_in_states(geoid: str) -> bool:
                    return geoid[0:2] in self.includes
                tracts = get_us_tracts(self.year).geoid
                return tracts[[is_in_states(x) for x in tracts]]

            case 'county':
                def is_in_counties(geoid: str) -> bool:
                    return geoid[0:5] in self.includes
                tracts = get_us_tracts(self.year).geoid
                return tracts[[is_in_counties(x) for x in tracts]]

            case 'tract':
                return np.array(self.includes, dtype=np.str_)


@dataclass(frozen=True)
class BlockGroupScope(CensusScope):
    """GeoScope at the Block Group granularity."""
    includes_granularity: Literal['state', 'county', 'tract', 'block group']
    includes: Sequence[str]
    year: CensusYear
    granularity: Literal['block group'] = field(init=False, default='block group')

    @staticmethod
    @verify_fips('state')
    def in_states(states_fips: Sequence[str], year: CensusYear = DEFAULT_YEAR) -> 'BlockGroupScope':
        """
        Create a scope including all block groups in a set of US states/state-equivalents.
        Raise GeographyError if any FIPS code is invalid.
        """
        return BlockGroupScope('state', states_fips, year)

    @staticmethod
    @convert_state_code_to_fips
    def in_states_by_code(states_code: Sequence[str], year: CensusYear = DEFAULT_YEAR) -> 'BlockGroupScope':
        """
        Create a scope including all block groups in a set of US states/state-equivalents,
        by postal code (two-letter abbreviation).
        Raise GeographyError if any FIPS code is invalid.
        """
        return BlockGroupScope('state', states_code, year)

    @staticmethod
    @verify_fips('county')
    def in_counties(counties_fips: Sequence[str], year: CensusYear = DEFAULT_YEAR) -> 'BlockGroupScope':
        """
        Create a scope including all block groups in a set of US counties, by FIPS code.
        Raise GeographyError if any FIPS code is invalid.
        """
        return BlockGroupScope('county', counties_fips, year)

    @staticmethod
    @verify_fips('tract')
    def in_tracts(counties_fips: Sequence[str], year: CensusYear = DEFAULT_YEAR) -> 'BlockGroupScope':
        """
        Create a scope including all block gropus in a set of US tracts, by FIPS code.
        Raise GeographyError if any FIPS code is invalid.
        """
        return BlockGroupScope('tract', counties_fips, year)

    @staticmethod
    @verify_fips('block group')
    def in_block_groups(counties_fips: Sequence[str], year: CensusYear = DEFAULT_YEAR) -> 'BlockGroupScope':
        """
        Create a scope including a set of US block groups, by FIPS code.
        Raise GeographyError if any FIPS code is invalid.
        """
        return BlockGroupScope('block group', counties_fips, year)

    def get_node_ids(self) -> NDArray[np.str_]:
        match self.includes_granularity:
            case 'state':
                def is_in_states(geoid: str) -> bool:
                    return geoid[0:2] in self.includes
                block_groups = get_us_block_groups(self.year).geoid
                return block_groups[[is_in_states(x) for x in block_groups]]

            case 'county':
                def is_in_counties(geoid: str) -> bool:
                    return geoid[0:5] in self.includes
                block_groups = get_us_block_groups(self.year).geoid
                return block_groups[[is_in_counties(x) for x in block_groups]]

            case 'tract':
                def is_in_tracts(geoid: str) -> bool:
                    return geoid[0:11] in self.includes
                block_groups = get_us_block_groups(self.year).geoid
                return block_groups[[is_in_tracts(x) for x in block_groups]]

            case 'block group':
                return np.array(self.includes, dtype=np.str_)
