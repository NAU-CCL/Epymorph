"""
Encodes the geographic system made up of US Census delineations.
This system comprises a set of perfectly-nested granularities,
and a structured ID system for labeling all delineations
(sometimes loosely called FIPS codes or GEOIDs).
"""

import re
from abc import ABC, abstractmethod
from typing import (
    Iterable,
    Literal,
    Self,
)

import numpy as np
from numpy.typing import NDArray

from epymorph.error import GeographyError
from epymorph.util import (
    prefix,
)

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
    _index: int
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
        self._index = CENSUS_HIERARCHY.index(name)
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

    # TODO: test operators
    def __lt__(self, other: Self) -> bool:
        return self._index < other._index

    def __gt__(self, other: Self) -> bool:
        return self._index > other._index

    def __le__(self, other: Self) -> bool:
        return self._index <= other._index

    def __ge__(self, other: Self) -> bool:
        return self._index >= other._index

    def __eq__(self, other) -> bool:
        if not isinstance(other, CensusGranularity):
            return False
        return self._index == other._index

    # TODO: remove?
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

    def truncate_unique(self, geoids: Iterable[str]) -> Iterable[str]:
        """
        Truncates an Iterable of GEOIDs to this level of granularity, returning only
        unique entries without changing the ordering of entries.
        """
        n = self.length
        seen = set[str]()
        for g in geoids:
            curr = g[:n]
            if curr not in seen:
                yield curr
                seen.add(curr)

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

    @staticmethod
    def of(name: CensusGranularityName) -> "CensusGranularity":
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
