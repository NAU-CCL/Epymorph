"""
Encodes the geographic system made up of US Census delineations.
This system comprises a set of perfectly-nested granularities,
and a structured ID system for labeling all delineations
(sometimes loosely called FIPS codes or GEOIDs).
"""

import re
from abc import ABC, abstractmethod
from typing import Iterable, Literal, Self

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.error import GeographyError
from epymorph.util import prefix

CensusGranularityName = Literal["state", "county", "tract", "block group", "block"]
"""Type alias: the name of a supported Census granularity."""

CENSUS_HIERARCHY = ("state", "county", "tract", "block group", "block")
"""The granularities in hierarchy order (largest to smallest)."""


class CensusGranularity(ABC):
    """
    Each `CensusGranularity` instance defines a set of utility functions for working
    with GEOIDs of that granularity, as well as inspecting and manipulating the
    granularity hierarchy itself.

    In typical usage, you will not construct this class directly, but use the
    `CensusGranularity.of(granularity)` static method to obtain an instance, or import
    one of the singleton instances: `STATE`, `COUNTY`, `TRACT`, `BLOCK_GROUP`, or
    `BLOCK`.

    Parameters
    ----------
    name :
        The granularity.
    length :
        The number of digits in the GEOIDs for this granularity.
    match_pattern :
        A regex pattern matching GEOIDs for this granularity.
    extract_pattern :
        A regex pattern with match groups for extracting this granularity
        segment from a GEOID of an equal or finer granularity.
    decompose_pattern :
        A regex pattern with match groups for decomposing a GEOID of this
        granularity into segments.
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

        Parameters
        ----------
        outer :
            The other granularity to consider.

        Returns
        -------
        :
            True if this granularity is inside or the same as `other`.
        """
        return CENSUS_HIERARCHY.index(outer) <= CENSUS_HIERARCHY.index(self.name)

    def matches(self, geoid: str) -> bool:
        """
        Test whether the given GEOID matches this granularity exactly.
        For example: "04" matches state granularity but not county granularity.

        Parameters
        ----------
        geoid :
            The GEOID to test.

        Returns
        -------
        :
            True if the GEOID given matches this granularity.
        """
        return self._match_pattern.match(geoid) is not None

    def extract(self, geoid: str) -> str:
        """
        Extract this level of granularity's GEOID segment, if the given GEOID is of
        this granularity or smaller.

        Parameters
        ----------
        geoid :
            The GEOID to operate on.

        Returns
        -------
        :
            The segment of the GEOID that matches this granularity.

        Raises
        ------
        GeographyError
            If the GEOID is unsuitable or poorly formatted.
        """
        if (m := self._extract_pattern.match(geoid)) is not None:
            return m[1]
        else:
            msg = f"Unable to extract {self._name} info from ID {id}; check its format."
            raise GeographyError(msg)

    def truncate(self, geoid: str) -> str:
        """
        Truncate the given GEOID to this level of granularity.
        If the given GEOID is for a granularity larger than this level,
        the GEOID will be returned unchanged.

        Parameters
        ----------
        geoid :
            The GEOID to operate on.

        Returns
        -------
        :
            The truncated GEOID.
        """
        return geoid[: self.length]

    def truncate_unique(self, geoids: Iterable[str]) -> Iterable[str]:
        """
        Truncate an Iterable of GEOIDs to this level of granularity, returning only
        unique entries without changing the ordering of entries.

        Parameters
        ----------
        geoids :
            The list of GEOIDs to operate on.

        Returns
        -------
        :
            The unique values contained in `geoids` after truncation
            to this level of granularity.
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
        Decompose a GEOID as a regex match.

        Raises
        ------
        GeographyError
            If the match fails.
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
        The GEOID must match this granularity exactly.

        Parameters
        ----------
        geoid :
            The GEOID to operate on.

        Returns
        -------
        :
            A tuple as long as the number of granularity segments used for this
            level of granularity.

        Raises
        ------
        GeographyError
            If the GEOID does not match this granularity exactly.
        """

    def grouped(self, sorted_geoids: NDArray[np.str_]) -> dict[str, NDArray[np.str_]]:
        """
        Group a list of GEOIDs by this level of granularity.

        Requires that the GEOID array has been sorted!

        Parameters
        ----------
        sorted_geoids :
            The GEOIDs to group, as a sorted array.

        Returns
        -------
        :
            A dictionary where the keys represent the unique groups
            present and the values are all GEOIDs contained in each group.
        """
        group_prefix = prefix(self.length)(sorted_geoids)
        uniques, splits = np.unique(group_prefix, return_index=True)
        grouped = np.split(sorted_geoids, splits[1:])
        return dict(zip(uniques, grouped))

    @staticmethod
    def of(name: CensusGranularityName) -> "CensusGranularity":
        """
        Get a CensusGranularity instance by name.

        Parameters
        ----------
        name :
            The name of the granularity.

        Returns
        -------
        :
            The CensusGranularity instance, whose type matches the named granularity.
        """
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

    @override
    def decompose(self, geoid: str) -> tuple[str]:
        """
        Decompose a GEOID into a tuple containing the state ID.
        The GEOID must be a state GEOID.

        Parameters
        ----------
        geoid :
            The GEOID to operate on.

        Returns
        -------
        :
            A tuple of state ID.

        Raises
        ------
        GeographyError
            If the GEOID does not match this granularity exactly.
        """
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

    @override
    def decompose(self, geoid: str) -> tuple[str, str]:
        """
        Decompose a GEOID into a tuple containing the state and county ID.
        The GEOID must be a county GEOID.

        Parameters
        ----------
        geoid :
            The GEOID to operate on.

        Returns
        -------
        :
            A tuple of state and county ID.

        Raises
        ------
        GeographyError
            If the GEOID does not match this granularity exactly.
        """
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

    @override
    def decompose(self, geoid: str) -> tuple[str, str, str]:
        """
        Decompose a GEOID into a tuple containing the state, county, and tract ID.
        The GEOID must be a tract GEOID.

        Parameters
        ----------
        geoid :
            The GEOID to operate on.

        Returns
        -------
        :
            A tuple of state, county, and tract ID.

        Raises
        ------
        GeographyError
            If the GEOID does not match this granularity exactly.
        """
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

    @override
    def decompose(self, geoid: str) -> tuple[str, str, str, str]:
        """
        Decompose a GEOID into a tuple containing the state, county, tract, and
        block group ID. The GEOID must be a block group GEOID.

        Parameters
        ----------
        geoid :
            The GEOID to operate on.

        Returns
        -------
        :
            A tuple of state, county, tract, and block group ID.

        Raises
        ------
        GeographyError
            If the GEOID does not match this granularity exactly.
        """
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

    @override
    def decompose(self, geoid: str) -> tuple[str, str, str, str, str]:
        """
        Decompose a GEOID into a tuple containing the state, county, tract,
        block group, and block ID. The GEOID must be a block GEOID.
        Note that block IDs are kind of strange -- the block group ID
        is a single digit, but the block ID also includes this digit.
        Thus, the returned tuple will include this digit in both of the last two parts.

        Parameters
        ----------
        geoid :
            The GEOID to operate on.

        Returns
        -------
        :
            A tuple of state, county, tract, block group, and block ID.

        Raises
        ------
        GeographyError
            If the GEOID does not match this granularity exactly.
        """
        # The block group ID is the first digit of the block ID,
        # but the block ID also includes this digit.
        m = self._decompose(geoid)
        return (m[1], m[2], m[3], m[4][0], m[4])


# Singletons for the CensusGranularity classes.
STATE = State()
"""A singleton `State` instance."""
COUNTY = County()
"""A singleton `County` instance."""
TRACT = Tract()
"""A singleton `Tract` instance."""
BLOCK_GROUP = BlockGroup()
"""A singleton `BlockGroup` instance."""
BLOCK = Block()
"""A singleton `Block` instance."""

CENSUS_GRANULARITY = (STATE, COUNTY, TRACT, BLOCK_GROUP, BLOCK)
"""`CensusGranularity` singletons in hierarchy order."""
