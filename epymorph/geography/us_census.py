"""
Geo scope instances that utilize US Census delineations.
Generally, each `CensusScope` describes the granularity of the nodes
in scope, plus some containing boundary at the same level of granularity
or higher in which all nodes in that boundary are considered in scope.
For instance, "give me all of the counties in Nebraska and South Dakota".
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal, Mapping, Never, Self, Sequence, TypeVar, cast

import numpy as np
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from pandas import DataFrame
from typing_extensions import override

import epymorph.geography.us_tiger as tiger
from epymorph.error import GeographyError
from epymorph.geography.custom import CustomScope
from epymorph.geography.scope import (
    GeoGroup,
    GeoGrouping,
    GeoScope,
    GeoSelection,
    GeoSelector,
    GeoStrategy,
    strategy_to_scope,
)
from epymorph.geography.us_geography import (
    COUNTY,
    STATE,
    TRACT,
    CensusGranularity,
    CensusGranularityName,
)
from epymorph.util import filter_unique, mask


@dataclass(frozen=True)
class CensusScope(ABC, GeoScope):
    """Base class for geo scopes using US Census delineations."""

    year: int
    """
    The Census delineation year. Census Bureau can (and does) define
    new delineations annually, especially at the smaller granularities. Hence, you must
    know the year in order to absolutely identify a set of delineations.
    """

    granularity: CensusGranularityName
    """Which granularity are the nodes in this scope?"""
    includes_granularity: CensusGranularityName
    """Which granularity defines the bounds of this scope?"""
    includes: tuple[str, ...]
    """Which nodes (of `includes_granularity`) are in scope?"""

    _node_ids: NDArray[np.str_] = field(init=False, compare=False, hash=False)

    def __post_init__(self):
        # NOTE: functionality depends on `includes` being sorted and unique.
        # Messing this up would be quite bad and hard to debug so I think it's
        # worth the overhead to verify it.
        xs = self.includes
        if not all(xs[i] < xs[i + 1] for i in range(len(xs) - 1)):
            raise ValueError("CensusScope.includes is not a sorted and unique list.")
        node_ids = self._compute_node_ids()
        object.__setattr__(self, "_node_ids", node_ids)

    @property
    @override
    def node_ids(self) -> NDArray[np.str_]:
        return self._node_ids

    def _compute_node_ids(self) -> NDArray[np.str_]:
        """Compute the GEOIDs for the nodes in scope."""
        if self.granularity == self.includes_granularity:
            return np.array(self.includes, dtype=np.str_)

        # return the nodes within a larger granularity
        g = CensusGranularity.of(self.includes_granularity)
        return np.array(
            [
                x
                for x in tiger.get_summary_of(self.granularity, self.year).geoid
                if g.truncate(x) in self.includes
            ],
            dtype=np.str_,
        )

    @abstractmethod
    def raise_granularity(self) -> "CensusScope":
        """
        Convert this scope to one that is one level higher in granularity.

        Returns
        -------
        :
            A scope with granularity one level higher than this.
            This may have the effect of widening the scope; for example,
            raising a `TractScope` which is filtered to a set of tracts will
            result in a `CountyScope` containing the counties that contained our
            tracts.

        Raises
        ------
        GeographyError
            If the granularity cannot be raised.
        """

    @abstractmethod
    def lower_granularity(self) -> "CensusScope":
        """
        Convert this scope to one that is one level lower in granularity.

        Returns
        -------
        :
            A scope with granularity one level lower than this.

        Raises
        ------
        GeographyError
            If the granularity cannot be lowered.
        """

    def as_granularity(self, granularity: CensusGranularityName) -> "CensusScope":
        """
        Convert this scope to the named granularity.

        Returns
        -------
        :
            A scope of the named granularity by either raising
            or lowering the original scope to match. If `granularity`
            already matches this scope, it is returned unchanged.

        Raises
        ------
        GeographyError
            If the granularity cannot be modified as requested.
        """
        target = CensusGranularity.of(granularity)
        scope = self
        while scope.granularity != granularity:
            if target < CensusGranularity.of(scope.granularity):
                scope = scope.raise_granularity()
            else:
                scope = scope.lower_granularity()
        return scope

    @staticmethod
    def of(
        name: CensusGranularityName,
        node_ids: Sequence[str],
        year: int,
    ) -> "CensusScope":
        """
        Create a `CensusScope` instance of the named granularity,
        using nodes IDs of the same granularity. This is the same as
        `StateScope.in_states(...)` for example.

        Parameters
        ----------
        name :
            The name of the Census granularity for the resulting scope.
        node_ids :
            The node IDs to include in the scope; these IDs must identify
            nodes of the same granularity as `name`.
        year :
            The Census delineation year.
        """
        match name:
            case "state":
                return StateScope.in_states(node_ids, year)
            case "county":
                return CountyScope.in_counties(node_ids, year)
            case "tract":
                return TractScope.in_tracts(node_ids, year)
            case "block group":
                return BlockGroupScope.in_block_groups(node_ids, year)
            case x:
                raise ValueError(f"Not a supported granularity: {x}")

    @abstractmethod
    def get_info(self) -> DataFrame:
        """
        Retrieve TIGER info for the nodes in this scope. This is very similar to the
        `geography` property of `CensusScope`s, but it omits the shape data so it's a
        little faster for use-cases that don't need it.
        """


##############################
# Mixins for creating scopes #
##############################


# NOTE: these mixins form a hierarchy of constructor methods.
# It is possible to create a StateScope using a set of states.
# It it possible to create a CountyScope using a set of states OR counties
# (where using states implies "all counties within these states").
# And so on.
# Scope instances will mix in the valid constructors.


class _InMixin(CensusScope):
    @classmethod
    def _in(
        cls,
        includes_granularity: CensusGranularityName,
        includes: Sequence[str],
        year: int,
    ) -> Self:
        # Utility classmethod for creating scopes.
        g = tiger.get_summary_of(includes_granularity, year)
        return cls(
            includes_granularity=includes_granularity,  # type: ignore
            includes=tuple(g.interpret(includes)),
            year=year,
        )


class _InStatesMixin(_InMixin):
    @classmethod
    def in_states(cls, states: Sequence[str], year: int) -> Self:
        """
        Create a scope including all nodes in a set of US states/state-equivalents.

        Parameters
        ----------
        states :
            The set of states to include. This can be a list of either state names,
            postal codes, or FIPs codes.
        year :
            The Census delineation year.

        Returns
        -------
        :
            The new scope.

        Raises
        ------
        GeographyError
            If the year or any entry in `states` is invalid.
        """
        return cls._in("state", states, year)


class _InCountiesMixin(_InMixin):
    @classmethod
    def in_counties(cls, counties: Sequence[str], year: int) -> Self:
        """
        Create a scope including all nodes in a set of US counties/county-equivalents.

        Parameters
        ----------
        counties :
            The set of counties to include. This can be a list of county names
            (in county-comma-postal-code format, e.g., "Maricopa, AZ") or FIPS codes.
        year :
            The Census delineation year.

        Returns
        -------
        :
            The new scope.

        Raises
        ------
        GeographyError
            If the year or any entry in `counties` is invalid.
        """
        return cls._in("county", counties, year)


class _InTractsMixin(_InMixin):
    @classmethod
    def in_tracts(cls, tracts: Sequence[str], year: int) -> Self:
        """
        Create a scope including all nodes in a set of US Census tracts.

        Parameters
        ----------
        tracts :
            The set of tracts to include, by ID.
        year :
            The Census delineation year.

        Returns
        -------
        :
            The new scope.

        Raises
        ------
        GeographyError
            If the year or any entry in `tracts` is invalid.
        """
        return cls._in("tract", tracts, year)


class _InBlockGroupsMixin(_InMixin):
    @classmethod
    def in_block_groups(cls, block_groups: Sequence[str], year: int) -> Self:
        """
        Create a scope including all nodes in a set of US Census block groups.

        Parameters
        ----------
        block_groups :
            The set of block groups to include, by ID.
        year :
            The Census delineation year.

        Returns
        -------
        :
            The new scope.

        Raises
        ------
        GeographyError
            If the year or any entry in `block_groups` is invalid.
        """
        return cls._in("block group", block_groups, year)


##################################
# Granularity-specific instances #
##################################


# TODO: (Tyler) it might be interesting to replace the "includes" concept
# with a "parent" scope, which is an actual scope instance. e.g., a TractScope
# with a parent StateScope would include all tracts in the states included.
# Possibly more elegant.


@dataclass(frozen=True)
class StateScope(_InStatesMixin, CensusScope):
    """
    A `CensusScope` at the US State granularity.

    Typically you'll create one of these using `all` or `in_states`.

    Parameters
    ----------
    year :
        The Census delineation year.
    includes_granularity :
        Which granularity defines the bounds of this scope?
    includes :
        Which nodes (of `includes_granularity`) are in scope?
    """

    year: int
    includes_granularity: Literal["state"]
    includes: tuple[str, ...]
    granularity: Literal["state"] = field(init=False, default="state")

    @classmethod
    def all(cls, year: int) -> Self:
        """
        Create a scope including all US states and state-equivalents.

        Parameters
        ----------
        year :
            The Census delineation year.

        Returns
        -------
        :
            The new scope.
        """
        return cls(
            includes_granularity="state",
            includes=tuple(tiger.get_states(year).geoid),
            year=year,
        )

    def _compute_node_ids(self) -> NDArray[np.str_]:
        # return all nodes
        return np.array(self.includes, dtype=np.str_)

    def is_all_states(self) -> bool:
        """
        Check if this scope includes every support US state.

        Returns
        -------
        :
            True if this scope includes all supported US states.
        """
        return np.array_equal(tiger.get_states(self.year).geoid, self.node_ids)

    @override
    def raise_granularity(self) -> Never:
        raise GeographyError("No granularity higher than state.")

    @override
    def lower_granularity(self) -> "CountyScope":
        return CountyScope(
            includes_granularity=self.includes_granularity,
            includes=self.includes,
            year=self.year,
        )

    @property
    def select(self) -> "StateSelector[StateScope, StateSelection]":
        """Create a selection from this scope."""
        return StateSelector(self, StateSelection)

    @property
    @override
    def labels_option(self) -> NDArray[np.str_]:
        mapping = tiger.get_states(self.year).state_fips_to_code
        return np.array([mapping[x] for x in self.node_ids], dtype=np.str_)

    @cached_property
    @override
    def geography(self) -> GeoDataFrame:  # type: ignore[override]
        gdf = tiger.get_states_geo(self.year)
        in_scope = gdf["GEOID"].isin(self.node_ids)
        return gdf[in_scope].sort_values(by="GEOID").reset_index(drop=True)

    @override
    def get_info(self) -> DataFrame:
        info_df = tiger.get_states_info(self.year)
        in_scope = info_df["GEOID"].isin(self.node_ids)
        return info_df[in_scope].sort_values(by="GEOID").reset_index(drop=True)


@dataclass(frozen=True)
class CountyScope(_InStatesMixin, _InCountiesMixin, CensusScope):
    """
    A `CensusScope` at the US County granularity.

    Typically you'll create one of these using `all`, `in_states`, or `in_counties`.

    Parameters
    ----------
    year :
        The Census delineation year.
    includes_granularity :
        Which granularity defines the bounds of this scope?
    includes :
        Which nodes (of `includes_granularity`) are in scope?
    """

    year: int
    includes_granularity: Literal["state", "county"]
    includes: tuple[str, ...]
    granularity: Literal["county"] = field(init=False, default="county")

    @classmethod
    def all(cls, year: int) -> Self:
        """
        Create a scope including all US counties and county-equivalents.

        Parameters
        ----------
        year :
            The Census delineation year.

        Returns
        -------
        :
            The new scope.
        """
        return cls(
            includes_granularity="county",
            includes=tuple(tiger.get_counties(year).geoid),
            year=year,
        )

    @override
    def raise_granularity(self) -> StateScope:
        if self.includes_granularity == "county":
            return StateScope(
                includes_granularity="state",
                includes=tuple(STATE.truncate_unique(self.includes)),
                year=self.year,
            )

        return StateScope(
            includes_granularity=self.includes_granularity,
            includes=self.includes,
            year=self.year,
        )

    @override
    def lower_granularity(self) -> "TractScope":
        return TractScope(
            includes_granularity=self.includes_granularity,
            includes=self.includes,
            year=self.year,
        )

    @property
    def select(self) -> "CountySelector[CountyScope, CountySelection]":
        """Create a selection from this scope."""
        return CountySelector(self, CountySelection)

    @property
    @override
    def labels_option(self) -> NDArray[np.str_]:
        mapping = tiger.get_counties(self.year).county_fips_to_name
        return np.array([mapping[x] for x in self.node_ids], dtype=np.str_)

    @cached_property
    @override
    def geography(self) -> GeoDataFrame:  # type: ignore[override]
        gdf = tiger.get_counties_geo(self.year)
        in_scope = gdf["GEOID"].isin(self.node_ids)
        return gdf[in_scope].sort_values(by="GEOID").reset_index(drop=True)

    @override
    def get_info(self) -> DataFrame:
        info_df = tiger.get_counties_info(self.year)
        in_scope = info_df["GEOID"].isin(self.node_ids)
        return info_df[in_scope].sort_values(by="GEOID").reset_index(drop=True)


@dataclass(frozen=True)
class TractScope(_InStatesMixin, _InCountiesMixin, _InTractsMixin, CensusScope):
    """
    A `CensusScope` at the US Tract granularity.

    Typically you will create one of these using `in_states`, `in_counties`, or
    `in_tracts`.

    Parameters
    ----------
    year :
        The Census delineation year.
    includes_granularity :
        Which granularity defines the bounds of this scope?
    includes :
        Which nodes (of `includes_granularity`) are in scope?
    """

    year: int
    includes_granularity: Literal["state", "county", "tract"]
    includes: tuple[str, ...]
    granularity: Literal["tract"] = field(init=False, default="tract")

    @override
    def raise_granularity(self) -> CountyScope:
        if self.includes_granularity == "tract":
            return CountyScope(
                includes_granularity="county",
                includes=tuple(COUNTY.truncate_unique(self.includes)),
                year=self.year,
            )

        return CountyScope(
            includes_granularity=self.includes_granularity,
            includes=self.includes,
            year=self.year,
        )

    @override
    def lower_granularity(self) -> "BlockGroupScope":
        return BlockGroupScope(
            year=self.year,
            includes_granularity=self.includes_granularity,
            includes=self.includes,
        )

    @property
    def select(self) -> "TractSelector[TractScope, TractSelection]":
        """Create a selection from this scope."""
        return TractSelector(self, TractSelection)

    @cached_property
    @override
    def geography(self) -> GeoDataFrame:  # type: ignore[override]
        states = list({STATE.extract(x) for x in self.node_ids})
        gdf = tiger.get_tracts_geo(self.year, states)
        in_scope = gdf["GEOID"].isin(self.node_ids)
        return gdf[in_scope].sort_values(by="GEOID").reset_index(drop=True)

    @override
    def get_info(self) -> DataFrame:
        states = list({STATE.extract(x) for x in self.node_ids})
        info_df = tiger.get_tracts_info(self.year, states)
        in_scope = info_df["GEOID"].isin(self.node_ids)
        return info_df[in_scope].sort_values(by="GEOID").reset_index(drop=True)


@dataclass(frozen=True)
class BlockGroupScope(
    _InStatesMixin, _InCountiesMixin, _InTractsMixin, _InBlockGroupsMixin, CensusScope
):
    """
    A `CensusScope` at the US Block Group granularity.

    Typically you will create one of these using `in_states`, `in_counties`,
    `in_tracts`, or `in_block_groups`.

    Parameters
    ----------
    year :
        The Census delineation year.
    includes_granularity :
        Which granularity defines the bounds of this scope?
    includes :
        Which nodes (of `includes_granularity`) are in scope?
    """

    year: int
    includes_granularity: Literal["state", "county", "tract", "block group"]
    includes: tuple[str, ...]
    granularity: Literal["block group"] = field(init=False, default="block group")

    @override
    def raise_granularity(self) -> TractScope:
        if self.includes_granularity == "block group":
            return TractScope(
                includes_granularity="tract",
                includes=tuple(TRACT.truncate_unique(self.includes)),
                year=self.year,
            )

        return TractScope(
            includes_granularity=self.includes_granularity,
            includes=self.includes,
            year=self.year,
        )

    @override
    def lower_granularity(self) -> Never:
        raise GeographyError("No valid granularity lower than block group.")

    @property
    def select(self) -> "BlockGroupSelector[BlockGroupScope, BlockGroupSelection]":
        """Create a selection from this scope."""
        return BlockGroupSelector(self, BlockGroupSelection)

    @cached_property
    @override
    def geography(self) -> GeoDataFrame:  # type: ignore[override]
        states = list({STATE.extract(x) for x in self.node_ids})
        gdf = tiger.get_block_groups_geo(self.year, states)
        in_scope = gdf["GEOID"].isin(self.node_ids)
        return gdf[in_scope].sort_values(by="GEOID").reset_index(drop=True)

    @override
    def get_info(self) -> DataFrame:
        states = list({STATE.extract(x) for x in self.node_ids})
        info_df = tiger.get_block_groups_info(self.year, states)
        in_scope = info_df["GEOID"].isin(self.node_ids)
        return info_df[in_scope].sort_values(by="GEOID").reset_index(drop=True)


#####################
# Geo axis strategy #
#####################


@strategy_to_scope.register
def _census_strategy_to_scope(
    scope: CensusScope,
    strategy: GeoStrategy[CensusScope],
) -> GeoScope:
    selected = scope.node_ids[strategy.selection]
    match (strategy.grouping, strategy.aggregation):
        case (None, None):
            # No grouping or aggregation: new scope is just a subselection.
            result_granularity = scope.granularity
            result_nodes = cast(list[str], selected.tolist())
        case (None, _):
            # No grouping, some aggregation reduces to one node.
            return CustomScope(["*"])
            # TODO: (Tyler) it would be possible to maintain a census scope
            # in the case where all the selected nodes share a common container,
            # e.g., I've selected only Arizona counties. We could detect the
            # maximal shared geoid prefix that corresponds to a granularity level
            # and use that. I need to think about whether or not this is desirable,
            # because it would be a pain to implement.
        case (CensusGrouping(), _):
            # Grouped by a CensusGrouping: create a new scope with nodes from the
            # group mapping of the selected nodes.
            g = strategy.grouping
            result_granularity = g.granularity
            result_nodes = cast(list[str], filter_unique(g.map(selected)))
        case (g, _):
            # Some other kind of grouping; fall back to CustomScope.
            # TODO: (Tyler) it may be possible to apply the same granularity detection
            # here as above... but again I dunno if that's a good idea.
            return CustomScope(g.map(selected))

    match result_granularity:
        case "state":
            return StateScope.in_states(result_nodes, scope.year)
        case "county":
            return CountyScope.in_counties(result_nodes, scope.year)
        case "tract":
            return TractScope.in_tracts(result_nodes, scope.year)
        case "block group":
            return BlockGroupScope.in_block_groups(result_nodes, scope.year)
        case x:
            err = f"Unsupported granularity {x}"
            raise GeographyError(err)


@dataclass(frozen=True)
class CensusGrouping(GeoGrouping):
    """
    A geo-axis grouping for Census geo scopes, for example, "group by state".

    Parameters
    ----------
    granularity :
        The granularity to group by.
    """

    granularity: CensusGranularityName
    """The granularity to group by."""

    @override
    def map(self, node_ids: NDArray[np.str_]) -> NDArray[np.str_]:
        gran = CensusGranularity.of(self.granularity)
        return np.array([gran.truncate(g) for g in node_ids], dtype=np.str_)


@dataclass(frozen=True)
class StateSelection(GeoSelection[CensusScope]):
    """A geo selection on a `StateScope`."""


@dataclass(frozen=True)
class CountySelection(GeoSelection[CensusScope]):
    """A geo selection on a `CountyScope`."""

    def group(self, grouping: Literal["state"] | GeoGrouping) -> GeoGroup[CensusScope]:
        """
        Group the geo series using the specified grouping.

        Parameters
        ----------
        grouping :
            The grouping to use. You can specify a supported string value --
            all of which act as shortcuts for common `GeoGrouping` instances --
            or you can provide a `GeoGrouping` instance to perform custom grouping.

            The shortcut values are:

            - "state": equivalent to `CensusGrouping("state")`

        Returns
        -------
        :
            The group strategy.
        """
        match grouping:
            case "state":
                grouping = CensusGrouping(grouping)
            case _:
                pass
        return GeoGroup(self.scope, self.selection, grouping)


@dataclass(frozen=True)
class TractSelection(GeoSelection[CensusScope]):
    """A geo selection on a `TractScope`."""

    def group(
        self, grouping: Literal["state", "county"] | GeoGrouping
    ) -> GeoGroup[CensusScope]:
        """
        Group the geo series using the specified grouping.

        Parameters
        ----------
        grouping :
            The grouping to use. You can specify a supported string value --
            all of which act as shortcuts for common `GeoGrouping` instances --
            or you can provide a `GeoGrouping` instance to perform custom grouping.

            The shortcut values are:

            - "state": equivalent to `CensusGrouping("state")`
            - "county": equivalent to `CensusGrouping("county")`

        Returns
        -------
        :
            The group strategy.
        """
        match grouping:
            case "state":
                grouping = CensusGrouping(grouping)
            case "county":
                grouping = CensusGrouping(grouping)
            case _:
                pass
        return GeoGroup(self.scope, self.selection, grouping)


@dataclass(frozen=True)
class BlockGroupSelection(GeoSelection[CensusScope]):
    """A geo selection on a `BlockGroupScope`."""

    def group(
        self, grouping: Literal["state", "county", "tract"] | GeoGrouping
    ) -> GeoGroup[CensusScope]:
        """
        Group the geo series using the specified grouping.

        Parameters
        ----------
        grouping :
            The grouping to use. You can specify a supported string value --
            all of which act as shortcuts for common `GeoGrouping` instances --
            or you can provide a `GeoGrouping` instance to perform custom grouping.

            The shortcut values are:

            - "state": equivalent to `CensusGrouping("state")`
            - "county": equivalent to `CensusGrouping("county")`
            - "tract": equivalent to `CensusGrouping("tract")`
        """
        match grouping:
            case "state":
                grouping = CensusGrouping(grouping)
            case "county":
                grouping = CensusGrouping(grouping)
            case "tract":
                grouping = CensusGrouping(grouping)
            case _:
                pass
        return GeoGroup(self.scope, self.selection, grouping)


# NOTE: these Selector classes form a hierarchy of shared functionality.
# With a StateScope, you can select by geoid and state.
# With a CountyScope, you can select by geoid, state, AND county.
# And so on.

CensusScopeT = TypeVar("CensusScopeT", bound=CensusScope)
"""The type of geo scope."""

CensusSelectionT = TypeVar(
    "CensusSelectionT",
    bound=(StateSelection | CountySelection | TractSelection | BlockGroupSelection),
)
"""The type of geo selection."""


@dataclass(frozen=True)
class _CensusSelector(GeoSelector[CensusScopeT, CensusSelectionT]):
    """Base class for Census scope selectors with shared functions."""

    _scope: CensusScopeT
    """The original scope."""
    _selection_class: type[CensusSelectionT]
    """The class of the selection produced."""
    _node_id_to_label: Mapping[str, str] | None = field(init=False, default=None)
    """The mapping to use for `node_id_to_label` (see: GeoStrategy)"""

    def by_geoid(self, *geoids: str) -> CensusSelectionT:
        """
        Select nodes by GEOID. It is possible to select all nodes within
        a coarser granularity by passing those GEOIDs, however all GEOIDs
        given must be the same granularity.

        Parameters
        ----------
        *geoids :
            The GEOIDs to select, as var-args.

        Returns
        -------
        :
            The geo selection.
        """
        scope_gran = CensusGranularity.of(self._scope.granularity)
        scope_geoid_len = scope_gran.length
        if any(len(x) > scope_geoid_len for x in geoids):
            err = f"Selection geoids must be {scope_gran.name}-level or coarser."
            raise GeographyError(err)

        mask = np.zeros_like(self._scope.node_ids, dtype=np.bool_)
        for i, node in enumerate(self._scope.node_ids):
            mask[i] = any(node.startswith(sel) for sel in geoids)
        return self._from_mask(mask)

    def by_slice(
        self,
        start: int,
        stop: int | None = None,
        step: int | None = None,
    ) -> CensusSelectionT:
        """
        Select nodes by specifying a slice on the node indices.

        Parameters
        ----------
        start :
            The initial index.
        stop :
            The stop index (exclusive of the endpoint), or `None`
            for no endpoint.
        step :
            The interval between selected indices, or `None` to
            default to every index.

        Returns
        -------
        :
            The geo selection.
        """
        return self._from_mask(mask(self._scope.nodes, slice(start, stop, step)))

    def by_indices(self, indices: list[int]) -> CensusSelectionT:
        """
        Select nodes by specifying the node indices to include.

        Parameters
        ----------
        indices :
            The indices to include.

        Returns
        -------
        :
            The geo selection.
        """
        return self._from_mask(mask(self._scope.nodes, indices))


@dataclass(frozen=True)
class StateSelector(_CensusSelector[CensusScopeT, CensusSelectionT]):
    """
    A geo selector on a `StateScope`.

    Most of the time you obtain one of these using a scope's `select` property.
    """

    def by_state(self, *identifiers: str) -> CensusSelectionT:
        """
        Select all nodes within a set of states.

        Parameters
        ----------
        *identifiers :
            The set of states to include, as var-args. This can be a list of either
            state names, postal codes, or FIPs codes.

        Returns
        -------
        :
            The geo selection.
        """
        states = tiger.get_states(self._scope.year)
        geoids = states.interpret(identifiers)
        return self.by_geoid(*geoids)


@dataclass(frozen=True)
class CountySelector(StateSelector[CensusScopeT, CensusSelectionT]):
    """
    A geo selector on a `CountyScope`.

    Most of the time you obtain one of these using a scope's `select` property.
    """

    def by_county(self, *identifiers: str) -> CensusSelectionT:
        """
        Select all nodes within a set of counties.

        Parameters
        ----------
        *identifiers :
            The set of counties to include, as var-args. This can be a list of
            county names (in county-comma-postal-code format, e.g., "Maricopa, AZ")
            or FIPS codes.

        Returns
        -------
        :
            The geo selection.
        """
        counties = tiger.get_counties(self._scope.year)
        geoids = counties.interpret(identifiers)
        return self.by_geoid(*geoids)


@dataclass(frozen=True)
class TractSelector(CountySelector[CensusScopeT, CensusSelectionT]):
    """
    A geo selector on a `TractScope`.

    Most of the time you obtain one of these using a scope's `select` property.
    """

    def by_tract(self, *identifiers: str) -> CensusSelectionT:
        """
        Select all nodes within a set of tracts.

        Parameters
        ----------
        *identifiers :
            The set of tracts to include by ID, as var-args.

        Returns
        -------
        :
            The geo selection.
        """
        tracts = tiger.get_tracts(self._scope.year)
        geoids = tracts.interpret(identifiers)
        return self.by_geoid(*geoids)


@dataclass(frozen=True)
class BlockGroupSelector(TractSelector[CensusScopeT, CensusSelectionT]):
    """
    A geo selector on a `BlockGroupScope`.

    Most of the time you obtain one of these using a scope's `select` property.
    """

    def by_block_group(self, *identifiers: str) -> CensusSelectionT:
        """
        Select all nodes within a set of block groups.

        Parameters
        ----------
        *identifiers :
            The set of block groups to include by ID, as var-args.

        Returns
        -------
        :
            The geo selection.
        """
        block_groups = tiger.get_block_groups(self._scope.year)
        geoids = block_groups.interpret(identifiers)
        return self.by_geoid(*geoids)


__all__ = [
    "CensusScope",
    "StateScope",
    "CountyScope",
    "TractScope",
    "BlockGroupScope",
    "CensusGrouping",
]
