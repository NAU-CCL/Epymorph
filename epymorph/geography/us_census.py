"""
GeoScope instances that utilize US Census delineations.
Generally, each CensusScope describes the granularity of the nodes
in scope, plus some containing boundary at the same level of granularity
or higher in which all nodes in that boundary are considered in scope.
For instance, "give me all of the counties in Nebraska and South Dakota".
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Mapping, Never, Self, Sequence, TypeVar, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

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
from epymorph.geography.us_tiger import (
    get_block_groups,
    get_counties,
    get_states,
    get_summary_of,
    get_tracts,
)
from epymorph.util import filter_unique, mask


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
    includes_granularity: CensusGranularityName
    """Which granularity defines the bounds of this scope?"""
    includes: tuple[str, ...]
    """Which nodes (of `includes_granularity`) are in scope?"""

    _node_ids: NDArray[np.str_] = field(init=False, compare=False, hash=False)

    def __post_init__(self):
        node_ids = self._compute_node_ids()
        object.__setattr__(self, "_node_ids", node_ids)

    @property
    @override
    def node_ids(self) -> NDArray[np.str_]:
        return self._node_ids

    def _compute_node_ids(self) -> NDArray[np.str_]:
        """Internal method to compute the GEOIDs for the nodes in scope."""
        if self.granularity == self.includes_granularity:
            return np.array(self.includes, dtype=np.str_)

        # return the nodes within a larger granularity
        g = CensusGranularity.of(self.includes_granularity)
        return np.array(
            [
                x
                for x in get_summary_of(self.granularity, self.year).geoid
                if g.truncate(x) in self.includes
            ],
            dtype=np.str_,
        )

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

    @staticmethod
    def of(
        name: CensusGranularityName,
        node_ids: Sequence[str],
        year: int,
    ) -> "CensusScope":
        """Creates a CensusScope instance with the named granularity,
        using nodes IDs of the same granularity. This is the same as
        `StateScope.in_states(...)` for example."""
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
        g = get_summary_of(includes_granularity, year)
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
        Raise GeographyError if any FIPS code is invalid.
        """
        return cls._in("state", states, year)


class _InCountiesMixin(_InMixin):
    @classmethod
    def in_counties(cls, counties: Sequence[str], year: int) -> Self:
        """
        Create a scope including all nodes in a set of US counties/county-equivalents.
        Raise GeographyError if any FIPS code is invalid.
        """
        return cls._in("county", counties, year)


class _InTractsMixin(_InMixin):
    @classmethod
    def in_tracts(cls, tracts: Sequence[str], year: int) -> Self:
        """
        Create a scope including all nodes in a set of US census tracts.
        Raise GeographyError if any FIPS code is invalid.
        """
        return cls._in("tract", tracts, year)


class _InBlockGroupsMixin(_InMixin):
    @classmethod
    def in_block_groups(cls, block_groups: Sequence[str], year: int) -> Self:
        """
        Create a scope including all nodes in a set of US census tracts.
        Raise GeographyError if any FIPS code is invalid.
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
    """GeoScope at the State granularity."""

    includes_granularity: Literal["state"]
    includes: tuple[str, ...]
    year: int
    granularity: Literal["state"] = field(init=False, default="state")

    @classmethod
    def all(cls, year: int) -> Self:
        """Create a scope including all US states and state-equivalents."""
        return cls(
            includes_granularity="state",
            includes=tuple(get_states(year).geoid),
            year=year,
        )

    def _compute_node_ids(self) -> NDArray[np.str_]:
        # return all nodes
        return np.array(self.includes, dtype=np.str_)

    def is_all_states(self) -> bool:
        return np.array_equal(get_states(self.year).geoid, self.node_ids)

    def raise_granularity(self) -> Never:
        raise GeographyError("No granularity higher than state.")

    def lower_granularity(self) -> "CountyScope":
        """Create and return CountyScope object with identical properties."""
        return CountyScope(
            includes_granularity=self.includes_granularity,
            includes=self.includes,
            year=self.year,
        )

    @property
    def select(self) -> "StateSelector[StateScope, StateSelection]":
        return StateSelector(self, StateSelection)

    @property
    @override
    def labels_option(self) -> NDArray[np.str_]:
        mapping = get_states(self.year).state_fips_to_code
        return np.array([mapping[x] for x in self.node_ids], dtype=np.str_)


@dataclass(frozen=True)
class CountyScope(_InStatesMixin, _InCountiesMixin, CensusScope):
    """GeoScope at the County granularity."""

    includes_granularity: Literal["state", "county"]
    includes: tuple[str, ...]
    year: int
    granularity: Literal["county"] = field(init=False, default="county")

    def raise_granularity(self) -> StateScope:
        """Create and return StateScope object with identical properties."""
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

    def lower_granularity(self) -> "TractScope":
        """Create and return TractScope object with identical properties."""
        return TractScope(
            includes_granularity=self.includes_granularity,
            includes=self.includes,
            year=self.year,
        )

    @property
    def select(self) -> "CountySelector[CountyScope, CountySelection]":
        return CountySelector(self, CountySelection)

    @property
    @override
    def labels_option(self) -> NDArray[np.str_]:
        mapping = get_counties(self.year).county_fips_to_name
        return np.array([mapping[x] for x in self.node_ids], dtype=np.str_)


@dataclass(frozen=True)
class TractScope(_InStatesMixin, _InCountiesMixin, _InTractsMixin, CensusScope):
    """GeoScope at the Tract granularity."""

    includes_granularity: Literal["state", "county", "tract"]
    includes: tuple[str, ...]
    year: int
    granularity: Literal["tract"] = field(init=False, default="tract")

    @override
    def raise_granularity(self) -> CountyScope:
        """Create and return CountyScope object with identical properties."""
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
        """Create and return BlockGroupScope object with identical properties."""
        return BlockGroupScope(
            year=self.year,
            includes_granularity=self.includes_granularity,
            includes=self.includes,
        )

    @property
    def select(self) -> "TractSelector[TractScope, TractSelection]":
        return TractSelector(self, TractSelection)


@dataclass(frozen=True)
class BlockGroupScope(
    _InStatesMixin, _InCountiesMixin, _InTractsMixin, _InBlockGroupsMixin, CensusScope
):
    """GeoScope at the Block Group granularity."""

    includes_granularity: Literal["state", "county", "tract", "block group"]
    includes: tuple[str, ...]
    year: int
    granularity: Literal["block group"] = field(init=False, default="block group")

    @override
    def raise_granularity(self) -> TractScope:
        """Create and return TractScope object with identical properties."""
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
        return BlockGroupSelector(self, BlockGroupSelection)


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
class StateSelection(GeoSelection[CensusScope]):
    pass


@dataclass(frozen=True)
class CountySelection(GeoSelection[CensusScope]):
    def group_by(self, grouping: Literal["state"]) -> GeoGroup[CensusScope]:
        return GeoGroup(
            self.scope,
            self.selection,
            CensusGrouping(grouping),
        )


@dataclass(frozen=True)
class TractSelection(GeoSelection[CensusScope]):
    def group_by(self, grouping: Literal["state", "county"]) -> GeoGroup[CensusScope]:
        return GeoGroup(
            self.scope,
            self.selection,
            CensusGrouping(grouping),
        )


@dataclass(frozen=True)
class BlockGroupSelection(GeoSelection[CensusScope]):
    def group_by(
        self, grouping: Literal["state", "county", "tract"]
    ) -> GeoGroup[CensusScope]:
        return GeoGroup(
            self.scope,
            self.selection,
            CensusGrouping(grouping),
        )


@dataclass(frozen=True)
class CensusGrouping(GeoGrouping):
    granularity: CensusGranularityName

    @override
    def map(
        self,
        node_ids: NDArray[np.str_],
    ) -> NDArray[np.str_]:
        gran = CensusGranularity.of(self.granularity)
        # return gran.truncate_array(node_ids)
        return np.array([gran.truncate(g) for g in node_ids], dtype=np.str_)


# NOTE: these Selector classes form a hierarchy of shared functionality.
# With a StateScope, you can select by geoid and state.
# With a CountyScope, you can select by geoid, state, AND county.
# And so on.

S = TypeVar("S", bound=CensusScope)
"""The type of geo scope."""

X = TypeVar(
    "X", bound=(StateSelection | CountySelection | TractSelection | BlockGroupSelection)
)
"""The type of geo selection."""


@dataclass(frozen=True)
class CensusSelector(GeoSelector[S, X]):
    _scope: S
    """The original scope."""
    _selection_class: type[X]
    """The class of the selection produced."""
    _node_id_to_label: Mapping[str, str] | None = field(init=False, default=None)
    """The mapping to use for `node_id_to_label` (see: GeoStrategy)"""

    def by_geoid(self, *geoids: str) -> X:
        """Select nodes by GEOID. It is possible to select all nodes within
        a coarser granularity by passing those GEOIDs, however all GEOIDs
        given must be the same granularity."""
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
    ) -> X:
        """Select nodes by specifying a slice on the node indices."""
        return self._from_mask(mask(self._scope.nodes, slice(start, stop, step)))

    def by_indices(self, indices: list[int]) -> X:
        """Select nodes by specifying the node indices to include."""
        return self._from_mask(mask(self._scope.nodes, indices))


@dataclass(frozen=True)
class StateSelector(CensusSelector[S, X]):
    def by_state(self, *identifiers: str) -> X:
        """Select all nodes within a set of states."""
        states = get_states(self._scope.year)
        geoids = states.interpret(identifiers)
        return self.by_geoid(*geoids)


@dataclass(frozen=True)
class CountySelector(StateSelector[S, X]):
    def by_county(self, *identifiers: str) -> X:
        """Select all nodes within a set of counties."""
        counties = get_counties(self._scope.year)
        geoids = counties.interpret(identifiers)
        return self.by_geoid(*geoids)


@dataclass(frozen=True)
class TractSelector(CountySelector[S, X]):
    def by_tract(self, *identifiers: str) -> X:
        """Select all nodes within a set of tracts."""
        tracts = get_tracts(self._scope.year)
        geoids = tracts.interpret(identifiers)
        return self.by_geoid(*geoids)


@dataclass(frozen=True)
class BlockGroupSelector(TractSelector[S, X]):
    def by_block_group(self, *identifiers: str) -> X:
        """Select all nodes within a set of block groups."""
        block_groups = get_block_groups(self._scope.year)
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
