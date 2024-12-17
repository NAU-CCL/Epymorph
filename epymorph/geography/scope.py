from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import singledispatch
from typing import Generic, Literal, Protocol, TypeVar, final, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class GeoScope(Protocol):
    """The common interface expected of all geo scopes."""

    @property
    @abstractmethod
    def node_ids(self) -> NDArray[np.str_]:
        """Retrieve the complete list of node IDs included in this scope."""

    @property
    def nodes(self) -> int:
        """The number of nodes in this scope."""
        return len(self.node_ids)

    @property
    def labels_option(self) -> NDArray[np.str_] | None:
        """An optional text label for each node. If this returns None,
        you should use the node_ids as the best labels."""
        # NOTE: override this method if friendly names are possible
        return None

    @property
    def labels(self) -> NDArray[np.str_]:
        """The best text label for each node.
        (This uses `labels_option` if available and falls back to `node_ids`.)"""
        if (labels := self.labels_option) is not None:
            return labels
        return self.node_ids


#############################################
# Geo scope quantity select/group/aggregate #
#############################################


D = TypeVar("D", bound=np.generic)
GeoScopeT = TypeVar("GeoScopeT", bound=GeoScope)
GeoScopeT_co = TypeVar("GeoScopeT_co", covariant=True, bound=GeoScope)
GeoScopeT_contra = TypeVar("GeoScopeT_contra", contravariant=True, bound=GeoScope)
GeoAggMethod = Literal["sum", "min", "max"]


@dataclass(frozen=True)
class GeoStrategy(ABC, Generic[GeoScopeT_co]):
    """A strategy for dealing with the spatial axis, e.g., in processing results.

    Strategies can include selection of a subset, grouping, and aggregation."""

    scope: GeoScopeT_co
    """The original scope."""
    selection: NDArray[np.bool_]
    """A boolean mask for selection of a subset of geo nodes."""
    grouping: "GeoGrouping | None"
    """A method for grouping geo nodes."""
    aggregation: GeoAggMethod | None
    """A method for aggregating by group
    (if no grouping is specified, selected nodes are treated as one group)."""

    @property
    def indices(self) -> tuple[int, ...]:
        """Get a tuple of indices for each selected node."""
        return tuple(i for i, selected in enumerate(self.selection) if selected)

    @final
    def to_scope(self) -> GeoScope:
        """Returns the scope that results from applying this strategy."""
        return strategy_to_scope(self.scope, self)


@singledispatch
def strategy_to_scope(scope: GeoScopeT, strategy: GeoStrategy[GeoScopeT]) -> GeoScope:
    raise NotImplementedError()


class GeoGrouping(ABC):
    """Defines a geo-axis grouping scheme. This is essentially a function that maps
    the simulation geo axis info (node IDs) into a new series which describes
    the group membership of each geo axis row.
    Certain groupings may only be valid for specific types of GeoScope."""

    @abstractmethod
    def map(self, node_ids: NDArray[np.str_]) -> NDArray[np.str_]:
        """Produce a column that describes the group membership of each "row",
        where rows of the geo axis are described by their geoid,
        producing a unique value per group. This column will be used as the basis
        of a `groupby` operation. The result must correspond element-wise to the given
        `node_ids` array."""


class _CanGeoAggregate(GeoStrategy[GeoScopeT_co]):
    def agg(self, agg: Literal["sum", "min", "max"]) -> "GeoAggregation[GeoScopeT_co]":
        """Apply the named aggregation for each geo node group."""
        return GeoAggregation(self.scope, self.selection, self.grouping, agg)

    def sum(self) -> "GeoAggregation[GeoScopeT_co]":
        """Perform a sum for each geo node group."""
        return self.agg("sum")

    def min(self) -> "GeoAggregation[GeoScopeT_co]":
        """Take the min value for each geo node group."""
        return self.agg("min")

    def max(self) -> "GeoAggregation[GeoScopeT_co]":
        """Take the max value for each geo node group."""
        return self.agg("max")


@dataclass(frozen=True)
class GeoSelection(_CanGeoAggregate[GeoScopeT_co], GeoStrategy[GeoScopeT_co]):
    """Describes a sub-selection operation on a geo scope
    (no grouping or aggregation)."""

    scope: GeoScopeT_co
    """The original scope."""
    selection: NDArray[np.bool_]
    """A boolean mask for selection of a subset of geo nodes."""
    grouping: None = field(init=False, default=None)
    """A method for grouping geo nodes."""
    aggregation: None = field(init=False, default=None)
    """A method for aggregating by group
    (if no grouping is specified, selected nodes are treated as one group)."""

    # NOTE: subclass this to provide appropriate grouping methods.


@dataclass(frozen=True)
class GeoGroup(_CanGeoAggregate[GeoScopeT_co], GeoStrategy[GeoScopeT_co]):
    """Describes a grouping operation on a geo scope,
    with an optional sub-selection."""

    scope: GeoScopeT_co
    """The original scope."""
    selection: NDArray[np.bool_]
    """A boolean mask for selection of a subset of geo nodes."""
    grouping: GeoGrouping
    """A method for grouping geo nodes."""
    aggregation: None = field(init=False, default=None)
    """A method for aggregating by group
    (if no grouping is specified, selected nodes are treated as one group)."""


@dataclass(frozen=True)
class GeoAggregation(GeoStrategy[GeoScopeT_co]):
    """Describes a group-and-aggregate operation on a geo scope,
    with an optional sub-selection."""

    scope: GeoScopeT_co
    """The original scope."""
    selection: NDArray[np.bool_]
    """A boolean mask for selection of a subset of geo nodes."""
    grouping: GeoGrouping | None
    """A method for grouping geo nodes."""
    aggregation: GeoAggMethod
    """A method for aggregating by group
    (if no grouping is specified, selected nodes are treated as one group)."""


GeoSelectionT_co = TypeVar("GeoSelectionT_co", covariant=True, bound=GeoSelection)
"""The type of geo selection."""


@dataclass(frozen=True)
class GeoSelector(Generic[GeoScopeT_co, GeoSelectionT_co]):
    """A utility class for making a selection on a particular kind of GeoScope."""

    _scope: GeoScopeT_co
    """The original scope."""
    _selection_class: type[GeoSelectionT_co]
    """The class of the selection produced."""

    def _from_mask(self, mask: NDArray[np.bool_]) -> GeoSelectionT_co:
        """Construct a geo selection instance of the proper type given a mask."""
        return self._selection_class(self._scope, mask)

    def all(self) -> GeoSelectionT_co:
        """Select all geo nodes."""
        mask = np.ones_like(self._scope.node_ids, dtype=np.bool_)
        return self._from_mask(mask)

    def by_id(self, *ids: str) -> GeoSelectionT_co:
        """Select geo nodes by their ID (exact matches only)."""
        mask = np.zeros_like(self._scope.node_ids, dtype=np.bool_)
        for i, node in enumerate(self._scope.node_ids):
            mask[i] = any(node == sel for sel in ids)
        return self._from_mask(mask)


__all__ = [
    "GeoScope",
    "GeoStrategy",
    "strategy_to_scope",
    "GeoGrouping",
    "GeoSelection",
    "GeoGroup",
    "GeoAggregation",
    "GeoSelector",
]
