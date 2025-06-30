"""
The geographic scope of a simulation describes the locations which are included in the
model. Each location is a discrete place. While interactions between locations may be
described by model data or arise from the movements of individuals, the geo scope itself
is agnostic to these dynamics.

`GeoScope` is generic, modeling geography in a very abstract way. Specialized classes
exist to describe geographic systems that model real world geography, e.g., the system
of delineations defined by the US Census Bureau.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import singledispatch
from typing import Generic, Literal, Never, Protocol, TypeVar, final, runtime_checkable

import numpy as np
from geopandas import GeoDataFrame
from numpy.typing import NDArray


@runtime_checkable
class GeoScope(Protocol):
    """The common interface expected of all geo scopes."""

    @property
    @abstractmethod
    def node_ids(self) -> NDArray[np.str_]:
        """The list of node IDs included in this scope."""

    @property
    def nodes(self) -> int:
        """The number of nodes in this scope."""
        return len(self.node_ids)

    def index_of(self, node_id: str) -> int:
        """
        Return the index of a given node by its ID string.

        Parameters
        ----------
        node_id :
            The ID to check for.

        Returns
        -------
        :
            The index if `node_id` exists in this scope.

        Raises
        ------
        ValueError
            If `node_id` is not found.
        """
        idxs, *_ = np.where(self.node_ids == node_id)
        if len(idxs) == 0:
            raise ValueError(f"'{node_id}' not present in geo scope.")
        return idxs[0]

    @property
    def labels_option(self) -> NDArray[np.str_] | None:
        """
        An optional text label for each node. If this returns `None`,
        convention is to use the node IDs as the labels.
        """
        # NOTE: override this method if friendly names are possible
        return None

    @property
    def labels(self) -> NDArray[np.str_]:
        """
        The best text label for each node.

        This uses `labels_option` if available and falls back to `node_ids`.
        """
        if (labels := self.labels_option) is not None:
            return labels
        return self.node_ids

    @property
    @abstractmethod
    def geography(self) -> GeoDataFrame | Never:
        """
        Retrieves the shapes corresponding to the nodes of this scope as a
        `GeoDataFrame`. Note that this is not possible for all types of `GeoScope`.

        Returns
        -------
        :
            The geography.

        Raises
        ------
        GeographyError
            If we cannot fetch geography for this type of scope.
        """
        # NOTE: implementations should:
        # - if geography is NOT supported:
        #   - override the return type to `Never`
        # - if geography IS supported:
        #   - override the return type to `GeoDataFrame`
        #   - use @cached_property as repeated access of the geography is expected;
        #     however Pylance errors this so you can mark it `typed: ignore[override]`


#############################################
# Geo scope quantity select/group/aggregate #
#############################################


GeoScopeT = TypeVar("GeoScopeT", bound=GeoScope)
"""A type of `GeoScope`."""
GeoScopeT_co = TypeVar("GeoScopeT_co", covariant=True, bound=GeoScope)
"""A type of `GeoScope`: covariant."""

GeoAggMethod = Literal["sum", "min", "max"]
"""Methods for aggregating results along the geo axis."""


@dataclass(frozen=True)
class GeoStrategy(ABC, Generic[GeoScopeT_co]):
    """
    A strategy for dealing with the spatial axis, e.g., in processing results.
    Strategies can include selection of a subset, grouping, and aggregation.

    Typically you will create one of these buy calling methods on a `GeoSelector`
    instance.

    `GeoStrategy` is generic in the type of `GeoScope` it works with (`GeoScopeT_co`).
    """

    scope: GeoScopeT_co
    """The original scope."""
    selection: NDArray[np.bool_]
    """A boolean mask for selection of a subset of geo nodes."""
    grouping: "GeoGrouping | None"
    """A method for grouping geo nodes."""
    aggregation: GeoAggMethod | None
    """
    A method for aggregating by group
    (if no grouping is specified, selected nodes are treated as one group).
    """

    @property
    def indices(self) -> tuple[int, ...]:
        """
        A tuple containing the indices from the original scope that are selected by this
        strategy.
        """
        return tuple(i for i, selected in enumerate(self.selection) if selected)

    @final
    def to_scope(self) -> GeoScope:
        """
        Convert this strategy to the scope that results from applying this strategy.

        For example, if your original scope included all tracts in the US and the
        strategy selects Arizona and groups by county, this method would return a scope
        containing all counties in Arizona.

        Returns
        -------
        :
            A new scope instance.

        Raises
        ------
        NotImplementedError
            If this type of scope does not support this operation.
        """
        # NOTE: in order to support this, you must provide a singledispatch
        # implementation of `strategy_to_scope` that covers the GeoScope subtype.
        return strategy_to_scope(self.scope, self)


@singledispatch
def strategy_to_scope(scope: GeoScopeT, strategy: GeoStrategy[GeoScopeT]) -> GeoScope:
    """
    Convert a `GeoStrategy` instance to the `GeoScope` that would result from
    applying the strategy.

    Warning
    -------
    This function is intended for epymorph's internal use; users should instead call
    `to_scope` on a `GeoStrategy` object.

    Parameters
    ----------
    scope :
        The original scope.
    strategy :
        The strategy to apply.

    Returns
    -------
    :
        The new scope instance.

    Raises
    ------
    NotImplementedError
        If this type of scope does not support this operation.
    """
    # This is a singledispatch function so scope implementations can opt to support
    # this feature if it makes sense.

    # NOTE: it might seem unnecessary to pass `scope` and `strategy` separately since
    # the strategy contains the scope. However, singledispatch mechanics can't use
    # the strategy's generic type declaration as the basis for dispatch, so we have
    # to pass the scope as the first argument.
    # In usage, be careful not to pass arguments that disagree with each other.
    raise NotImplementedError()


class GeoGrouping(ABC):
    """
    Defines a geo-axis grouping scheme. This is essentially a function that maps
    the simulation geo axis info (node IDs) into a new series which describes
    the group membership of each geo axis row.

    Certain groupings may only be valid for specific types of `GeoScope`.
    """

    @abstractmethod
    def map(self, node_ids: NDArray[np.str_]) -> NDArray[np.str_]:
        """
        Produce a column that describes the group membership of each node.

        The returned column will be used as the basis of a `groupby` operation.

        Parameters
        ----------
        node_ids :
            The node IDs to group.

        Returns
        -------
        :
            An array of the same length as `node_ids` where each value defines which
            group the original node ID belongs to.
        """


class _CanGeoAggregate(GeoStrategy[GeoScopeT_co]):
    """Base functionality for geo strategies that support aggregation operations."""

    def agg(self, agg: GeoAggMethod) -> "GeoAggregation[GeoScopeT_co]":
        """
        Apply the named aggregation for each geo node group.

        Parameters
        ----------
        agg :
            The aggregation to apply.

        Returns
        -------
        :
            The geo aggregation.
        """
        return GeoAggregation(self.scope, self.selection, self.grouping, agg)

    def sum(self) -> "GeoAggregation[GeoScopeT_co]":
        """
        Perform a sum for each geo node group.

        Returns
        -------
        :
            The geo aggregation.
        """
        return self.agg("sum")

    def min(self) -> "GeoAggregation[GeoScopeT_co]":
        """
        Take the min value for each geo node group.

        Returns
        -------
        :
            The geo aggregation.
        """
        return self.agg("min")

    def max(self) -> "GeoAggregation[GeoScopeT_co]":
        """
        Take the max value for each geo node group.

        Returns
        -------
        :
            The geo aggregation.
        """
        return self.agg("max")


@dataclass(frozen=True)
class GeoSelection(_CanGeoAggregate[GeoScopeT_co], GeoStrategy[GeoScopeT_co]):
    """
    A kind of `GeoStrategy` describing a sub-selection operation on a geo scope.
    A selection performs no grouping or aggregation.

    Typically you will create one of these by calling methods on a `GeoSelector`
    instance.

    Parameters
    ----------
    scope :
        The original scope.
    selection :
        A boolean mask for selection of a subset of geo nodes.
    """

    scope: GeoScopeT_co
    """The original scope."""
    selection: NDArray[np.bool_]
    """A boolean mask for selection of a subset of geo nodes."""
    grouping: None = field(init=False, default=None)
    """A method for grouping geo nodes."""
    aggregation: None = field(init=False, default=None)
    """
    A method for aggregating by group
    (if no grouping is specified, selected nodes are treated as one group).
    """

    # NOTE: subclass this to provide appropriate grouping methods.


@dataclass(frozen=True)
class GeoGroup(_CanGeoAggregate[GeoScopeT_co], GeoStrategy[GeoScopeT_co]):
    """
    A kind of `GeoStrategy` describing a grouping operation on a geo scope,
    with an optional sub-selection.

    Typically you will create one of these by calling methods on a `GeoSelection`
    instance.

    Parameters
    ----------
    scope :
        The original scope.
    selection :
        A boolean mask for selection of a subset of geo nodes.
    grouping :
        A method for grouping geo nodes.
    """

    scope: GeoScopeT_co
    """The original scope."""
    selection: NDArray[np.bool_]
    """A boolean mask for selection of a subset of geo nodes."""
    grouping: GeoGrouping
    """A method for grouping geo nodes."""
    aggregation: None = field(init=False, default=None)
    """
    A method for aggregating by group
    (if no grouping is specified, selected nodes are treated as one group).
    """


@dataclass(frozen=True)
class GeoAggregation(GeoStrategy[GeoScopeT_co]):
    """
    Describes a group-and-aggregate operation on a geo scope,
    with an optional sub-selection.

    Typically you will create one of these by calling methods on a `GeoSelection`
    instance.

    Parameters
    ----------
    scope :
        The original scope.
    selection :
        A boolean mask for selection of a subset of geo nodes.
    grouping :
        A method for grouping geo nodes.
    aggregation :
        A method for aggregating by group
        (if no grouping is specified, selected nodes are treated as one group).
    """

    scope: GeoScopeT_co
    """The original scope."""
    selection: NDArray[np.bool_]
    """A boolean mask for selection of a subset of geo nodes."""
    grouping: GeoGrouping | None
    """A method for grouping geo nodes."""
    aggregation: GeoAggMethod
    """
    A method for aggregating by group
    (if no grouping is specified, selected nodes are treated as one group).
    """


GeoSelectionT_co = TypeVar("GeoSelectionT_co", covariant=True, bound=GeoSelection)
"""The type of geo selection."""


@dataclass(frozen=True)
class GeoSelector(Generic[GeoScopeT_co, GeoSelectionT_co]):
    """
    A utility class for making a selection on a particular kind of GeoScope.
    Most of the time you obtain one of these using a `GeoScope`'s `select` property.
    """

    _scope: GeoScopeT_co
    """The original scope."""
    _selection_class: type[GeoSelectionT_co]
    """The class of the selection produced."""

    def _from_mask(self, mask: NDArray[np.bool_]) -> GeoSelectionT_co:
        """Construct a geo selection instance of the proper type given a mask."""
        return self._selection_class(self._scope, mask)

    def all(self) -> GeoSelectionT_co:
        """
        Select all geo nodes.

        Returns
        -------
        :
            The geo selection.
        """
        mask = np.ones_like(self._scope.node_ids, dtype=np.bool_)
        return self._from_mask(mask)

    def by_id(self, *ids: str) -> GeoSelectionT_co:
        """
        Select geo nodes by their ID (exact matches only).

        Parameters
        ----------
        *ids :
            Node IDs to include in the selection, as var-args.

        Returns
        -------
        :
            The geo selection.
        """
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
