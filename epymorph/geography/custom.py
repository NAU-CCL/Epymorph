"""Implements a generic geo scope capable of representing arbitrary geographies."""

from dataclasses import dataclass
from typing import Never

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.error import GeographyError
from epymorph.geography.scope import (
    GeoGroup,
    GeoGrouping,
    GeoScope,
    GeoSelection,
    GeoSelector,
    GeoStrategy,
    strategy_to_scope,
)


class CustomScope(GeoScope):
    """
    A scope with no logical connection to existing geographic systems.

    Parameters
    ----------
    nodes :
        The identifiers for all nodes in the scope.
    """

    _nodes: NDArray[np.str_]

    def __init__(self, nodes: NDArray[np.str_] | list[str]):
        if isinstance(nodes, list):
            nodes = np.array(nodes, dtype=np.str_)
        self._nodes = np.sort(np.unique(nodes))

    @property
    @override
    def node_ids(self) -> NDArray[np.str_]:
        return self._nodes

    @property
    def select(self) -> "CustomSelector":
        """Create a selection from this scope."""
        return CustomSelector(self, CustomSelection)

    @property
    @override
    def geography(self) -> Never:
        """Automatic geography is not supported for custom scopes."""
        raise GeographyError("Geography is not possible for CustomScopes.")


@dataclass(frozen=True)
class CustomSelection(GeoSelection[CustomScope]):
    """
    A `GeoSelection` on a `CustomScope`.

    Typically you will create one of these by calling methods on a `GeoSelector`
    instance.
    """

    def group(self, grouping: GeoGrouping) -> GeoGroup[CustomScope]:
        """
        Group the geo axis using the specified grouping.

        Parameters
        ----------
        grouping :
            The grouping to use.

        Returns
        -------
        :
            The grouping strategy.
        """
        return GeoGroup(self.scope, self.selection, grouping)


@dataclass(frozen=True)
class CustomSelector(GeoSelector[CustomScope, CustomSelection]):
    """A `GeoSelector` for `CustomScopes`."""


@strategy_to_scope.register
def _custom_strategy_to_scope(
    scope: CustomScope,
    strategy: GeoStrategy[CustomScope],
) -> GeoScope:
    selected = scope.node_ids[strategy.selection]
    return CustomScope(selected)
