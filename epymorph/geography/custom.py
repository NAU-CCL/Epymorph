from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

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
    You simply specify a list of IDs, one for each node in the scope.
    The order in which you specify them will be the canonical node order.
    """

    _nodes: NDArray[np.str_]

    def __init__(self, nodes: NDArray[np.str_] | list[str]):
        if isinstance(nodes, list):
            nodes = np.array(nodes, dtype=np.str_)
        self._nodes = nodes

    @property
    @override
    def node_ids(self) -> NDArray[np.str_]:
        return self._nodes

    @property
    def select(self) -> "CustomSelector":
        return CustomSelector(self, CustomSelection)


@dataclass(frozen=True)
class CustomSelection(GeoSelection[CustomScope]):
    """A GeoSelection on a CustomScope."""

    def group(self, grouping: GeoGrouping) -> GeoGroup[CustomScope]:
        return GeoGroup(self.scope, self.selection, grouping)


@dataclass(frozen=True)
class CustomSelector(GeoSelector[CustomScope, CustomSelection]):
    """A GeoSelector for CustomScopes."""


@strategy_to_scope.register
def _custom_strategy_to_scope(
    scope: CustomScope,
    strategy: GeoStrategy[CustomScope],
) -> GeoScope:
    selected = scope.node_ids[strategy.selection]
    return CustomScope(selected)
