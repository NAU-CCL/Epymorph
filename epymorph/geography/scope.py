from abc import abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class GeoScope(Protocol):
    """The common interface expected of all geo scopes."""

    @property
    def nodes(self) -> int:
        """The number of nodes in this scope."""
        return len(self.get_node_ids())

    @abstractmethod
    def get_node_ids(self) -> NDArray[np.str_]:
        """Retrieve the complete list of node IDs included in this scope."""


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

    def get_node_ids(self) -> NDArray[np.str_]:
        return self._nodes


class ScopeFilter(GeoScope):
    """
    A scope for filtering out specific nodes from another scope instance.
    """
    # NOTE: I'm not convinced this is the _best_ way to do this but it works for now.

    _parent: GeoScope
    _nodes: NDArray[np.str_]

    def __init__(self, parent: GeoScope, remove: NDArray[np.str_]):
        self._parent = parent
        orig_nodes = parent.get_node_ids()
        self._nodes = orig_nodes[~np.isin(orig_nodes, remove)]

    def get_node_ids(self) -> NDArray[np.str_]:
        return self._nodes
