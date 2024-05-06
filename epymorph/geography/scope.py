from abc import abstractmethod
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class GeoScope(Protocol):
    """The common interface expected of all geo scopes."""

    @abstractmethod
    def get_node_ids(self) -> NDArray[np.str_]:
        """Retrieve the complete list of node IDs included in this scope."""
