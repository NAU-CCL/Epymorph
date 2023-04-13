from abc import ABC
from typing import Callable, TypeVar

import numpy as np
from numpy.typing import NDArray

from epymorph.clock import Tick

# NOTE: I'm struggling a bit with how best to represent the geo model.
# On the one hand, it's most flexible just to give direct access to the underlying
# NDArrays. However this doesn't provide anything in the way of call-site guidance
# on how to correctly access the data.
# (Is this a 1D or 2D array? If it's time-series which axis is time and in what increment?)
# And with this approach I might be stretching Python's type checking abilities.
# So caveat emptor: these interfaces are far from solid.

# TODO: would be ideal if these GeoParam classes were generic in the
# foundational type (int, double, etc.). Tried it but ran into some issues.

T = TypeVar('T')

# A 1D array of data by node.
ParamN = Callable[[int], T]
# A time-series array of data by node.
ParamNT = Callable[[int, Tick], T]
# An matrix of data node-by-node.
ParamNN = Callable[[int, int], T]


class GeoParam(ABC):
    def __init__(self, name: str):
        self.name = name


class GeoParamN(GeoParam):
    """Parameter by node."""

    def __init__(self, name: str, data: NDArray):
        super().__init__(name)
        self.data = data

    def __call__(self, node_idx: int):
        return self.data[node_idx]


class GeoParamNT(GeoParam):
    """Parameter by node which is time-series by day."""

    def __init__(self, name: str, data: NDArray):
        super().__init__(name)
        self.data = data

    def __call__(self, node_idx: int, tick: Tick):
        return self.data[tick.day][node_idx]


class GeoParamNN(GeoParam):
    """Parameter by pairwise nodes."""

    def __init__(self, name: str, data: NDArray):
        super().__init__(name)
        self.data = data

    def __call__(self, node1_idx: int, node2_idx):
        return self.data[node1_idx][node2_idx]


class Geo:
    """The geo model is a collection of parameters organized by nodes."""

    # TODO: there is some work to do reconciling time-series parameters
    # with the calendar run-time of the simulation. At the very least
    # we should check whether or not we have sufficient values
    # for the duration of the simulation. But we probably also need to know
    # the calendar date on which each time-series parameter begins, and check
    # with that knowledge.

    def __init__(self, node_labels: list[str], params: list[GeoParam]):
        self.node_labels = node_labels
        self.num_nodes = len(node_labels)
        self.params = params

    def get_param(self, name: str) -> GeoParam:
        p = next((x for x in self.params if x.name == name), None)
        if p == None:
            raise Exception(f"No such parameter: {name}")
        return p

    def get_paramn(self, name: str, dtype) -> GeoParamN:
        p = next((x for x in self.params if x.name == name), None)
        if p == None:
            raise Exception(f"No such parameter (by name): {name}")
        if not isinstance(p, GeoParamN):
            raise Exception(f"No such parameter (by class): {name}")
        if not p.data.dtype == dtype:
            raise Exception(f"No such parameter (by type): {name}")
        return p

    def get_paramnt(self, name: str, dtype) -> GeoParamNT:
        p = next((x for x in self.params if x.name == name), None)
        if p == None:
            raise Exception(f"No such parameter (by name): {name}")
        if not isinstance(p, GeoParamNT):
            raise Exception(f"No such parameter (by class): {name}")
        if not p.data.dtype == dtype:
            raise Exception(f"No such parameter (by type): {name}")
        return p

    def get_paramnn(self, name: str, dtype) -> GeoParamNN:
        p = next((x for x in self.params if x.name == name), None)
        if p == None:
            raise Exception(f"No such parameter (by name): {name}")
        if not isinstance(p, GeoParamNN):
            raise Exception(f"No such parameter (by class): {name}")
        if not p.data.dtype == dtype:
            raise Exception(f"No such parameter (by type): {name}")
        return p
