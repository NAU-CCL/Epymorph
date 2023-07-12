"""
NOTE: This Initializer system is currently a prototype. It has to be manually integrated into an IPM to work,
and the only IPM that does this so far is the SDH IPM.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from epymorph.context import SimContext
from epymorph.util import Compartments


def get_param(ctx: SimContext, name: str, override_value: Any | None = None) -> Any:
    """
    Get a parameter from the context, unless an override value is provided.
    If we need to take the parameter from the context and it's not provided,
    throw an exception.
    """
    if override_value is not None:
        return override_value
    elif name not in ctx.param:
        raise Exception(f"params missing {name}")
    else:
        return ctx.param[name]


class Initializer(ABC):
    """Abstract class responsible for initializing the compartments at the start of the simulation."""
    @abstractmethod
    def apply(self, ctx: SimContext, out: list[Compartments]) -> None:
        """
        The arguments are:
        1. the sim context
        2. a "template" of compartments for each location;
        each location starts with its entire population in the first compartment (susceptible).
        your function should modify this template in-place, it is the output value.
        """
        pass


class DefaultInitializer(Initializer):
    """Default initialization logic: infect a fixed number of people at a single location (by index)."""
    infection_seed_size: int | None
    infection_seed_loc: int | None

    def __init__(self, infection_seed_size: int | None = None, infection_seed_loc: int | None = None):
        self.infection_seed_size = infection_seed_size
        self.infection_seed_loc = infection_seed_loc

    def apply(self, ctx: SimContext, out: list[Compartments]) -> None:
        # Seed an infection in one location.
        si = get_param(ctx, "infection_seed_loc", self.infection_seed_loc)
        sn = get_param(ctx, "infection_seed_size", self.infection_seed_size)
        out[si][0] -= sn
        out[si][1] += sn


class SelectionInitializer(Initializer, ABC):
    """
    Abstract initialization logic: infect a fixed number of people distributed (proportional to their population)
    across a selection of nodes. Implementations provide the selection logic.
    A multivariate hypergeometric draw using the available populations is used to distribute to infections.
    """
    infection_seed_size: int | None

    def __init__(self, infection_seed_size: int | None = None):
        self.infection_seed_size = infection_seed_size

    def apply(self, ctx: SimContext, out: list[Compartments]) -> None:
        size = get_param(ctx, "infection_seed_size", self.infection_seed_size)
        # Seed an infection in `locs` random locations
        indices = self.select(ctx)
        pops = ctx.geo['population'][indices]
        available = pops.sum()
        if available < size:
            raise Exception(f"{self.__class__.__name__} was asked to infect {size} people but only had {available} available")
        infected = ctx.rng.multivariate_hypergeometric(pops, size)
        for j, n in enumerate(infected):
            out[indices[j]][0] -= n
            out[indices[j]][1] += n
        # print([(i, c[1]) for i, c in enumerate(out) if c[1] > 0])

    @abstractmethod
    def select(self, ctx: SimContext) -> NDArray[np.int_]:
        """Given the context, return the indices of which nodes to infect."""
        pass


class IndexedLocsInitializer(SelectionInitializer):
    """Initializer to infect a fixed number of people across a specified selection of locations (by index)."""
    infection_locations: list[int] | None

    def __init__(self, infection_seed_size: int | None = None, infection_locations: list[int] | None = None):
        super().__init__(infection_seed_size)
        self.infection_locations = infection_locations

    def select(self, ctx: SimContext) -> NDArray[np.int_]:
        return get_param(ctx, "infection_locations", self.infection_locations)


class LabeledLocsInitializer(SelectionInitializer):
    """Initializer to infect a fixed number of people across a specified selection of locations (by label)."""
    infection_locations: list[str] | None

    def __init__(self, infection_seed_size: int | None = None, infection_locations: list[str] | None = None):
        super().__init__(infection_seed_size)
        self.infection_locations = infection_locations

    def select(self, ctx: SimContext) -> NDArray[np.int_]:
        labels = get_param(ctx, "infection_locations", self.infection_locations)
        return np.where(np.isin(ctx.geo['labels'], labels))[0]


class RandomLocsInitializer(SelectionInitializer):
    """Initializer to infect a fixed number of people across a fixed number of randomly chosen locations."""
    infection_locations: int | None

    def __init__(self, infection_seed_size: int | None = None, infection_locations: int | None = None):
        super().__init__(infection_seed_size)
        self.infection_locations = infection_locations

    def select(self, ctx: SimContext) -> NDArray[np.int_]:
        num_locs = get_param(ctx, "infection_locations", self.infection_locations)
        return ctx.rng.integers(0, ctx.nodes, num_locs)


class TopLocsInitializer(SelectionInitializer):
    """
    Initializer to infect a fixed number of people across a fixed number of locations,
    selecting the top locations as measured by a given geo attribute.
    """
    infection_locations: int | None
    infection_attribute: str | None

    def __init__(self, infection_seed_size: int | None = None, infection_locations: int | None = None, infection_attribute: str | None = None):
        super().__init__(infection_seed_size)
        self.infection_locations = infection_locations
        self.infection_attribute = infection_attribute

    def select(self, ctx: SimContext) -> NDArray[np.int_]:
        num_locs = get_param(ctx, "infection_locations", self.infection_locations)
        attribute = get_param(ctx, "infection_attribute", self.infection_attribute)
        if attribute not in ctx.geo:
            raise Exception(f"{self.__class__.__name__} attempted to pick nodes by geo[{attribute}], but that attribute is missing.")
        arr = ctx.geo[attribute]
        # `argpartition` chops an array in two halves (yielding indices of the original array):
        # all indices whose values are smaller than the kth element,
        # followed by the index of the kth element,
        # followed by all indices whose values are larger.
        # So by using -k we create a partition of the largest k elements at the end of the array,
        # then slice using [-k:] to get just those indices.
        # This should be O(k log k) and saves us from copying+sorting (or some-such)
        selected = np.argpartition(arr, -num_locs)[-num_locs:]
        print(f"Selected {num_locs} locations by geo[{attribute}]:")
        print(f"   labels: {ctx.geo['labels'][selected]}")
        print(f"  indices: {selected}")
        print(f"   values: {arr[selected]}")
        return selected


class BottomLocsInitializer(SelectionInitializer):
    """
    Initializer to infect a fixed number of people across a fixed number of locations,
    selecting the bottom locations as measured by a given geo attribute.
    """
    infection_locations: int | None
    infection_attribute: str | None

    def __init__(self, infection_seed_size: int | None = None, infection_locations: int | None = None, infection_attribute: str | None = None):
        super().__init__(infection_seed_size)
        self.infection_locations = infection_locations
        self.infection_attribute = infection_attribute

    def select(self, ctx: SimContext) -> NDArray[np.int_]:
        num_locs = get_param(ctx, "infection_locations", self.infection_locations)
        attribute = get_param(ctx, "infection_attribute", self.infection_attribute)
        if attribute not in ctx.geo:
            raise Exception(f"{self.__class__.__name__} attempted to pick nodes by geo[{attribute}], but that attribute is missing.")
        arr = ctx.geo[attribute]
        # `argpartition` chops an array in two halves (yielding indices of the original array):
        # all indices whose values are smaller than the kth element,
        # followed by the index of the kth element,
        # followed by all indices whose values are larger.
        # So by using k we create a partition of the smallest k elements at the start of the array,
        # then slice using [:k] to get just those indices.
        # This should be O(k log k) and saves us from copying+sorting (or some-such)
        selected = np.argpartition(arr, num_locs)[:num_locs]
        print(f"Selected {num_locs} locations by geo[{attribute}]:")
        print(f"   labels: {ctx.geo['labels'][selected]}")
        print(f"  indices: {selected}")
        print(f"   values: {arr[selected]}")
        return selected
