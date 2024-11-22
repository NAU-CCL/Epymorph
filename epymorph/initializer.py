"""
Initializers are responsible for setting up the initial conditions for the simulation:
the populations at each node, and which disease compartments they belong to.
There are potentially many ways to do this, driven by source data from
the geo or simulation parameters, so this module provides a uniform interface
to accomplish the task, as well as a few common implementations.
"""

from abc import ABC
from typing import cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.data_shape import DataShapeMatcher, Shapes, SimDimensions
from epymorph.data_type import SimArray, SimDType
from epymorph.error import InitException
from epymorph.simulation import (
    AttributeDef,
    SimulationFunction,
)
from epymorph.util import NumpyTypeError, check_ndarray, match


class Initializer(SimulationFunction[SimArray], ABC):
    """
    Represents an initialization routine responsible for determining the initial values
    of populations by IPM compartment for every simulation node.
    """

    @override
    def validate(self, result) -> None:
        # Must be an NxC array of integers, none less than zero.
        try:
            check_ndarray(
                result,
                dtype=match.dtype(SimDType),
                shape=DataShapeMatcher(Shapes.NxC, self.dim, allow_broadcast=False),
            )
        except NumpyTypeError as e:
            msg = f"Invalid return type from '{self.__class__.__name__}'"
            raise InitException(msg) from e

        if np.min(result) < 0:
            msg = (
                f"Initializer '{self.__class__.__name__}' returned "
                "values less than zero."
            )
            raise InitException(msg)


# Initializer utility functions


def _empty_array(dim: SimDimensions) -> SimArray:
    return np.zeros((dim.nodes, dim.compartments), dtype=SimDType)


def _pop_array(dim: SimDimensions, pop: SimArray, initial_compartment: int) -> SimArray:
    array = _empty_array(dim)
    array[:, initial_compartment] = pop
    return array


def _transition_1(
    array: SimArray, from_compartment: int, to_compartment: int, row: int, count: int
) -> None:
    array[row, from_compartment] -= count
    array[row, to_compartment] += count


def _transition_N(
    array: SimArray, from_compartment: int, to_compartment: int, count: SimArray
) -> None:
    array[:, from_compartment] -= count
    array[:, to_compartment] += count


# Pre-baked initializer implementations


_POPULATION_ATTR = AttributeDef(
    "population", int, Shapes.N, comment="The population at each geo node."
)

_LABEL_ATTR = AttributeDef(
    "label", str, Shapes.N, comment="A label associated with each geo node."
)


class NoInfection(Initializer):
    """
    An initializer that places all 'population' individuals in a single compartment
    (default: the first compartment).
    """

    requirements = (_POPULATION_ATTR,)

    initial_compartment: int
    """The IPM compartment index where people should start."""

    def __init__(self, initial_compartment: int = 0):
        self.initial_compartment = initial_compartment

    def evaluate(self) -> SimArray:
        pop = self.data(_POPULATION_ATTR)
        return _pop_array(self.dim, pop, self.initial_compartment)


class Explicit(Initializer):
    """
    An initializer that sets all compartment populations directly.
    You provide the (N,C) array and we use a copy of it.
    """

    initials: SimArray
    """The initial compartment values to use."""

    def __init__(self, initials: SimArray):
        self.initials = initials

    def evaluate(self) -> SimArray:
        return self.initials.copy()


class Proportional(Initializer):
    """
    Set all compartments as a proportion of their population according to the geo.
    The parameter array provided to this initializer will be normalized, then multiplied
    element-wise by the geo population. So you're free to express the proportions
    in any way, (as long as no row is zero). Parameters:
    - `ratios` a (C,) or (N,C) numpy array describing the ratios for each compartment
    """

    requirements = (_POPULATION_ATTR,)

    ratios: NDArray[np.int64 | np.float64]
    """The initialization ratios to use."""

    def __init__(self, ratios: NDArray[np.int64 | np.float64]):
        self.ratios = ratios

    def evaluate(self) -> SimArray:
        try:
            check_ndarray(
                self.ratios,
                dtype=match.dtype(np.int64, np.float64),
                shape=DataShapeMatcher(Shapes.NxC, self.dim, True),
            )
        except NumpyTypeError as e:
            raise InitException(
                f"Initializer argument 'ratios' is not properly specified. {e}"
            ) from None

        try:
            ratios = Shapes.NxC.adapt(self.dim, self.ratios, True)
        except ValueError:
            raise InitException(
                "Initializer argument 'ratios' is not properly specified."
            )
        ratios = ratios.astype(np.float64, copy=False)
        row_sums = cast(NDArray[np.float64], np.sum(ratios, axis=1, dtype=np.float64))

        if np.any(row_sums <= 0):
            msg = "One or more rows sum to zero or less."
            raise InitException(msg)

        pop = self.data(_POPULATION_ATTR)
        result = pop[:, np.newaxis] * (ratios / row_sums[:, np.newaxis])
        return result.round().astype(SimDType, copy=True)


class SeededInfection(Initializer, ABC):
    """
    Abstract base class for initializers which seed an infection to start. We assume
    most people start out in a particular compartment (default: the first) and if chosen
    for infection, are placed in another compartment (default: the second). The user
    can identify which two compartments to use, but it can only be two compartments.
    Parameters:
    - `initial_compartment` which compartment (by index) is "not infected", where most
        individuals start out
    - `infection_compartment` which compartment (by index) will be seeded as the
        initial infection
    """

    DEFAULT_INITIAL = 0
    DEFAULT_INFECTION = 1

    initial_compartment: int
    """The IPM compartment index for non-infected individuals."""
    infection_compartment: int
    """The IPM compartment index for infected individuals."""

    def __init__(
        self,
        initial_compartment: int = DEFAULT_INITIAL,
        infection_compartment: int = DEFAULT_INFECTION,
    ):
        self.initial_compartment = initial_compartment
        self.infection_compartment = infection_compartment

    def validate_compartments(self):
        """
        Child classes should call this during `initialize` to check that the given
        compartment indices are valid.
        """
        C = self.dim.compartments
        if not 0 <= self.initial_compartment < C:
            msg = (
                "Initializer argument `initial_compartment` must be an index "
                f"of the IPM compartments (0 to {C-1}, inclusive)."
            )
            raise InitException(msg)
        if not 0 <= self.infection_compartment < C:
            msg = (
                "Initializer argument `infection_compartment` must be an index "
                f"of the IPM compartments (0 to {C-1}, inclusive)."
            )
            raise InitException(msg)


class IndexedLocations(SeededInfection):
    """
    Infect a fixed number of people distributed (proportional to their population)
    across a selection of nodes. A multivariate hypergeometric draw using the available
    populations is used to distribute infections. Parameters:
    - `selection` a one-dimensional array of indices;
        all values must be in range (-N,+N)
    - `seed_size` the number of individuals to infect in total
    """

    requirements = (_POPULATION_ATTR,)

    selection: NDArray[np.intp]
    """Which locations to infect."""
    seed_size: int
    """How many individuals to infect, randomly distributed to selected locations."""

    def __init__(
        self,
        selection: NDArray[np.intp],
        seed_size: int,
        initial_compartment: int = SeededInfection.DEFAULT_INITIAL,
        infection_compartment: int = SeededInfection.DEFAULT_INFECTION,
    ):
        super().__init__(initial_compartment, infection_compartment)

        if len(selection.shape) != 1:
            msg = "Initializer argument 'selection' must be a 1-dimensional array."
            raise InitException(msg)
        if seed_size < 0:
            msg = (
                "Initializer argument 'seed_size' must be a non-negative integer value."
            )
            raise InitException(msg)

        self.selection = selection
        self.seed_size = seed_size

    def evaluate(self) -> SimArray:
        N = self.dim.nodes
        sel = self.selection

        if not np.all((-N < sel) & (sel < N)):
            msg = (
                "Initializer argument 'selection' invalid: "
                f"some indices are out of range ({-N}, {N})."
            )
            raise InitException(msg)

        self.validate_compartments()

        pop = self.data(_POPULATION_ATTR)
        selected = pop[sel]
        available = selected.sum()
        if available < self.seed_size:
            msg = (
                f"Attempted to infect {self.seed_size} individuals "
                f"but only had {available} available."
            )
            raise InitException(msg)

        # Randomly select individuals from each of the selected locations.
        if len(selected) == 1:
            infected = np.array([self.seed_size], dtype=SimDType)
        else:
            infected = self.rng.multivariate_hypergeometric(selected, self.seed_size)

        result = _pop_array(self.dim, pop, self.initial_compartment)

        # Special case: the "no" IPM has only one compartment!
        # Technically it would be more "correct" to choose a different initializer,
        # but it's convenient to allow this special case for ease of testing.
        if self.dim.compartments == 1:
            return result

        for i, n in zip(sel, infected):
            _transition_1(
                result, self.initial_compartment, self.infection_compartment, i, n
            )
        return result


class SingleLocation(IndexedLocations):
    """
    Infect a fixed number of people at a single location (by index). Parameters:
    - `location` the index of the node in which to seed an initial infection
    - `seed_size` the number of individuals to infect in total
    """

    requirements = (_POPULATION_ATTR,)

    def __init__(
        self,
        location: int,
        seed_size: int,
        initial_compartment: int = SeededInfection.DEFAULT_INITIAL,
        infection_compartment: int = SeededInfection.DEFAULT_INFECTION,
    ):
        super().__init__(
            selection=np.array([location], dtype=np.intp),
            seed_size=seed_size,
            initial_compartment=initial_compartment,
            infection_compartment=infection_compartment,
        )

    def evaluate(self) -> SimArray:
        N = self.dim.nodes
        if not -N < self.selection[0] < N:
            msg = (
                "Initializer argument 'location' must be a valid index "
                f"to an array of {N} populations."
            )
            raise InitException(msg)
        return super().evaluate()


class LabeledLocations(SeededInfection):
    """
    Infect a fixed number of people distributed to a selection of locations (by label).
    Parameters:
    - `labels` the labels of the locations to select for infection
    - `seed_size` the number of individuals to infect in total
    """

    requirements = (_POPULATION_ATTR, _LABEL_ATTR)

    labels: NDArray[np.str_]
    """Which locations to infect."""
    seed_size: int
    """How many individuals to infect, randomly distributed to selected locations."""

    def __init__(
        self,
        labels: NDArray[np.str_],
        seed_size: int,
        initial_compartment: int = SeededInfection.DEFAULT_INITIAL,
        infection_compartment: int = SeededInfection.DEFAULT_INFECTION,
    ):
        super().__init__(initial_compartment, infection_compartment)
        self.labels = labels
        self.seed_size = seed_size

    def evaluate(self) -> SimArray:
        geo_labels = self.data(_LABEL_ATTR)

        if not np.all(np.isin(self.labels, geo_labels)):
            msg = (
                "Initializer argument 'labels' invalid: "
                "some labels are not in the geography."
            )
            raise InitException(msg)

        (selection,) = np.isin(geo_labels, self.labels).nonzero()
        sub = IndexedLocations(
            selection=selection,
            seed_size=self.seed_size,
            initial_compartment=self.initial_compartment,
            infection_compartment=self.infection_compartment,
        )
        return self.defer(sub)


class RandomLocations(SeededInfection):
    """
    Seed an infection in a number of randomly selected locations. Parameters:
    - `num_locations` the number of locations to choose
    - `seed_size` the number of individuals to infect in total
    """

    requirements = (_POPULATION_ATTR,)

    num_locations: int
    """The number of locations to choose (randomly)."""
    seed_size: int
    """How many individuals to infect, randomly distributed to selected locations."""

    def __init__(
        self,
        num_locations: int,
        seed_size: int,
        initial_compartment: int = SeededInfection.DEFAULT_INITIAL,
        infection_compartment: int = SeededInfection.DEFAULT_INFECTION,
    ):
        super().__init__(initial_compartment, infection_compartment)
        self.num_locations = num_locations
        self.seed_size = seed_size

    def evaluate(self) -> SimArray:
        N = self.dim.nodes

        if not 0 < self.num_locations <= N:
            msg = (
                "Initializer argument 'num_locations' must be "
                f"a value from 1 up to the number of locations ({N})."
            )
            raise InitException(msg)

        indices = np.arange(N, dtype=np.intp)
        selection = self.rng.choice(indices, self.num_locations)
        sub = IndexedLocations(
            selection=selection,
            seed_size=self.seed_size,
            initial_compartment=self.initial_compartment,
            infection_compartment=self.infection_compartment,
        )
        return self.defer(sub)


class TopLocations(SeededInfection):
    """
    Infect a fixed number of people across a fixed number of locations,
    selecting the top locations as measured by a given geo attribute.
    """

    # attributes is set in constructor

    top_attribute: AttributeDef[type[int]] | AttributeDef[type[float]]
    """
    The attribute to by which to judge the 'top' locations.
    Must be an N-shaped attribute.
    """
    num_locations: int
    """The number of locations to choose (randomly)."""
    seed_size: int
    """
    How many individuals to infect, randomly distributed between all selected locations.
    """

    def __init__(
        self,
        top_attribute: AttributeDef[type[int]] | AttributeDef[type[float]],
        num_locations: int,
        seed_size: int,
        initial_compartment: int = SeededInfection.DEFAULT_INITIAL,
        infection_compartment: int = SeededInfection.DEFAULT_INFECTION,
    ):
        super().__init__(initial_compartment, infection_compartment)

        if not top_attribute.shape == Shapes.N:
            msg = "Initializer argument `top_locations` must be an N-shaped attribute."
            raise InitException(msg)

        self.top_attribute = top_attribute
        self.requirements = (_POPULATION_ATTR, top_attribute)
        self.num_locations = num_locations
        self.seed_size = seed_size

    def evaluate(self) -> SimArray:
        N = self.dim.nodes

        if not 0 < self.num_locations <= N:
            msg = (
                "Initializer argument 'num_locations' must be "
                f"a value from 1 up to the number of locations ({N})."
            )
            raise InitException(msg)

        # `argpartition` chops an array in two halves
        # (yielding indices of the original array):
        # all indices whose values are smaller than the kth element,
        # followed by the index of the kth element,
        # followed by all indices whose values are larger.
        # So by using -k we create a partition of the largest k elements
        # at the end of the array, then slice using [-k:] to get just those indices.
        # This should be O(k log k) and saves us from copying+sorting (or some-such)
        arr = self.data(self.top_attribute)
        selection = np.argpartition(arr, -self.num_locations)[-self.num_locations :]
        sub = IndexedLocations(
            selection=selection,
            seed_size=self.seed_size,
            initial_compartment=self.initial_compartment,
            infection_compartment=self.infection_compartment,
        )
        return self.defer(sub)


class BottomLocations(SeededInfection):
    """
    Infect a fixed number of people across a fixed number of locations,
    selecting the bottom locations as measured by a given geo attribute.
    """

    # attributes is set in constructor

    bottom_attribute: AttributeDef[type[int]] | AttributeDef[type[float]]
    """
    The attribute to by which to judge the 'bottom' locations.
    Must be an N-shaped attribute.
    """
    num_locations: int
    """The number of locations to choose (randomly)."""
    seed_size: int
    """
    How many individuals to infect, randomly distributed between all selected locations.
    """

    def __init__(
        self,
        bottom_attribute: AttributeDef[type[int]] | AttributeDef[type[float]],
        num_locations: int,
        seed_size: int,
        initial_compartment: int = SeededInfection.DEFAULT_INITIAL,
        infection_compartment: int = SeededInfection.DEFAULT_INFECTION,
    ):
        super().__init__(initial_compartment, infection_compartment)

        if not bottom_attribute.shape == Shapes.N:
            msg = (
                "Initializer argument `bottom_locations` must be an N-shaped attribute."
            )
            raise InitException(msg)

        self.bottom_attribute = bottom_attribute
        self.requirements = (_POPULATION_ATTR, bottom_attribute)
        self.num_locations = num_locations
        self.seed_size = seed_size

    def evaluate(self) -> SimArray:
        N = self.dim.nodes

        if not 0 < self.num_locations <= N:
            msg = (
                "Initializer argument 'num_locations' must be "
                f"a value from 1 up to the number of locations ({N})."
            )
            raise InitException(msg)

        # `argpartition` chops an array in two halves
        # (yielding indices of the original array):
        # all indices whose values are smaller than the kth element,
        # followed by the index of the kth element,
        # followed by all indices whose values are larger.
        # So by using k we create a partition of the smallest k elements
        # at the start of the array, then slice using [:k] to get just those indices.
        # This should be O(k log k) and saves us from copying+sorting (or some-such)
        arr = self.data(self.bottom_attribute)
        selection = np.argpartition(arr, self.num_locations)[: self.num_locations]
        sub = IndexedLocations(
            selection=selection,
            seed_size=self.seed_size,
            initial_compartment=self.initial_compartment,
            infection_compartment=self.infection_compartment,
        )
        return self.defer(sub)


initializer_library: dict[str, type[Initializer]] = {
    "no_infection": NoInfection,
    "explicit": Explicit,
    "proportional": Proportional,
    "indexed_locations": IndexedLocations,
    "single_location": SingleLocation,
    "labeled_locations": LabeledLocations,
    "random_locations": RandomLocations,
    "top_locations": TopLocations,
    "bottom_locations": BottomLocations,
}
"""A library for the built-in initializer functions."""
