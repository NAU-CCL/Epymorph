"""
Initializers are responsible for setting up the initial conditions for the simulation:
the populations at each node, and which disease compartments they belong to.
It may draw from simulation parameters to do so. As there are many valid ways to do
this, this module provides a uniform interface to accomplish the task as well as a few
common implementations.
"""

from abc import ABC
from typing import ClassVar, cast

import numpy as np
from numpy.typing import DTypeLike, NDArray
from typing_extensions import override

from epymorph.attribute import AttributeDef
from epymorph.data_shape import DataShape, DataShapeMatcher, Shapes
from epymorph.data_type import SimArray, SimDType
from epymorph.error import InitError
from epymorph.simulation import (
    SimulationFunction,
)
from epymorph.util import Matcher, NumpyTypeError, check_ndarray, dtype_name, match


class Initializer(SimulationFunction[SimArray], ABC):
    """
    An initialization routine responsible for determining the initial values
    of populations by IPM compartment for every simulation node.
    """

    def as_compartment(self, name_or_index: int | str) -> int:
        """
        Convert a compartment identifier to a compartment index.

        Parameters
        ----------
        name_or_index :
            Identifies the compartment; usually a compartment name.
            However this can also be an integer index, in which case we check that the
            index is valid and then return it.

        Raises
        ------
        InitError
            If the compartment identifier is not valid.
        """
        try:
            if isinstance(name_or_index, int):
                if name_or_index < 0:
                    raise ValueError()
                if name_or_index >= self.ipm.num_compartments:
                    raise ValueError()
                return name_or_index
            else:
                return self.ipm.select.compartments(name_or_index).compartment_index
        except ValueError:
            err = f"Unknown compartment '{name_or_index}' specified in initializer."
            raise InitError(err)

    @override
    def validate(self, result) -> None:
        # Must be an NxC array of integers, none less than zero.
        try:
            check_ndarray(
                result,
                dtype=match.dtype(SimDType),
                shape=DataShapeMatcher(Shapes.NxC, self.dim, exact=True),
            )
        except NumpyTypeError as e:
            err = f"Invalid return type from Initializer '{self.__class__.__name__}'"
            raise InitError(err) from e

        if np.min(result) < 0:
            err = (
                f"Initializer '{self.__class__.__name__}' returned "
                "values less than zero."
            )
            raise InitError(err)

    @property
    def _nxc(self) -> tuple[int, int]:
        return (self.scope.nodes, self.ipm.num_compartments)

    def _condition_input_array(
        self,
        value: NDArray | list,
        arg_name: str,
        *,
        dtype: DTypeLike,
        shape: DataShape | Matcher,
        exact: bool = True,
    ) -> NDArray:
        # A general validation utility for input arrays.
        try:
            shape_match = (
                DataShapeMatcher(shape, self.dim, exact=exact)
                if isinstance(shape, DataShape)
                else shape
            )
            result = np.array(value).astype(
                dtype=dtype,
                copy=True,
                casting="safe",
            )
            check_ndarray(
                result,
                dtype=match.dtype(dtype),
                shape=shape_match,
            )
            if isinstance(shape, DataShape) and not exact:
                result = shape.adapt(self.dim, result)
            return result
        except (ValueError, TypeError, NumpyTypeError):
            err = (
                f"Initializer {self.__class__.__name__} `{arg_name}` is not valid, "
                "check that you have provided the expected number and type of values "
                "for this simulation.\n"
                f"Expected: {shape}-shaped array of {dtype_name(np.dtype(SimDType))}"
            )
            raise InitError(err)


#######################
# Initializer Library #
#######################


_POPULATION_ATTR = AttributeDef(
    "population", int, Shapes.N, comment="The population at each geo node."
)

_LABEL_ATTR = AttributeDef(
    "label", str, Shapes.N, comment="A label associated with each geo node."
)


class NoInfection(Initializer):
    """
    An initializer that places all individuals in a single compartment.
    Requires "population" as a data attribute.

    Parameters
    ----------
    initial_compartment :
        The compartment name or index in which to start the population.
    """

    requirements = (_POPULATION_ATTR,)

    initial_compartment: int | str
    """The IPM compartment where people should start, as either a name or index."""

    def __init__(self, initial_compartment: int | str = 0):
        self.initial_compartment = initial_compartment

    @override
    def evaluate(self) -> SimArray:
        """
        Evaluate the initializer in the current context.

        Returns
        -------
        :
            The initial populations for each node and IPM compartment.
        """
        pop = self.data(_POPULATION_ATTR)
        result = np.zeros(self._nxc, dtype=SimDType)
        initial = self.as_compartment(self.initial_compartment)
        result[:, initial] = pop
        return result


class Explicit(Initializer):
    """
    An initializer that sets all compartment populations directly.
    You provide the (N,C)-shaped array and the initializer returns a copy of it.

    Parameters
    ----------
    initials :
        The literal initial values to use.
    """

    initials: NDArray | list[list[int]]
    """The initial compartment values to use."""

    def __init__(self, initials: NDArray | list[list[int]]):
        self.initials = initials

    @override
    def evaluate(self) -> SimArray:
        """
        Evaluate the initializer in the current context.

        Returns
        -------
        :
            The initial populations for each node and IPM compartment.
        """
        return self._condition_input_array(
            self.initials,
            "initials",
            dtype=SimDType,
            shape=Shapes.NxC,
            exact=True,
        )


class Proportional(Initializer):
    """
    An initializer that sets all compartments as a proportion of their population.
    Requires "population" as a data attribute.

    Parameters
    ----------
    ratios :
        A (C,)- or (N,C)-shaped array describing the ratios for each compartment.
        Row values will be normalized, such that they sum to 1.
    """

    requirements = (_POPULATION_ATTR,)

    ratios: (
        NDArray[np.int64 | np.float64]
        | list[int]
        | list[float]
        | list[list[int]]
        | list[list[float]]
    )
    """The initialization ratios to use."""

    def __init__(
        self,
        ratios: NDArray[np.int64 | np.float64]
        | list[int]
        | list[float]
        | list[list[int]]
        | list[list[float]],
    ):
        self.ratios = ratios

    @override
    def evaluate(self) -> SimArray:
        """
        Evaluate the initializer in the current context.

        Returns
        -------
        :
            The initial populations for each node and IPM compartment.
        """
        ratios = self._condition_input_array(
            self.ratios,
            "ratios",
            dtype=np.float64,
            shape=Shapes.NxC,
            exact=False,
        )

        row_sums = cast(NDArray[np.float64], np.sum(ratios, axis=1, dtype=np.float64))
        if np.any(row_sums <= 0):
            err = "One or more rows sum to zero or less."
            raise InitError(err)

        pop = self.data(_POPULATION_ATTR)
        result = pop[:, np.newaxis] * (ratios / row_sums[:, np.newaxis])
        return result.round().astype(SimDType)


class SeededInfection(Initializer, ABC):
    """
    Abstract base class for initializers which seed an initial infection.
    It assumes most people start out in a particular compartment (default: the first)
    and if chosen for infection, are moved to another compartment (default: the second).

    You can customize which two compartments to use, but it can only be two.

    Parameters
    ----------
    initial_compartment :
        Which compartment (by index or name) is "not infected", where most individuals
        start out.
    infection_compartment :
        Which compartment (by index or name) will be seeded as the initial infection.
    """

    DEFAULT_INITIAL: ClassVar = 0
    DEFAULT_INFECTION: ClassVar = 1

    initial_compartment: int | str
    """The IPM compartment for non-infected individuals."""
    infection_compartment: int | str
    """The IPM compartment for infected individuals."""

    def __init__(
        self,
        initial_compartment: int | str = DEFAULT_INITIAL,
        infection_compartment: int | str = DEFAULT_INFECTION,
    ):
        self.initial_compartment = initial_compartment
        self.infection_compartment = infection_compartment


class IndexedLocations(SeededInfection):
    """
    Infect a fixed number of people distributed (proportional to their population)
    across a selection of nodes. A multivariate hypergeometric draw using the available
    populations in each node is used to distribute infections.

    Requires "population" as a data attribute.

    Parameters
    ----------
    selection :
        The list of node indices to infect; all values must be in range (-N,+N).
    seed_size :
        The number of individuals to infect in total.
    initial_compartment :
        Which compartment (by index or name) is "not infected", where most individuals
        start out.
    infection_compartment :
        Which compartment (by index or name) will be seeded as the initial infection.
    """

    requirements = (_POPULATION_ATTR,)

    selection: NDArray[np.intp] | list[int]
    """Which locations to infect."""
    seed_size: int
    """How many individuals to infect, randomly distributed to selected locations."""

    def __init__(
        self,
        selection: NDArray[np.intp] | list[int],
        seed_size: int,
        initial_compartment: int | str = SeededInfection.DEFAULT_INITIAL,
        infection_compartment: int | str = SeededInfection.DEFAULT_INFECTION,
    ):
        super().__init__(initial_compartment, infection_compartment)
        if seed_size < 0:
            err = (
                "Initializer argument 'seed_size' must be a non-negative integer value."
            )
            raise InitError(err)

        self.selection = selection
        self.seed_size = seed_size

    @override
    def evaluate(self) -> SimArray:
        """
        Evaluate the initializer in the current context.

        Returns
        -------
        :
            The initial populations for each node and IPM compartment.
        """
        initial = self.as_compartment(self.initial_compartment)
        infection = self.as_compartment(self.infection_compartment)

        sel = self._condition_input_array(
            self.selection,
            "selection",
            dtype=np.intp,
            shape=match.dimensions(1),
        )

        N = self.scope.nodes
        if not np.all((-N < sel) & (sel < N)):
            err = (
                "Initializer argument 'selection' invalid: "
                f"some indices are out of range ({-N}, {N})."
            )
            raise InitError(err)

        pop = self.data(_POPULATION_ATTR)
        selected = pop[sel]
        available = selected.sum()
        if available < self.seed_size:
            err = (
                f"Attempted to infect {self.seed_size} individuals "
                f"but only had {available} available."
            )
            raise InitError(err)

        # Randomly select individuals from each of the selected locations.
        if len(selected) == 1:
            infected = np.array([self.seed_size], dtype=SimDType)
        else:
            infected = self.rng.multivariate_hypergeometric(selected, self.seed_size)

        result = np.zeros(self._nxc, dtype=SimDType)
        result[:, initial] = pop

        # Special case: the "no" IPM has only one compartment!
        # Technically it would be more "correct" to choose a different initializer,
        # but it's convenient to allow this special case for ease of testing.
        if self.ipm.num_compartments == 1:
            return result

        for i, n in zip(sel, infected):
            result[i, initial] -= n
            result[i, infection] += n
        return result


class SingleLocation(IndexedLocations):
    """
    Infect a fixed number of people at a single location (by index).

    Requires "population" as a data attribute.

    Parameters
    ----------
    location :
        The index of the node in which to seed an initial infection.
    seed_size :
        The number of individuals to infect in total.
    initial_compartment :
        Which compartment (by index or name) is "not infected", where most individuals
        start out.
    infection_compartment :
        Which compartment (by index or name) will be seeded as the initial infection.
    """

    requirements = (_POPULATION_ATTR,)

    def __init__(
        self,
        location: int,
        seed_size: int,
        initial_compartment: int | str = SeededInfection.DEFAULT_INITIAL,
        infection_compartment: int | str = SeededInfection.DEFAULT_INFECTION,
    ):
        super().__init__(
            selection=np.array([location], dtype=np.intp),
            seed_size=seed_size,
            initial_compartment=initial_compartment,
            infection_compartment=infection_compartment,
        )

    @override
    def evaluate(self) -> SimArray:
        """
        Evaluate the initializer in the current context.

        Returns
        -------
        :
            The initial populations for each node and IPM compartment.
        """
        N = self.scope.nodes
        if not -N < self.selection[0] < N:
            err = (
                "Initializer argument 'location' must be a valid index "
                f"to an array of {N} populations."
            )
            raise InitError(err)
        return super().evaluate()


class LabeledLocations(SeededInfection):
    """
    Infect a fixed number of people distributed to a selection of locations (by label).

    Requires "population" and "label" as data attributes.

    Parameters
    ----------
    labels :
        The labels of the locations to select for infection.
    seed_size :
        The number of individuals to infect in total.
    initial_compartment :
        Which compartment (by index or name) is "not infected", where most individuals
        start out.
    infection_compartment :
        Which compartment (by index or name) will be seeded as the initial infection.
    """

    requirements = (_POPULATION_ATTR, _LABEL_ATTR)

    labels: NDArray[np.str_] | list[str]
    """Which locations to infect."""
    seed_size: int
    """How many individuals to infect, randomly distributed to selected locations."""

    def __init__(
        self,
        labels: NDArray[np.str_] | list[str],
        seed_size: int,
        initial_compartment: int | str = SeededInfection.DEFAULT_INITIAL,
        infection_compartment: int | str = SeededInfection.DEFAULT_INFECTION,
    ):
        super().__init__(initial_compartment, infection_compartment)
        self.labels = labels
        self.seed_size = seed_size

    @override
    def evaluate(self) -> SimArray:
        """
        Evaluate the initializer in the current context.

        Returns
        -------
        :
            The initial populations for each node and IPM compartment.
        """
        geo_labels = self.data(_LABEL_ATTR)
        labels = self._condition_input_array(
            self.labels,
            "labels",
            dtype=np.str_,
            shape=match.dimensions(1),
        )

        if not np.all(np.isin(labels, geo_labels)):
            err = (
                "Initializer argument 'labels' invalid: "
                "some labels are not in the geography."
            )
            raise InitError(err)

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
    Seed an infection in a number of randomly selected locations.

    Requires "population" as a data attribute.

    Parameters
    ----------
    num_locations :
        The number of locations to choose.
    seed_size :
        The number of individuals to infect in total.
    initial_compartment :
        Which compartment (by index or name) is "not infected", where most individuals
        start out.
    infection_compartment :
        Which compartment (by index or name) will be seeded as the initial infection.
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
        initial_compartment: int | str = SeededInfection.DEFAULT_INITIAL,
        infection_compartment: int | str = SeededInfection.DEFAULT_INFECTION,
    ):
        super().__init__(initial_compartment, infection_compartment)
        self.num_locations = num_locations
        self.seed_size = seed_size

    @override
    def evaluate(self) -> SimArray:
        """
        Evaluate the initializer in the current context.

        Returns
        -------
        :
            The initial populations for each node and IPM compartment.
        """
        N = self.scope.nodes
        if not 0 < self.num_locations <= N:
            err = (
                "Initializer argument 'num_locations' must be "
                f"a value from 1 up to the number of locations ({N})."
            )
            raise InitError(err)

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
    Infect a fixed number of people across a fixed number of locations, selecting the
    top locations as measured by a given data attribute.

    Requires "population" and the top attribute as data attributes.

    Parameters
    ----------
    top_attribute :
        The attribute to use in determining the "top" locations. Must be numeric data.
    num_locations :
        The number of locations to choose.
    seed_size :
        The number of individuals to infect in total.
    initial_compartment :
        Which compartment (by index or name) is "not infected", where most individuals
        start out.
    infection_compartment :
        Which compartment (by index or name) will be seeded as the initial infection.
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
        initial_compartment: int | str = SeededInfection.DEFAULT_INITIAL,
        infection_compartment: int | str = SeededInfection.DEFAULT_INFECTION,
    ):
        super().__init__(initial_compartment, infection_compartment)

        if not top_attribute.shape == Shapes.N:
            err = "Initializer argument `top_locations` must be an N-shaped attribute."
            raise InitError(err)

        self.top_attribute = top_attribute
        self.requirements = (_POPULATION_ATTR, top_attribute)
        self.num_locations = num_locations
        self.seed_size = seed_size

    @override
    def evaluate(self) -> SimArray:
        """
        Evaluate the initializer in the current context.

        Returns
        -------
        :
            The initial populations for each node and IPM compartment.
        """
        N = self.scope.nodes
        if not 0 < self.num_locations <= N:
            err = (
                "Initializer argument 'num_locations' must be "
                f"a value from 1 up to the number of locations ({N})."
            )
            raise InitError(err)

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
    Infect a fixed number of people across a fixed number of locations, selecting the
    bottom locations as measured by a given geo attribute.

    Requires "population" and the bottom attribute as data attributes.

    Parameters
    ----------
    bottom_attribute :
        The attribute to use in determining the "bottom" locations. Must be numeric
        data.
    num_locations :
        The number of locations to choose.
    seed_size :
        The number of individuals to infect in total.
    initial_compartment :
        Which compartment (by index or name) is "not infected", where most individuals
        start out.
    infection_compartment :
        Which compartment (by index or name) will be seeded as the initial infection.
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
        initial_compartment: int | str = SeededInfection.DEFAULT_INITIAL,
        infection_compartment: int | str = SeededInfection.DEFAULT_INFECTION,
    ):
        super().__init__(initial_compartment, infection_compartment)

        if not bottom_attribute.shape == Shapes.N:
            err = (
                "Initializer argument `bottom_locations` must be an N-shaped attribute."
            )
            raise InitError(err)

        self.bottom_attribute = bottom_attribute
        self.requirements = (_POPULATION_ATTR, bottom_attribute)
        self.num_locations = num_locations
        self.seed_size = seed_size

    @override
    def evaluate(self) -> SimArray:
        """
        Evaluate the initializer in the current context.

        Returns
        -------
        :
            The initial populations for each node and IPM compartment.
        """
        N = self.scope.nodes
        if not 0 < self.num_locations <= N:
            err = (
                "Initializer argument 'num_locations' must be "
                f"a value from 1 up to the number of locations ({N})."
            )
            raise InitError(err)

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
