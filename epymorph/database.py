"""
A `Database` in epymorph organizes data attributes in a hierarchical namespace of three
components, as in: "strata::module::attribute_id".

The goal is to be as flexibile as possible for users specifying requirements and values
which fulfill those requirements. For example, you can provide a value for
"*::*::population" to indicate that every strata and every module should use the same
value if they require a "population" attribute. Or you can provide "*::init::population"
to indicate that only initializers should use this value, and, presumably, another value
is more appropriate for other modules like movement.

Specialized database instances provide the ability to "layer" data for a simulation --
if the outermost database has a matching value, that value is used, otherwise the search
for a match proceeds to the inner layers (recursively).
"""

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import singledispatch
from typing import (
    Callable,
    Generic,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    NamedTuple,
    Protocol,
    Sequence,
    TypeGuard,
    TypeVar,
    overload,
)

import numpy as np
from typing_extensions import override

from epymorph.attribute import (
    AbsoluteName,
    AttributeDef,
    NamePattern,
)
from epymorph.compartment_model import BaseCompartmentModel
from epymorph.data_shape import (
    DataShape,
    Dimensions,
    Node,
    Scalar,
    Shapes,
    Time,
    TimeAndNode,
)
from epymorph.data_type import (
    AttributeArray,
    AttributeType,
    AttributeValue,
    dtype_as_np,
    dtype_str,
)
from epymorph.error import DataAttributeError, DataAttributeErrorGroup
from epymorph.geography.scope import GeoScope
from epymorph.time import TimeFrame
from epymorph.util import (
    ANSIColor,
    ANSIStyle,
    ansi_stylize,
    filter_unique,
)

############
# Database #
############


T = TypeVar("T")
"""Type of database values."""


class Match(NamedTuple, Generic[T]):
    """The result of a database query."""

    pattern: NamePattern
    """The name pattern of the value that matched."""
    value: T
    """The matched value."""


class Database(Generic[T]):
    """
    A dictionary of key/value pairs where keys obey epymorph's namespace system.

    Namespaces are in the form "a::b::c", where "a" is a strata, "b" is a module,
    and "c" is an attribute name. Values can be assigned with wildcards (specified by
    asterisks), so that key "*::b::c" matches queries for both "a::b::c" and "z::b::c".

    See Also
    --------
    `Database` is mostly used to store raw-form parameter values declared by the user.
    In constrast [epymorph.database.DataResolver][] stores fully-evaluated parameters
    optimized for use by simulation logic.

    Parameters
    ----------
    data :
        The key/value pairs to store.
    """

    _data: dict[NamePattern, T]
    """The data values."""

    def __init__(self, data: dict[NamePattern, T]):
        self._data = data

        # Check for key ambiguity:
        conflicts = filter_unique(
            [
                key
                for key, _ in self._data.items()
                for other, _ in self._data.items()
                if key != other and key.match(other)
            ]
        )

        if len(conflicts) > 0:
            conflicts = ", ".join(map(str, conflicts))
            msg = f"Keys in data source are ambiguous; conflicts:\n{conflicts}"
            raise ValueError(msg)

    def query(self, key: str | AbsoluteName) -> Match[T] | None:
        """
        Query this database for a key match.

        Parameters
        ----------
        key :
            The name to find. If given as a string, it is first parsed as an
            `AbsoluteName`.

        Returns
        -------
        :
            The found value, if any, else `None`.
        """
        if not isinstance(key, AbsoluteName):
            key = AbsoluteName.parse(key)
        for pattern, value in self._data.items():
            if pattern.match(key):
                return Match(pattern, value)
        return None

    def to_dict(self) -> dict[NamePattern, T]:
        """
        Return the database's data as a dictionary.

        Returns
        -------
        :
            A copy of this database's data.
        """
        return {**self._data}


class DatabaseWithFallback(Database[T]):
    """
    A specialization of `Database` which has its own key/value pairs as well as a
    fallback `Database`.

    If a match is not found in this database, the fallback is checked (recursively).

    Parameters
    ----------
    data :
        The highest priority values in the database.
    fallback :
        A database containing fallback values; if a match cannot be found
        in `data`, this database will be checked.
    """

    _fallback: Database[T]
    """The fallback."""

    def __init__(self, data: dict[NamePattern, T], fallback: Database[T]):
        super().__init__(data)
        self._fallback = fallback

    @override
    def query(self, key: str | AbsoluteName) -> Match[T] | None:
        if not isinstance(key, AbsoluteName):
            key = AbsoluteName.parse(key)
        # First check "our" data:
        matched = super().query(key)
        # Otherwise check fallback data:
        if matched is None:
            matched = self._fallback.query(key)
        return matched

    @override
    def to_dict(self) -> dict[NamePattern, T]:
        def is_overridden(fb_key: NamePattern) -> bool:
            for override_key in self._data.keys():
                if override_key.match(fb_key):
                    return True
            return False

        results = {
            fb_key: fb_value
            for fb_key, fb_value in self._fallback.to_dict().items()
            if not is_overridden(fb_key)
        }
        results.update(self._data)
        return results


class DatabaseWithStrataFallback(Database[T]):
    """
    A specialization of `Database` which has per-strata fallback databases.

    For example consider querying this database for "a::b::c". If there is no match in
    this database but there is a fallback for strata "a", the search proceeds to that
    database (which could continue recursively).

    Parameters
    ----------
    data :
        The highest-priority values in the database.
    children :
        Fallback databases by strata.
    """

    _children: dict[str, Database[T]]
    """The fallbacks by strata name."""

    def __init__(self, data: dict[NamePattern, T], children: dict[str, Database[T]]):
        super().__init__(data)
        self._children = children

    @override
    def query(self, key: str | AbsoluteName) -> Match[T] | None:
        if not isinstance(key, AbsoluteName):
            key = AbsoluteName.parse(key)
        # First check "our" data:
        matched = super().query(key)
        # Otherwise check strata fallback for match:
        if matched is None and (db := self._children.get(key.strata)) is not None:
            matched = db.query(key)
        return matched

    @override
    def to_dict(self) -> dict[NamePattern, T]:
        def is_overridden(fb_key: NamePattern) -> bool:
            for override_key in self._data.keys():
                if override_key.match(fb_key):
                    return True
            return False

        results = {
            fb_key: fb_value
            for strata_db in self._children.values()
            for fb_key, fb_value in strata_db.to_dict().items()
            if not is_overridden(fb_key)
        }
        results.update(self._data)
        return results


############
# Resolver #
############


def assert_can_adapt(
    data_type: AttributeType,
    data_shape: DataShape,
    dim: Dimensions,
    value: AttributeArray,
) -> None:
    """
    Check that we can adapt the given `value` to the given type and shape,
    given dimensional information.

    Parameters
    ----------
    data_type :
        The final data type to adapt to.
    data_shape :
        The final data shape to adapt to.
    dim :
        Simulation dimensions.
    value :
        The value to adapt.

    Raises
    ------
    DataAttributeError
        If not.
    """
    if not np.can_cast(value, dtype_as_np(data_type)):
        raise DataAttributeError("Not a compatible type.")
    if not data_shape.matches(dim, value):
        raise DataAttributeError("Not a compatible shape.")


def adapt(
    data_type: AttributeType,
    data_shape: DataShape,
    dim: Dimensions,
    value: AttributeArray,
) -> AttributeArray:
    """
    Adapt the given `value` to the given type and shape, given dimensional
    information.

    Parameters
    ----------
    data_type :
        The final data type to adapt to.
    data_shape :
        The final data shape to adapt to.
    dim :
        Simulation dimensions.
    value :
        The value to adapt.

    Returns
    -------
    :
        The adapated array.

    Raises
    ------
    DataAttributeError
        If this fails.
    """
    assert_can_adapt(data_type, data_shape, dim, value)
    try:
        typed = value.astype(
            dtype_as_np(data_type),
            casting="safe",
            subok=False,
            copy=False,
        )
        return data_shape.adapt(dim, typed)
    except Exception as e:
        raise DataAttributeError("Failed to adapt value.") from e


class DataResolver:
    """
    Stores evaluated data attributes for a simulation.

    Data (typically parameter values) are provided by the user as a collection
    of key-value pairs, with the values in several forms. These are evaluated
    to turn them into our internal data representation which is optimized for
    simulation execution.

    Users typically provide parameter values with keys in `NamePattern` form, but
    during parameter resolution values are stored by `AbsoluteName` to facilitate
    lookups by the systems that use them. It is possible that multiple `AbsoluteNames`
    resolve to the same value (which is done in a memory-efficient way).

    Data requirements also specify the expected type and shape of the data,
    and `DataResolver` also manages the resulting complexity. Two requirements could be
    fulfilled by the same value while having different type and shape specifications.
    Instead of rejecting this as an error, epymorph allows this provided the given value
    can be successfully coerced to fit both requirements independently. For example, a
    scalar integer value can be coerced to fit requirements for an N-shaped array of
    floats as well as a T-shaped array of integers. `DataResolver` accomplishes this
    in an efficient way by storing both the original values and all adapted values.
    If multiple requirements specify the same type/shape adaptation for one value,
    the adaptation only needs to happen once.

    `DataResolver` is partially mutable -- new values can be added but values cannot
    be overwritten or removed.

    Parameters
    ----------
    dim :
        The simulation dimensions; needed to perform shape adaptations.
    values :
        The values that should be in the resolver to begin with.
    """

    Key = tuple[AbsoluteName, AttributeType, DataShape]
    """The type of the key used to track adapted values."""

    _dim: Dimensions
    """Simulation dimensions."""
    _raw_values: dict[AbsoluteName, AttributeArray]
    """Values in their 'input' form."""
    _adapted_values: dict[Key, AttributeArray]
    """Values in their 'output' form, after adaptation."""

    def __init__(
        self,
        dim: Dimensions,
        values: dict[AbsoluteName, AttributeArray] | None = None,
    ):
        self._dim = dim
        self._raw_values = values or {}
        self._adapted_values = {}

    def has(self, name: AbsoluteName) -> bool:
        """
        Test whether or not a given name is in this resolver.

        Parameters
        ----------
        name :
            The name to test.

        Returns
        -------
        :
            True if `name` is in this resolver.
        """
        return name in self._raw_values

    @property
    def raw_values(self) -> Mapping[AbsoluteName, AttributeArray]:
        """
        The mapping of raw values in the resolver, by absolute name.

        Warning:
        -------
        It is not safe to modify this mapping!
        """
        return self._raw_values

    def get_raw(self, name: str | NamePattern | AbsoluteName) -> AttributeArray:
        """
        Retrieve a raw value that matches the given name.

        Parameters
        ----------
        name :
            The name of the value to retrieve. This can be an `AbsoluteName`,
            a `NamePattern`, or a string. A string will be parsed as an
            `AbsoluteName` if possible, and fall back to parsing it as a `NamePattern`.
            In any case, the name must match exactly one value in order to
            return successfully.

        Returns
        -------
        :
            The requested value, if found.

        Raises
        ------
        ValueError
            If the name is not present in the resolver, or if the name
            is ambiguous and matches more than one value.
        """
        if isinstance(name, str):
            # if we can parse string as an absolute name, do that
            # otherwise, parse it as a name pattern
            try:
                name = AbsoluteName.parse(name)
            except ValueError:
                name = NamePattern.parse(name)

        if isinstance(name, NamePattern):
            # if we have a name pattern, find matching absolute names
            # if there is exactly one match we can proceed
            matches = [x for x in self._raw_values if name.match(x)]
            match len(matches):
                case 0:
                    err = f"Name '{name}' does not match any values."
                    raise ValueError(err)
                case 1:
                    name = matches[0]
                case _:
                    err = f"Name '{name}' matches more than one value."
                    raise ValueError(err)

        # at this point we must have an absolute name, try to fetch it
        if name not in self._raw_values:
            err = f"Name '{name}' does not match any values."
            raise ValueError(err)
        return self._raw_values[name]

    def add(self, name: AbsoluteName, value: AttributeArray) -> None:
        """
        Add a value to this resolver. You may not overwrite an existing name.

        Parameters
        ----------
        name :
            The name for the value.
        value : AttributeArray
            The value to add.

        Raises
        ------
        ValueError
            If `name` is already in this resolver.
        """
        if name in self._raw_values:
            err = (
                f"A value for '{name}' already exists in this resolver; "
                "values cannot be overwritten."
            )
            raise ValueError(err)
        self._raw_values[name] = value

    def resolve(self, name: AbsoluteName, definition: AttributeDef) -> AttributeArray:
        """
        Resolve a value known by `name` to fit the given requirement `definition`.

        Parameters
        ----------
        name :
            The name of the value to resolve.
        definition :
            The definition of the requirement being fulfilled; this provides the
            necessary type and shape information.

        Returns
        -------
        :
            The resolved value in adapted form.

        Raises
        ------
        DataAttributeError
            If resolution of adaptation fails.
        """
        key = (name, definition.type, definition.shape)
        if key in self._adapted_values:
            return self._adapted_values[key]

        if name not in self._raw_values:
            raise DataAttributeError(f"No value for name '{name}'")
        value = self._raw_values[name]
        adapted_value = adapt(definition.type, definition.shape, self._dim, value)
        self._adapted_values[key] = adapted_value
        return adapted_value

    def resolve_txn_series(
        self,
        requirements: Iterable[tuple[AbsoluteName, AttributeDef]],
        tau_steps: int,
    ) -> Iterator[list[AttributeValue]]:
        """
        Generate an iterator on the series of values for the given requirements.
        Each item produced by the generator is a sequence of scalar values, one for
        each attribute (in the given order).

        The sequence of items is generated in simulation order --

        - day=0, tau step=0, node=0 => [beta, gamma, xi]
        - day=0, tau_step=0; node=1 => [beta, gamma, xi]
        - day=0, tau_step=1; node=0 => [beta, gamma, xi]
        - day=0, tau_step=1; node=1 => [beta, gamma, xi]
        - and so on.

        This is a convenient alternative to resolving all of the TxN arrays separately,
        and managing the iteration yourself.

        Parameters
        ----------
        requirements :
            The name-definition pairs for all of the attributes to include.
        tau_steps :
            The number of tau steps per day; since T in a TxN array is simulation days,
            this simply repeats values such that all of a day's tau steps
            see the same value.

        Returns
        -------
        :
            An iterator by tau step where each value is the list of attribute values for
            that step.
        """

        def as_txn(attr_name: AbsoluteName, attr_def: AttributeDef) -> AttributeDef:
            # We want to resolve every attribute as TxN, however it's defined.
            # But we should check that this is possible first.
            if not isinstance(attr_def.shape, Scalar | Time | Node | TimeAndNode):
                err = (
                    "Cannot generate a TxN series unless all attributes "
                    f"broadcast to TxN. {attr_name} is {attr_def.shape}"
                )
                raise DataAttributeError(err)
            return dataclasses.replace(attr_def, shape=Shapes.TxN)

        resolved = [self.resolve(n, as_txn(n, d)) for n, d in requirements]

        T, N = (self._dim.T, self._dim.N)
        for t in range(T):
            # values-per-node repeats if tau_steps > 1, so compute it outside that loop.
            vals_by_node = [[v[t, n] for v in resolved] for n in range(N)]
            for _ in range(tau_steps):
                for vals in vals_by_node:  # implicitly: range(N)
                    yield vals

    @overload
    def to_dict(
        self, *, simplify_names: Literal[False] = False
    ) -> dict[AbsoluteName, AttributeArray]: ...

    @overload
    def to_dict(
        self, *, simplify_names: Literal[True]
    ) -> dict[str, AttributeArray]: ...

    def to_dict(
        self, *, simplify_names: bool = False
    ) -> dict[AbsoluteName, AttributeArray] | dict[str, AttributeArray]:
        """
        Extract a dictionary from this resolver containing all of its raw (non-adapted)
        key-value pairs.

        Parameters
        ----------
        simplify_names :
            True to return stringified names. By default, names are returned in
            `AbsoluteName` form.

        Returns
        -------
        :
            The dictionary of all values in this resolver, with names
            either simplified or not according to `simplify_names`.
        """
        if simplify_names:
            return {str(k): v for k, v in self._raw_values.items()}
        return {**self._raw_values}


class Requirement(NamedTuple):
    """A RUME data requirement: a name and a definition."""

    name: AbsoluteName
    """The requirement name."""
    definition: AttributeDef
    """The requirement definition."""

    def __str__(self) -> str:
        properties = [dtype_str(self.definition.type), str(self.definition.shape)]
        return f"{self.name} ({', '.join(properties)})"


class Resolution(ABC):
    """Describes the source used to resolve a requirement in a RUME."""

    # NOTE: it is required that all Resolution instances be hashable!
    # NOTE: This should be treated as a sealed type.

    cacheable: bool
    """True if this resolution can be cached."""


@dataclass(frozen=True)
class MissingValue(Resolution):
    """Requirement was not resolved."""

    cacheable: bool = field(init=False, default=False)

    def __str__(self) -> str:
        return "missing"


@dataclass(frozen=True)
class ParameterValue(Resolution):
    """
    Requirement was resolved by a RUME parameter.

    Parameters
    ----------
    cacheable :
        True if this resolution can be cached.
    pattern :
        The name by which the value was resolved.
    """

    cacheable: bool
    pattern: NamePattern
    """The name by which the value was resolved."""

    def __str__(self) -> str:
        return f"parameter value '{self.pattern}'"


@dataclass(frozen=True)
class DefaultValue(Resolution):
    """
    Requirement was resolved by an attribute default value.

    Parameters
    ----------
    default_value :
        The value that was resolved.
    """

    cacheable: bool = field(init=False, default=True)
    default_value: AttributeValue
    """The value that was resolved."""

    # NOTE: we need default_value here so that default resolutions
    # are only cached as one if they use the same value;
    # this is critical for conflicting defaults detection.
    # Since this also needs to be hashable, that means default values
    # cannot be numpy arrays (unless we add some code to handle those).

    def __str__(self) -> str:
        return "default value"


@dataclass(frozen=True)
class ResolutionTree:
    """
    In resolving parameters, parameter values might have their own requirements.
    This potentially forms a tree of requirements that must be resolved first in order
    to resolve the root of the tree.

    If two values share the same resolution tree, and if the tree is comprised entirely
    of pure functions and constant values, then the resolution result would be the same.

    Parameters
    ----------
    resolution :
        The root node resolution.
    children :
        The resolution of the root's requirements (empty if it has no requirements).


    See Also
    --------
    A resolution tree is a dual of the [epymorph.database.ReqTree][], but contains
    only the resolution results and not any resolved values.
    """

    resolution: Resolution
    """The root node resolution."""
    children: "tuple[ResolutionTree, ...]"
    """The resolution of the root's requirements (empty if it has no requirements)."""

    @property
    def is_tree_cacheable(self) -> bool:
        """Is the entire tree cacheable? Only true if all resolutions are cacheable."""
        return (
            self.resolution.cacheable  #
            and all(x.is_tree_cacheable for x in self.children)
        )


class RecursiveValue(Protocol):
    """The interface for parameter values that may depend on other parameter values."""

    requirements: Sequence[AttributeDef]
    """Defines the data requirements for this value."""
    randomized: bool
    """
    Should this value be re-evaluated every time it's referenced?
    Mostly useful for results that incorporate random values.
    """


@singledispatch
def is_recursive_value(value: object) -> TypeGuard[RecursiveValue]:
    """
    Check if a value is recursive, as a type guard.

    Implemented by single dispatch.

    Parameters
    ----------
    value :
        The value to check.

    Returns
    -------
    :
        True if the value can be recursive, as a type-guard.
    """
    return False


V = TypeVar("V")
"""The value type for a requirements tree."""


@dataclass(frozen=True)
class ReqTree(Generic[V]):
    """
    A requirements tree describes how data requirements are resolved for a RUME.

    Models used in the RUME have a set of requirements, each of which may be fulfilled
    by RUME parameters or default values. RUME parameters may also have data
    requirements, and those requirements may have requirements, etc., hence the need
    to represent this as a tree structure. The top of a `ReqTree` is a `ReqRoot` and
    each requirement is a `ReqNode`. Each `ReqNode` tracks the attribute itself
    (its `AbsoluteName` and `AttributeDef`), whether it is fulfilled, and if so how.

    Parameters
    ----------
    children :
        The child nodes of this node.
    """

    children: "tuple[ReqNode, ...]"
    """The child nodes of this node."""

    def __str__(self) -> str:
        # Built-in stringify is just to_string() with default args.
        return self.to_string()

    @abstractmethod
    def to_string(
        self,
        /,
        format_name: Callable[[AbsoluteName], str] = str,
        depth: int = 0,
    ) -> str:
        """
        Convert this tree to a string.

        Parameters
        ----------
        format_name :
            A method to convert an absolute name into a string.
            This allows you to control how absolute names are rendered.
        depth :
            The resolution "depth" of this node in the tree. Top-level requirements have
            a depth of 0. A requirement that itself is required by one other requirement
            would have a depth of 1.

        Returns
        -------
        :
            The stringified tree.
        """
        return "\n".join(
            [x.to_string(format_name=format_name, depth=depth) for x in self.children]
        )

    def traverse(self) -> "Iterable[ReqNode]":
        """
        Perform a depth-first traversal of the nodes of this tree.

        Yields
        ------
        :
            Each node in the tree in depth-first order.
        """
        for x in self.children:
            yield from x.traverse()
        # NOTE: method overridden by ReqNode; yields children and then itself

    def missing(self) -> Iterable[Requirement]:
        """
        Return the missing requirements from this tree.

        Returns
        -------
        :
            All requirements which were determined by resolution to be missing.
        """
        return (
            Requirement(node.name, node.definition)
            for node in self.traverse()
            if isinstance(node.resolution, MissingValue)
        )

    def evaluate(
        self,
        scope: GeoScope | None,
        time_frame: TimeFrame | None,
        ipm: BaseCompartmentModel | None,
        rng: np.random.Generator | None,
    ) -> DataResolver:
        """
        Evaluate this tree. Defers to `evaluate_requirements()`.

        Parameters
        ----------
        scope :
            The geographic scope information of this simulation context, if available.
        time_frame :
            The temporal scope information of this simulation context, if available.
        ipm :
            The disease model for this simulation context, if available.
        rng :
            The random number generator to use, if available.

        Returns
        -------
        :
            The resolver containing all evaluated parameters.

        Raises
        ------
        DataAttributeError
            If any attribute fails evaluation.
        """
        return evaluate_requirements(self, scope, time_frame, ipm, rng)

    @staticmethod
    def of(
        requirements: Mapping[AbsoluteName, AttributeDef],
        params: Database[V],
    ) -> "ReqTree":
        """
        Compute the requirements tree for the given set of requirements and a database
        supplying values. Note that missing values do not stop us from computing the
        tree -- these nodes will have `MissingValue` as the resolution.

        Parameters
        ----------
        requirements :
            The top-level requirements of the tree.
        params :
            The database of values, where each value may be recursive in the sense of
            having its own data requirements.

        Returns
        -------
        :
            The requirements tree.

        Raises
        ------
        DataAttributeError
            If the tree cannot be evaluated, for instance, due to it containing
            circular dependencies.
        """

        def recurse(
            name: AbsoluteName,
            definition: AttributeDef,
            chain: list[AbsoluteName],
        ) -> ReqNode:
            if name in chain:
                err = f"Circular dependency in evaluation of parameter {name}"
                raise DataAttributeError(err)

            if (val_match := params.query(name)) is not None:
                # User provided a parameter value to use.
                pattern, value = val_match
                if is_recursive_value(value):
                    # might have its own requirements
                    ns = name.to_namespace()
                    children = tuple(
                        recurse(ns.to_absolute(r.name), r, [name, *chain])
                        for r in value.requirements
                    )
                    # is this resolution node cacheable?
                    cacheable = not value.randomized
                else:
                    children = ()
                    cacheable = True
                resolution = ParameterValue(
                    cacheable=cacheable,
                    pattern=pattern,
                )

            elif definition.default_value is not None:
                # User did not provide a value, but we have a default.
                value = definition.default_value
                resolution = DefaultValue(value)
                children = ()

            else:
                # Missing value!
                value = None
                resolution = MissingValue()
                children = ()

            return ReqNode(children, name, definition, resolution, value)

        top_level_reqs = tuple(recurse(r, d, []) for r, d in requirements.items())
        return ReqTree(top_level_reqs)


@dataclass(frozen=True)
class ReqNode(ReqTree[V]):
    """
    A non-root node of a requirements tree, identifying a requirement, how (or if)
    it was resolved, and its value (if available).

    Parameters
    ----------
    children :
        The dependencies of this node (if any).
    name :
        The name of this requirement.
    definition :
        The definition of this requirement.
    resolution :
        How this requirement was resolved (or not).
    value :
        The value for this requirement if available.
    """

    children: "tuple[ReqNode, ...]"
    """The dependencies of this node (if any)."""
    name: AbsoluteName
    """The name of this requirement."""
    definition: AttributeDef
    """The definition of this requirement."""
    resolution: Resolution
    """How this requirement was resolved (or not)."""
    value: V | None
    """The value for this requirement if available."""

    @override
    def to_string(
        self,
        /,
        format_name: Callable[[AbsoluteName], str] = str,
        depth: int = 0,
    ) -> str:
        attr = self.definition
        properties = [dtype_str(attr.type), str(attr.shape)]

        match self.resolution:
            case ParameterValue(_cacheable, value_name):
                color = None
                resd = f" <- {value_name}"
            case DefaultValue():
                properties.append(f"default={self.value}")
                color = ANSIColor.CYAN
                resd = ""
            case MissingValue():
                color = ANSIColor.RED
                resd = ""
            case x:
                # have to handle the impossible case
                # because Python doesn't have sealed types yet
                raise NotImplementedError(f"Unsupported resolution type ({type(x)})")

        indent = "  " * depth
        corner = "└╴" if depth > 0 else ""
        name = ansi_stylize(format_name(self.name), color)
        prop = ansi_stylize(f"({', '.join(properties)})", color, ANSIStyle.ITALIC)

        return "\n".join(
            [
                f"{indent}{corner}{name} {prop}{resd}",
                *(
                    x.to_string(format_name=format_name, depth=depth + 1)
                    for x in self.children
                ),
            ]
        )

    @override
    def traverse(self) -> "Iterable[ReqNode]":
        yield from super().traverse()  # traverse children
        yield self  # and then self

    def as_res_tree(self) -> ResolutionTree:
        """
        Extract the resolution tree from this requirements tree.

        Returns
        -------
        :
            The resolution tree.
        """
        return ResolutionTree(
            self.resolution,
            tuple(x.as_res_tree() for x in self.children),
        )


@singledispatch
def evaluate_param(
    value: object,
    name: AbsoluteName,
    data: DataResolver,
    scope: GeoScope | None,
    time_frame: TimeFrame | None,
    ipm: BaseCompartmentModel | None,
    rng: np.random.Generator | None,
) -> AttributeArray:
    """
    Evaluate a parameter, transforming acceptable input values (type: `ParamValue`) to
    the form required internally by epymorph (`AttributeArray`).

    Different types of parameters are supported by single dispatch, so there should be a
    registered implementation for every supported type of parameter.

    It is possible that the user is attempting to evaluate parameters with a
    partial context (`scope`, `time_frame`, `ipm`, `rng`), and so one or more of these
    may be missing. In that case, parameter evaluation is expected to happen on a
    "best effort" basis -- if no parameter accesses the missing scope elements,
    parameter evaluation succeeds. Otherwise, it raises a `DataAttributeError`.

    Parameters
    ----------
    value :
        The value being evaluated.
    name :
        The full name given to the value in the simulation context.
    data :
        A resolved instance which should contain values for all of the data
        requirements needed by this value (only `RecursiveValues` have requirements).
    scope :
        The geographic scope information of this simulation context, if available.
    time_frame :
        The temporal scope information of this simulation context, if available.
    ipm :
        The disease model for this simulation context, if available.
    rng :
        The random number generator to use, if available.

    Returns
    -------
    :
        The evaluated value.

    Raises
    ------
    DataAttributeError
        If parameter evaluation fails.
    """
    err = f"Parameter not a supported type (found: {type(value)})"
    raise DataAttributeError(err)


@evaluate_param.register
def _(
    value: np.ndarray,
    name: AbsoluteName,
    data: DataResolver,
    scope: GeoScope | None,
    time_frame: TimeFrame | None,
    ipm: BaseCompartmentModel | None,
    rng: np.random.Generator | None,
) -> AttributeArray:
    # Evaluate numpy arrays.
    # numpy array: make a copy so we don't risk unexpected mutations
    return value.copy()


@evaluate_param.register
def _(
    value: int | float | str | tuple | list,
    name: AbsoluteName,
    data: DataResolver,
    scope: GeoScope | None,
    time_frame: TimeFrame | None,
    ipm: BaseCompartmentModel | None,
    rng: np.random.Generator | None,
) -> AttributeArray:
    # Evluate scalars.
    # scalar value or python collection: re-pack it as a numpy array
    return np.asarray(value, dtype=None)


@evaluate_param.register
def _(
    value: type,
    name: AbsoluteName,
    data: DataResolver,
    scope: GeoScope | None,
    time_frame: TimeFrame | None,
    ipm: BaseCompartmentModel | None,
    rng: np.random.Generator | None,
) -> AttributeArray:
    # Evaluate Python type instances.
    # forgot to instantiate? a common error worth checking for
    err = (
        "Parameter was given as a class instead of an instance. "
        "Did you forget to instantiate it?"
    )
    raise DataAttributeError(err)


def evaluate_requirements(
    req: ReqTree,
    scope: GeoScope | None,
    time_frame: TimeFrame | None,
    ipm: BaseCompartmentModel | None,
    rng: np.random.Generator | None,
) -> DataResolver:
    """
    Evaluate all parameters in the tree, using the given simulation context
    (`scope`, `time_frame`, `ipm`, `rng`).

    You may attempt to evaluate parameters with a partial context, so one or more of
    these may be missing. In that case, parameter evaluation happens on a "best effort"
    basis -- if no parameter accesses the missing scope elements, parameter evaluation
    succeeds; otherwise it will raise an `DataAttributeError`.

    Parameters
    ----------
    req :
        The requirements tree.
    scope :
        The geographic scope information of this simulation context, if available.
    time_frame :
        The temporal scope information of this simulation context, if available.
    ipm :
        The disease model for this simulation context, if available.
    rng :
        The random number generator to use, if available.

    Returns
    -------
    :
        The resolver containing all evaluated parameters.

    Raises
    ------
    DataAttributeError
        If any attribute fails evaluation.
    """

    # First make sure there are no missing values.
    missing = list(req.missing())
    if len(missing) > 0:
        err = "\n".join(
            [
                "Cannot evaluate parameters, there are missing values:",
                *(str(r.name) for r in missing),
            ]
        )
        raise DataAttributeError(err)

    # For each node in the requirements tree (traversing depth-first),
    # evaluate each value and validate it against the requirement's definition;
    # fulfilled requirements get stored to `resolved` and we use `evaluated` as
    # an evaluation cache, to avoid evaluating the same parameter (input value)
    # twice. Accumulate validation errors into `errors` to be raised as a group.
    dim = Dimensions.of(
        T=time_frame.duration_days if time_frame is not None else None,
        N=scope.nodes if scope is not None else None,
        C=ipm.num_compartments if ipm is not None else None,
        E=ipm.num_events if ipm is not None else None,
    )
    errors = list[DataAttributeError]()
    resolved = DataResolver(dim)
    resolved_by = dict[AbsoluteName, ResolutionTree]()
    evaluated = dict[ResolutionTree, AttributeArray]()

    for node in req.traverse():
        try:
            res_tree = node.as_res_tree()
            if res_tree in evaluated:
                # Use cached value
                value = evaluated[res_tree]
            else:
                # Evaluate
                value = evaluate_param(
                    node.value,
                    node.name,
                    resolved,
                    scope,
                    time_frame,
                    ipm,
                    rng,
                )

                if not isinstance(value, np.ndarray):
                    err = (
                        f"Attribute '{node.name}' ({node.resolution}) did "
                        f"not evaluate to a numpy array."
                    )
                    raise DataAttributeError(err)

                if res_tree.is_tree_cacheable:
                    evaluated[res_tree] = value

            # If we have the necessary info, check that the raw value
            # can be successfully adapted to fulfill this requirement.
            # Raise DataAttributeError if not.
            try:
                d = node.definition
                assert_can_adapt(d.type, d.shape, dim, value)
            except DataAttributeError as e:
                err = (
                    f"Attribute '{node.name}' ({node.resolution}) is "
                    f"not properly specified: {e}"
                )
                raise DataAttributeError(err)

            # Store value
            if not resolved.has(node.name):
                resolved.add(node.name, value)
                resolved_by[node.name] = res_tree
            elif res_tree != resolved_by[node.name]:
                # Raise error for conflicting resolutions.
                err = (
                    "Detected conflicting resolutions for requirement "
                    f"'{node.name}'\n"
                    "This is likely caused by AttributeDefs with different default "
                    "values both resolving to the same name. If this is the case, "
                    "you will need to provide an explicit value instead."
                )
                raise DataAttributeError(err)

        except DataAttributeError as e:
            errors.append(e)

    if len(errors) > 0:
        raise DataAttributeErrorGroup("Errors found evaluating parameters.", errors)

    return resolved
