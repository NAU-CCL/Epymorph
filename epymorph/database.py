"""
A Database in epymorph is a way to organize values with our namespace system
of three hierarchical components, as in: "strata::module::attribute_id".
This gives us a lot of flexibility when specifying data requirements and values
which fulfill those requirements. For example, you can provide a value for
"*::*::population" to indicate that every strata and every module should use the same
value if they need a "population" attribute. Or you can provide "*::init::population"
to indicate that only initializers should use this value, and, presumably, another value
is more appropriate for other modules like movement.

Hierarchical Database instances are also included which provide the ability to "layer"
data for a simulation -- if the outermost database has a matching value, that value is
used, otherwise the search for a match proceeds to the inner layers (recursively).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import singledispatch
from typing import (
    Callable,
    Generic,
    Iterable,
    Literal,
    Mapping,
    NamedTuple,
    Protocol,
    Self,
    Sequence,
    TypeGuard,
    TypeVar,
    overload,
)

import numpy as np
from typing_extensions import override

from epymorph.data_shape import DataShape, Dimensions, SimDimensions
from epymorph.data_type import (
    AttributeArray,
    AttributeType,
    AttributeValue,
    dtype_as_np,
    dtype_check,
    dtype_str,
)
from epymorph.error import AttributeException, AttributeExceptionGroup
from epymorph.geography.scope import GeoScope
from epymorph.util import (
    AnsiColor,
    AnsiStyle,
    acceptable_name,
    ansi_stylize,
    filter_unique,
)

########################
# Names and Namespaces #
########################


def _validate_name_segments(*names: str) -> None:
    for n in names:
        if len(n) == 0:
            raise ValueError("Invalid name: cannot use empty strings.")
        if n == "*":
            raise ValueError("Invalid name: cannot use wildcards (*).")
        if "::" in n:
            raise ValueError("Invalid name: cannot contain '::'.")


def _validate_pattern_segments(*names: str) -> None:
    for n in names:
        if len(n) == 0:
            raise ValueError("Invalid pattern: cannot use empty strings.")
        if "::" in n:
            raise ValueError("Invalid pattern: cannot contain '::'.")


@dataclass(frozen=True)
class ModuleNamespace:
    """A namespace with a specified strata and module."""

    strata: str
    module: str

    def __post_init__(self):
        _validate_name_segments(self.strata, self.module)

    @classmethod
    def parse(cls, name: str) -> Self:
        """Parse a module name from a ::-delimited string."""
        parts = name.split("::")
        if len(parts) != 2:
            raise ValueError("Invalid number of parts for namespace.")
        return cls(*parts)

    def __str__(self) -> str:
        return f"{self.strata}::{self.module}"

    def to_absolute(self, attrib_id: str) -> "AbsoluteName":
        """Creates an absolute name by providing the attribute ID."""
        return AbsoluteName(self.strata, self.module, attrib_id)


STRATA_PLACEHOLDER = "(unspecified)"
MODULE_PLACEHOLDER = "(unspecified)"
NAMESPACE_PLACEHOLDER = ModuleNamespace(STRATA_PLACEHOLDER, MODULE_PLACEHOLDER)
"""A namespace to use when we don't need to be specific about strata or module names."""


@dataclass(frozen=True)
class AbsoluteName:
    """A fully-specified name (strata, module, and attribute ID)."""

    strata: str
    module: str
    id: str

    def __post_init__(self):
        _validate_name_segments(self.strata, self.module, self.id)

    @classmethod
    def parse(cls, name: str) -> Self:
        """Parse a module name from a ::-delimited string."""
        parts = name.split("::")
        if len(parts) != 3:
            raise ValueError("Invalid number of parts for absolute name.")
        return cls(*parts)

    def __str__(self) -> str:
        return f"{self.strata}::{self.module}::{self.id}"

    def in_strata(self, strata: str) -> "AbsoluteName":
        """
        Creates a new AbsoluteName that is a copy of this name
        but with the given strata.
        """
        return AbsoluteName(strata, self.module, self.id)

    def to_namespace(self) -> ModuleNamespace:
        """Extracts the module namespace of this name."""
        return ModuleNamespace(self.strata, self.module)

    def to_pattern(self) -> "NamePattern":
        """Converts this name to a pattern that is an exact match for this name."""
        return NamePattern(self.strata, self.module, self.id)


@dataclass(frozen=True)
class ModuleName:
    """A partially-specified name with module and attribute ID."""

    module: str
    id: str

    def __post_init__(self):
        _validate_name_segments(self.module, self.id)

    @classmethod
    def parse(cls, name: str) -> Self:
        """Parse a module name from a ::-delimited string."""
        parts = name.split("::")
        if len(parts) != 2:
            raise ValueError("Invalid number of parts for module name.")
        return cls(*parts)

    def __str__(self) -> str:
        return f"{self.module}::{self.id}"

    def to_absolute(self, strata: str) -> AbsoluteName:
        """Creates an absolute name by providing the strata."""
        return AbsoluteName(strata, self.module, self.id)


@dataclass(frozen=True)
class AttributeName:
    """A partially-specified name with just an attribute ID."""

    id: str

    def __post_init__(self):
        _validate_name_segments(self.id)

    def __str__(self) -> str:
        return self.id


@dataclass(frozen=True)
class NamePattern:
    """
    A name with a strata, module, and attribute ID that allows wildcards (*) so it can
    act as a pattern to match against AbsoluteNames.
    """

    strata: str
    module: str
    id: str

    def __post_init__(self):
        _validate_pattern_segments(self.strata, self.module, self.id)

    @classmethod
    def parse(cls, name: str) -> Self:
        """
        Parse a pattern from a ::-delimited string. As a shorthand, you can omit
        preceding wildcard segments and they will be automatically filled in,
        e.g., "a" will become "*::*::a" and "a::b" will become "*::a::b".
        """
        parts = name.split("::")
        match len(parts):
            case 1:
                return cls("*", "*", *parts)
            case 2:
                return cls("*", *parts)
            case 3:
                return cls(*parts)
            case _:
                raise ValueError("Invalid number of parts for name pattern.")

    @staticmethod
    def of(name: "str | NamePattern") -> "NamePattern":
        """Coerce the given value to a NamePattern.
        If it's already a NamePattern, return it; if it's a string, parse it."""
        return name if isinstance(name, NamePattern) else NamePattern.parse(name)

    def match(self, name: "AbsoluteName | NamePattern") -> bool:
        """
        Test this pattern to see if it matches the given AbsoluteName or NamePattern.
        The ability to match against NamePatterns is useful to see if two patterns
        conflict with each other and would create ambiguity.
        """
        match name:
            case AbsoluteName(s, m, i):
                if self.strata != "*" and self.strata != s:
                    return False
                if self.module != "*" and self.module != m:
                    return False
                if self.id != "*" and self.id != i:
                    return False
                return True
            case NamePattern(s, m, i):
                if self.strata != "*" and s != "*" and self.strata != s:
                    return False
                if self.module != "*" and m != "*" and self.module != m:
                    return False
                if self.id != "*" and i != "*" and self.id != i:
                    return False
                return True
            case _:
                raise ValueError(f"Unsupported match: {type(name)}")

    def __str__(self) -> str:
        return f"{self.strata}::{self.module}::{self.id}"


@dataclass(frozen=True)
class ModuleNamePattern:
    """
    A name with a module and attribute ID that allows wildcards (*).
    Mostly this is useful to provide parameters to GPMs, which don't have
    a concept of which strata they belong to. A ModuleNamePattern can be
    transformed into a full NamePattern by adding the strata.
    """

    module: str
    id: str

    def __post_init__(self):
        _validate_pattern_segments(self.module, self.id)

    @classmethod
    def parse(cls, name: str) -> Self:
        """
        Parse a pattern from a ::-delimited string. As a shorthand, you can omit
        a preceding wildcard segment and it will be automatically filled in,
        e.g.,"a" will become "*::a".
        """
        if len(name) == 0:
            raise ValueError("Empty string is not a valid name.")
        parts = name.split("::")
        match len(parts):
            case 1:
                return cls("*", *parts)
            case 2:
                return cls(*parts)
            case _:
                raise ValueError("Invalid number of parts for module name pattern.")

    def to_absolute(self, strata: str) -> NamePattern:
        """Creates a full name pattern by providing the strata."""
        return NamePattern(strata, self.module, self.id)

    def __str__(self) -> str:
        return f"{self.module}::{self.id}"


##############
# Attributes #
##############


AttributeT = TypeVar("AttributeT", bound=AttributeType)
"""The data type of an attribute; maps to the numpy type of the attribute array."""

# NOTE: I had AttributeT as covariant originally but that seems to cause pyright
# some headache interpreting typed method overloads (which is practically the
# whole point of making AttributeDef generic).
# So for now this must be invariant, but I think that's okay.
# Rather than being able to say things like:
# `AttributeDef[type[int] | type[float]]`
# you have to say `AttributeDef[type[int]] | AttributeDef[type[float]]`.


@dataclass(frozen=True)
class AttributeDef(Generic[AttributeT]):
    """
    Definition of a simulation attribute;
    the identity plus optional default value and comment.
    """

    name: str
    type: AttributeT
    shape: DataShape
    default_value: AttributeValue | None = field(default=None, compare=False)
    comment: str | None = field(default=None, compare=False)

    def __post_init__(self):
        if acceptable_name.match(self.name) is None:
            raise ValueError(f"Invalid attribute name: {self.name}")
        try:
            dtype_as_np(self.type)
        except Exception as e:
            msg = (
                f"AttributeDef's type is not correctly specified: {self.type}\n"
                "See documentation for appropriate type designations."
            )
            raise ValueError(msg) from e

        if (
            self.default_value is not None  #
            and not dtype_check(self.type, self.default_value)
        ):
            msg = (
                "AttributeDef's default value does not align with its dtype "
                f"('{self.name}')."
            )
            raise ValueError(msg)


############
# Database #
############


T = TypeVar("T")
"""Type of database values."""


class Match(NamedTuple, Generic[T]):
    """The result of a database query."""

    pattern: NamePattern
    value: T


class Database(Generic[T]):
    """
    A simple database implementation which provides namespaced key/value pairs.
    Namespaces are in the form "a::b::c", where "a" is the strata,
    "b" is the module, and "c" is the attribute name.
    Values are permitted to be assigned with wildcards (specified by asterisks),
    so that key "*::b::c" matches queries for "a::b::c" as well as "z::b::c".

    This is intended for tracking parameter values as given by the user,
    in constrast to `DataResolver` which is for tracking fully-evaluated parameters.
    """

    _data: dict[NamePattern, T]

    def __init__(self, data: dict[NamePattern, T]):
        """Constructor.

        Parameters
        ----------
        data : dict[NamePattern, T]
            the values in this database
        """
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
        """Query this database for a key match.

        Parameters
        ----------
        key : str | AbsoluteName
            the name to find; if given as a string, we must be able to parse it as a
            valid AbsoluteName

        Returns
        -------
        Match[T] | None
            the found value, if any, else None
        """
        if not isinstance(key, AbsoluteName):
            key = AbsoluteName.parse(key)
        for pattern, value in self._data.items():
            if pattern.match(key):
                return Match(pattern, value)
        return None

    def to_dict(self) -> dict[NamePattern, T]:
        """Return a copy of this database's data."""
        return {**self._data}


class DatabaseWithFallback(Database[T]):
    """
    A specialization of Database which has a fallback Database.
    If a match is not found in this DB's keys, the fallback is checked (recursively).
    """

    _fallback: Database[T]

    def __init__(self, data: dict[NamePattern, T], fallback: Database[T]):
        """Constructor.

        Parameters
        ----------
        data : dict[NamePattern, T]
            the highest priority values in the database
        fallback : Database[T]
            a database containing fallback values; if a match cannot be found
            in `data`, this database will be checked
        """
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
    A specialization of Database which has a set of fallback Databases, one per strata.
    For example, we might query this DB for "a::b::c". If we do not have a match in our
    own key/values, but we do have a fallback DB for "a", we will query that fallback
    (which could continue recursively).
    """

    _children: dict[str, Database[T]]

    def __init__(self, data: dict[NamePattern, T], children: dict[str, Database[T]]):
        """Constructor.

        Parameters
        ----------
        data : dict[NamePattern, T]
            the highest-priority values in the database
        children : dict[str, Database[T]]
            fallback databases by strata; if a match can't be found in `data`,
            the database for the matching strata (if any) will be checked
        """
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
    """Check that we can adapt the given `value` to the given type and shape,
    given dimensional information. Raises AttributeException if not."""
    if not np.can_cast(value, dtype_as_np(data_type)):
        raise AttributeException("Not a compatible type.")
    if not data_shape.matches(dim, value):
        raise AttributeException("Not a compatible shape.")


def adapt(
    data_type: AttributeType,
    data_shape: DataShape,
    dim: Dimensions,
    value: AttributeArray,
) -> AttributeArray:
    """Adapt the given `value` to the given type and shape, given dimensional
    information. Raises AttributeException if this fails."""
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
        raise AttributeException("Failed to adapt value.") from e


class DataResolver:
    """A sort of database for data attributes in the context of a simulation.
    Data (typically parameter values) are provided by the user as a collection
    of key-value pairs, with the values in several forms. These are evaluated
    to turn them into our internal data representation which is most useful
    for simulation execution (a numpy array, to be exact).

    While the keys can be described using NamePatterns, when they are resolved
    they are tracked by their full AbsoluteName to facilitate lookups by the
    systems that use them. It is possible that different AbsoluteNames actually
    resolve to the same value (which is done in a memory-efficient way).

    Meanwhile the usage of values adds its own complexity. When a value is used
    to fulfill the data requirements of a system, we want that value to be of
    a known type and shape. If two requirements are fulfilled by the same value,
    it it possible that the requirements will have different specifications for
    type and shape. Rather than be over-strict, and enforce that this can never happen,
    we allow this provided the given value can be successfully coerced to fit
    both requirements independently. For example, a scalar integer value can be coerced
    to both an N-shaped array of floats as well as a T-shaped array of integers.
    DataResolver accomplishes this flexibility in an efficient way by storing
    both the original values and all adapted values. If multiple requirements
    specify the same type/shape adaptation for one value, the adaptation only needs
    to happen once.

    DataResolver is partially mutable -- new values can be added but values cannot
    be overwritten or removed.
    """

    Key = tuple[AbsoluteName, AttributeType, DataShape]

    _dim: Dimensions
    _raw_values: dict[AbsoluteName, AttributeArray]
    _adapted_values: dict[Key, AttributeArray]

    def __init__(
        self,
        dim: Dimensions,
        values: dict[AbsoluteName, AttributeArray] | None = None,
    ):
        """Constructs a resolver.

        Parameters
        ----------
        dim: Dimensions
            the critical dimensions of the context in which these values have
            been evaluated; this is needed to perform shape adaptations
        values : dict[AbsoluteName, AttributeArray], optional
            a collection of values that should be in the resolver to begin with
        """
        self._dim = dim
        self._raw_values = values or {}
        self._adapted_values = {}

    def has(self, name: AbsoluteName) -> bool:
        """Tests whether or not a given name is in this resolver.

        Parameters
        ----------
        name : AbsoluteName
            the name to test

        Returns
        -------
        bool
            True if `name` is in this resolver"""
        return name in self._raw_values

    def add(self, name: AbsoluteName, value: AttributeArray) -> None:
        """Adds a value to this resolver. You may not overwrite an existing name.

        Parameters
        ----------
        name : AbsoluteName
            the name for the value
        value : AttributeArray
            the value to add

        Raises
        ------
        ValueError
            if `name` is already in this resolver
        """
        if name in self._raw_values:
            err = (
                f"A value for '{name}' already exists in this resolver; "
                "values cannot be overwritten."
            )
            raise ValueError(err)
        self._raw_values[name] = value

    def resolve(self, name: AbsoluteName, definition: AttributeDef) -> AttributeArray:
        """Resolves a value known by `name` to fit the given requirement `definition`.

        Parameters
        ----------
        name : AbsoluteName
            the name of the value to resolve
        definition : AttributeDef
            the definition of the requirement being fulfilled (which is needed because
            it contains the type and shape information)

        Returns
        -------
        AttributeArray
            the resolved value, adapted if necessary

        Raises
        ------
        AttributeException
            if the resolution fails
        """
        key = (name, definition.type, definition.shape)
        if key in self._adapted_values:
            return self._adapted_values[key]

        if name not in self._raw_values:
            raise AttributeException(f"No value for name '{name}'")
        value = self._raw_values[name]
        adapted_value = adapt(definition.type, definition.shape, self._dim, value)
        self._adapted_values[key] = adapted_value
        return adapted_value

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
        """Extract a dictionary from this DataResolver of all of its
        (non-adapted) keys and values.

        Parameters
        ----------
        simplify_names : bool, default=False
            by default, names are returned as `AbsoluteName` objects; if True,
            return stringified names as a convenience

        Returns
        -------
        dict[AbsoluteName, AttributeArray] | dict[str, AttributeArray]
            the dictionary of all values in this resolver, with names
            either simplified or not according to `simplify_names`
        """
        if simplify_names:
            return {str(k): v for k, v in self._raw_values.items()}
        return {**self._raw_values}


class Requirement(NamedTuple):
    """A RUME data requirement: a name and a definition."""

    name: AbsoluteName
    definition: AttributeDef

    def __str__(self) -> str:
        properties = [dtype_str(self.definition.type), str(self.definition.shape)]
        return f"{self.name} ({', '.join(properties)})"


class Resolution(ABC):
    """What source was used to resolve a requirement in a RUME?"""

    # NOTE: it is required that all Resolution instances be hashable!
    # NOTE: This should be treated as a sealed type.

    cacheable: bool


@dataclass(frozen=True)
class MissingValue(Resolution):
    """Requirement was not resolved."""

    cacheable: bool = field(init=False, default=False)

    def __str__(self) -> str:
        return "missing"


@dataclass(frozen=True)
class ParameterValue(Resolution):
    """Requirement was resolved by a RUME parameter."""

    cacheable: bool
    pattern: NamePattern

    def __str__(self) -> str:
        return f"parameter value '{self.pattern}'"


@dataclass(frozen=True)
class DefaultValue(Resolution):
    """Requirement was resolved by an attribute default value."""

    cacheable: bool = field(init=False, default=True)
    default_value: AttributeValue

    # NOTE: we need default_value here so that default resolutions
    # are only cached as one if they use the same value;
    # this is critical for conflicting defaults detection.
    # Since this also needs to be hashable, that means default values
    # cannot be numpy arrays (unless we add some code to handle those).

    def __str__(self) -> str:
        return "default value"


@dataclass(frozen=True)
class ResolutionTree:
    """Just the resolution part of a requirements tree; children are the dependencies
    of this resolution. If two values share the same resolution tree, and if the tree
    is comprised entirely of pure functions and values, then the resolution result
    would be the same."""

    resolution: Resolution
    children: "tuple[ResolutionTree, ...]"

    @property
    def is_tree_cacheable(self) -> bool:
        return (
            self.resolution.cacheable  #
            and all(x.is_tree_cacheable for x in self.children)
        )


class RecursiveValue(Protocol):
    """A parameter value that itself may depend on other parameter values."""

    requirements: Sequence[AttributeDef]
    """Defines the data requirements for this value."""
    randomized: bool
    """Should this value be re-evaluated every time it's referenced?
    (Mostly useful for randomized results.)"""


@singledispatch
def is_recursive_value(value: object) -> TypeGuard[RecursiveValue]:
    """TypeGuard for RecursiveValues, implemented by single dispatch."""
    return False


V = TypeVar("V")
"""The value type for a requirements tree."""


@dataclass(frozen=True)
class ReqTree(Generic[V]):
    """A requirements tree describes how data requirements are resolved for a RUME.
    Models used in the RUME have a set of requirements, each of which may be fulfilled
    by RUME parameters or default values. RUME parameters may also have data
    requirements, and those requirements may have requirements, etc., hence the need
    to represent this as a tree structure. The top of a `ReqTree` is a `ReqRoot` and
    each requirement is a `ReqNode`. Each `ReqNode` tracks the attribute itself
    (its `AbsoluteName` and `AttributeDef`) and whether it is fulfilled and if so how.
    """

    children: "tuple[ReqNode, ...]"

    @abstractmethod
    def to_string(
        self,
        /,
        format_name: Callable[[AbsoluteName], str] = str,
        depth: int = 0,
    ) -> str:
        """Convert this ReqTree to a string.

        Parameters
        ----------
        format_name : Callable[[AbsoluteName], str], default=str
            A method to convert an absolute name into a string.
            This allows you to control how absolute names are rendered.
        depth : int, default=0
            The resolution "depth" of this node in the tree.
            A requirement that itself is required by one other requirement
            would have a depth of 1. Top-level requirements have a depth of 0.
        """
        return "\n".join(
            [x.to_string(format_name=format_name, depth=depth) for x in self.children]
        )

    def traverse(self) -> "Iterable[ReqNode]":
        """Perform a depth-first traversal of the nodes of this tree."""
        for x in self.children:
            yield from x.traverse()
        # NOTE: method overridden by ReqNode; yields children and then itself

    def missing(self) -> Iterable[Requirement]:
        """Return missing requirements."""
        return (
            Requirement(node.name, node.definition)
            for node in self.traverse()
            if isinstance(node.resolution, MissingValue)
        )

    def evaluate(
        self,
        dim: SimDimensions | None,
        scope: GeoScope | None,
        rng: np.random.Generator | None,
    ) -> DataResolver:
        """Evaluate this tree. See: `evaluate_requirements()`."""
        return evaluate_requirements(self, dim, scope, rng)

    @staticmethod
    def of(
        requirements: Mapping[AbsoluteName, AttributeDef],
        params: Database[V],
    ) -> "ReqTree":
        """Compute the requirements tree for the given set of requirements
        and a database supplying values. Note that missing values do not
        stop us from computing the tree -- these nodes will have `MissingValue`
        as the resolution.

        Parameters
        ----------
        requirements : Mapping[AbsoluteName, AttributeDef]
            the top-level requirements of the tree
        params : Database[V]
            the database of values, where each value may be "recursive"
            in the sense of having its own data requirements

        Raises
        ------
        AttributeException
            if the tree cannot be evaluated, for instance, due to containing
            circular dependencies
        """

        def recurse(
            name: AbsoluteName,
            definition: AttributeDef,
            chain: list[AbsoluteName],
        ) -> ReqNode:
            if name in chain:
                err = f"Circular dependency in evaluation of parameter {name}"
                raise AttributeException(err)

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
    """A non-root node of a requirements tree, identifying a requirement,
    how (or if) it was resolved, and its value (if available)."""

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
                color = AnsiColor.CYAN
                resd = ""
            case MissingValue():
                color = AnsiColor.RED
                resd = ""
            case x:
                # have to handle the impossible case
                # because Python doesn't have sealed types yet
                raise NotImplementedError(f"Unsupported resolution type ({type(x)})")

        indent = "  " * depth
        corner = "└╴" if depth > 0 else ""
        name = ansi_stylize(format_name(self.name), color)
        prop = ansi_stylize(f"({', '.join(properties)})", color, AnsiStyle.ITALIC)

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
        """Extracts the resolution tree from this requirements tree."""
        return ResolutionTree(
            self.resolution,
            tuple(x.as_res_tree() for x in self.children),
        )


@singledispatch
def evaluate_param(
    value: object,
    name: AbsoluteName,
    data: DataResolver,
    dim: SimDimensions | None,
    scope: GeoScope | None,
    rng: np.random.Generator | None,
) -> AttributeArray:
    """Evaluate a parameter, transforming acceptable input values (type: ParamValue) to
    the form required internally by epymorph (AttributeArray). This handles different
    types of parameters by single dispatch, so there should be a registered
    implementation for every unique type. It is possible that the user is attempting to
    evaluate parameters with a partial context (`dim`, `scope`, `rng`), and so one or
    more of these may be missing. In that case, parameter evaluation is expected to
    happen on a "best effort" basis -- if no parameter requires the missing scope
    elements, parameter evaluation succeeds. Otherwise, this is expected to raise
    an `AttributeException`.

    Parameters
    ----------
    value : object
        the value being evaluated
    name : AbsoluteName
        the full name given to the value in the simulation context
    data : DataResolver
        a DataResolver instance which should contain values for all of the data
        requirements needed by this value (only `RecursiveValues` have requirements)
    dim : SimDimensions, optional
        the dimensional information of this simulation context, if available
    scope : GeoScope, optional
        the geographic scope information of this simulation context, if available
    rng : np.random.Generator, optional
        the random number generator to use, if available

    Returns
    -------
    AttributeArray
        the evaluated value
    """
    err = f"Parameter not a supported type (found: {type(value)})"
    raise AttributeException(err)


@evaluate_param.register
def _(
    value: np.ndarray,
    name: AbsoluteName,
    data: DataResolver,
    dim: SimDimensions | None,
    scope: GeoScope | None,
    rng: np.random.Generator | None,
) -> AttributeArray:
    # numpy array: make a copy so we don't risk unexpected mutations
    return value.copy()


@evaluate_param.register
def _(
    value: int | float | str | tuple | list,
    name: AbsoluteName,
    data: DataResolver,
    dim: SimDimensions | None,
    scope: GeoScope | None,
    rng: np.random.Generator | None,
) -> AttributeArray:
    # scalar value or python collection: re-pack it as a numpy array
    return np.asarray(value, dtype=None)


@evaluate_param.register
def _(
    value: type,
    name: AbsoluteName,
    data: DataResolver,
    dim: SimDimensions | None,
    scope: GeoScope | None,
    rng: np.random.Generator | None,
) -> AttributeArray:
    # forgot to instantiate? a common error worth checking for
    err = (
        "Parameter was given as a class instead of an instance. "
        "Did you forget to instantiate it?"
    )
    raise AttributeException(err)


def evaluate_requirements(
    req: ReqTree,
    dim: SimDimensions | None,
    scope: GeoScope | None,
    rng: np.random.Generator | None,
) -> DataResolver:
    """Evaluate all parameters in `req`, using the given simulation context
    (`dim`, `scope`, `rng`). You may attempt to evaluate parameters with a partial
    context, so one or more of these may be missing. In that case, parameter evaluation
    is happens on a "best effort" basis -- if no parameter requires the missing scope
    elements, parameter evaluation succeeds; otherwise raises an `AttributeException`.

    Parameters
    ----------
    req : ReqTree
        the requirements tree
    dim : SimDimensions, optional
        the dimensional information of this simulation context, if available
    scope : GeoScope, optional
        the geographic scope information of this simulation context, if available
    rng : np.random.Generator, optional
        the random number generator to use, if available

    Returns
    -------
    DataResolver
        the resolver containing all evaluated parameters
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
        raise AttributeException(err)

    # For each node in the requirements tree (traversing depth-first),
    # evaluate each value and validate it against the requirement's definition;
    # fulfilled requirements get stored to `resolved` and we use `evaluated` as
    # an evaluation cache, to avoid evaluating the same parameter (input value)
    # twice. Accumulate validation errors into `errors` to be raised as a group.
    errors = list[AttributeException]()
    resolved = DataResolver(dim or Dimensions.of())
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
                value = evaluate_param(node.value, node.name, resolved, dim, scope, rng)
                if res_tree.is_tree_cacheable:
                    evaluated[res_tree] = value

            # If we have the necessary info, check that the raw value
            # can be successfully adapted to fulfill this requirement.
            # Raise AttributeException if not.
            if dim is not None:
                try:
                    d = node.definition
                    assert_can_adapt(d.type, d.shape, dim, value)
                except AttributeException as e:
                    err = (
                        f"Attribute '{node.name}' ({node.resolution}) is "
                        f"not properly specified: {e}"
                    )
                    raise AttributeException(err)

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
                raise AttributeException(err)

        except AttributeException as e:
            errors.append(e)

    if len(errors) > 0:
        raise AttributeExceptionGroup("Errors found evaluating parameters.", errors)

    return resolved
