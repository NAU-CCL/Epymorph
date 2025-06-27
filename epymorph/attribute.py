"""
Data attributes are how epymorph keeps track of input data for simulations.
An IPM's requirements are expressed as attributes, and when supplying data
values to a RUME you use names to match values to the requirement(s) they fulfill.

This module provides a foundation for that, encoding systems for how
names work and defining the `DataAttribute` object itself.
"""

from dataclasses import dataclass, field
from typing import Generic, Self, TypeVar

from epymorph.data_shape import DataShape
from epymorph.data_type import AttributeType, AttributeValue, dtype_as_np, dtype_check
from epymorph.util import acceptable_name

########################
# Names and Namespaces #
########################


def _validate_pattern_segments(*names: str) -> None:
    for n in names:
        if len(n) == 0:
            raise ValueError("Invalid pattern: cannot use empty strings.")
        if "::" in n:
            raise ValueError("Invalid pattern: cannot contain '::'.")


def _validate_name_segments(*names: str) -> None:
    for n in names:
        if len(n) == 0:
            raise ValueError("Invalid name: cannot use empty strings.")
        if n == "*":
            raise ValueError("Invalid name: cannot use wildcards (*).")
        if "::" in n:
            raise ValueError("Invalid name: cannot contain '::'.")


@dataclass(frozen=True)
class ModuleNamespace:
    """
    A namespace which specifies strata and module.

    Parameters
    ----------
    strata :
        The strata name.
    module :
        The module name.
    """

    strata: str
    """The strata name."""
    module: str
    """The module name."""

    def __post_init__(self):
        _validate_name_segments(self.strata, self.module)

    @classmethod
    def parse(cls, name: str) -> Self:
        """
        Parse a module namespace from a ::-delimited string.

        Parameters
        ----------
        name :
            The string to parse.

        Returns
        -------
        :
            The new namespace instance.

        Raises
        ------
        ValueError
            If the given string cannot be parsed.

        Examples
        --------
        >>> ModuleNamespace.parse("gpm:human::ipm")
        ModuleNamespace(strata="gpm:human", module="ipm")
        """
        parts = name.split("::")
        if len(parts) != 2:
            raise ValueError("Invalid number of parts for namespace.")
        return cls(*parts)

    def __str__(self) -> str:
        return f"{self.strata}::{self.module}"

    def to_absolute(self, attrib_id: str) -> "AbsoluteName":
        """
        Create an absolute name by providing the attribute ID.

        Parameters
        ----------
        attrib_id :
            The attribute ID to append to the namespace.

        Returns
        -------
        :
            The new absolute name instance.
        """
        return AbsoluteName(self.strata, self.module, attrib_id)


@dataclass(frozen=True)
class AbsoluteName:
    """
    A fully-specified name: strata, module, and attribute ID.

    Parameters
    ----------
    strata :
        The strata name.
    module :
        The module name.
    id :
        The attribute ID.
    """

    strata: str
    """The strata name."""
    module: str
    """The module name."""
    id: str
    """The attribute ID."""

    def __post_init__(self):
        _validate_name_segments(self.strata, self.module, self.id)

    @classmethod
    def parse(cls, name: str) -> Self:
        """
        Parse a module name from a ::-delimited string.

        Parameters
        ----------
        name :
            The string to parse.

        Returns
        -------
        :
            The new name instance.

        Raises
        ------
        ValueError
            If the given string cannot be parsed.

        Examples
        --------
        >>> AbsoluteName.parse("gpm:human::ipm::beta")
        AbsoluteName(strata="gpm:human", module="ipm", id="beta")
        """
        parts = name.split("::")
        if len(parts) != 3:
            raise ValueError("Invalid number of parts for absolute name.")
        return cls(*parts)

    def __str__(self) -> str:
        return f"{self.strata}::{self.module}::{self.id}"

    def in_strata(self, new_strata: str) -> "AbsoluteName":
        """
        Create a new `AbsoluteName` that is a copy of this name
        but with the given strata.

        Parameters
        ----------
        new_strata :
            The strata to use to replace this name's strata.

        Returns
        -------
        :
            The new name instance.
        """
        return AbsoluteName(new_strata, self.module, self.id)

    def with_id(self, new_id: str) -> "AbsoluteName":
        """
        Create a new `AbsoluteName` that is a copy of this name
        but with the given ID.

        Parameters
        ----------
        new_id :
            The attribute ID to use to replace this name's ID.

        Returns
        -------
        :
            The new name instance.
        """
        return AbsoluteName(self.strata, self.module, new_id)

    def to_namespace(self) -> ModuleNamespace:
        """
        Extract the module namespace part of this name.

        Returns
        -------
        :
            The new module namespace instance.
        """
        return ModuleNamespace(self.strata, self.module)

    def to_pattern(self) -> "NamePattern":
        """
        Convert this name to a pattern that is an exact match for this name.

        Returns
        -------
        :
            The new name pattern instance.
        """
        return NamePattern(self.strata, self.module, self.id)


STRATA_PLACEHOLDER = "(unspecified)"
"""The strata name to use when one has not been specified."""
MODULE_PLACEHOLDER = "(unspecified)"
"""The module name to use when one has not been specified."""
ID_PLACEHOLDER = "(unspecified)"
"""The attribute ID to use when one has not been specified."""
NAMESPACE_PLACEHOLDER = ModuleNamespace(STRATA_PLACEHOLDER, MODULE_PLACEHOLDER)
"""A namespace to use when we don't need to be specific."""
NAME_PLACEHOLDER = NAMESPACE_PLACEHOLDER.to_absolute(ID_PLACEHOLDER)
"""An absolute name to use when we don't need to be specific."""


@dataclass(frozen=True)
class ModuleName:
    """
    A partially-specified name with module and attribute ID.

    Parameters
    ----------
    module :
        The module name.
    id :
        The attribute ID.
    """

    module: str
    """The module name."""
    id: str
    """The attribute ID."""

    def __post_init__(self):
        _validate_name_segments(self.module, self.id)

    @classmethod
    def parse(cls, name: str) -> Self:
        """
        Parse a module name from a ::-delimited string.

        Parameters
        ----------
        name :
            The string to parse.

        Returns
        -------
        :
            The new name instance.

        Raises
        ------
        ValueError
            If the given string cannot be parsed.

        Examples
        --------
        >>> ModuleName.parse("ipm::beta")
        ModuleName(module="ipm", id="beta")
        """
        parts = name.split("::")
        if len(parts) != 2:
            raise ValueError("Invalid number of parts for module name.")
        return cls(*parts)

    def __str__(self) -> str:
        return f"{self.module}::{self.id}"

    def to_absolute(self, strata: str) -> AbsoluteName:
        """
        Create an absolute name by providing the strata.

        Parameters
        ----------
        strata :
            The strata name to use.

        Returns
        -------
        :
            The new absolute name instance.
        """
        return AbsoluteName(strata, self.module, self.id)


@dataclass(frozen=True)
class AttributeName:
    """
    A partially-specified name with just an attribute ID.

    Parameters
    ----------
    id :
        The attribute ID.
    """

    id: str
    """The attribute ID."""

    def __post_init__(self):
        _validate_name_segments(self.id)

    def __str__(self) -> str:
        return self.id


@dataclass(frozen=True)
class NamePattern:
    """
    A name with a strata, module, and attribute ID that allows wildcards (*) so it can
    act as a pattern to match against `AbsoluteNames`.

    Parameters
    ----------
    strata :
        The strata name or wildcard.
    module :
        The module name or wildcard.
    id :
        The attribute ID or wildcard.
    """

    strata: str
    """The strata name or wildcard."""
    module: str
    """The module name or wildcard."""
    id: str
    """The attribute ID or wildcard."""

    def __post_init__(self):
        _validate_pattern_segments(self.strata, self.module, self.id)

    @classmethod
    def parse(cls, name: str) -> Self:
        """
        Parse a pattern from a ::-delimited string. As a shorthand, you can omit
        preceding wildcard segments and they will be automatically filled in,
        e.g., "a" will become "*::*::a" and "a::b" will become "*::a::b".

        Parameters
        ----------
        name :
            The string to parse.

        Returns
        -------
        :
            The new name pattern instance.

        Raises
        ------
        ValueError
            If the given string cannot be parsed.

        Examples
        --------
        >>> NamePattern.parse("gpm:human::ipm::beta")
        NamePattern(strata="gpm:human", module="ipm", id="beta")

        >>> NamePattern.parse("*::*::beta")
        NamePattern(strata="*", module="*", id="beta")

        >>> NamePattern.parse("beta")
        NamePattern(strata="*", module="*", id="beta")
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
        """
        Coerce the given value to a `NamePattern`.

        Parameters
        ----------
        name :
            The name to coerce. This will be parsed if given as a string or
            returned as-is if it's already a `NamePattern` instance.

        Returns
        -------
        :
            The name pattern instance.

        Raises
        ------
        ValueError
            If `name` is given as a string but cannot be parsed.
        """
        return name if isinstance(name, NamePattern) else NamePattern.parse(name)

    def match(self, name: "AbsoluteName | NamePattern") -> bool:
        """
        Test this pattern to see if it matches the given `AbsoluteName` or
        `NamePattern`. The ability to match against `NamePatterns` is useful to see if
        two patterns conflict with each other and would create ambiguity.

        Parameters
        ----------
        name :
            The name to check against this pattern.

        Returns
        -------
        :
            True if there is a match, false otherwise.
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
    a concept of which strata they belong to. A `ModuleNamePattern` can be
    transformed into a full `NamePattern` by adding the strata.

    Parameters
    ----------
    module :
        The module name or wildcard.
    id :
        The attribute ID or wildcard.
    """

    module: str
    """The module name or wildcard."""
    id: str
    """The attribute ID or wildcard."""

    def __post_init__(self):
        _validate_pattern_segments(self.module, self.id)

    @classmethod
    def parse(cls, name: str) -> Self:
        """
        Parse a pattern from a ::-delimited string. As a shorthand, you can omit
        a preceding wildcard segment and it will be automatically filled in,
        e.g.,"a" will become "*::a".

        Parameters
        ----------
        name :
            The string to parse.

        Returns
        -------
        :
            The new name pattern instance.

        Raises
        ------
        ValueError
            If the given string cannot be parsed.

        Examples
        --------
        >>> ModuleNamePattern.parse("ipm::beta")
        ModuleNamePattern(module="ipm", id="beta")

        >>> ModuleNamePattern.parse("*::beta")
        ModuleNamePattern(module="*", id="beta")

        >>> ModuleNamePattern.parse("beta")
        ModuleNamePattern(module="*", id="beta")
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
        """
        Create a full name pattern by providing the strata.

        Parameters
        ----------
        strata :
            The strata to use.

        Returns
        -------
        :
            The new name pattern instance.
        """
        return NamePattern(strata, self.module, self.id)

    def __str__(self) -> str:
        return f"{self.module}::{self.id}"


##############
# Attributes #
##############


AttributeT = TypeVar("AttributeT", bound=AttributeType)
"""The data type of an attribute; maps to the numpy type of the attribute array."""


@dataclass(frozen=True)
class AttributeDef(Generic[AttributeT]):
    """
    The definition of a data attribute. Attributes are frequently used to define the
    data requirements of things like IPMs and parameter functions.

    `AttributeDef` is generic on the `AttributeType` which describes the type
    of the data (`AttributeT`).

    Parameters
    ----------
    name :
        The name used to identify the attribute.
    type :
        The type of the data.
    shape :
        The expected array shape of the data.
    default_value :
        An optional default value.
    comment :
        An optional description of the attribute.
    """

    name: str
    """The name used to identify the attribute."""
    type: AttributeT
    """The type of the data."""
    shape: DataShape
    """The expected array shape of the data."""
    default_value: AttributeValue | None = field(default=None, compare=False)
    """An optional default value."""
    comment: str | None = field(default=None, compare=False)
    """An optional description of the attribute."""

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
