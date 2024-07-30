"""
A Database in epymorph is a way to organize values with our namespace system
of three hierarchical components, as in: "strata::module::attribute_id".
This gives us a lot of flexibility when specifying data requirements and values
which fulfill those requirements. For example, you can provide a value for "*::*::population"
to indicate that every strata and every module should use the same value if they need a "population"
attribute. Or you can provide "*::init::population" to indicate that only initializers should use this
value, and, presumably, another value is more appropriate for other modules like movement.

Hierarchical Database instances are also included which provide the ability to "layer" data
for a simulation -- if the outermost database has a matching value, that value is used, otherwise
the search for a match proceeds to the inner layers (recursively).
"""
from dataclasses import dataclass
from typing import Generic, NamedTuple, Self, TypeVar, overload

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

    def to_absolute(self, attrib_id: str) -> 'AbsoluteName':
        """Creates an absolute name by providing the attribute ID."""
        return AbsoluteName(self.strata, self.module, attrib_id)


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

    @classmethod
    def parse_with_defaults(cls, name: str, strata: str, module: str) -> Self:
        """
        Parse a module name from a ::-delimited string, where strata and module
        can be omitted to be filled from defaults.
        """
        def replace_star(string: str, default_value: str) -> str:
            return default_value if string == "*" else string

        parts = name.split("::")
        match parts:
            case [i]:
                return cls(strata, module, i)
            case [m, i]:
                return cls(strata, replace_star(m, module), i)
            case [s, m, i]:
                return cls(replace_star(s, strata), replace_star(m, module), i)
            case _:
                raise ValueError("Invalid number of parts for absolute name.")

    def __str__(self) -> str:
        return f"{self.strata}::{self.module}::{self.id}"

    def in_strata(self, strata: str) -> 'AbsoluteName':
        """Creates a new AbsoluteName that is a copy of this name but with the given strata."""
        return AbsoluteName(strata, self.module, self.id)

    def to_namespace(self) -> ModuleNamespace:
        """Extracts the module namespace of this name."""
        return ModuleNamespace(self.strata, self.module)

    def to_pattern(self) -> 'NamePattern':
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
    A name with a strata, module, and attribute ID that allows wildcards (*) so it can act
    as a pattern to match against AbsoluteNames.
    """
    strata: str
    module: str
    id: str

    def __post_init__(self):
        _validate_pattern_segments(self.strata, self.module, self.id)

    @classmethod
    def parse(cls, name: str) -> Self:
        """
        Parse a pattern from a ::-delimited string. As a shorthand, you can omit preceding wildcard
        segments and they will be automatically filled in, e.g., "a" will become "*::*::a" and
        "a::b" will become "*::a::b".
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

    def match(self, name: 'AbsoluteName | NamePattern') -> bool:
        """
        Test this pattern to see if it matches the given AbsoluteName or NamePattern.
        The ability to match against NamePatterns is useful to see if two patterns conflict
        with each other and would create ambiguity.
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
        Parse a pattern from a ::-delimited string. As a shorthand, you can omit a preceding wildcard
        segment and it will be automatically filled in, e.g., "a" will become "*::a".
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


############
# Database #
############


T = TypeVar('T')
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
    """
    _data: dict[NamePattern, T]

    def __init__(self, data: dict[NamePattern, T]):
        self._data = data

        # Check for key ambiguity:
        conflicts = [
            key
            for key, _ in self._data.items()
            for other, _ in self._data.items()
            if key != other and key.match(other)
        ]

        if len(conflicts) > 0:
            msg = f"Keys in data source are ambiguous; conflicts:\n{', '.join(map(str, conflicts))}"
            raise ValueError(msg)

    @overload
    def query(self, key: str) -> Match[T] | None: ...
    @overload
    def query(self, key: AbsoluteName) -> Match[T] | None: ...

    def query(self, key: str | AbsoluteName) -> Match[T] | None:
        """Query this database for a key match."""
        if not isinstance(key, AbsoluteName):
            key = AbsoluteName.parse(key)
        for pattern, value in self._data.items():
            if pattern.match(key):
                return Match(pattern, value)
        return None

    def update(self, pattern: NamePattern, value: T) -> None:
        """Update (or add) a database value."""
        # If this is a new key, we need to check for conflicts:
        if pattern not in self._data:
            conflicts = [
                existing_key
                for existing_key, _ in self._data.items()
                if existing_key != pattern and existing_key.match(pattern)
            ]

            if len(conflicts) > 0:
                msg = "Adding this key would make key resolution ambiguous; " \
                    f"conflicts:\n{', '.join(map(str, conflicts))}"
                raise ValueError(msg)

        self._data[pattern] = value


class DatabaseWithFallback(Database[T]):
    """
    A specialization of Database which has a fallback Database.
    If a match is not found in this DB's keys, the fallback is checked (recursively).
    """
    _fallback: Database[T]

    def __init__(self, data: dict[NamePattern, T], fallback: Database[T]):
        super().__init__(data)
        self._fallback = fallback

    @overload
    def query(self, key: str) -> Match[T] | None: ...
    @overload
    def query(self, key: AbsoluteName) -> Match[T] | None: ...

    def query(self, key: str | AbsoluteName) -> Match[T] | None:
        if not isinstance(key, AbsoluteName):
            key = AbsoluteName.parse(key)
        # First check "our" data:
        matched = super().query(key)
        # Otherwise check fallback data:
        if matched is None:
            matched = self._fallback.query(key)
        return matched


class DatabaseWithStrataFallback(Database[T]):
    """
    A specialization of Database which has a set of fallback Databases, one per strata.
    For example, we might query this DB for "a::b::c". If we do not have a match in our 
    own key/values, but we do have a fallback DB for "a", we will query that fallback 
    (which could continue recursively).
    """
    _children: dict[str, Database[T]]

    def __init__(self, data: dict[NamePattern, T], children: dict[str, Database[T]]):
        super().__init__(data)
        self._children = children

    @overload
    def query(self, key: str) -> Match[T] | None: ...
    @overload
    def query(self, key: AbsoluteName) -> Match[T] | None: ...

    def query(self, key: str | AbsoluteName) -> Match[T] | None:
        if not isinstance(key, AbsoluteName):
            key = AbsoluteName.parse(key)
        # First check "our" data:
        matched = super().query(key)
        # Otherwise check strata fallback for match:
        if matched is None and (db := self._children.get(key.strata)) is not None:
            matched = db.query(key)
        return matched


__all__ = [
    'ModuleNamespace',
    'AbsoluteName',
    'ModuleName',
    'AttributeName',
    'NamePattern',
    'ModuleNamePattern',
    'Match',
    'Database',
    'DatabaseWithFallback',
    'DatabaseWithStrataFallback',
]
