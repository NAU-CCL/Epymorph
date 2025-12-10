"""
The module to organize epymorph's configuration settings and related functionality.
"""

import os
from pathlib import Path
from typing import Callable, Generic, NamedTuple, TypeVar, overload
from warnings import warn


class InvalidBooleanError(Exception):
    """Raised when a value cannot be interpreted as a boolean."""


def strtobool(value: str) -> bool:
    """
    Interpret a string as a boolean, if possible.

    We use the widely-adopted `distutils.util.strtobool` convention for
    strings which can be interpreted as booleans. To quote the documentation:
    "True values are y, yes, t, true,  on,  and 1;
    false values are n, no,  f, false, off, and 0."
    Whitespace will be stripped and values are case-insensitive.

    Parameters
    ----------
    value :
        The string to interpret.

    Returns
    -------
    :
        The boolean value if valid.

    Raises
    ------
    InvalidBooleanError
        If the string is not an accepted boolean value format.
    """
    value = value.strip().lower()
    if value in {"y", "yes", "t", "true", "on", "1"}:
        return True
    if value in {"n", "no", "f", "false", "off", "0"}:
        return False
    raise InvalidBooleanError()


@overload
def env_flag(name: str, default_value: bool) -> bool: ...


@overload
def env_flag(name: str, default_value: None = None) -> bool | None: ...


def env_flag(name: str, default_value: bool | None = None) -> bool | None:
    """
    Load an environment variable assuming it represents a boolean setting.

    See the `strtobool` function for the set of strings which are allowed
    for True and False.

    Parameters
    ----------
    name :
        The name of the environment variable to load.
    default_value :
        A default value to use in case the variable is not present or can't be
        interpreted as a boolean.

    Returns
    -------
    :
        If the named variable is present and if the value can be interpreted
        as a boolean, return the boolean. Else return `default_value`.
    """
    value = os.getenv(name)
    if value is None:
        return default_value
    try:
        return strtobool(value)
    except InvalidBooleanError:
        warn(
            f"Environment variable {name} was specified with a value that cannot be "
            f"interpreted as a boolean string. Received: '{value}'. "
            "Prefer 'true' for True and 'false' for False."
        )
        return default_value


@overload
def env_path(
    name: str,
    default_value: Path,
    *,
    ensure_exists: bool = False,
) -> Path: ...


@overload
def env_path(
    name: str,
    default_value: None = None,
    *,
    ensure_exists: bool = False,
) -> Path | None: ...


def env_path(
    name: str,
    default_value: Path | None = None,
    *,
    ensure_exists: bool = False,
) -> Path | None:
    """
    Load an environment variable assuming it represents a Path.

    Parameters
    ----------
    name :
        The name of the environment variable to load.
    default_value :
        A default value to use in case the variable is not present or can't be
        interpreted as a Path.
    ensure_exists :
        True to attempt to create the Path (as a directory) if it doesn't already exist.

    Returns
    -------
    :
        If the named variable is present and if the value can be interpreted
        as a Path, return the Path. Else return `default_value`.
    """
    value = os.getenv(name)
    if value is None:
        path = default_value
    else:
        try:
            path = Path(value)
        except TypeError:
            path = default_value

    if path is not None and ensure_exists:
        path.mkdir(parents=True, exist_ok=True)

    return path


def env_path_list(name: str) -> list[Path]:
    """
    Load an environment variable assuming it represents a semicolon-separated list of
    Paths.

    Note: empty string paths will be ignored, since these are likely to be mistakes.

    Parameters
    ----------
    name :
        The name of the environment variable to load.

    Returns
    -------
    :
        If the named variable is present and if the value can be interpreted
        as a list of Paths, return the Paths. Else return an empty list.
    """
    value = os.getenv(name)
    if value is None:
        return []
    try:
        return [Path(x.strip()) for x in value.split(";") if x.strip() != ""]
    except TypeError:
        return []


ValueT = TypeVar("ValueT")
"""The type of the value of a setting."""


class Setting(NamedTuple, Generic[ValueT]):
    """An epymorph configuration setting."""

    name: str
    """The name of the setting."""
    description: str
    """The description of the setting."""
    getter: Callable[[], ValueT]
    """A function to get the value of the setting."""

    def get(self) -> ValueT:
        """Get the current value of the setting."""
        return self.getter()


SETTINGS = list[Setting]()
"""A list of epymorph configuration settings."""


def declare_setting(
    name: str,
    description: str,
    getter: Callable[[], ValueT],
) -> Setting[ValueT]:
    """
    Declare an application configuration setting.

    Parameters
    ----------
    name :
        The setting name.
    description :
        The setting description.
    getter :
        A function to get the value of the setting.

    Returns
    -------
    :
        The Setting object.

    Raises
    ------
    ValueError
        If a setting of the same name has been previously declared.
    """
    for x in SETTINGS:
        if x.name == name:
            raise ValueError(f"Duplicate Setting name: {name}")
    setting = Setting(name, description, getter)
    SETTINGS.append(setting)
    return setting
