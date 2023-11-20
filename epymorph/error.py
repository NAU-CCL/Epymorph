"""
A common exception framework for epymorph.
"""
from __future__ import annotations

from abc import ABC
from typing import Self


class ValidationException(Exception, ABC):
    """Superclass for exceptions which happen during simulation validation."""


class GeoValidationException(ValidationException):
    """Exception for invalid GEO."""
    attribute_errors: list[str] | None

    def __init__(self, message: str, attribute_errors: list[str] | None = None):
        super().__init__(message)
        self.attribute_errors = attribute_errors

    def pretty(self) -> str:
        """Returns a nicely-formatted string for this exception."""
        if self.attribute_errors is None:
            return str(self)
        return str(self) + "".join((f"\n- {e}" for e in self.attribute_errors))


class IpmValidationException(ValidationException):
    """Exception for invalid IPM."""


class MmValidationException(ValidationException):
    """Exception for invalid MM."""


class SimValidationException(ValidationException):
    """
    Exception for cases where a simulation is invalid as configured,
    typically because the MM, IPM, or Initializer require data attributes
    that are not available.
    """


class SimulationException(Exception, ABC):
    """Superclass for exceptions which happen during simulation runtime."""


class InitException(SimulationException):
    """Exception for invalid initialization."""

    @classmethod
    def for_arg(cls, name: str, addendum: str | None = None) -> Self:
        """Init is invalid due to improper/missing argument."""
        if addendum is None:
            return cls(f"Invalid argument '{name}'.")
        else:
            return cls(f"Invalid argument '{name}': {addendum}")


class IpmSimException(SimulationException):
    """Exception during IPM processing."""


class MmSimException(SimulationException):
    """Exception during MM processing."""
