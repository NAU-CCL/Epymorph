"""
A common exception framework for epymorph.
"""
from abc import ABC
from typing import Self


class UnknownModel(Exception):
    """Exception for the inability to load a model as specified."""
    model_type: str
    name_or_path: str

    def __init__(self, model_type: str, name_or_path: str):
        super().__init__(f"Unable to load {model_type} model at {name_or_path}")
        self.model_type = model_type
        self.name_or_path = name_or_path


class ModelRegistryException(Exception):
    """Exception during model library initialization."""


class ValidationException(Exception, ABC):
    """Superclass for exceptions which happen during simulation validation."""


class AttributeException(ValidationException):
    """Exception handling data attributes."""


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


class SimCompileException(SimulationException):
    """Exception during the compilation phase of the simulation."""


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


class SimStateException(SimulationException):
    """Exception when the simulation is not in a valid state to perform the requested operation."""
