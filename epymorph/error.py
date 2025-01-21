"""
A common exception framework for epymorph.
"""

from contextlib import contextmanager
from textwrap import dedent


class GeographyError(Exception):
    """Exception working with geographic system representations."""

    # NOTE: this is *not* for general errors related to the epymorph GEO module,
    # but instead for things like utility functions for working with
    # US Census delineations.


class DimensionError(Exception):
    """Raised when epymorph needed dimensional information that was not provided."""


class UnknownModel(Exception):
    """Exception for the inability to load a model as specified."""

    model_type: str
    name_or_path: str

    def __init__(self, model_type: str, name_or_path: str):
        super().__init__(f"Unable to load {model_type} model '{name_or_path}'")
        self.model_type = model_type
        self.name_or_path = name_or_path


class ModelRegistryException(Exception):
    """Exception during model library initialization."""


class ParseException(Exception):
    """Superclass for exceptions which happen while parsing a model."""


class MmParseException(ParseException):
    """Exception parsing a Movement Model."""


class ValidationException(Exception):
    """Superclass for exceptions which happen during simulation validation."""


class AttributeException(ValidationException):
    """Exception handling data attributes."""


class AttributeExceptionGroup(ExceptionGroup, AttributeException):
    """Multiple exceptions encountered handling data attributes."""


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


class DataResourceException(Exception):
    """Exception during resource loading from ADRIOs."""


class IpmValidationException(ValidationException):
    """Exception for invalid IPM."""


class MmValidationException(ValidationException):
    """Exception for invalid MM."""


class InitValidationException(ValidationException):
    """Exception for invalid Init."""


class SimValidationException(ValidationException):
    """
    Exception for cases where a simulation is invalid as configured,
    typically because the MM, IPM, or Initializer require data attributes
    that are not available.
    """


class CompilationException(Exception):
    """Exception during the compilation phase of the simulation."""


class IpmCompileException(CompilationException):
    """Exception during the compilation of the IPM."""


class MmCompileException(CompilationException):
    """Exception during the compilation of the MM."""


class SimulationException(Exception):
    """Superclass for exceptions which happen during simulation runtime."""


class InitException(SimulationException):
    """Exception for invalid initialization."""


class IpmSimException(SimulationException):
    """Exception during IPM processing."""


class IpmSimExceptionWithFields(IpmSimException):
    """
    Exception during IPM processing where it is appropriate to show specific
    fields within the simulation.
    To create a new error with fields, create a subclass of this and set the
    displayed error message along with the fields to print.
    See 'IpmSimNaNException' for an example.
    """

    display_fields: list[tuple[str, dict]]

    def __init__(
        self, message: str, display_fields: tuple[str, dict] | list[tuple[str, dict]]
    ):
        super().__init__(message)
        if isinstance(display_fields, tuple):
            display_fields = [display_fields]
        self.display_fields = display_fields

    def __str__(self):
        msg = super().__str__()
        fields = ""
        for name, values in self.display_fields:
            fields += f"Showing current {name}\n"
            for key, value in values.items():
                fields += f"{key}: {value}\n"
            fields += "\n"
        return f"{msg}\n{fields}"


class IpmSimNaNException(IpmSimExceptionWithFields):
    """Exception for handling NaN (not a number) rate values"""

    def __init__(self, display_fields: tuple[str, dict] | list[tuple[str, dict]]):
        msg = (
            "NaN (not a number) rate detected. This is often the result of a "
            "divide by zero error.\n"
            "When constructing the IPM, ensure that no edge transitions can "
            "result in division by zero\n"
            "This commonly occurs when defining an S->I edge that is "
            "(some rate / sum of the compartments)\n"
            "To fix this, change the edge to define the S->I edge as "
            "(some rate / Max(1/sum of the the compartments))\n"
            "See examples of this in the provided example ipm definitions "
            "in the data/ipms folder."
        )
        msg = dedent(msg)
        super().__init__(msg, display_fields)


class IpmSimLessThanZeroException(IpmSimExceptionWithFields):
    """Exception for handling less than 0 rate values"""

    def __init__(self, display_fields: tuple[str, dict] | list[tuple[str, dict]]):
        msg = (
            "Less than zero rate detected. When providing or defining ipm parameters, "
            "ensure that they will not result in a negative rate.\n"
            "Note: this can often happen unintentionally if a function is given "
            "as a parameter."
        )
        msg = dedent(msg)
        super().__init__(msg, display_fields)


class IpmSimInvalidProbsException(IpmSimExceptionWithFields):
    """Exception for handling invalid probability values"""

    def __init__(self, display_fields: tuple[str, dict] | list[tuple[str, dict]]):
        msg = """
              Invalid probabilities for fork definition detected. Probabilities for a
              given tick should always be nonnegative and sum to 1
              """
        msg = dedent(msg)
        super().__init__(msg, display_fields)


class MmSimException(SimulationException):
    """Exception during MM processing."""


class SimStateException(SimulationException):
    """
    Exception when the simulation is not in a valid state to perform
    the requested operation.
    """


@contextmanager
def error_gate(
    description: str, exception_type: type[Exception], *reraises: type[Exception]
):
    """
    Provide nice error messaging linked to a phase of the simulation.
    `description` should describe the phase in gerund form.
    If an exception of type `exception_type` is caught, it will be re-raised as-is.
    If an exception is caught from the list of exception types in `reraises`,
    the exception will be stringified and re-raised as `exception_type`.
    All other exceptions will be labeled "unknown errors" with the given description.
    """
    try:
        yield
    except exception_type as e:
        raise e
    except Exception as e:  # pylint: disable=broad-exception-caught
        if any(isinstance(e, r) for r in reraises):
            raise exception_type(str(e)) from e

        msg = f"Unknown error {description}."
        raise exception_type(msg) from e
