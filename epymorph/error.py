"""
A common exception framework for epymorph.
"""

from contextlib import contextmanager
from textwrap import dedent


class ExternalDependencyError(Exception):
    """Exception when a native program is required but not found."""

    missing: list[str]
    """Which programs are missing?"""

    def __init__(self, msg: str, missing: list[str]):
        super().__init__(msg)
        self.missing = missing


class GeographyError(Exception):
    """Exception working with geographic system representations."""

    # NOTE: this is *not* for general errors related to the epymorph GEO module,
    # but instead for things like utility functions for working with
    # US Census delineations.


class DimensionError(Exception):
    """Raised when epymorph needed dimensional information that was not provided."""


class ValidationError(Exception):
    """Superclass for exceptions which happen during simulation validation."""


class DataAttributeError(ValidationError):
    """Exception handling data attributes."""


class DataAttributeErrorGroup(ExceptionGroup, DataAttributeError):  # noqa: N818
    """Multiple exceptions encountered handling data attributes."""


class DataResourceError(Exception):
    """Exception during resource loading from ADRIOs."""


class IPMValidationError(ValidationError):
    """Exception for invalid IPM."""


class MMValidationError(ValidationError):
    """Exception for invalid MM."""


class InitValidationError(ValidationError):
    """Exception for invalid Init."""


class SimValidationError(ValidationError):
    """
    Exception for cases where a simulation is invalid as configured,
    typically because the MM, IPM, or Initializer require data attributes
    that are not available.
    """


class SimCompilationError(Exception):
    """Exception during the compilation phase of the simulation."""


class SimulationError(Exception):
    """Superclass for exceptions which happen during simulation runtime."""


class InitError(SimulationError):
    """Exception for invalid initialization."""


class IPMSimError(SimulationError):
    """Exception during IPM processing."""


class IPMSimWithFieldsError(IPMSimError):
    """
    Exception during IPM processing where it is appropriate to show specific
    fields within the simulation.
    To create a new error with fields, create a subclass of this and set the
    displayed error message along with the fields to print.
    See 'IpmSimNaNException' for an example.
    """

    display_fields: list[tuple[str, dict]]

    def __init__(self, message: str, display_fields: list[tuple[str, dict]]):
        super().__init__(message)
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


class IPMSimNaNError(IPMSimWithFieldsError):
    """Exception for handling NaN (not a number) rate values"""

    def __init__(self, display_fields: list[tuple[str, dict]]):
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


class IPMSimLessThanZeroError(IPMSimWithFieldsError):
    """Exception for handling less than 0 rate values"""

    def __init__(self, display_fields: list[tuple[str, dict]]):
        msg = (
            "Less than zero rate detected. When providing or defining ipm parameters, "
            "ensure that they will not result in a negative rate.\n"
            "Note: this can often happen unintentionally if a function is given "
            "as a parameter."
        )
        msg = dedent(msg)
        super().__init__(msg, display_fields)


class IPMSimInvalidProbsError(IPMSimWithFieldsError):
    """Exception for handling invalid probability values"""

    def __init__(self, display_fields: list[tuple[str, dict]]):
        msg = """
              Invalid probabilities for fork definition detected. Probabilities for a
              given tick should always be nonnegative and sum to 1
              """
        msg = dedent(msg)
        super().__init__(msg, display_fields)


class MMSimError(SimulationError):
    """Exception during MM processing."""


@contextmanager
def error_gate(
    description: str,
    exception_type: type[Exception],
    *reraises: type[Exception],
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
