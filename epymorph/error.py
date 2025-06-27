"""Common epymorph exceptions."""

from contextlib import contextmanager


class ExternalDependencyError(Exception):
    """
    Exception when epymorph requires a native program (or programs) to perform an
    application function but the program is not found.

    Parameters
    ----------
    msg :
        The exception message.
    missing :
        The list of external programs which are missing.
    """

    missing: list[str]
    """The list of external programs which are missing."""

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
    """Exception encountered handling a data attribute."""


class DataAttributeErrorGroup(ExceptionGroup, DataAttributeError):  # noqa: N818
    """Multiple exceptions encountered handling data attributes."""


class MissingContextError(Exception):
    """
    Exception during simulation function evaluation, where the function required
    context elements that were not provided.
    """


class IPMValidationError(ValidationError):
    """Exception for invalid IPM."""


class MMValidationError(ValidationError):
    """Exception for invalid MM."""


class InitValidationError(ValidationError):
    """Exception for invalid Initializer."""


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


class _WithFieldsMixin:
    """
    A mixin class for exceptions that include diagnostic information about
    the simulation step being processed.
    """

    display_fields: list[tuple[str, dict]]

    def __str__(self):
        msg = super().__str__()
        fields = ""
        for name, values in self.display_fields:
            fields += f"Showing current {name}\n"
            for key, value in values.items():
                fields += f"{key}: {value}\n"
            fields += "\n"
        return f"{msg}\n{fields}"


class IPMSimNaNError(_WithFieldsMixin, IPMSimError):
    """Exception raised when an IPM transition rate is NaN."""

    def __init__(self, display_fields: list[tuple[str, dict]]):
        msg = [
            "An IPM transition rate was NaN (not a number).",
            "This is often the result of a divide by zero error.",
            "When constructing an IPM, ensure that no edge transitions can result in "
            "division by zero.",
            "This commonly occurs when defining an edge that is divided by "
            "a population total, since populations in nodes can be zero.",
            "Possible fixes include wrapping the divisor with a 'Max' expression so as "
            "to guarantee it is never less than 1. Commonly in this situation, the "
            "numerator must be 0, making the divisor's actual value irrelevant.",
        ]
        super().__init__("\n".join(msg))
        self.display_fields = display_fields


class IPMSimLessThanZeroError(_WithFieldsMixin, IPMSimError):
    """Exception raised when an IPM transition rate is less than zero."""

    def __init__(self, display_fields: list[tuple[str, dict]]):
        msg = [
            "An IPM transition rate was less than zero.",
            "When providing IPM parameters ensure that this will not result in a "
            "negative rate.",
        ]
        super().__init__("\n".join(msg))
        self.display_fields = display_fields


class IPMSimInvalidForkError(_WithFieldsMixin, IPMSimError):
    """Exception raised when an IPM fork transition's probabilities are invalid."""

    def __init__(self, display_fields: list[tuple[str, dict]]):
        msg = (
            "An IPM fork transition is invalid. The computed probabilities for the "
            "fork must always be non-negative and sum to 1."
        )
        super().__init__(msg)
        self.display_fields = display_fields


class MMSimError(SimulationError):
    """Exception during MM processing."""


@contextmanager
def error_gate(
    description: str,
    exception_type: type[Exception],
    *reraises: type[Exception],
):
    """
    Begin a context that standardizes errors linked to a particular simulation phase.
    Wrap the phase in an error gate and all exceptions raised within will be normalized.

    If an exception of type `exception_type` is caught, it will be re-raised as-is.
    If an exception is caught from the list of exception types in `reraises`,
    the exception will be stringified and re-raised as `exception_type`.
    All other exceptions will be labeled as "unknown errors".

    Parameters
    ----------
    description :
        Describe the phase in gerund form, e.g., "executing the IPM".
    exception_type :
        The best exception type for this phase.
    reraises :
        Additional exception types which will be stringified and raised as an
        `exception_type`.
    """
    try:
        yield
    except exception_type as e:
        raise e
    except Exception as e:
        if any(isinstance(e, r) for r in reraises):
            raise exception_type(str(e)) from e

        msg = f"Unknown error {description}."
        raise exception_type(msg) from e
