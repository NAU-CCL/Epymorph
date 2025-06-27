"""
The basis of the movement system in epymorph.

Movement models are responsible for dividing up the day into one or more parts, in
accordance with their desired tempo of movement. For example, commuting patterns of
work day versus night.

Movement mechanics are expressed as a set of clauses which calculate the "requested"
number of individuals that should move between geospatial nodes at a particular time
step of the simulation. This is "requested" movement because fewer individuals may wind
up moving if there are not enough to satisfy the request.
"""

import re
from abc import ABC, ABCMeta, abstractmethod
from math import isclose
from typing import Any, Literal, Sequence, Type, TypeVar, cast

from numpy.typing import NDArray
from typing_extensions import override

from epymorph.attribute import AttributeDef
from epymorph.data_type import SimDType
from epymorph.simulation import (
    NEVER,
    SimulationFunctionClass,
    SimulationTickFunction,
    Tick,
    TickDelta,
    TickIndex,
)
from epymorph.util import are_instances

DayOfWeek = Literal["M", "T", "W", "Th", "F", "Sa", "Su"]
"""Type for days of the week values."""

ALL_DAYS: tuple[DayOfWeek, ...] = ("M", "T", "W", "Th", "F", "Sa", "Su")
"""All days of the week values."""

_day_of_week_pattern = r"\b(M|T|W|Th|F|Sa|Su)\b"


def parse_days_of_week(dow: str) -> tuple[DayOfWeek, ...]:
    """
    Parse a list of days of the week using our standard abbreviations:
    M, T, W, Th, F, Sa, Su.

    Parameters
    ----------
    dow :
        A string containing a list of days of the week.
        The parser is pretty permissive; it ignores invalid parts of the input while
        keeping the valid parts. Any separator is allowed between the day of the week
        themselves.

    Returns
    -------
    :
        The days of the week parsed from the string. Empty if there are none.

    Examples
    --------
    >>> parse_days_of_week("M,W,F")
    ('M', 'W', 'F')
    """
    ds = re.findall(_day_of_week_pattern, dow)
    # return the matched days in standard order
    return tuple(x for x in ALL_DAYS if x in ds)


class MovementPredicate(ABC):
    """
    Checks the current tick and responds with true or false.
    Movement predicates are used to determine whether a movement clause
    should be applied for this simulation tick or not.
    """

    @abstractmethod
    def evaluate(self, tick: Tick) -> bool:
        """
        Check the given tick.

        Parameters
        ----------
        tick :
            The current simulation tick.

        Returns
        -------
        :
            The result of the predicate.
        """


class EveryDay(MovementPredicate):
    """Return true for every day."""

    @override
    def evaluate(self, tick: Tick) -> bool:
        return True


class DayIs(MovementPredicate):
    """
    Checks that the day is in the given set of days of the week.

    Parameters
    ----------
    week_days :
        The week days for which this predicate should evaluate true.
        Either a list of `DayOfWeek` or a string which will be parsed
        as in `parse_days_of_week`.
    """

    week_days: tuple[DayOfWeek, ...]

    def __init__(self, week_days: Sequence[DayOfWeek] | str):
        if isinstance(week_days, str):
            self.week_days = parse_days_of_week(week_days)
        else:
            self.week_days = tuple(week_days)

    @override
    def evaluate(self, tick: Tick) -> bool:
        return tick.date.weekday() in self.week_days


##################
# MovementClause #
##################


_TypeT = TypeVar("_TypeT")


class MovementClauseClass(SimulationFunctionClass):
    """The metaclass for `MovementClause` classes; enforces proper implementation."""

    def __new__(
        cls: Type[_TypeT],
        name: str,
        bases: tuple[type, ...],
        dct: dict[str, Any],
    ) -> _TypeT:
        # Skip these checks for classes we want to treat as abstract:
        if dct.get("_abstract_simfunc", False):
            return super().__new__(cls, name, bases, dct)

        # Check predicate.
        predicate = dct.get("predicate")
        if predicate is None or not isinstance(predicate, MovementPredicate):
            raise TypeError(
                f"Invalid predicate in {name}: "
                "please specify a MovementPredicate instance."
            )
        # Check leaves.
        leaves = dct.get("leaves")
        if leaves is None or not isinstance(leaves, TickIndex):
            raise TypeError(
                f"Invalid leaves in {name}: please specify a TickIndex instance."
            )
        if leaves.step < 0:
            raise TypeError(
                f"Invalid leaves in {name}: step indices cannot be less than zero."
            )
        # Check returns.
        returns = dct.get("returns")
        if returns is None or not isinstance(returns, TickDelta):
            raise TypeError(
                f"Invalid returns in {name}: please specify a TickDelta instance."
            )
        if returns != NEVER:
            if returns.step < 0:
                raise TypeError(
                    f"Invalid returns in {name}: step indices cannot be less than zero."
                )
            if returns.days < 0:
                raise TypeError(
                    f"Invalid returns in {name}: days cannot be less than zero."
                )

        return super().__new__(cls, name, bases, dct)


class MovementClause(
    SimulationTickFunction[NDArray[SimDType]], ABC, metaclass=MovementClauseClass
):
    """
    A movement clause is basically a function which calculates _how many_ individuals
    should move between all of the geo nodes.

    It's up to epymorph's internal mechanisms to decide by random draw _which_
    individuals move (as identified by their disease status, or IPM compartment).

    The clause also has various settings which determine when the clause is active
    (for example, only move people Monday-Friday at the start of the day)
    and when the individuals that were moved by the clause should return home
    (for example, stay for two days and then return at the end of the day).
    """

    _abstract_simfunc = True  # marking this abstract skips metaclass validation

    # in addition to requirements (from super), movement clauses must also specify:

    predicate: MovementPredicate
    """When does this movement clause apply?"""

    leaves: TickIndex
    """On which tau step does this movement clause apply?"""

    returns: TickDelta
    """When do the movers from this clause return home?"""

    def is_active(self, tick: Tick) -> bool:
        """
        Check if this movement clause should be applied this tick.

        Parameters
        ----------
        tick :
            The current simulation tick.

        Returns
        -------
        :
            True if the clause should be applied this tick.
        """
        return self.leaves.step == tick.step and self.predicate.evaluate(tick)

    @property
    def clause_name(self) -> str:
        """A display name to use for the clause."""
        return self.__class__.__name__

    @abstractmethod
    def evaluate(self, tick: Tick) -> NDArray[SimDType]:
        """
        Implement this method to provide logic for the clause.
        Use self methods and properties to access the simulation context or defer
        processing to another function.

        Parameters
        ----------
        tick :
            The simulation tick being evaluated.

        Returns
        -------
        :
            An array describing the requested number of individuals to move from origin
            location (row; axis 0) to destination location (column; axis 1).
        """


#################
# MovementModel #
#################


class MovementModelClass(ABCMeta):
    """The metaclass for `MovementModel` classes; enforces proper implementation."""

    def __new__(
        cls: Type[_TypeT],
        name: str,
        bases: tuple[type, ...],
        dct: dict[str, Any],
    ) -> _TypeT:
        # Skip these checks for known base classes:
        if name in ("MovementModel",):
            return super().__new__(cls, name, bases, dct)

        # Check tau steps.
        steps = dct.get("steps")
        if steps is None or not isinstance(steps, (list, tuple)):
            raise TypeError(
                f"Invalid steps in {name}: please specify as a list or tuple."
            )
        if not are_instances(steps, float):
            raise TypeError(f"Invalid steps in {name}: must be floating point numbers.")
        if len(steps) == 0:
            raise TypeError(
                f"Invalid steps in {name}: please specify at least one tau step length."
            )
        if not isclose(sum(steps), 1.0, abs_tol=1e-6):
            raise TypeError(f"Invalid steps in {name}: steps must sum to 1.")
        if any(x <= 0 for x in steps):
            raise TypeError(
                f"Invalid steps in {name}: steps must all be greater than 0."
            )
        dct["steps"] = tuple(steps)

        # Check clauses.
        clauses = dct.get("clauses")
        if clauses is None or not isinstance(clauses, (list, tuple)):
            raise TypeError(
                f"Invalid clauses in {name}: please specify as a list or tuple."
            )
        if not are_instances(clauses, MovementClause):
            raise TypeError(
                f"Invalid clauses in {name}: must be instances of MovementClause."
            )
        if len(clauses) == 0:
            raise TypeError(
                f"Invalid clauses in {name}: please specify at least one clause."
            )
        for c in cast(Sequence[MovementClause], clauses):
            # Check that clause steps are valid.
            num_steps = len(steps)
            if c.leaves.step >= num_steps:
                raise TypeError(
                    f"Invalid clauses in {name}: {c.__class__.__name__} "
                    f"uses a leave step ({c.leaves.step}) "
                    f"which is not a valid index. (steps: {steps})"
                )
            if c.returns.step >= num_steps:
                raise TypeError(
                    f"Invalid clauses in {name}: {c.__class__.__name__} "
                    f"uses a return step ({c.returns.step}) "
                    f"which is not a valid index. (steps: {steps})"
                )
        dct["clauses"] = tuple(clauses)

        return super().__new__(cls, name, bases, dct)


class MovementModel(ABC, metaclass=MovementModelClass):
    """
    A `MovementModel` (MM) describes a pattern of geospatial movement for
    individuals in the model.

    The MM defines the tau steps which chop the day up into one or more parts,
    as well as the set of movement clauses which may apply throughout each day.

    To create a custom MM, you will write an implementation of this class
    and of any required clause classes.
    """

    steps: Sequence[float]
    """The length and order of tau steps."""
    clauses: Sequence[MovementClause]
    """The movement clauses that make up the model."""

    @property
    def requirements(self) -> Sequence[AttributeDef]:
        """The combined requirements of all of the clauses in this model."""
        return [req for clause in self.clauses for req in clause.requirements]
