# ruff: noqa: PT009,PT027
import numpy as np
import pytest
from numpy.typing import NDArray

from epymorph.data_type import SimDType
from epymorph.movement_model import (
    EveryDay,
    MovementClause,
    MovementModel,
    parse_days_of_week,
)
from epymorph.simulation import NEVER, Tick, TickDelta, TickIndex


@pytest.mark.parametrize(
    ("input_str", "expected"),
    [
        ("M", ("M",)),
        ("T", ("T",)),
        ("W", ("W",)),
        ("Th", ("Th",)),
        ("F", ("F",)),
        ("Sa", ("Sa",)),
        ("Su", ("Su",)),
        ("M T W Th F Sa Su", ("M", "T", "W", "Th", "F", "Sa", "Su")),
        ("Su Sa F Th W T M", ("M", "T", "W", "Th", "F", "Sa", "Su")),  # order is fixed
        ("M, T;W|Th/F-Sa.Su", ("M", "T", "W", "Th", "F", "Sa", "Su")),
        ("M;F;W", ("M", "W", "F")),
        ("I would like T and Th please", ("T", "Th")),
        ("a house divided against itself cannot stand", ()),
    ],
)
def test_parse_day_of_week(input_str, expected):
    assert parse_days_of_week(input_str) == expected


def test_movement_clause_create_01():
    class MyClause(MovementClause):
        leaves = TickIndex(step=0)
        returns = TickDelta(days=0, step=1)
        predicate = EveryDay()

        def evaluate(self, tick: Tick) -> NDArray[SimDType]:
            return np.array([0])

    clause = MyClause()
    assert clause.leaves == TickIndex(step=0)
    assert clause.returns == TickDelta(days=0, step=1)


def test_movement_clause_create_02_missing_leaves():
    with pytest.raises(TypeError, match=r"(?i)invalid leaves in myclause"):

        class MyClause(MovementClause):
            # leaves = TickIndex(step=0)  # noqa: ERA001
            returns = TickDelta(days=0, step=1)
            predicate = EveryDay()

            def evaluate(self, tick: Tick) -> NDArray[SimDType]:
                return np.array([0])


def test_movement_clause_create_03_missing_returns():
    with pytest.raises(TypeError, match=r"(?i)invalid returns in myclause"):

        class MyClause(MovementClause):
            leaves = TickIndex(step=0)
            # returns = TickDelta(days=0, step=1)  # noqa: ERA001
            predicate = EveryDay()

            def evaluate(self, tick: Tick) -> NDArray[SimDType]:
                return np.array([0])


def test_movement_clause_create_04_missing_predicate():
    with pytest.raises(TypeError, match=r"(?i)invalid predicate in myclause"):

        class MyClause(MovementClause):
            leaves = TickIndex(step=0)
            returns = TickDelta(days=0, step=1)
            # predicate = EveryDay()  # noqa: ERA001

            def evaluate(self, tick: Tick) -> NDArray[SimDType]:
                return np.array([0])


def test_movement_clause_create_05_invalid_leaves_index():
    with pytest.raises(TypeError, match=r"(?i)step indices cannot be less than zero"):

        class MyClause(MovementClause):
            leaves = TickIndex(step=-23)
            returns = TickDelta(days=0, step=1)
            predicate = EveryDay()

            def evaluate(self, tick: Tick) -> NDArray[SimDType]:
                return np.array([0])


def test_movement_clause_create_06_invalid_returns_index():
    with pytest.raises(TypeError, match=r"(?i)step indices cannot be less than zero"):

        class MyClause(MovementClause):
            leaves = TickIndex(step=0)
            returns = TickDelta(days=0, step=-23)
            predicate = EveryDay()

            def evaluate(self, tick: Tick) -> NDArray[SimDType]:
                return np.array([0])


def test_movement_clause_create_07_never_returns():
    class MyClause(MovementClause):
        leaves = TickIndex(step=0)
        returns = NEVER
        predicate = EveryDay()

        def evaluate(self, tick: Tick) -> NDArray[SimDType]:
            return np.array([0])

    clause = MyClause()
    assert clause.leaves == TickIndex(step=0)
    assert clause.returns == TickDelta(days=-1, step=-1)


class MyClause(MovementClause):
    leaves = TickIndex(step=0)
    returns = TickDelta(days=0, step=1)
    predicate = EveryDay()

    def evaluate(self, tick: Tick) -> NDArray[SimDType]:
        return np.array([0])


def test_movement_model_create_01():
    class MyModel(MovementModel):
        steps = [1 / 3, 2 / 3]
        clauses = [MyClause()]

    model = MyModel()
    assert model.steps == (1 / 3, 2 / 3)
    assert len(model.clauses) == 1
    assert model.clauses[0].__class__.__name__ == "MyClause"


def test_movement_model_create_02_missing_steps():
    with pytest.raises(TypeError, match=r"(?i)invalid steps in mymodel"):

        class MyModel(MovementModel):
            # steps = [1 / 3, 2 / 3]  # noqa: ERA001
            clauses = [MyClause()]


def test_movement_model_create_03_steps_sum_error():
    with pytest.raises(TypeError, match=r"(?i)steps must sum to 1"):

        class MyModel1(MovementModel):
            steps = [1 / 3, 1 / 3]
            clauses = [MyClause()]

    with pytest.raises(TypeError, match=r"(?i)steps must sum to 1"):

        class MyModel2(MovementModel):
            steps = [0.1, 0.75, 0.3, 0.2]
            clauses = [MyClause()]


def test_movement_model_create_04_steps_positive_check():
    with pytest.raises(TypeError, match=r"(?i)steps must all be greater than 0"):

        class MyModel(MovementModel):
            steps = [1 / 3, -1 / 3, 1 / 3, 2 / 3]
            clauses = [MyClause()]


def test_movement_model_create_05_missing_clauses():
    with pytest.raises(TypeError, match=r"(?i)invalid clauses"):

        class MyModel(MovementModel):
            steps = [1 / 3, 2 / 3]
            # clauses = [MyClause()]  # noqa: ERA001


def test_movement_model_create_06_clause_references_invalid_step():
    class MyClauseLocal(MovementClause):
        leaves = TickIndex(0)
        returns = TickDelta(days=0, step=9)
        predicate = EveryDay()

        def evaluate(self, tick: Tick) -> NDArray[SimDType]:
            return np.array([0])

    with pytest.raises(
        TypeError,
        match=r"(?i)return step \(9\) which is not a valid index",
    ):

        class MyModel(MovementModel):
            steps = (1 / 3, 2 / 3)
            clauses = (MyClauseLocal(),)
