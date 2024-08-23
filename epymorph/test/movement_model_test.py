# pylint: disable=missing-docstring,unused-variable
import unittest

import numpy as np
from numpy.typing import NDArray

from epymorph.data_type import SimDType
from epymorph.movement_model import EveryDay, MovementClause, MovementModel
from epymorph.simulation import Tick, TickDelta, TickIndex


class MovementClauseTest(unittest.TestCase):

    def test_create_01(self):
        # Successful clause!
        class MyClause(MovementClause):
            leaves = TickIndex(step=0)
            returns = TickDelta(days=0, step=1)
            predicate = EveryDay()

            def evaluate(self, tick: Tick) -> NDArray[SimDType]:
                return np.array([0])

        clause = MyClause()

        self.assertEqual(clause.leaves, TickIndex(step=0))
        self.assertEqual(clause.returns, TickDelta(days=0, step=1))

    def test_create_02(self):
        # Test for error: forgot 'leaves'
        with self.assertRaises(TypeError) as e:
            class MyClause(MovementClause):
                # leaves = TickIndex(step=0)
                returns = TickDelta(days=0, step=1)
                predicate = EveryDay()

                def evaluate(self, tick: Tick) -> NDArray[SimDType]:
                    return np.array([0])
        self.assertIn("invalid leaves in myclause", str(e.exception).lower())

    def test_create_03(self):
        # Test for error: forgot 'returns'
        with self.assertRaises(TypeError) as e:
            class MyClause(MovementClause):
                leaves = TickIndex(step=0)
                # returns = TickDelta(days=0, step=1)
                predicate = EveryDay()

                def evaluate(self, tick: Tick) -> NDArray[SimDType]:
                    return np.array([0])
        self.assertIn("invalid returns in myclause", str(e.exception).lower())

    def test_create_04(self):
        # Test for error: forgot 'predicate'
        with self.assertRaises(TypeError) as e:
            class MyClause(MovementClause):
                leaves = TickIndex(step=0)
                returns = TickDelta(days=0, step=1)
                # predicate = EveryDay()

                def evaluate(self, tick: Tick) -> NDArray[SimDType]:
                    return np.array([0])
        self.assertIn("invalid predicate in myclause", str(e.exception).lower())

    def test_create_05(self):
        # Test for error: invalid 'leaves' index
        with self.assertRaises(TypeError) as e:
            class MyClause(MovementClause):
                leaves = TickIndex(step=-23)
                returns = TickDelta(days=0, step=1)
                predicate = EveryDay()

                def evaluate(self, tick: Tick) -> NDArray[SimDType]:
                    return np.array([0])
        self.assertIn("step indices cannot be less than zero", str(e.exception).lower())

    def test_create_06(self):
        # Test for error: invalid 'returns' index
        with self.assertRaises(TypeError) as e:
            class MyClause(MovementClause):
                leaves = TickIndex(step=0)
                returns = TickDelta(days=0, step=-23)
                predicate = EveryDay()

                def evaluate(self, tick: Tick) -> NDArray[SimDType]:
                    return np.array([0])
        self.assertIn("step indices cannot be less than zero", str(e.exception).lower())


class MovementModelTest(unittest.TestCase):

    class MyClause(MovementClause):
        leaves = TickIndex(step=0)
        returns = TickDelta(days=0, step=1)
        predicate = EveryDay()

        def evaluate(self, tick: Tick) -> NDArray[SimDType]:
            return np.array([0])

    def test_create_01(self):
        class MyModel(MovementModel):
            steps = [1 / 3, 2 / 3]
            clauses = [MovementModelTest.MyClause()]

        model = MyModel()
        self.assertEqual(model.steps, (1 / 3, 2 / 3))
        self.assertEqual(len(model.clauses), 1)
        self.assertEqual(model.clauses[0].__class__.__name__, "MyClause")

    def test_create_02(self):
        # Test for error: forgot 'steps'
        with self.assertRaises(TypeError) as e:
            class MyModel(MovementModel):
                # steps = [1 / 3, 2 / 3]
                clauses = [MovementModelTest.MyClause()]
        self.assertIn("invalid steps in mymodel", str(e.exception).lower())

    def test_create_03(self):
        # Test for error: 'steps' don't sum to 1
        with self.assertRaises(TypeError) as e:
            class MyModel1(MovementModel):
                steps = [1 / 3, 1 / 3]
                clauses = [MovementModelTest.MyClause()]
        self.assertIn("steps must sum to 1", str(e.exception).lower())

        with self.assertRaises(TypeError) as e:
            class MyModel2(MovementModel):
                steps = [0.1, 0.75, 0.3, 0.2]
                clauses = [MovementModelTest.MyClause()]
        self.assertIn("steps must sum to 1", str(e.exception).lower())

    def test_create_04(self):
        # Test for error: 'steps' aren't all greater than zero
        with self.assertRaises(TypeError) as e:
            class MyModel(MovementModel):
                steps = [1 / 3, -1 / 3, 1 / 3, 2 / 3]
                clauses = [MovementModelTest.MyClause()]
        self.assertIn("steps must all be greater than 0", str(e.exception).lower())

    def test_create_05(self):
        # Test for error: forgot 'clauses'
        with self.assertRaises(TypeError) as e:
            class MyModel(MovementModel):
                steps = [1 / 3, 2 / 3]
                # clauses = [MovementModelTest.MyClause()]
        self.assertIn("invalid clauses", str(e.exception).lower())

    def test_create_06(self):
        # Test for error: clauses reference steps which don't exist
        with self.assertRaises(TypeError) as e:
            class MyClause(MovementClause):
                leaves = TickIndex(0)
                returns = TickDelta(days=0, step=9)
                predicate = EveryDay()

                def evaluate(self, tick: Tick) -> NDArray[SimDType]:
                    return np.array([0])

            class MyModel(MovementModel):
                steps = (1 / 3, 2 / 3)
                clauses = (MyClause(),)
        self.assertIn("return step (9)", str(e.exception).lower())
        self.assertIn("not a valid index", str(e.exception).lower())
