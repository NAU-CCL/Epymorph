# pylint: disable=missing-docstring
import unittest
from unittest.mock import MagicMock

from epymorph.code import compile_function, parse_function
from epymorph.movement.compile import (ClauseFunctionTransformer,
                                       PredefFunctionTransformer)
from epymorph.movement.movement_model import MovementContext, PredefData


class TestMovementClauseTransformer(unittest.TestCase):

    def test_transform_clause(self):
        source = """
            def foo(t, src, dst):
                return t * src * dst
        """

        ast1 = parse_function(source)
        f1 = compile_function(ast1, {})

        transformer = ClauseFunctionTransformer([])
        ast2 = transformer.visit_and_fix(ast1)

        f2 = compile_function(ast2, {
            'MovementContext': MovementContext,
            'PredefData': PredefData,
        })

        ctx = MagicMock(spec=MovementContext)
        predef = {}

        self.assertEqual(f1(3, 5, 7), 105)
        self.assertEqual(f2(ctx, predef, 3, 5, 7), 105)
        with self.assertRaises(TypeError):
            f2(3, 5, 7)

    def test_transform_predef(self):
        source = """
            def my_predef():
                return {'foo': 42}
        """

        ast1 = parse_function(source)
        f1 = compile_function(ast1, {})

        transformer = PredefFunctionTransformer([])
        ast2 = transformer.visit_and_fix(ast1)

        f2 = compile_function(ast2, {
            'MovementContext': MovementContext,
            'PredefData': PredefData,
        })

        ctx = MagicMock(spec=MovementContext)

        self.assertEqual(f1(), {'foo': 42})
        self.assertEqual(f2(ctx), {'foo': 42})
        with self.assertRaises(TypeError):
            f2()
