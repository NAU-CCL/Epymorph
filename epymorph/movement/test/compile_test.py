# pylint: disable=missing-docstring
import textwrap
import unittest
from unittest.mock import MagicMock

from epymorph.code import compile_function, parse_function
from epymorph.movement.compile import (transform_movement_ast,
                                       transform_predef_ast)
from epymorph.movement.movement_model import MovementContext, PredefParams


class TestMovementClauseTransformer(unittest.TestCase):

    def test_transform_clause(self):
        source = textwrap.dedent("""
            def foo(t, src, dst):
                return t * src * dst
        """)

        ast1 = parse_function(source)
        f1 = compile_function(ast1, {})

        ast2 = transform_movement_ast(ast1)
        f2 = compile_function(ast2, {
            'MovementContext': MovementContext,
            'PredefParams': PredefParams,
        })

        ctx = MagicMock(spec=MovementContext)
        predef = {}

        self.assertEqual(f1(3, 5, 7), 105)
        self.assertEqual(f2(ctx, predef, 3, 5, 7), 105)
        with self.assertRaises(TypeError):
            f2(3, 5, 7)

    def test_transform_predef(self):
        source = textwrap.dedent("""
            def my_predef():
                return {'foo': 42}
        """)

        ast1 = parse_function(source)
        f1 = compile_function(ast1, {})

        ast2 = transform_predef_ast(ast1)
        f2 = compile_function(ast2, {
            'MovementContext': MovementContext,
            'PredefParams': PredefParams,
        })

        ctx = MagicMock(spec=MovementContext)

        self.assertEqual(f1(), {'foo': 42})
        self.assertEqual(f2(ctx), {'foo': 42})
        with self.assertRaises(TypeError):
            f2()
