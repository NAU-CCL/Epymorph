# pylint: disable=missing-docstring
import unittest
from unittest.mock import MagicMock

import numpy as np

from epymorph.compartment_model import (BIRTH, DEATH, CompartmentModel,
                                        compartment, create_model,
                                        create_symbols, edge, param)
from epymorph.engine.context import RumeContext
from epymorph.engine.ipm_exec import StandardIpmExecutor, _make_apply_matrix
from epymorph.simulation import SimDType


def model1() -> CompartmentModel:
    symbols = create_symbols(
        compartments=[
            compartment('S', tags=['test_tag']),
            compartment('I'),
            compartment('R'),
        ],
        attributes=[
            param('beta'),
            param('gamma'),
        ]
    )

    [S, I, R] = symbols.compartment_symbols
    [beta, gamma] = symbols.attribute_symbols

    return create_model(
        symbols=symbols,
        transitions=[
            edge(S, I, rate=beta * S * I),
            edge(I, R, rate=gamma * I),
        ],
    )


def model2() -> CompartmentModel:
    symbols = create_symbols(
        compartments=[
            compartment('S'),
            compartment('I'),
            compartment('R'),
        ],
        attributes=[
            param('beta'),
            param('gamma'),
            param('b'),  # birth rate
            param('d'),  # death rate
        ]
    )

    [S, I, R] = symbols.compartment_symbols
    [beta, gamma, b, d] = symbols.attribute_symbols

    return create_model(
        symbols=symbols,
        transitions=[
            edge(S, I, rate=beta * S * I),
            edge(BIRTH, S, rate=b),
            edge(I, R, rate=gamma * I),
            edge(S, DEATH, rate=d * S),
            edge(I, DEATH, rate=d * I),
            edge(R, DEATH, rate=d * R),
        ],
    )


class IpmExecutorTest(unittest.TestCase):

    def test_make_apply_matrix_01(self):
        self.assertTrue(np.array_equal(
            _make_apply_matrix(model1()),
            np.array([[-1, +1, 0], [0, -1, +1]], dtype=SimDType)
        ))

    def test_make_apply_matrix_02(self):
        self.assertTrue(np.array_equal(
            _make_apply_matrix(model2()),
            np.array([
                [-1, +1, 0],
                [+1, 0, 0],
                [0, -1, +1],
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ], dtype=SimDType)
        ))


class StandardIpmExecutorTest(unittest.TestCase):

    def test_init_01(self):
        ctx = MagicMock(spec=RumeContext)
        ctx.ipm = model1()

        exe = StandardIpmExecutor(ctx)

        self.assertEqual(exe._events_leaving_compartment, [[0], [1], []])
        self.assertEqual(exe._source_compartment_for_event, [0, 1])

    def test_init_02(self):
        ctx = MagicMock(spec=RumeContext)
        ctx.ipm = model2()

        exe = StandardIpmExecutor(ctx)

        self.assertEqual(exe._source_compartment_for_event, [0, -1, 1, 0, 1, 2])
