# pylint: disable=missing-docstring
import unittest
from datetime import date
from unittest.mock import MagicMock

import numpy as np
import numpy.testing as npt

from epymorph.compartment_model import (BIRTH, DEATH, CompartmentModel,
                                        compartment, create_model,
                                        create_symbols, edge)
from epymorph.data_shape import Shapes, SimDimensions
from epymorph.data_type import AttributeArray, SimDType
from epymorph.database import Database
from epymorph.rume import Rume
from epymorph.simulation import AttributeDef
from epymorph.simulator.basic.ipm_exec import IpmExecutor
from epymorph.simulator.world_list import ListWorld


def _model1() -> CompartmentModel:
    symbols = create_symbols(
        compartments=[
            compartment('S', tags=['test_tag']),
            compartment('I'),
            compartment('R'),
        ],
        attributes=[
            AttributeDef('beta', float, Shapes.TxN),
            AttributeDef('gamma', float, Shapes.TxN),
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


def _model2() -> CompartmentModel:
    symbols = create_symbols(
        compartments=[
            compartment('S'),
            compartment('I'),
            compartment('R'),
        ],
        attributes=[
            AttributeDef('beta', float, Shapes.TxN),
            AttributeDef('gamma', float, Shapes.TxN),
            AttributeDef('b', float, Shapes.TxN),  # birth rate
            AttributeDef('d', float, Shapes.TxN),  # death rate
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


class StandardIpmExecutorTest(unittest.TestCase):

    def test_init_01(self):
        ipm = _model1()

        rume = MagicMock(spec=Rume)
        rume.ipm = ipm
        rume.dim = SimDimensions.build(
            tau_step_lengths=[1.0],
            start_date=date(2021, 1, 1),
            days=100,
            nodes=1,
            compartments=ipm.num_compartments,
            events=ipm.num_events)

        world = MagicMock(spec=ListWorld)
        data = MagicMock(spec=Database[AttributeArray])
        rng = np.random.default_rng()

        exe = IpmExecutor(rume, world, data, rng)

        self.assertEqual(exe._events_leaving_compartment, [[0], [1], []])
        self.assertEqual(exe._source_compartment_for_event, [0, 1])
        npt.assert_array_equal(
            exe._apply_matrix,
            np.array([
                [-1, +1, 0],
                [0, -1, +1],
            ], dtype=SimDType)
        )

    def test_init_02(self):
        ipm = _model2()

        rume = MagicMock(spec=Rume)
        rume.ipm = ipm
        rume.dim = SimDimensions.build(
            tau_step_lengths=[1.0],
            start_date=date(2021, 1, 1),
            days=100,
            nodes=1,
            compartments=ipm.num_compartments,
            events=ipm.num_events)

        world = MagicMock(spec=ListWorld)
        data = MagicMock(spec=Database[AttributeArray])
        rng = np.random.default_rng()

        exe = IpmExecutor(rume, world, data, rng)

        self.assertEqual(exe._source_compartment_for_event, [0, -1, 1, 0, 1, 2])
        npt.assert_array_equal(
            exe._apply_matrix,
            np.array([
                [-1, +1, 0],
                [+1, 0, 0],
                [0, -1, +1],
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ], dtype=SimDType)
        )
