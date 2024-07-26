# pylint: disable=missing-docstring
import unittest
from datetime import date
from unittest.mock import MagicMock

import numpy as np
import numpy.testing as npt

from epymorph.compartment_model import (BIRTH, DEATH, CompartmentModel,
                                        compartment, edge)
from epymorph.data_shape import Shapes, SimDimensions
from epymorph.data_type import AttributeArray, SimDType
from epymorph.database import Database
from epymorph.rume import Rume
from epymorph.simulation import AttributeDef
from epymorph.simulator.basic.ipm_exec import IpmExecutor
from epymorph.simulator.world_list import ListWorld


class Sir(CompartmentModel):
    compartments = [
        compartment('S', tags=['test_tag']),
        compartment('I'),
        compartment('R'),
    ]
    requirements = [
        AttributeDef('beta', float, Shapes.TxN),
        AttributeDef('gamma', float, Shapes.TxN),
    ]

    def edges(self, symbols):
        [S, I, R] = symbols.all_compartments
        [beta, gamma] = symbols.all_requirements
        return [
            edge(S, I, rate=beta * S * I),
            edge(I, R, rate=gamma * I),
        ]


class Sirbd(CompartmentModel):
    compartments = [
        compartment('S'),
        compartment('I'),
        compartment('R'),
    ]

    requirements = [
        AttributeDef('beta', float, Shapes.TxN),
        AttributeDef('gamma', float, Shapes.TxN),
        AttributeDef('b', float, Shapes.TxN),  # birth rate
        AttributeDef('d', float, Shapes.TxN),  # death rate
    ]

    def edges(self, symbols):
        [S, I, R] = symbols.all_compartments
        [beta, gamma, b, d] = symbols.all_requirements

        return [
            edge(S, I, rate=beta * S * I),
            edge(BIRTH, S, rate=b),
            edge(I, R, rate=gamma * I),
            edge(S, DEATH, rate=d * S),
            edge(I, DEATH, rate=d * I),
            edge(R, DEATH, rate=d * R),
        ]


class StandardIpmExecutorTest(unittest.TestCase):

    def test_init_01(self):
        ipm = Sir()

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
        ipm = Sirbd()

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
