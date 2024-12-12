# pylint: disable=missing-docstring
import unittest
from datetime import date
from unittest.mock import MagicMock

import numpy as np
import numpy.testing as npt

from epymorph.compartment_model import BIRTH, DEATH, CompartmentModel, compartment, edge
from epymorph.data_shape import Shapes, SimDimensions
from epymorph.data_type import SimDType
from epymorph.database import DataResolver
from epymorph.rume import Rume
from epymorph.simulation import AttributeDef, Tick
from epymorph.simulator.basic.ipm_exec import IpmExecutor
from epymorph.simulator.world_list import ListWorld


class Sir(CompartmentModel):
    compartments = [
        compartment("S", tags=["test_tag"]),
        compartment("I"),
        compartment("R"),
    ]
    requirements = [
        AttributeDef("beta", float, Shapes.TxN),
        AttributeDef("gamma", float, Shapes.TxN),
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
        compartment("S"),
        compartment("I"),
        compartment("R"),
    ]

    requirements = [
        AttributeDef("beta", float, Shapes.TxN),
        AttributeDef("gamma", float, Shapes.TxN),
        AttributeDef("b", float, Shapes.TxN),  # birth rate
        AttributeDef("d", float, Shapes.TxN),  # death rate
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
        dim = SimDimensions.build(
            tau_step_lengths=[1.0],
            start_date=date(2021, 1, 1),
            days=100,
            nodes=1,
            compartments=ipm.num_compartments,
            events=ipm.num_events,
        )

        exe = IpmExecutor(
            rume=MagicMock(spec=Rume, ipm=ipm, dim=dim),
            world=MagicMock(spec=ListWorld),
            data=MagicMock(spec=DataResolver),
            rng=np.random.default_rng(),
        )

        self.assertEqual(exe._events_leaving_compartment, [[0], [1], []])
        self.assertEqual(exe._source_compartment_for_event, [0, 1])
        npt.assert_array_equal(
            exe._apply_matrix,
            np.array(
                [
                    [-1, +1, 0],
                    [0, -1, +1],
                ],
                dtype=SimDType,
            ),
        )

    def test_init_02(self):
        ipm = Sirbd()
        dim = SimDimensions.build(
            tau_step_lengths=[1.0],
            start_date=date(2021, 1, 1),
            days=100,
            nodes=1,
            compartments=ipm.num_compartments,
            events=ipm.num_events,
        )

        exe = IpmExecutor(
            rume=MagicMock(spec=Rume, ipm=ipm, dim=dim),
            world=MagicMock(spec=ListWorld),
            data=MagicMock(spec=DataResolver),
            rng=np.random.default_rng(),
        )

        self.assertEqual(exe._source_compartment_for_event, [0, -1, 1, 0, 1, 2])
        npt.assert_array_equal(
            exe._apply_matrix,
            np.array(
                [
                    [-1, +1, 0],
                    [+1, 0, 0],
                    [0, -1, +1],
                    [-1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1],
                ],
                dtype=SimDType,
            ),
        )

    def test_apply(self):
        # For simplicity, use a variant of SIRBD model with hard-coded parameters.
        class SIRBD2(CompartmentModel):
            compartments = [
                compartment("S"),
                compartment("I"),
                compartment("R"),
            ]

            def edges(self, symbols):
                [S, I, R] = symbols.all_compartments
                beta, gamma, b, d = 0.4, 1 / 10, 100, 0.05
                return [
                    edge(S, I, rate=beta * S * I),
                    edge(BIRTH, S, rate=b),
                    edge(I, R, rate=gamma * I),
                    edge(S, DEATH, rate=d * S),
                    edge(I, DEATH, rate=d * I),
                    edge(R, DEATH, rate=d * R),
                ]

        ipm = SIRBD2()

        # Create a simple world and have some folks move around.
        initials = np.array(
            [
                [100, 0, 0],
                [200, 0, 0],
                [300, 0, 0],
            ],
            dtype=SimDType,
        )
        world = ListWorld.from_initials(initials)

        # fmt: off
        world.apply_travel(np.array([
            [ [0, 0, 0], [21, 0, 0], [31, 0, 0]],
            [[12, 0, 0],  [0, 0, 0], [32, 0, 0]],
            [[13, 0, 0], [23, 0, 0],  [0, 0, 0]],
        ], dtype=SimDType), return_tick=0)
        # fmt: on

        dim = SimDimensions.build(
            tau_step_lengths=[1.0],
            start_date=date(2021, 1, 1),
            days=100,
            nodes=world.nodes,
            compartments=ipm.num_compartments,
            events=ipm.num_events,
        )

        exe = IpmExecutor(
            rume=MagicMock(spec=Rume, ipm=ipm, dim=dim),
            world=world,
            data=MagicMock(spec=DataResolver),
            rng=np.random.default_rng(),
        )

        cs_before = np.vstack(
            [world.get_cohort_array(n).sum(axis=0) for n in range(world.nodes)]
        )

        # Apply the IPM
        (_, es, _, _) = exe.apply(
            Tick(
                sim_index=0,
                day=0,
                date=date(2021, 1, 1),
                step=0,
                tau=1.0,
            )
        )

        cs_after = np.vstack(
            [world.get_cohort_array(n).sum(axis=0) for n in range(world.nodes)]
        )

        # Compartments after the application should be equal to
        # the compartments from before plus (the events * apply_matrix)
        np.testing.assert_equal(
            cs_before + np.matmul(es, exe._apply_matrix, dtype=SimDType),
            cs_after,
        )
