# ruff: noqa: PT009,PT027
import math
import unittest

import numpy as np
import numpy.testing as npt
import sympy
from numpy.typing import NDArray

from epymorph.attribute import (
    AbsoluteName,
    AttributeDef,
    ModuleNamePattern,
)
from epymorph.compartment_model import MultiStrataModelSymbols, edge
from epymorph.data.ipm.sirs import SIRS
from epymorph.data.mm.centroids import Centroids
from epymorph.data_shape import Shapes
from epymorph.data_type import AttributeArray, CentroidDType
from epymorph.error import DataAttributeError
from epymorph.geography.us_census import StateScope
from epymorph.initializer import SingleLocation
from epymorph.params import (
    ParamFunctionNode,
    ParamFunctionNumpy,
    ParamFunctionScalar,
    ParamFunctionTimeAndNode,
    simulation_symbols,
)
from epymorph.rume import GPM, RUME, MultiStrataRUME
from epymorph.simulation import ParamValue
from epymorph.time import TimeFrame


class EvaluateParamsTest(unittest.TestCase):
    def assert_db(
        self,
        db: dict[AbsoluteName, AttributeArray],
        key: str,
        value: AttributeArray,
    ) -> None:
        matched = db.get(AbsoluteName.parse(key))
        if matched is None:
            self.fail(f"Database did not contain the expected key: {key}")
        else:
            msg = f"Database value at key {key} did not match expected."
            if value.dtype == np.float64:
                npt.assert_array_almost_equal(matched, value, err_msg=msg)  # type: ignore
            else:
                npt.assert_array_equal(matched, value, err_msg=msg)

    def _default_params(self) -> dict[str, ParamValue]:
        return {
            "gpm:aaa::ipm::beta": 0.4,
            "gpm:bbb::ipm::beta": 0.3,
            "gamma": 1 / 10,  # gamma for all strata will be the same
            "gpm:aaa::ipm::xi": 0,
            "gpm:bbb::ipm::xi": 1 / 90,
            "ipm::beta_bbb_aaa": 0.2,
            # use the same population values for init and mm modules
            # test input as lists and np arrays
            "gpm:aaa::*::population": [100, 200],
            "gpm:bbb::*::population": np.array([300, 400], dtype=np.int64),
            # param names can also include leading stars explicitly
            "*::*::centroid": np.array([(1.0, 1.0), (2.0, 2.0)], dtype=CentroidDType),
        }

    def _create_rume(self, rume_params: dict[str, ParamValue] | None = None) -> RUME:
        meta_requirements = [
            AttributeDef("beta_bbb_aaa", float, Shapes.TxN),
        ]

        def meta_edges(s: MultiStrataModelSymbols):
            [S_aaa, I_aaa, R_aaa] = s.strata_compartments("aaa")  # noqa: N806
            [S_bbb, I_bbb, R_bbb] = s.strata_compartments("bbb")  # noqa: N806
            [beta_bbb_aaa] = s.all_meta_requirements
            N_aaa = sympy.Max(1, S_aaa + I_aaa + R_aaa)  # noqa: N806
            return [
                edge(S_bbb, I_bbb, beta_bbb_aaa * S_bbb * I_aaa / N_aaa),
            ]

        return MultiStrataRUME.build(
            strata=[
                GPM(
                    name="aaa",
                    ipm=SIRS(),
                    mm=Centroids(),
                    init=SingleLocation(location=0, seed_size=100),
                    params={
                        # leave phi unspecified to test default value resolution
                    },
                ),
                GPM(
                    name="bbb",
                    ipm=SIRS(),
                    mm=Centroids(),
                    init=SingleLocation(location=0, seed_size=100),
                    params={
                        ModuleNamePattern.parse(k): v
                        for k, v in {
                            "beta": 99.0,  # we'll override this value to test shadowing
                            "phi": 33.0,  # test GPM value resolution
                        }.items()
                    },
                ),
            ],
            meta_requirements=meta_requirements,
            meta_edges=meta_edges,
            scope=StateScope.in_states(["04", "35"], year=2020),
            time_frame=TimeFrame.of("2021-01-01", 180),
            params=rume_params or self._default_params(),
        )

    def test_eval_1(self):
        rume = self._create_rume()

        db = rume.evaluate_params(rng=np.random.default_rng(1)).to_dict()

        # We should have as many entries in our DB as we have attributes in the RUME,
        # plus 1 (for geo labels).
        self.assertEqual(len(db), len(rume.requirements) + 1)

        self.assert_db(db, "gpm:aaa::ipm::beta", np.array(0.4, dtype=np.float64))
        self.assert_db(db, "gpm:bbb::ipm::beta", np.array(0.3, dtype=np.float64))
        self.assert_db(db, "gpm:aaa::ipm::gamma", np.array(0.1, dtype=np.float64))
        self.assert_db(db, "gpm:bbb::ipm::gamma", np.array(0.1, dtype=np.float64))
        self.assert_db(db, "gpm:aaa::ipm::xi", np.array(0.0, dtype=np.float64))
        self.assert_db(db, "gpm:bbb::ipm::xi", np.array(1 / 90, dtype=np.float64))
        self.assert_db(db, "meta::ipm::beta_bbb_aaa", np.array(0.2, dtype=np.float64))

        self.assert_db(
            db, "gpm:aaa::init::population", np.array([100, 200], dtype=np.int64)
        )
        self.assert_db(
            db, "gpm:bbb::init::population", np.array([300, 400], dtype=np.int64)
        )

        self.assert_db(
            db, "gpm:aaa::mm::population", np.array([100, 200], dtype=np.int64)
        )
        self.assert_db(
            db, "gpm:bbb::mm::population", np.array([300, 400], dtype=np.int64)
        )

        self.assert_db(db, "gpm:aaa::mm::phi", np.array(40.0, dtype=np.float64))
        self.assert_db(db, "gpm:bbb::mm::phi", np.array(33.0, dtype=np.float64))

        self.assert_db(
            db,
            "gpm:aaa::mm::centroid",
            np.array([(1.0, 1.0), (2.0, 2.0)], dtype=CentroidDType),
        )
        self.assert_db(
            db,
            "gpm:bbb::mm::centroid",
            np.array([(1.0, 1.0), (2.0, 2.0)], dtype=CentroidDType),
        )

        # When params are provided as the same literal value,
        # they should evaluate to the same object.
        x1 = db[AbsoluteName.parse("gpm:aaa::ipm::gamma")]
        x2 = db[AbsoluteName.parse("gpm:bbb::ipm::gamma")]
        self.assertIs(x1, x2)

    def test_eval_2(self):
        # Test with override values.
        rume = self._create_rume()

        db = rume.evaluate_params(
            override_params={"*::*::beta": 0.5},
            rng=np.random.default_rng(1),
        ).to_dict()

        # Beta should be overridden from test case 1,
        self.assert_db(db, "gpm:aaa::ipm::beta", np.array(0.5, dtype=np.float64))
        self.assert_db(db, "gpm:bbb::ipm::beta", np.array(0.5, dtype=np.float64))
        # the rest should be the same.
        self.assert_db(db, "gpm:aaa::ipm::gamma", np.array(0.1, dtype=np.float64))
        self.assert_db(db, "gpm:bbb::ipm::gamma", np.array(0.1, dtype=np.float64))
        self.assert_db(db, "gpm:aaa::ipm::xi", np.array(0.0, dtype=np.float64))
        self.assert_db(db, "gpm:bbb::ipm::xi", np.array(1 / 90, dtype=np.float64))

    def test_eval_3(self):
        # Test for missing attribute.
        # Use the default params but delete one of the attributes.
        params = self._default_params()
        del params["gamma"]

        rume = self._create_rume(params)

        with self.assertRaises(DataAttributeError) as ctx:
            rume.evaluate_params(rng=np.random.default_rng(1))

        err = str(ctx.exception).lower()
        self.assertIn("there are missing values", err)
        self.assertIn("gpm:aaa::ipm::gamma", err)
        self.assertIn("gpm:bbb::ipm::gamma", err)

    def test_eval_sympy_expression(self):
        # Test param as sympy expression
        t, T, n = simulation_symbols("day", "duration_days", "node_index")
        beta_expr = 0.04 * sympy.sin(8 * sympy.pi * t / T) + 0.34 + (0.02 * n)

        rume = self._create_rume()

        db = rume.evaluate_params(
            override_params={"gpm:aaa::ipm::beta": beta_expr},
            rng=np.random.default_rng(1),
        ).to_dict()

        expected = np.stack(
            [
                0.04 * np.sin(8 * np.pi * np.arange(180) / 180) + 0.34,
                0.04 * np.sin(8 * np.pi * np.arange(180) / 180) + 0.36,
            ],
            axis=1,
            dtype=np.float64,
        )
        self.assert_db(db, "gpm:aaa::ipm::beta", expected)

    def test_eval_param_function_1(self):
        # Test param as shaped function
        class Beta(ParamFunctionTimeAndNode):
            GAMMA = AttributeDef("gamma", float, Shapes.TxN)

            requirements = [GAMMA]

            r_0: float

            def __init__(self, r_0: float):
                self.r_0 = r_0

            def evaluate1(self, day: int, node_index: int) -> float:
                T = self.time_frame.days
                gamma = self.data(self.GAMMA)[day, node_index]
                magnitude = self.r_0 * gamma
                return (
                    0.1 * magnitude * math.sin(8 * math.pi * day / T)
                    + (0.85 * magnitude)
                    + (0.05 * magnitude * node_index)
                )

        rume = self._create_rume()

        db = rume.evaluate_params(
            override_params={"gpm:aaa::ipm::beta": Beta(4.0)},
            rng=np.random.default_rng(1),
        ).to_dict()

        expected = np.stack(
            [
                0.04 * np.sin(8 * np.pi * np.arange(180) / 180) + 0.34,
                0.04 * np.sin(8 * np.pi * np.arange(180) / 180) + 0.36,
            ],
            axis=1,
            dtype=np.float64,
        )
        self.assert_db(db, "gpm:aaa::ipm::beta", expected)

    def test_eval_param_function_2(self):
        # Test param as shaped function, with difference between strata
        class Xi(ParamFunctionNode):
            BETA = AttributeDef("beta", float, Shapes.TxN)

            requirements = [BETA]

            def evaluate1(self, node_index: int) -> float:
                beta = self.data(self.BETA)[0, node_index]
                return beta / (5 * (node_index + 1))

        rume = self._create_rume()

        db = rume.evaluate_params(
            override_params={"ipm::xi": Xi()},
            rng=np.random.default_rng(1),
        ).to_dict()

        expected_aaa = np.array([(0.4 / 5), (0.4 / 10)], dtype=np.float64)
        expected_bbb = np.array([(0.3 / 5), (0.3 / 10)], dtype=np.float64)
        self.assert_db(db, "gpm:aaa::ipm::xi", expected_aaa)
        self.assert_db(db, "gpm:bbb::ipm::xi", expected_bbb)

    def test_eval_param_function_chained(self):
        class Gamma(ParamFunctionScalar):
            BETA = AttributeDef("beta", float, Shapes.Scalar)

            requirements = [BETA]

            def evaluate1(self) -> float:
                beta = self.data(self.BETA)
                return float(beta) / 4.0

        class Xi(ParamFunctionNumpy):
            ALPHA = AttributeDef("alpha", float, Shapes.Scalar)
            GAMMA = AttributeDef("gamma", float, Shapes.Scalar)

            requirements = [ALPHA, GAMMA]

            def evaluate(self) -> NDArray[np.float64]:
                # alpha and gamma are both scalars,
                # but I'm using ParamFunctionNumpy
                # so it's on me to make sure my result is an NDArray
                alpha = self.data(self.ALPHA)
                gamma = self.data(self.GAMMA)
                return np.asarray(gamma / alpha)

        rume = self._create_rume()

        db = rume.evaluate_params(
            override_params={
                "gpm:aaa::ipm::alpha": 9,
                "gpm:aaa::ipm::beta": 0.4,
                "gpm:aaa::ipm::gamma": Gamma(),
                "gpm:aaa::ipm::xi": Xi(),
            },
            rng=np.random.default_rng(1),
        ).to_dict()

        self.assert_db(db, "gpm:aaa::ipm::alpha", np.array(9))
        self.assert_db(db, "gpm:aaa::ipm::beta", np.array(0.4))
        self.assert_db(db, "gpm:aaa::ipm::gamma", np.array(0.1))
        self.assert_db(db, "gpm:aaa::ipm::xi", np.array(1 / 90))

    def test_eval_param_function_circular(self):
        class Gamma(ParamFunctionNumpy):
            XI = AttributeDef("xi", float, Shapes.Scalar)

            requirements = [XI]

            def evaluate(self) -> NDArray[np.float64]:
                return np.array(0)

        class Xi(ParamFunctionNumpy):
            GAMMA = AttributeDef("gamma", float, Shapes.Scalar)

            requirements = [GAMMA]

            def evaluate(self) -> NDArray[np.float64]:
                return np.array(0)

        rume = self._create_rume()

        with self.assertRaises(DataAttributeError) as ctx:
            rume.evaluate_params(
                override_params={
                    "gpm:aaa::ipm::gamma": Gamma(),
                    "gpm:aaa::ipm::xi": Xi(),
                },
                rng=np.random.default_rng(1),
            )

        err = str(ctx.exception).lower()
        self.assertIn("circular dependency", err)
        self.assertIn("gpm:aaa::ipm::gamma", err)
