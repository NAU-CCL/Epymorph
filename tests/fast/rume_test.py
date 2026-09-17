# ruff: noqa: PT009,PT027
from typing import Sequence

import numpy as np
import pytest
import sympy
from numpy.typing import NDArray

from epymorph.attribute import (
    AbsoluteName,
    AttributeDef,
    ModuleNamePattern,
    NamePattern,
)
from epymorph.compartment_model import (
    CompartmentModel,
    MultiStrataModelSymbols,
    compartment,
    edge,
)
from epymorph.data.ipm.sirs import SIRS
from epymorph.data.mm.centroids import Centroids
from epymorph.data.mm.no import No
from epymorph.data_shape import Shapes
from epymorph.data_type import CentroidDType
from epymorph.database import (
    DefaultValue,
    MissingValue,
    ParameterValue,
    ReqNode,
    ReqTree,
)
from epymorph.error import DataAttributeError
from epymorph.geography.us_census import StateScope
from epymorph.initializer import NoInfection, SingleLocation
from epymorph.movement_model import EveryDay, MovementClause, MovementModel
from epymorph.params import ParamFunctionNode, ParamFunctionNumpy, ParamFunctionScalar
from epymorph.rume import (
    GPM,
    RUME,
    MultiStrataRUME,
    SingleStrataRUME,
    combine_tau_steps,
    remap_taus,
)
from epymorph.simulation import NEVER, ParamValue, Tick, TickDelta, TickIndex
from epymorph.strata import DEFAULT_STRATA
from epymorph.time import TimeFrame


def _assert_tau_steps_equal(list1: Sequence[float], list2: Sequence[float]) -> None:
    """Check that two lists of tau steps are approximately equal."""
    assert len(list1) == len(list2)
    for a, b in zip(list1, list2, strict=True):
        assert abs(a - b) < 1e-10


###################
# Combine MM test #
###################


def test_combine_tau_steps_1():
    new_taus, start_map, stop_map = combine_tau_steps(
        {
            "a": [1 / 3, 2 / 3],
            "b": [1 / 2, 1 / 2],
        }
    )
    _assert_tau_steps_equal(new_taus, [1 / 3, 1 / 6, 1 / 2])
    assert start_map == {
        "a": {0: 0, 1: 1},
        "b": {0: 0, 1: 2},
    }
    assert stop_map == {
        "a": {0: 0, 1: 2},
        "b": {0: 1, 1: 2},
    }


def test_combine_tau_steps_2():
    new_taus, start_map, stop_map = combine_tau_steps(
        {
            "a": [1 / 3, 2 / 3],
        }
    )
    _assert_tau_steps_equal(new_taus, [1 / 3, 2 / 3])
    assert start_map == {
        "a": {0: 0, 1: 1},
    }
    assert stop_map == {
        "a": {0: 0, 1: 1},
    }


def test_combine_tau_steps_3():
    new_taus, start_map, stop_map = combine_tau_steps(
        {
            "a": [1 / 3, 2 / 3],
            "b": [1 / 3, 2 / 3],
        }
    )
    _assert_tau_steps_equal(new_taus, [1 / 3, 2 / 3])
    assert start_map == {
        "a": {0: 0, 1: 1},
        "b": {0: 0, 1: 1},
    }
    assert stop_map == {
        "a": {0: 0, 1: 1},
        "b": {0: 0, 1: 1},
    }


def test_combine_tau_steps_4():
    new_taus, start_map, stop_map = combine_tau_steps(
        {
            "a": [0.5, 0.5],
            "b": [0.2, 0.4, 0.4],
            "c": [0.1, 0.7, 0.2],
            "d": [0.5, 0.5],
        }
    )
    _assert_tau_steps_equal(new_taus, [0.1, 0.1, 0.3, 0.1, 0.2, 0.2])
    assert start_map == {
        "a": {0: 0, 1: 3},
        "b": {0: 0, 1: 2, 2: 4},
        "c": {0: 0, 1: 1, 2: 5},
        "d": {0: 0, 1: 3},
    }
    assert stop_map == {
        "a": {0: 2, 1: 5},
        "b": {0: 1, 1: 3, 2: 5},
        "c": {0: 0, 1: 4, 2: 5},
        "d": {0: 2, 1: 5},
    }


def test_remap_taus_1():
    class Clause1(MovementClause):
        leaves = TickIndex(0)
        returns = TickDelta(days=0, step=1)
        predicate = EveryDay()

        def evaluate(self, tick: Tick) -> NDArray[np.int64]:
            return np.array([])

    class Model1(MovementModel):
        steps = (1 / 3, 2 / 3)
        clauses = (Clause1(),)

    class Clause2(MovementClause):
        leaves = TickIndex(1)
        returns = TickDelta(days=0, step=1)
        predicate = EveryDay()

        def evaluate(self, tick: Tick) -> NDArray[np.int64]:
            return np.array([])

    class Model2(MovementModel):
        steps = (1 / 2, 1 / 2)
        clauses = (Clause2(),)

    new_mms = remap_taus([("a", Model1()), ("b", Model2())])

    new_taus = new_mms["a"].steps
    _assert_tau_steps_equal(new_taus, [1 / 3, 1 / 6, 1 / 2])
    assert len(new_mms) == 2

    new_mm1 = new_mms["a"]
    assert new_mm1.clauses[0].leaves.step == 0
    assert new_mm1.clauses[0].returns.step == 2

    new_mm2 = new_mms["b"]
    assert new_mm2.clauses[0].leaves.step == 2
    assert new_mm2.clauses[0].returns.step == 2


def test_remap_taus_2():
    class Clause1(MovementClause):
        leaves = TickIndex(0)
        returns = TickDelta(days=0, step=1)
        predicate = EveryDay()

        def evaluate(self, tick: Tick) -> NDArray[np.int64]:
            return np.array([])

    class Model1(MovementModel):
        steps = (1 / 3, 2 / 3)
        clauses = (Clause1(),)

    class Clause2(MovementClause):
        leaves = TickIndex(1)
        returns = NEVER
        predicate = EveryDay()

        def evaluate(self, tick: Tick) -> NDArray[np.int64]:
            return np.array([])

    class Model2(MovementModel):
        steps = (1 / 2, 1 / 2)
        clauses = (Clause2(),)

    new_mms = remap_taus([("a", Model1()), ("b", Model2())])

    new_taus = new_mms["a"].steps
    _assert_tau_steps_equal(new_taus, [1 / 3, 1 / 6, 1 / 2])
    assert len(new_mms) == 2

    new_mm1 = new_mms["a"]
    assert new_mm1.clauses[0].leaves.step == 0
    assert new_mm1.clauses[0].returns.step == 2

    new_mm2 = new_mms["b"]
    assert new_mm2.clauses[0].leaves.step == 2
    assert new_mm2.clauses[0].returns.step == -1


#############
# RUME test #
#############


class Sir(CompartmentModel):
    compartments = [
        compartment("S"),
        compartment("I"),
        compartment("R"),
    ]

    requirements = [
        AttributeDef("beta", float, Shapes.TxN),
        AttributeDef("gamma", float, Shapes.TxN),
    ]

    def edges(self, symbols):
        [S, I, R] = symbols.all_compartments  # noqa: N806
        [beta, gamma] = symbols.all_requirements
        return [
            edge(S, I, rate=beta * S * I),
            edge(I, R, rate=gamma * I),
        ]


def test_rume_create_monostrata_1():
    # A single-strata RUME uses the IPM without modification.
    sir = Sir()
    centroids = Centroids()
    # Make sure centroids has the tau steps we will expect later...
    _assert_tau_steps_equal(centroids.steps, [1 / 3, 2 / 3])

    rume = SingleStrataRUME.build(
        ipm=sir,
        mm=centroids,
        init=NoInfection(),
        scope=StateScope.in_states(["04", "35"], year=2020),
        time_frame=TimeFrame.of("2021-01-01", 180),
        params={},
    )
    assert sir is rume.ipm

    assert rume.num_ticks == 360
    _assert_tau_steps_equal(rume.tau_step_lengths, [1 / 3, 2 / 3])

    assert rume.compartment_mask[DEFAULT_STRATA].tolist() == [True, True, True]
    assert rume.compartment_mobility[DEFAULT_STRATA].tolist() == [True, True, True]


def test_rume_create_multistrata_1():
    # Test a multi-strata model.

    sir = Sir()
    no = No()
    # Make sure 'no' has the tau steps we will expect later...
    _assert_tau_steps_equal(no.steps, [1.0])

    rume = MultiStrataRUME.build(
        strata=[
            GPM(
                name="aaa",
                ipm=sir,
                mm=no,
                init=SingleLocation(location=0, seed_size=100),
            ),
            GPM(
                name="bbb",
                ipm=sir,
                mm=no,
                init=SingleLocation(location=0, seed_size=100),
            ),
        ],
        meta_requirements=[],
        meta_edges=lambda _: [],
        scope=StateScope.in_states(["04", "35"], year=2020),
        time_frame=TimeFrame.of("2021-01-01", 180),
        params={},
    )

    assert rume.num_ticks == 180
    _assert_tau_steps_equal(rume.tau_step_lengths, [1.0])

    # fmt: off
    assert rume.compartment_mask["aaa"].tolist() \
        == [True, True, True, False, False, False]
    assert rume.compartment_mask["bbb"].tolist() \
        == [False, False, False, True, True, True]
    assert rume.compartment_mobility["aaa"].tolist() \
        == [True, True, True, False, False, False]
    assert rume.compartment_mobility["bbb"].tolist() \
        == [False, False, False, True, True, True]
    # fmt: on

    # NOTE: these tests will break if someone alters the MM or Init definition;
    # even just the comments
    assert rume.requirements == {
        AbsoluteName("gpm:aaa", "ipm", "beta"): AttributeDef("beta", float, Shapes.TxN),
        AbsoluteName("gpm:aaa", "ipm", "gamma"): AttributeDef(
            "gamma", float, Shapes.TxN
        ),
        AbsoluteName("gpm:bbb", "ipm", "beta"): AttributeDef("beta", float, Shapes.TxN),
        AbsoluteName("gpm:bbb", "ipm", "gamma"): AttributeDef(
            "gamma", float, Shapes.TxN
        ),
        AbsoluteName("gpm:aaa", "init", "population"): AttributeDef(
            "population",
            int,
            Shapes.N,
            comment="The population at each geo node.",
        ),
        AbsoluteName("gpm:bbb", "init", "population"): AttributeDef(
            "population",
            int,
            Shapes.N,
            comment="The population at each geo node.",
        ),
    }


def test_rume_create_multistrata_2():
    # Test special case: a multi-strata model but with only one strata.

    sir = Sir()
    centroids = Centroids()
    # Make sure centroids has the tau steps we will expect later...
    _assert_tau_steps_equal(centroids.steps, [1 / 3, 2 / 3])

    rume = MultiStrataRUME.build(
        strata=[
            GPM(
                name="aaa",
                ipm=sir,
                mm=centroids,
                init=NoInfection(),
            ),
        ],
        meta_requirements=[],
        meta_edges=lambda _: [],
        scope=StateScope.in_states(["04", "35"], year=2020),
        time_frame=TimeFrame.of("2021-01-01", 180),
        params={},
    )

    assert rume.num_ticks == 360
    _assert_tau_steps_equal(rume.tau_step_lengths, [1 / 3, 2 / 3])

    # NOTE: these tests will break if someone alters the MM or Init definition;
    # even just the comments
    assert rume.requirements == {
        AbsoluteName("gpm:aaa", "ipm", "beta"): AttributeDef("beta", float, Shapes.TxN),
        AbsoluteName("gpm:aaa", "ipm", "gamma"): AttributeDef(
            "gamma", float, Shapes.TxN
        ),
        AbsoluteName("gpm:aaa", "mm", "population"): AttributeDef(
            "population",
            int,
            Shapes.N,
            comment="The total population at each node.",
        ),
        AbsoluteName("gpm:aaa", "mm", "centroid"): AttributeDef(
            "centroid",
            (("longitude", float), ("latitude", float)),
            Shapes.N,
            comment=("The centroids for each node as (longitude, latitude) tuples."),
        ),
        AbsoluteName("gpm:aaa", "mm", "phi"): AttributeDef(
            "phi",
            float,
            Shapes.Scalar,
            comment="Influences the distance that movers tend to travel.",
            default_value=40.0,
        ),
        AbsoluteName("gpm:aaa", "mm", "commuter_proportion"): AttributeDef(
            "commuter_proportion",
            float,
            Shapes.Scalar,
            default_value=0.1,
            comment="The proportion of the total population that commutes.",
        ),
        AbsoluteName("gpm:aaa", "init", "population"): AttributeDef(
            "population",
            int,
            Shapes.N,
            comment="The population at each geo node.",
        ),
    }


def _default_params() -> dict[str, ParamValue]:
    return {
        "gpm:aaa::ipm::beta": 0.4,
        "gpm:bbb::ipm::beta": 0.3,
        "gamma": 1 / 10,
        "gpm:aaa::ipm::xi": 0,
        "gpm:bbb::ipm::xi": 1 / 90,
        "ipm::beta_bbb_aaa": 0.2,
        "gpm:aaa::*::population": [100, 200],
        "gpm:bbb::*::population": np.array([300, 400], dtype=np.int64),
        "*::*::centroid": np.array([(1.0, 1.0), (2.0, 2.0)], dtype=CentroidDType),
    }


def _rume(rume_params: dict[str, ParamValue] | None = None) -> RUME:
    meta_requirements = [AttributeDef("beta_bbb_aaa", float, Shapes.TxN)]

    def meta_edges(s: MultiStrataModelSymbols):
        [S_aaa, I_aaa, R_aaa] = s.strata_compartments("aaa")  # noqa: N806
        [S_bbb, I_bbb, R_bbb] = s.strata_compartments("bbb")  # noqa: N806
        [beta_bbb_aaa] = s.all_meta_requirements
        N_aaa = sympy.Max(1, S_aaa + I_aaa + R_aaa)  # noqa: N806
        return [edge(S_bbb, I_bbb, beta_bbb_aaa * S_bbb * I_aaa / N_aaa)]

    return MultiStrataRUME.build(
        strata=[
            GPM(
                name="aaa",
                ipm=SIRS(),
                mm=Centroids(),
                init=SingleLocation(location=0, seed_size=100),
            ),
            GPM(
                name="bbb",
                ipm=SIRS(),
                mm=Centroids(),
                init=SingleLocation(location=0, seed_size=100),
                params={
                    ModuleNamePattern.parse("beta"): 99.0,
                    ModuleNamePattern.parse("phi"): 33.0,
                },
            ),
        ],
        meta_requirements=meta_requirements,
        meta_edges=meta_edges,
        scope=StateScope.in_states(["04", "35"], year=2020),
        time_frame=TimeFrame.of("2021-01-01", 180),
        params=rume_params or _default_params(),
    )


def _assert_has_node(tree: ReqTree, name: str) -> ReqNode:
    expected = AbsoluteName.parse(name)
    for node in tree.traverse():
        if node.name == expected:
            return node
    else:
        pytest.fail(f"ReqTree did not contain the expected node: {name}")


def test_requirements_tree_contains_all_requirements():
    rume = _rume()
    tree = rume.requirements_tree()

    # The tree's first level should one node for each RUME requirement.
    # There could be second-level or deeper requirements (transitives),
    # but that wouldn't change this.
    assert {node.name for node in tree.children} == set(rume.requirements)
    # Check that the definitions line up as expected.
    for node in tree.children:
        assert node.definition == rume.requirements[node.name]


def test_requirements_tree_basic_resolution():
    tree = _rume().requirements_tree()

    beta_aaa = _assert_has_node(tree, "gpm:aaa::ipm::beta")
    assert beta_aaa.value == 0.4
    assert beta_aaa.resolution == ParameterValue(
        cacheable=True,
        pattern=NamePattern.parse("gpm:aaa::ipm::beta"),
    )

    beta_bbb = _assert_has_node(tree, "gpm:bbb::ipm::beta")
    assert beta_bbb.value == 0.3
    assert beta_bbb.resolution == ParameterValue(
        cacheable=True,
        pattern=NamePattern.parse("gpm:bbb::ipm::beta"),
    )

    population_bbb = _assert_has_node(tree, "gpm:bbb::mm::population")
    assert population_bbb.value is not None
    np.testing.assert_array_equal(
        population_bbb.value,
        np.array([300, 400], dtype=np.int64),
    )
    assert population_bbb.resolution == ParameterValue(
        cacheable=True,
        pattern=NamePattern.parse("gpm:bbb::*::population"),
    )

    phi_aaa = _assert_has_node(tree, "gpm:aaa::mm::phi")
    assert phi_aaa.value == 40.0
    assert phi_aaa.resolution == DefaultValue(40.0)

    phi_bbb = _assert_has_node(tree, "gpm:bbb::mm::phi")
    assert phi_bbb.value == 33.0
    assert phi_bbb.resolution == ParameterValue(
        cacheable=True,
        pattern=NamePattern.parse("gpm:bbb::*::phi"),
    )


def test_requirements_tree_override_parameters():
    tree = _rume().requirements_tree({"*::*::beta": 0.5})

    for name in ("gpm:aaa::ipm::beta", "gpm:bbb::ipm::beta"):
        node = _assert_has_node(tree, name)
        assert node.value == 0.5
        assert node.resolution == ParameterValue(
            cacheable=True,
            pattern=NamePattern.parse("*::*::beta"),
        )


def test_requirements_tree_missing_values():
    params = _default_params()
    del params["gamma"]
    tree = _rume(params).requirements_tree()

    missing_names = {requirement.name for requirement in tree.missing()}
    assert missing_names == {
        AbsoluteName.parse("gpm:aaa::ipm::gamma"),
        AbsoluteName.parse("gpm:bbb::ipm::gamma"),
    }
    for name in missing_names:
        node = _assert_has_node(tree, str(name))
        assert isinstance(node.resolution, MissingValue)


def test_requirements_tree_transitive_deps_1():
    class Xi(ParamFunctionNode):
        BETA = AttributeDef("beta", float, Shapes.TxN)

        requirements = [BETA]

        def evaluate1(self, node_index: int) -> float:
            return float(node_index)

    tree = _rume().requirements_tree({"ipm::xi": Xi()})

    for strata in ("aaa", "bbb"):
        node = _assert_has_node(tree, f"gpm:{strata}::ipm::xi")
        assert isinstance(node.resolution, ParameterValue)
        assert node.resolution.cacheable
        assert node.resolution.pattern == NamePattern.parse("ipm::xi")
        child_reqs = [child.name for child in node.children]
        assert child_reqs == [AbsoluteName.parse(f"gpm:{strata}::ipm::beta")]


def test_requirements_tree_transitive_deps_2():
    class Gamma(ParamFunctionScalar):
        BETA = AttributeDef("beta", float, Shapes.Scalar)

        requirements = [BETA]

        def evaluate1(self) -> float:
            beta = self.data(self.BETA)
            return float(beta) * 4.0

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
            return np.asarray(alpha * gamma, dtype=np.float64)

    tree = _rume().requirements_tree(
        override_params={
            "gpm:aaa::ipm::alpha": 10,
            "gpm:bbb::ipm::alpha": 20,
            "gpm:aaa::ipm::beta": 0.4,
            "gpm:bbb::ipm::beta": 0.6,
            "*::ipm::gamma": Gamma(),
            "*::ipm::xi": Xi(),
        }
    )

    for strata in ("aaa", "bbb"):
        xi = _assert_has_node(tree, f"gpm:{strata}::ipm::xi")
        assert xi.resolution == ParameterValue(
            cacheable=True,
            pattern=NamePattern.parse("*::ipm::xi"),
        )
        assert [n.resolution for n in xi.children] == [
            ParameterValue(
                cacheable=True,
                pattern=NamePattern.parse(f"gpm:{strata}::ipm::alpha"),
            ),
            ParameterValue(
                cacheable=True,
                pattern=NamePattern.parse("*::ipm::gamma"),
            ),
        ]

        gamma = _assert_has_node(tree, f"gpm:{strata}::ipm::gamma")
        assert [n.resolution for n in gamma.children] == [
            ParameterValue(
                cacheable=True,
                pattern=NamePattern.parse(f"gpm:{strata}::ipm::beta"),
            ),
        ]


def test_requirements_tree_circular_dependencies_1():
    class Beta(ParamFunctionScalar):
        BETA = AttributeDef("beta", float, Shapes.Scalar)

        requirements = [BETA]

        def evaluate1(self) -> float:
            return 0.0

    with pytest.raises(DataAttributeError, match="Circular dependency"):
        _rume().requirements_tree({"gpm:aaa::ipm::beta": Beta()})


def test_requirements_tree_circular_dependencies_2():
    class Gamma(ParamFunctionScalar):
        XI = AttributeDef("xi", float, Shapes.Scalar)

        requirements = [XI]

        def evaluate1(self) -> float:
            return 0.0

    class Xi(ParamFunctionScalar):
        GAMMA = AttributeDef("gamma", float, Shapes.Scalar)

        requirements = [GAMMA]

        def evaluate1(self) -> float:
            return 0.0

    with pytest.raises(DataAttributeError, match="Circular dependency"):
        _rume().requirements_tree(
            {
                "gpm:aaa::ipm::gamma": Gamma(),
                "gpm:aaa::ipm::xi": Xi(),
            }
        )
