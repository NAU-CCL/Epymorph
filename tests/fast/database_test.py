# ruff: noqa: PT009,PT027
import math
from typing import Any, TypeVar, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
import sympy
from numpy.typing import NDArray

from epymorph.attribute import (
    AbsoluteName,
    AttributeDef,
    AttributeName,
    ModuleName,
    ModuleNamePattern,
    ModuleNamespace,
    NamePattern,
)
from epymorph.compartment_model import CombinedCompartmentModel
from epymorph.data_shape import Dimensions, Shapes
from epymorph.data_type import (
    AttributeData,
    SingleStratumAttributeArray,
    StratifiedAttributeArray,
)
from epymorph.database import (
    Database,
    DataResolver,
    DefaultValue,
    Match,
    MissingValue,
    ParameterValue,
    ReqNode,
    ReqTree,
    evaluate_requirements,
)
from epymorph.error import DataAttributeError, DataAttributeErrorGroup
from epymorph.geography.scope import GeoScope
from epymorph.params import ParamFunction, ParamFunctionTimeAndNode, simulation_symbols
from epymorph.time import TimeFrame

AD = AttributeDef
AN = AbsoluteName.parse
NP = NamePattern.parse


@pytest.fixture(scope="module")
def time_frame():
    return TimeFrame.of("2020-01-01", 3)


@pytest.fixture(scope="module")
def scope():
    return MagicMock(spec=GeoScope, nodes=2)


@pytest.fixture(scope="module")
def multistrata_ipm():
    return MagicMock(spec=CombinedCompartmentModel, strata=("aaa", "bbb"))


###################
# ModuleNamespace #
###################


def test_module_namespace_post_init_empty():
    with pytest.raises(ValueError, match="Invalid name"):
        ModuleNamespace("", "module")
    with pytest.raises(ValueError, match="Invalid name"):
        ModuleNamespace("strata", "")


def test_module_namespace_post_init_wildcards():
    with pytest.raises(ValueError, match="Invalid name"):
        ModuleNamespace("*", "module")
    with pytest.raises(ValueError, match="Invalid name"):
        ModuleNamespace("strata", "*")


def test_module_namespace_post_init_delimeters():
    with pytest.raises(ValueError, match="Invalid name"):
        ModuleNamespace("::", "module")
    with pytest.raises(ValueError, match="Invalid name"):
        ModuleNamespace("strata", "::")


def test_module_namespace_parse_valid_string():
    ns = ModuleNamespace.parse("strata::module")
    assert ns.strata == "strata"
    assert ns.module == "module"


def test_module_namespace_parse_invalid_string():
    with pytest.raises(ValueError, match="Invalid number of parts"):
        ModuleNamespace.parse("invalid_string")


def test_module_namespace_parse_with_more_parts():
    with pytest.raises(ValueError, match="Invalid number of parts"):
        ModuleNamespace.parse("too::many::parts")


def test_module_namespace_str_representation():
    ns = ModuleNamespace("strata", "module")
    assert str(ns) == "strata::module"


def test_module_namespace_to_absolute():
    ns = ModuleNamespace("strata", "module")
    pattern = ns.to_absolute("id")
    assert isinstance(pattern, AbsoluteName)
    assert pattern.strata == "strata"
    assert pattern.module == "module"
    assert pattern.id == "id"


################
# AbsoluteName #
################


def test_absolute_name_post_init_empty():
    with pytest.raises(ValueError, match="Invalid name"):
        AbsoluteName("", "module", "id")
    with pytest.raises(ValueError, match="Invalid name"):
        AbsoluteName("strata", "", "id")
    with pytest.raises(ValueError, match="Invalid name"):
        AbsoluteName("strata", "module", "")


def test_absolute_name_post_init_wildcards():
    with pytest.raises(ValueError, match="Invalid name"):
        AbsoluteName("*", "module", "id")
    with pytest.raises(ValueError, match="Invalid name"):
        AbsoluteName("strata", "*", "id")
    with pytest.raises(ValueError, match="Invalid name"):
        AbsoluteName("strata", "module", "*")


def test_absolute_name_post_init_delimeters():
    with pytest.raises(ValueError, match="Invalid name"):
        AbsoluteName("::", "module", "id")
    with pytest.raises(ValueError, match="Invalid name"):
        AbsoluteName("strata", "::", "id")
    with pytest.raises(ValueError, match="Invalid name"):
        AbsoluteName("strata", "module", "::")


def test_absolute_name_parse_valid_string():
    name = AbsoluteName.parse("strata::module::id")
    assert name.strata == "strata"
    assert name.module == "module"
    assert name.id == "id"


def test_absolute_name_parse_invalid_string():
    with pytest.raises(ValueError, match="Invalid number of parts"):
        AbsoluteName.parse("invalid_string")


def test_absolute_name_str_representation():
    name = AbsoluteName("strata", "module", "id")
    assert str(name) == "strata::module::id"


def test_absolute_name_in_strata():
    name = AbsoluteName("strata", "module", "id")
    new_name = name.in_strata("new_strata")
    assert isinstance(new_name, AbsoluteName)
    assert new_name.strata == "new_strata"
    assert new_name.module == "module"
    assert new_name.id == "id"


def test_absolute_name_to_namespace():
    name = AbsoluteName("strata", "module", "id")
    namespace = name.to_namespace()
    assert isinstance(namespace, ModuleNamespace)
    assert namespace.strata == "strata"
    assert namespace.module == "module"


def test_absolute_name_to_pattern():
    name = AbsoluteName("strata", "module", "id")
    pattern = name.to_pattern()
    assert isinstance(pattern, NamePattern)
    assert pattern.strata == "strata"
    assert pattern.module == "module"
    assert pattern.id == "id"


##############
# ModuleName #
##############


def test_module_name_post_init_empty():
    with pytest.raises(ValueError, match="Invalid name"):
        ModuleName("module", "")
    with pytest.raises(ValueError, match="Invalid name"):
        ModuleName("", "id")


def test_module_name_post_init_wildcards():
    with pytest.raises(ValueError, match="Invalid name"):
        ModuleName("*", "id")
    with pytest.raises(ValueError, match="Invalid name"):
        ModuleName("module", "*")


def test_module_name_post_init_delimeters():
    with pytest.raises(ValueError, match="Invalid name"):
        ModuleName("::", "id")
    with pytest.raises(ValueError, match="Invalid name"):
        ModuleName("module", "::")


def test_module_name_parse_empty():
    with pytest.raises(ValueError, match="Invalid number of parts"):
        ModuleName.parse("")


def test_module_name_parse_valid_string():
    name = ModuleName.parse("module::id")
    assert name.module == "module"
    assert name.id == "id"


def test_module_name_parse_invalid_string():
    with pytest.raises(ValueError, match="Invalid number of parts"):
        ModuleName.parse("invalid_string")


def test_module_name_parse_with_more_parts():
    with pytest.raises(ValueError, match="Invalid number of parts"):
        ModuleName.parse("too::many::parts")


def test_module_name_str_representation():
    name = ModuleName("module", "id")
    assert str(name) == "module::id"


def test_module_name_to_absolute():
    name = ModuleName("module", "id")
    absolute_name = name.to_absolute("strata")
    assert isinstance(absolute_name, AbsoluteName)
    assert absolute_name.strata == "strata"
    assert absolute_name.module == "module"
    assert absolute_name.id == "id"


#################
# AttributeName #
#################


def test_attribute_name_post_init_empty():
    with pytest.raises(ValueError, match="Invalid name"):
        AttributeName("")


def test_attribute_name_post_init_wildcard_id():
    with pytest.raises(ValueError, match="Invalid name"):
        AttributeName("*")


def test_attribute_name_post_init_delimiters():
    with pytest.raises(ValueError, match="Invalid name"):
        AttributeName("invalid::id")


def test_attribute_name_str_representation():
    attr_name = AttributeName("id")
    assert str(attr_name) == "id"


###############
# NamePattern #
###############


def test_name_pattern_post_init_empty():
    with pytest.raises(ValueError, match="Invalid pattern"):
        NamePattern("", "module", "id")
    with pytest.raises(ValueError, match="Invalid pattern"):
        NamePattern("strata", "", "id")
    with pytest.raises(ValueError, match="Invalid pattern"):
        NamePattern("strata", "module", "")


def test_name_pattern_post_init_delimeters():
    with pytest.raises(ValueError, match="Invalid pattern"):
        NamePattern("::", "module", "id")
    with pytest.raises(ValueError, match="Invalid pattern"):
        NamePattern("strata", "::", "id")
    with pytest.raises(ValueError, match="Invalid pattern"):
        NamePattern("strata", "module", "::")


def test_name_pattern_parse_one_part():
    pattern = NamePattern.parse("id")
    assert pattern.strata == "*"
    assert pattern.module == "*"
    assert pattern.id == "id"


def test_name_pattern_parse_two_parts():
    pattern = NamePattern.parse("module::id")
    assert pattern.strata == "*"
    assert pattern.module == "module"
    assert pattern.id == "id"


def test_name_pattern_parse_three_parts():
    pattern = NamePattern.parse("strata::module::id")
    assert pattern.strata == "strata"
    assert pattern.module == "module"
    assert pattern.id == "id"


def test_name_pattern_parse_invalid_string():
    with pytest.raises(ValueError, match="Invalid number of parts"):
        NamePattern.parse("too::many::parts::here")


def test_name_pattern_match_absolute_name():
    valid_patterns = [
        NamePattern("strata", "module", "*"),
        NamePattern("strata", "*", "id"),
        NamePattern("*", "module", "id"),
        NamePattern("*", "*", "id"),
        NamePattern("*", "module", "*"),
        NamePattern("strata", "*", "*"),
        NamePattern("*", "*", "*"),
    ]
    for pattern in valid_patterns:
        absolute_name = AbsoluteName("strata", "module", "id")
        assert pattern.match(absolute_name)


def test_name_pattern_no_match_absolute_name():
    pattern = NamePattern("strata", "module", "*")
    absolute_name = AbsoluteName("other_strata", "module", "id")
    assert not pattern.match(absolute_name)

    pattern = NamePattern("strata", "*", "id")
    absolute_name = AbsoluteName("other_strata", "module", "id")
    assert not pattern.match(absolute_name)

    pattern = NamePattern("*", "module", "id")
    absolute_name = AbsoluteName("strata", "other_module", "id")
    assert not pattern.match(absolute_name)


def test_name_pattern_match_name_pattern():
    pattern1 = NamePattern("strata", "*", "id")
    pattern2 = NamePattern("strata", "module", "id")
    assert pattern1.match(pattern2)


def test_name_pattern_no_match_name_pattern():
    pattern1 = NamePattern("strata", "module", "id")
    pattern2 = NamePattern("*", "other_module", "id")
    assert not pattern1.match(pattern2)


def test_name_pattern_str_representation():
    pattern = NamePattern("strata", "module", "id")
    assert str(pattern) == "strata::module::id"


#####################
# ModuleNamePattern #
#####################


def test_module_name_pattern_post_init_empty():
    with pytest.raises(ValueError, match="Invalid pattern"):
        ModuleNamePattern("", "id")
    with pytest.raises(ValueError, match="Invalid pattern"):
        ModuleNamePattern("module", "")


def test_module_name_pattern_post_init_delimeters():
    with pytest.raises(ValueError, match="Invalid pattern"):
        ModuleNamePattern("::", "id")
    with pytest.raises(ValueError, match="Invalid pattern"):
        ModuleNamePattern("module", "::")


def test_module_name_pattern_parse_one_part():
    pattern = ModuleNamePattern.parse("id")
    assert pattern.module == "*"
    assert pattern.id == "id"


def test_module_name_pattern_parse_two_parts():
    pattern = ModuleNamePattern.parse("module::id")
    assert pattern.module == "module"
    assert pattern.id == "id"


def test_module_name_pattern_parse_invalid_string():
    with pytest.raises(ValueError, match="Invalid number of parts"):
        ModuleNamePattern.parse("too::many::parts::here")


def test_module_name_pattern_parse_empty():
    with pytest.raises(ValueError, match="Empty string"):
        ModuleNamePattern.parse("")


def test_module_name_pattern_to_absolute():
    pattern = ModuleNamePattern("module", "id")
    absolute_pattern = pattern.to_absolute("strata")
    assert isinstance(absolute_pattern, NamePattern)
    assert absolute_pattern.strata == "strata"
    assert absolute_pattern.module == "module"
    assert absolute_pattern.id == "id"


def test_module_name_pattern_str_representation():
    pattern = ModuleNamePattern("module", "id")
    assert str(pattern) == "module::id"


############
# Database #
############


T = TypeVar("T")


def _assert_match(expected: T, test: Match[T] | None):
    if test is None:
        raise AssertionError("Expected a match, but it was None.")
    assert expected == test.value


def test_database_query():
    db = Database[int](
        {
            NamePattern("gpm:1", "ipm", "beta"): 1,
            NamePattern("*", "ipm", "delta"): 2,
            NamePattern("*", "*", "gamma"): 3,
            NamePattern("gpm:2", "ipm", "beta"): 4,
        }
    )

    _assert_match(1, db.query(AbsoluteName("gpm:1", "ipm", "beta")))
    _assert_match(1, db.query("gpm:1::ipm::beta"))
    _assert_match(4, db.query("gpm:2::ipm::beta"))
    assert db.query("gpm:3::ipm::beta") is None

    _assert_match(2, db.query("gpm:1::ipm::delta"))
    _assert_match(2, db.query("gpm:2::ipm::delta"))
    _assert_match(2, db.query("gpm:9::ipm::delta"))
    assert db.query("gpm:1::mm::delta") is None

    _assert_match(3, db.query("gpm:1::ipm::gamma"))
    _assert_match(3, db.query("gpm:2::ipm::gamma"))
    _assert_match(3, db.query("gpm:1::mm::gamma"))
    _assert_match(3, db.query("gpm:1::init::gamma"))


def test_database_query_ambiguous():
    with pytest.raises(ValueError, match="ambiguous"):
        Database[int](
            {
                NamePattern("*", "*", "beta"): 1,
                NamePattern("gpm:1", "*", "beta"): 2,
                NamePattern("*", "ipm", "beta"): 3,
            }
        )


def test_database_query_all():
    primary = Database[int](
        {
            NamePattern("gpm:1", "ipm", "beta"): 11,
            NamePattern("gpm:2", "*", "beta"): 44,
            NamePattern("gpm:3", "*", "*"): 55,
        }
    )

    secondary = Database[int](
        {
            NamePattern("gpm:1", "ipm", "beta"): 1,
            NamePattern("*", "ipm", "delta"): 2,
            NamePattern("*", "ipm", "gamma"): 3,
            NamePattern("gpm:2", "ipm", "beta"): 4,
            NamePattern("gpm:3", "init", "alpha"): 6,
        }
    )

    db = [primary, secondary]

    _assert_match(11, Database.query_all(db, "gpm:1::ipm::beta"))
    _assert_match(44, Database.query_all(db, "gpm:2::ipm::beta"))

    _assert_match(55, Database.query_all(db, "gpm:3::ipm::beta"))
    _assert_match(55, Database.query_all(db, "gpm:3::init::alpha"))
    _assert_match(55, Database.query_all(db, "gpm:3::foo::bar"))

    _assert_match(2, Database.query_all(db, "gpm:1::ipm::delta"))
    _assert_match(2, Database.query_all(db, "gpm:2::ipm::delta"))
    _assert_match(55, Database.query_all(db, "gpm:3::ipm::delta"))

    assert Database.query_all(db, "gpm:1::init::alpha") is None


########################
# Parameter evaluation #
########################


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(1)


def _to_txn(
    value: float,
    time_frame: TimeFrame,
    scope: GeoScope,
) -> NDArray[np.float64]:
    return np.broadcast_to(value, shape=(time_frame.days, scope.nodes))


def _assert_is_single_strata(
    values: dict[str, AttributeData],
    name: str,
) -> SingleStratumAttributeArray:
    val = values.get(name)
    if val is None:
        err = f"Expected value for {name} not found in values."
        raise AssertionError(err)
    if isinstance(val, StratifiedAttributeArray):
        err = f"Expected single-stratum value for {name}, but found a stratified value."
        raise AssertionError(err)
    return val


def _assert_is_stratified(
    values: dict[str, AttributeData],
    name: str,
) -> StratifiedAttributeArray:
    val = values.get(name)
    if val is None:
        err = f"Expected value for {name} not found in values."
        raise AssertionError(err)
    if not isinstance(val, StratifiedAttributeArray):
        err = f"Expected stratified value for {name}, but found a single-stratum value."
        raise AssertionError(err)
    return val


def test_param_eval_01(time_frame, scope):
    eval_calls = 0

    # Test that a function can be used for multiple strata
    # but will only be evaluated once if all of its dependencies
    # do not vary by strata.
    class F(ParamFunction):
        requirements = (AD("gamma", float, Shapes.TxN),)

        def evaluate(self):
            nonlocal eval_calls
            eval_calls += 1
            return 2.0 * self.data("gamma")

    reqs = ReqTree.of(
        requirements={
            AN("gpm:a::ipm::beta"): AD("beta", float, Shapes.TxN),
            AN("gpm:b::ipm::beta"): AD("beta", float, Shapes.TxN),
        },
        params=Database(
            {
                NP("*::ipm::beta"): F(),
                NP("*::ipm::gamma"): 0.7,
            }
        ),
    )

    values = reqs.evaluate(scope, time_frame, None, None).to_dict(simplify_names=True)

    # F evaluated once; beta is 1.4 for both strata
    assert 1 == eval_calls
    exp = _to_txn(1.4, time_frame, scope)
    a_val = _assert_is_single_strata(values, "gpm:a::ipm::beta")
    b_val = _assert_is_single_strata(values, "gpm:b::ipm::beta")
    np.testing.assert_array_equal(exp, a_val)
    np.testing.assert_array_equal(exp, b_val)


def test_param_eval_02(time_frame, scope):
    eval_calls = 0

    # Test that a function declared "randomized" will be evaluated
    # every time it's referenced, even if it otherwise wouldn't need to be.
    class F(ParamFunction):
        requirements = (AD("gamma", float, Shapes.TxN),)
        randomized = True

        def evaluate(self):
            nonlocal eval_calls
            eval_calls += 1
            return 2.0 * self.data("gamma")

    reqs = ReqTree.of(
        requirements={
            AN("gpm:a::ipm::beta"): AD("beta", float, Shapes.TxN),
            AN("gpm:b::ipm::beta"): AD("beta", float, Shapes.TxN),
        },
        params=Database(
            {
                NP("*::ipm::beta"): F(),
                NP("*::ipm::gamma"): 0.7,
            }
        ),
    )

    values = reqs.evaluate(scope, time_frame, None, None).to_dict(simplify_names=True)

    # F evaluated twice, even though it produces the same value each time
    # beta is 1.4 for both strata
    assert 2 == eval_calls
    exp = _to_txn(1.4, time_frame, scope)
    a_val = _assert_is_single_strata(values, "gpm:a::ipm::beta")
    b_val = _assert_is_single_strata(values, "gpm:b::ipm::beta")
    np.testing.assert_array_equal(exp, a_val)
    np.testing.assert_array_equal(exp, b_val)


def test_param_eval_03(time_frame, scope):
    eval_calls = 0

    # Test a single function resolving to different values
    # due to dependencies that differ between strata.
    class F(ParamFunction):
        requirements = (AD("gamma", float, Shapes.TxN),)

        def evaluate(self):
            nonlocal eval_calls
            eval_calls += 1
            return 2.0 * self.data("gamma")

    reqs = ReqTree.of(
        requirements={
            AN("gpm:a::ipm::beta"): AD("beta", float, Shapes.TxN),
            AN("gpm:b::ipm::beta"): AD("beta", float, Shapes.TxN),
        },
        params=Database(
            {
                NP("*::ipm::beta"): F(),
                NP("gpm:a::ipm::gamma"): 0.3,
                NP("gpm:b::ipm::gamma"): 0.7,
            }
        ),
    )

    values = reqs.evaluate(scope, time_frame, None, None).to_dict(simplify_names=True)

    # F evaluated twice
    # beta is 0.6 for strata a
    # and 1.4 for strata b
    assert 2 == eval_calls
    a_val = _assert_is_single_strata(values, "gpm:a::ipm::beta")
    b_val = _assert_is_single_strata(values, "gpm:b::ipm::beta")
    np.testing.assert_array_equal(_to_txn(0.6, time_frame, scope), a_val)
    np.testing.assert_array_equal(_to_txn(1.4, time_frame, scope), b_val)


def test_param_eval_04(time_frame, scope, rng):
    eval_calls = 0

    # Test a single shared random value when a single instance is used.
    class F(ParamFunction):
        requirements = (AD("gamma", float, Shapes.TxN),)

        def evaluate(self):
            nonlocal eval_calls
            eval_calls += 1
            return 2.0 * self.data("gamma") * self.rng.random()

    reqs = ReqTree.of(
        requirements={
            AN("gpm:a::ipm::beta"): AD("beta", float, Shapes.TxN),
            AN("gpm:b::ipm::beta"): AD("beta", float, Shapes.TxN),
        },
        params=Database(
            {
                NP("*::ipm::beta"): F(),
                NP("*::ipm::gamma"): 0.7,
            }
        ),
    )

    values = reqs.evaluate(scope, time_frame, None, rng).to_dict(simplify_names=True)

    # F evaluated once
    # beta is random, but the same value is shared between strata
    assert 1 == eval_calls

    a_val = _assert_is_single_strata(values, "gpm:a::ipm::beta")
    b_val = _assert_is_single_strata(values, "gpm:b::ipm::beta")
    np.testing.assert_array_equal(a_val, b_val)


def test_param_eval_05(time_frame, scope, rng):
    eval_calls = 0

    # Test unique random values by virtue of providing different instances.
    class F(ParamFunction):
        requirements = (AD("gamma", float, Shapes.TxN),)

        def evaluate(self):
            nonlocal eval_calls
            eval_calls += 1
            return 2.0 * self.data("gamma") * self.rng.random()

    reqs = ReqTree.of(
        requirements={
            AN("gpm:a::ipm::beta"): AD("beta", float, Shapes.TxN),
            AN("gpm:b::ipm::beta"): AD("beta", float, Shapes.TxN),
        },
        params=Database(
            {
                NP("gpm:a::ipm::beta"): F(),
                NP("gpm:b::ipm::beta"): F(),
                NP("*::ipm::gamma"): 0.7,
            }
        ),
    )

    values = reqs.evaluate(scope, time_frame, None, rng).to_dict(simplify_names=True)

    # Fs evaluated once each
    # beta is two unique random numbers
    assert 2 == eval_calls
    a_val = _assert_is_single_strata(values, "gpm:a::ipm::beta")
    b_val = _assert_is_single_strata(values, "gpm:b::ipm::beta")
    assert not np.array_equal(a_val, b_val)


def test_param_eval_06(time_frame, scope):
    # Test input broadcasting and TxN functions.
    class Beta(ParamFunctionTimeAndNode):
        GAMMA = AD("gamma", float, Shapes.TxN)

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

    reqs = ReqTree.of(
        requirements={
            AN("gpm:a::ipm::beta"): AD("beta", float, Shapes.TxN),
            AN("gpm:b::ipm::beta"): AD("beta", float, Shapes.TxN),
            AN("gpm:a::ipm::gamma"): AD("gamma", float, Shapes.TxN),
            AN("gpm:b::ipm::gamma"): AD("gamma", float, Shapes.TxN),
        },
        params=Database(
            {
                NP("beta"): Beta(4),
                NP("gamma"): 0.1,
            }
        ),
    )

    values = reqs.evaluate(scope, time_frame, None, None).to_dict(simplify_names=True)

    T = time_frame.days
    N = scope.nodes
    assert values["gpm:a::ipm::gamma"] == 0.1
    assert values["gpm:b::ipm::gamma"] == 0.1
    a_val = _assert_is_single_strata(values, "gpm:a::ipm::beta")
    b_val = _assert_is_single_strata(values, "gpm:b::ipm::beta")
    assert a_val.shape == (T, N)
    assert b_val.shape == (T, N)


def test_param_eval_07(time_frame, scope, rng):
    f_eval_calls = 0
    g_eval_calls = 0

    # Test when a dependent function is not randomized,
    # it and its parent will only be evaluated once.
    class F(ParamFunction):
        requirements = (AD("gamma", float, Shapes.TxN),)

        def evaluate(self):
            nonlocal f_eval_calls
            f_eval_calls += 1
            return 2.0 * self.data("gamma")

    class G(ParamFunction):
        randomized = False

        def evaluate(self):
            nonlocal g_eval_calls
            g_eval_calls += 1
            return np.asarray(3.0 * self.rng.random())

    reqs = ReqTree.of(
        requirements={
            AN("gpm:a::ipm::beta"): AD("beta", float, Shapes.TxN),
            AN("gpm:b::ipm::beta"): AD("beta", float, Shapes.TxN),
        },
        params=Database(
            {
                NP("*::ipm::beta"): F(),
                NP("*::ipm::gamma"): G(),
            }
        ),
    )

    values = reqs.evaluate(scope, time_frame, None, rng).to_dict(simplify_names=True)

    # F and G(randomized=False) evaluated once
    # same random values for both strata
    assert 1 == f_eval_calls
    assert 1 == g_eval_calls
    a_beta_val = _assert_is_single_strata(values, "gpm:a::ipm::beta")
    b_beta_val = _assert_is_single_strata(values, "gpm:b::ipm::beta")
    np.testing.assert_array_equal(a_beta_val, b_beta_val)
    a_gamma_val = _assert_is_single_strata(values, "gpm:a::ipm::gamma")
    b_gamma_val = _assert_is_single_strata(values, "gpm:b::ipm::gamma")
    np.testing.assert_array_equal(a_gamma_val, b_gamma_val)


def test_param_eval_08(time_frame, scope, rng):
    f_eval_calls = 0
    g_eval_calls = 0

    # Test when a dependent function is randomized,
    # it and its parent will be evaluated every time.
    class F(ParamFunction):
        requirements = (AD("gamma", float, Shapes.TxN),)

        def evaluate(self):
            nonlocal f_eval_calls
            f_eval_calls += 1
            return 2.0 * self.data("gamma")

    class G(ParamFunction):
        randomized = True

        def evaluate(self):
            nonlocal g_eval_calls
            g_eval_calls += 1
            return np.asarray(3.0 * self.rng.random())

    reqs = ReqTree.of(
        requirements={
            AN("gpm:a::ipm::beta"): AD("beta", float, Shapes.TxN),
            AN("gpm:b::ipm::beta"): AD("beta", float, Shapes.TxN),
        },
        params=Database(
            {
                NP("*::ipm::beta"): F(),
                NP("*::ipm::gamma"): G(),
            }
        ),
    )

    values = reqs.evaluate(scope, time_frame, None, rng).to_dict(simplify_names=True)

    # F and G(randomized=True) evaluated twice
    # different random values for the strata
    assert 2 == f_eval_calls
    assert 2 == g_eval_calls
    a_beta_val = _assert_is_single_strata(values, "gpm:a::ipm::beta")
    b_beta_val = _assert_is_single_strata(values, "gpm:b::ipm::beta")
    assert not np.array_equal(a_beta_val, b_beta_val)
    a_gamma_val = _assert_is_single_strata(values, "gpm:a::ipm::gamma")
    b_gamma_val = _assert_is_single_strata(values, "gpm:b::ipm::gamma")
    assert not np.array_equal(a_gamma_val, b_gamma_val)


def test_param_eval_09(time_frame, scope):
    # Test that different AttributeDefs can specify different shapes
    # and resolve correctly, even when they use the same value,
    # as long as that value can successfully broadcast to both shapes.
    class F(ParamFunction):
        # F wants a TxN alpha
        requirements = (AD("alpha", float, Shapes.TxN),)

        def evaluate(self):
            # NOTE: it would also be possible to pull T and N from
            # the shape of alpha, however this "hides" the dependency
            # on the `dim` context; if dim is not given alpha will not
            # be shape-adapted (it remains scalar), which causes this logic to fail.
            t = self.time_frame.days
            n = self.scope.nodes
            alpha = self.data("alpha")
            assert (t, n) == alpha.shape
            return np.arange(t * n).reshape((t, n)) * alpha

    class G(ParamFunction):
        # G wants a scalar alpha
        requirements = (AD("alpha", float, Shapes.Scalar),)

        def evaluate(self):
            alpha = self.data("alpha")
            assert () == alpha.shape
            return np.asarray(3.0 * alpha)

    req_a = AN("gpm:a::ipm::beta"), AD("beta", float, Shapes.TxN)
    req_b = AN("gpm:b::ipm::beta"), AD("beta", float, Shapes.TxN)

    reqs = ReqTree.of(
        requirements={
            req_a[0]: req_a[1],
            req_b[0]: req_b[1],
        },
        params=Database(
            {
                NP("gpm:a::ipm::beta"): F(),
                NP("gpm:b::ipm::beta"): G(),
                NP("*::*::alpha"): 0.5,
            }
        ),
    )

    data = reqs.evaluate(scope, time_frame, None, None)

    # alpha should be interpreted differently for F and G:
    beta_a = data.resolve(req_a[0], req_a[1])
    beta_b = data.resolve(req_b[0], req_b[1])

    T = time_frame.days
    N = scope.nodes
    # (gpm:a) F should get a TxN view of alpha,
    # allowing it to produce a varying result
    assert (T, N) == beta_a.shape
    assert np.unique(beta_a).size == T * N

    # (gpm:b) G should get a scalar view of alpha,
    # producing a constant result over TxN
    assert (T, N) == beta_b.shape
    assert np.all(beta_b == beta_b[0])


def test_param_eval_err_01(time_frame, scope):
    # Tests what happens when default values wind up conflicting
    # due to different AttributeDefs managing to resolve to the same
    # AbsoluteName. And test that this can be resolved by providing
    # explicit values.
    class F(ParamFunction):
        requirements = (AD("gamma", float, Shapes.TxN, default_value=0.9),)

        def evaluate(self):
            return 2.0 * self.data("gamma")

    requirements = {
        AN("gpm:a::ipm::beta"): AD("beta", float, Shapes.TxN),
        AN("gpm:b::ipm::beta"): AD("beta", float, Shapes.TxN),
        AN("gpm:a::ipm::gamma"): AD("gamma", float, Shapes.TxN, default_value=0.3),
        AN("gpm:b::ipm::gamma"): AD("gamma", float, Shapes.TxN, default_value=0.7),
    }

    # detect conflicting defaults!
    with pytest.raises(DataAttributeErrorGroup) as exc:
        ReqTree.of(
            requirements=requirements,
            params=Database({NP("*::ipm::beta"): F()}),
        ).evaluate(scope, time_frame, None, None)

    err = "\n".join([str(e).lower() for e in exc.value.exceptions])
    assert "conflicting resolutions for requirement 'gpm:a::ipm::gamma'" in err
    assert "conflicting resolutions for requirement 'gpm:b::ipm::gamma'" in err

    # Now test resolution:
    ReqTree.of(
        requirements=requirements,
        params=Database(
            {
                NP("*::ipm::beta"): F(),
                # Providing these two values prevents the error.
                NP("gpm:a::ipm::gamma"): 0.4,
                NP("gpm:b::ipm::gamma"): 0.5,
            }
        ),
    ).evaluate(scope, time_frame, None, None)


def test_param_eval_err_02(time_frame, scope):
    # Test circular dependency detection.
    class F(ParamFunction):
        requirements = (AD("gamma", float, Shapes.TxN),)

        def evaluate(self):
            return 2.0 * self.data("gamma")

    class G(ParamFunction):
        requirements = (AD("beta", float, Shapes.TxN),)

        def evaluate(self):
            return 3.0 * self.data("beta")

    with pytest.raises(DataAttributeError) as exc:
        ReqTree.of(
            requirements={
                AN("gpm:a::ipm::beta"): AD("beta", float, Shapes.TxN),
                AN("gpm:b::ipm::beta"): AD("beta", float, Shapes.TxN),
            },
            params=Database(
                {
                    NP("*::ipm::beta"): F(),
                    NP("*::ipm::gamma"): G(),
                }
            ),
        ).evaluate(scope, time_frame, None, None)

    err = str(exc.value).lower()
    assert "circular dependency" in err
    assert "gpm:a::ipm::beta" in err


def test_param_eval_err_03(time_frame, scope):
    # Test that independent validation failures are collected together.
    requirements = {
        AN("gpm:a::ipm::beta"): AD("beta", int, Shapes.Scalar),
        AN("gpm:a::ipm::gamma"): AD("gamma", int, Shapes.Scalar),
    }

    with pytest.raises(DataAttributeErrorGroup) as exc:
        ReqTree.of(
            requirements=requirements,
            params=Database(
                {
                    NP("gpm:a::ipm::beta"): 0.5,
                    NP("gpm:a::ipm::gamma"): 0.7,
                }
            ),
        ).evaluate(scope, time_frame, None, None)

    errors = "\n".join(str(e).lower() for e in exc.value.exceptions)
    assert 2 == len(exc.value.exceptions)
    assert "gpm:a::ipm::beta" in errors
    assert "gpm:a::ipm::gamma" in errors
    assert "not a compatible type" in errors


def test_param_eval_err_04(time_frame, scope):
    # Test the targeted error for a class supplied in place of an instance.
    with pytest.raises(DataAttributeErrorGroup) as exc:
        ReqTree.of(
            requirements={
                AN("gpm:a::ipm::beta"): AD("beta", float, Shapes.Scalar),
            },
            params=Database({NP("gpm:a::ipm::beta"): ParamFunction}),
        ).evaluate(scope, time_frame, None, None)

    assert 1 == len(exc.value.exceptions)
    error = str(exc.value.exceptions[0]).lower()
    assert "class instead of an instance" in error


#########################
# evaluate_requirements #
#########################


def _req_tree(*nodes: ReqNode) -> ReqTree:
    return ReqTree(nodes)


def _resolved_as_default(
    definition: AttributeDef,
    name: AbsoluteName,
) -> ReqNode:
    val = definition.default_value
    if val is None:
        raise ValueError("Test setup issue: attribute has no default value.")
    return ReqNode(
        children=(),
        name=name,
        definition=definition,
        resolution=DefaultValue(val),
        value=val,
    )


def _resolved_by_param(
    definition: AttributeDef,
    name: AbsoluteName,
    param_pattern: NamePattern | None,
    param_value: Any,
    cacheable: bool = True,
) -> ReqNode:
    if param_pattern is None:
        param_pattern = name.to_pattern()
    return ReqNode(
        children=(),
        name=name,
        definition=definition,
        resolution=ParameterValue(cacheable=cacheable, pattern=param_pattern),
        value=param_value,
    )


BETA_ATTRIB = AttributeDef("beta", float, Shapes.TxN)
GAMMA_ATTRIB = AttributeDef("gamma", float, Shapes.TxN)
PHI_ATTRIB = AttributeDef("phi", float, Shapes.Scalar, default_value=40.0)


def test_evaluate_reqs_literals_and_defaults(scope, time_frame):
    tree = _req_tree(
        _resolved_by_param(BETA_ATTRIB, AN("gpm:aaa::ipm::beta"), None, 0.4),
        _resolved_as_default(PHI_ATTRIB, AN("gpm:aaa::mm::phi")),
    )

    data = evaluate_requirements(tree, scope, time_frame, None, None)

    np.testing.assert_array_equal(
        data.resolve(AN("gpm:aaa::ipm::beta"), BETA_ATTRIB),
        np.full((3, 2), 0.4),
    )
    np.testing.assert_array_equal(
        data.resolve(AN("gpm:aaa::mm::phi"), PHI_ATTRIB),
        np.array(40.0),
    )


def test_evaluate_reqs_stratified(scope, time_frame, multistrata_ipm):
    value = StratifiedAttributeArray.from_dict(
        {
            "aaa": np.array(0.4),
            "bbb": np.array(0.5),
        }
    )

    tree = _req_tree(
        _resolved_by_param(
            BETA_ATTRIB,
            AN("gpm:aaa::ipm::beta"),
            NP("*::ipm::beta"),
            value,
        ),
        _resolved_by_param(
            BETA_ATTRIB,
            AN("gpm:bbb::ipm::beta"),
            NP("*::ipm::beta"),
            value,
        ),
    )

    data = evaluate_requirements(tree, scope, time_frame, multistrata_ipm, None)

    np.testing.assert_array_equal(
        data.resolve(AN("gpm:aaa::ipm::beta"), BETA_ATTRIB),
        np.full((3, 2), 0.4),
    )
    np.testing.assert_array_equal(
        data.resolve(AN("gpm:bbb::ipm::beta"), BETA_ATTRIB),
        np.full((3, 2), 0.5),
    )


def test_evaluate_reqs_sympy(scope, time_frame):
    t, T, n = simulation_symbols("day", "duration_days", "node_index")
    expression = 0.04 * sympy.sin(8 * sympy.pi * t / T) + 0.34 + 0.02 * n

    tree = _req_tree(
        _resolved_by_param(BETA_ATTRIB, AN("gpm:aaa::ipm::beta"), None, expression),
    )

    data = evaluate_requirements(tree, scope, time_frame, None, None)

    np.testing.assert_allclose(
        cast(NDArray[np.float64], data.resolve(AN("gpm:aaa::ipm::beta"), BETA_ATTRIB)),
        np.stack(
            [
                0.04 * np.sin(8 * np.pi * np.arange(3) / 3) + 0.34,
                0.04 * np.sin(8 * np.pi * np.arange(3) / 3) + 0.36,
            ],
            axis=1,
        ),
    )


def test_evaluate_reqs_caching(scope, time_frame):
    beta_eval_calls = 0

    class Beta(ParamFunctionTimeAndNode):
        requirements = (GAMMA_ATTRIB,)

        def evaluate1(self, day: int, node_index: int) -> float:
            nonlocal beta_eval_calls
            # beta should only be evaluated once (per day/node)
            if day == 0 and node_index == 0:
                beta_eval_calls += 1
            gamma = self.data(GAMMA_ATTRIB)[day, node_index]
            return 4.0 * gamma + 0.1 * math.sin(day) + node_index

    # beta value provided as *::ipm::beta, so is shared
    beta_value = Beta()
    # gamma value provided as *::ipm::gamma (also shared)
    gamma_value = 0.1

    tree = _req_tree(
        ReqNode(
            name=AN("gpm:aaa::ipm::beta"),
            definition=BETA_ATTRIB,
            resolution=ParameterValue(cacheable=True, pattern=NP("*::ipm::beta")),
            value=beta_value,
            children=(
                _resolved_by_param(
                    GAMMA_ATTRIB,
                    AN("gpm:aaa::ipm::gamma"),
                    NP("*::ipm::gamma"),
                    gamma_value,
                ),
            ),
        ),
        ReqNode(
            name=AN("gpm:bbb::ipm::beta"),
            definition=BETA_ATTRIB,
            value=beta_value,
            resolution=ParameterValue(cacheable=True, pattern=NP("*::ipm::beta")),
            children=(
                _resolved_by_param(
                    GAMMA_ATTRIB,
                    AN("gpm:bbb::ipm::gamma"),
                    NP("*::ipm::gamma"),
                    gamma_value,
                ),
            ),
        ),
    )

    data = evaluate_requirements(tree, scope, time_frame, None, None)

    assert beta_eval_calls == 1

    raw_beta_aaa = data.get_raw(AN("gpm:aaa::ipm::beta"))
    raw_beta_bbb = data.get_raw(AN("gpm:bbb::ipm::beta"))
    assert raw_beta_aaa is raw_beta_bbb

    np.testing.assert_allclose(
        cast(NDArray[np.float64], data.resolve(AN("gpm:aaa::ipm::beta"), BETA_ATTRIB)),
        np.array(
            [
                [0.4, 1.4],
                [0.4 + 0.1 * math.sin(1), 1.4 + 0.1 * math.sin(1)],
                [0.4 + 0.1 * math.sin(2), 1.4 + 0.1 * math.sin(2)],
            ]
        ),
    )


def test_evaluate_reqs_missing_values(scope, time_frame):
    tree = _req_tree(
        ReqNode((), AN("gpm:aaa::ipm::beta"), BETA_ATTRIB, MissingValue(), None),
    )

    with pytest.raises(DataAttributeError, match="there are missing values"):
        evaluate_requirements(tree, scope, time_frame, None, None)


def test_evaluate_reqs_collects_errors(scope, time_frame):
    tree = _req_tree(
        _resolved_by_param(
            AttributeDef("beta", int, Shapes.Scalar),  # <-- attribute is int
            AN("gpm:aaa::ipm::beta"),
            None,
            0.5,  # <-- but value is a float (can't convert without loss)
        ),
        _resolved_by_param(
            AttributeDef("gamma", int, Shapes.Scalar),  # <-- attribute is scalar
            AN("gpm:aaa::ipm::gamma"),
            None,
            [0.5, 0.6, 0.7],  # <-- but value is an array
        ),
    )

    with pytest.raises(DataAttributeErrorGroup) as exc:
        evaluate_requirements(tree, scope, time_frame, None, None)

    assert len(exc.value.exceptions) == 2
    errors = "\n".join(str(error).lower() for error in exc.value.exceptions)
    assert "gpm:aaa::ipm::beta" in errors
    assert "gpm:aaa::ipm::gamma" in errors


################
# DataResolver #
################


@pytest.fixture(scope="module")
def data_resolver():
    return DataResolver(
        dim=MagicMock(Dimensions),
        values={
            AbsoluteName.parse("gpm:all::ipm::beta"): np.array(0.4),
            AbsoluteName.parse("gpm:all::mm::population"): np.array([100, 200, 300]),
            AbsoluteName.parse("gpm:one::ipm::xi"): np.array(0.5),
            AbsoluteName.parse("gpm:two::ipm::xi"): np.array(0.6),
        },
    )


def test_data_resolver_has(data_resolver):
    assert data_resolver.has(AbsoluteName.parse("gpm:all::ipm::beta"))
    assert data_resolver.has(AbsoluteName.parse("gpm:all::mm::population"))
    assert not data_resolver.has(AbsoluteName.parse("gpm:all::ipm::gamma"))
    assert not data_resolver.has(AbsoluteName.parse("gpm:all::mm::beta"))
    assert not data_resolver.has(AbsoluteName.parse("gpm:one::ipm::beta"))


def test_data_resolver_get_raw(data_resolver):
    def test(name, expected):
        actual = data_resolver.get_raw(name)
        np.testing.assert_array_equal(actual, expected)

    beta = np.array(0.4)
    test(AbsoluteName.parse("gpm:all::ipm::beta"), beta)
    test("gpm:all::ipm::beta", beta)
    test("ipm::beta", beta)
    test("beta", beta)

    pop = np.array([100, 200, 300])
    test(AbsoluteName.parse("gpm:all::mm::population"), pop)
    test("gpm:all::mm::population", pop)
    test("mm::population", pop)
    test("population", pop)

    with pytest.raises(ValueError, match="does not match any values"):
        data_resolver.get_raw("gpm:all::ipm::gamma")
    with pytest.raises(ValueError, match="does not match any values"):
        data_resolver.get_raw("gpm:all::mm::beta")
    with pytest.raises(ValueError, match="does not match any values"):
        data_resolver.get_raw("gpm:one::ipm::beta")
    with pytest.raises(ValueError, match="matches more than one value"):
        data_resolver.get_raw("xi")


def test_data_resolver_resolve_adapts_and_caches():
    dim = Dimensions.of(T=2, N=3)
    beta_name = AbsoluteName.parse("gpm:all::ipm::beta")

    resolver_a = DataResolver(dim, {beta_name: np.array(7, dtype=np.int64)})

    # resolving the same an attribute with the shape/type should yield the same object
    # certain shape adaptations are allowed (e.g., scalar -> TxN)
    # type adaptations which do not lose information are allowed (e.g., int -> float)
    first = resolver_a.resolve(beta_name, AttributeDef("beta", float, Shapes.TxN))
    second = resolver_a.resolve(beta_name, AttributeDef("beta", float, Shapes.TxN))
    assert first is second
    assert first.dtype == np.float64
    np.testing.assert_array_equal(first, np.full((2, 3), 7.0))

    with pytest.raises(DataAttributeError, match="No value"):
        resolver_a.resolve(
            AbsoluteName.parse("gpm:all::ipm::gamma"),
            AttributeDef("gamma", float, Shapes.Scalar),
        )

    # type adaptations which might lose information are not allowed (e.g., float -> int)
    resolver_b = DataResolver(dim, {beta_name: np.array(7.5, dtype=np.float64)})
    with pytest.raises(DataAttributeError, match="Not a compatible type"):
        resolver_b.resolve(beta_name, AttributeDef("beta", int, Shapes.Scalar))

    # shape adaptations which are not compatible are not allowed
    # (here, value is length 2, eval as shape N, but N=3)
    resolver_c = DataResolver(dim, {beta_name: np.array([1, 2], dtype=np.int64)})
    with pytest.raises(DataAttributeError, match="Not a compatible shape"):
        resolver_c.resolve(beta_name, AttributeDef("beta", int, Shapes.N))


def test_data_resolver_resolves_stratified_values():
    value = StratifiedAttributeArray(
        np.array(
            [
                [[1, 2], [3, 4], [5, 6]],
                [[7, 8], [9, 10], [11, 12]],
            ],
            dtype=np.int64,
        ),
        strata=["aaa", "bbb"],
    )

    # We presume that `value` resolves for both strata,
    # as it would for a parameter specified as "*::ipm::beta".
    resolver = DataResolver(
        dim=Dimensions.of(T=3, N=2),
        values={
            AN("gpm:aaa::ipm::beta"): value,
            AN("gpm:bbb::ipm::beta"): value,
        },
    )

    definition = AD("beta", float, Shapes.TxN)
    beta_aaa = resolver.resolve(AN("gpm:aaa::ipm::beta"), definition)
    beta_bbb = resolver.resolve(AN("gpm:bbb::ipm::beta"), definition)

    assert beta_aaa.dtype == np.float64
    assert beta_bbb.dtype == np.float64
    np.testing.assert_array_equal(beta_aaa, [[1, 2], [3, 4], [5, 6]])
    np.testing.assert_array_equal(beta_bbb, [[7, 8], [9, 10], [11, 12]])


def test_eval_copies_numpy_parameter_values(time_frame, scope):
    source = np.array([3, 5], dtype=np.int64)
    name = AN("gpm:all::ipm::beta")
    resolver = ReqTree.of(
        requirements={name: AttributeDef("beta", int, Shapes.N)},
        params=Database({NP("gpm:all::ipm::beta"): source}),
    ).evaluate(scope, time_frame, None, None)

    source[0] = 99

    val = resolver.get_raw(name)
    assert not isinstance(val, StratifiedAttributeArray)
    np.testing.assert_array_equal(val, np.array([3, 5]))


def test_data_resolver_resolve_txn_series():
    dim = Dimensions.of(T=2, N=2)
    beta = AbsoluteName.parse("gpm:all::ipm::beta")
    gamma = AbsoluteName.parse("gpm:all::ipm::gamma")
    resolver = DataResolver(
        dim,
        {
            beta: np.array(
                [
                    [1, 2],  # day 0
                    [3, 4],  # day 1
                ],
                dtype=np.int64,
            ),
            gamma: np.array(0.5, dtype=np.float64),
        },
    )

    values = list(
        resolver.resolve_txn_series(
            [
                (beta, AttributeDef("beta", float, Shapes.TxN)),
                (gamma, AttributeDef("gamma", float, Shapes.Scalar)),
            ],
            tau_steps=2,
        )
    )

    assert values == [
        [1.0, 0.5],  # day 0, step 0, node 0
        [2.0, 0.5],  # day 0, step 0, node 1
        [1.0, 0.5],  # day 0, step 1, node 0
        [2.0, 0.5],  # day 0, step 1, node 1
        [3.0, 0.5],  # day 1, step 0, node 0
        [4.0, 0.5],  # day 1, step 0, node 1
        [3.0, 0.5],  # day 1, step 1, node 0
        [4.0, 0.5],  # day 1, step 1, node 1
    ]

    with pytest.raises(DataAttributeError, match="broadcast to TxN"):
        list(
            resolver.resolve_txn_series(
                [(beta, AttributeDef("beta", float, Shapes.NxN))],
                tau_steps=1,
            )
        )
