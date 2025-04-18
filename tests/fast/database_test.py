# ruff: noqa: PT009,PT027
import math
import unittest
from functools import wraps
from typing import Callable, ParamSpec, TypeVar
from unittest.mock import MagicMock

import numpy as np
import pytest
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
from epymorph.data_shape import Dimensions, Shapes
from epymorph.database import (
    Database,
    DatabaseWithFallback,
    DatabaseWithStrataFallback,
    DataResolver,
    Match,
    ReqTree,
)
from epymorph.error import DataAttributeError, DataAttributeErrorGroup
from epymorph.geography.scope import GeoScope
from epymorph.params import ParamFunction, ParamFunctionTimeAndNode
from epymorph.time import TimeFrame


class ModuleNamespaceTest(unittest.TestCase):
    def test_post_init_empty(self):
        with self.assertRaises(ValueError):
            ModuleNamespace("", "module")
        with self.assertRaises(ValueError):
            ModuleNamespace("strata", "")

    def test_post_init_wildcards(self):
        with self.assertRaises(ValueError):
            ModuleNamespace("*", "module")
        with self.assertRaises(ValueError):
            ModuleNamespace("strata", "*")

    def test_post_init_delimeters(self):
        with self.assertRaises(ValueError):
            ModuleNamespace("::", "module")
        with self.assertRaises(ValueError):
            ModuleNamespace("strata", "::")

    def test_parse_valid_string(self):
        ns = ModuleNamespace.parse("strata::module")
        self.assertEqual(ns.strata, "strata")
        self.assertEqual(ns.module, "module")

    def test_parse_invalid_string(self):
        with self.assertRaises(ValueError):
            ModuleNamespace.parse("invalid_string")

    def test_parse_with_more_parts(self):
        with self.assertRaises(ValueError):
            ModuleNamespace.parse("too::many::parts")

    def test_str_representation(self):
        ns = ModuleNamespace("strata", "module")
        self.assertEqual(str(ns), "strata::module")

    def test_to_absolute(self):
        ns = ModuleNamespace("strata", "module")
        pattern = ns.to_absolute("id")
        self.assertIsInstance(pattern, AbsoluteName)
        self.assertEqual(pattern.strata, "strata")
        self.assertEqual(pattern.module, "module")
        self.assertEqual(pattern.id, "id")


class AbsoluteNameTest(unittest.TestCase):
    def test_post_init_empty(self):
        with self.assertRaises(ValueError):
            AbsoluteName("", "module", "id")
        with self.assertRaises(ValueError):
            AbsoluteName("strata", "", "id")
        with self.assertRaises(ValueError):
            AbsoluteName("strata", "module", "")

    def test_post_init_wildcards(self):
        with self.assertRaises(ValueError):
            AbsoluteName("*", "module", "id")
        with self.assertRaises(ValueError):
            AbsoluteName("strata", "*", "id")
        with self.assertRaises(ValueError):
            AbsoluteName("strata", "module", "*")

    def test_post_init_delimeters(self):
        with self.assertRaises(ValueError):
            AbsoluteName("::", "module", "id")
        with self.assertRaises(ValueError):
            AbsoluteName("strata", "::", "id")
        with self.assertRaises(ValueError):
            AbsoluteName("strata", "module", "::")

    def test_parse_valid_string(self):
        name = AbsoluteName.parse("strata::module::id")
        self.assertEqual(name.strata, "strata")
        self.assertEqual(name.module, "module")
        self.assertEqual(name.id, "id")

    def test_parse_invalid_string(self):
        with self.assertRaises(ValueError):
            AbsoluteName.parse("invalid_string")

    def test_str_representation(self):
        name = AbsoluteName("strata", "module", "id")
        self.assertEqual(str(name), "strata::module::id")

    def test_in_strata(self):
        name = AbsoluteName("strata", "module", "id")
        new_name = name.in_strata("new_strata")
        self.assertIsInstance(new_name, AbsoluteName)
        self.assertEqual(new_name.strata, "new_strata")
        self.assertEqual(new_name.module, "module")
        self.assertEqual(new_name.id, "id")

    def test_to_namespace(self):
        name = AbsoluteName("strata", "module", "id")
        namespace = name.to_namespace()
        self.assertIsInstance(namespace, ModuleNamespace)
        self.assertEqual(namespace.strata, "strata")
        self.assertEqual(namespace.module, "module")

    def test_to_pattern(self):
        name = AbsoluteName("strata", "module", "id")
        pattern = name.to_pattern()
        self.assertIsInstance(pattern, NamePattern)
        self.assertEqual(pattern.strata, "strata")
        self.assertEqual(pattern.module, "module")
        self.assertEqual(pattern.id, "id")


class ModuleNameTest(unittest.TestCase):
    def test_post_init_empty(self):
        with self.assertRaises(ValueError):
            ModuleName("module", "")
        with self.assertRaises(ValueError):
            ModuleName("", "id")

    def test_post_init_wildcards(self):
        with self.assertRaises(ValueError):
            ModuleName("*", "id")
        with self.assertRaises(ValueError):
            ModuleName("module", "*")

    def test_post_init_delimeters(self):
        with self.assertRaises(ValueError):
            ModuleName("::", "id")
        with self.assertRaises(ValueError):
            ModuleName("module", "::")

    def test_empty(self):
        with self.assertRaises(ValueError):
            ModuleName.parse("")

    def test_parse_valid_string(self):
        name = ModuleName.parse("module::id")
        self.assertEqual(name.module, "module")
        self.assertEqual(name.id, "id")

    def test_parse_invalid_string(self):
        with self.assertRaises(ValueError):
            ModuleName.parse("invalid_string")

    def test_parse_with_more_parts(self):
        with self.assertRaises(ValueError):
            ModuleName.parse("too::many::parts")

    def test_str_representation(self):
        name = ModuleName("module", "id")
        self.assertEqual(str(name), "module::id")

    def test_to_absolute(self):
        name = ModuleName("module", "id")
        absolute_name = name.to_absolute("strata")
        self.assertIsInstance(absolute_name, AbsoluteName)
        self.assertEqual(absolute_name.strata, "strata")
        self.assertEqual(absolute_name.module, "module")
        self.assertEqual(absolute_name.id, "id")


class AttributeNameTest(unittest.TestCase):
    def test_post_init_empty(self):
        with self.assertRaises(ValueError):
            AttributeName("")

    def test_post_init_wildcard_id(self):
        with self.assertRaises(ValueError):
            AttributeName("*")

    def test_post_init_delimiters(self):
        with self.assertRaises(ValueError):
            AttributeName("invalid::id")

    def test_str_representation(self):
        attr_name = AttributeName("id")
        self.assertEqual(str(attr_name), "id")


class NamePatternTest(unittest.TestCase):
    def test_post_init_empty(self):
        with self.assertRaises(ValueError):
            NamePattern("", "module", "id")
        with self.assertRaises(ValueError):
            NamePattern("strata", "", "id")
        with self.assertRaises(ValueError):
            NamePattern("strata", "module", "")

    def test_post_init_delimeters(self):
        with self.assertRaises(ValueError):
            NamePattern("::", "module", "id")
        with self.assertRaises(ValueError):
            NamePattern("strata", "::", "id")
        with self.assertRaises(ValueError):
            NamePattern("strata", "module", "::")

    def test_parse_one_part(self):
        pattern = NamePattern.parse("id")
        self.assertEqual(pattern.strata, "*")
        self.assertEqual(pattern.module, "*")
        self.assertEqual(pattern.id, "id")

    def test_parse_two_parts(self):
        pattern = NamePattern.parse("module::id")
        self.assertEqual(pattern.strata, "*")
        self.assertEqual(pattern.module, "module")
        self.assertEqual(pattern.id, "id")

    def test_parse_three_parts(self):
        pattern = NamePattern.parse("strata::module::id")
        self.assertEqual(pattern.strata, "strata")
        self.assertEqual(pattern.module, "module")
        self.assertEqual(pattern.id, "id")

    def test_parse_invalid_string(self):
        with self.assertRaises(ValueError):
            NamePattern.parse("too::many::parts::here")

    def test_match_absolute_name(self):
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
            self.assertTrue(pattern.match(absolute_name))

    def test_no_match_absolute_name(self):
        pattern = NamePattern("strata", "module", "*")
        absolute_name = AbsoluteName("other_strata", "module", "id")
        self.assertFalse(pattern.match(absolute_name))

        pattern = NamePattern("strata", "*", "id")
        absolute_name = AbsoluteName("other_strata", "module", "id")
        self.assertFalse(pattern.match(absolute_name))

        pattern = NamePattern("*", "module", "id")
        absolute_name = AbsoluteName("strata", "other_module", "id")
        self.assertFalse(pattern.match(absolute_name))

    def test_match_name_pattern(self):
        pattern1 = NamePattern("strata", "*", "id")
        pattern2 = NamePattern("strata", "module", "id")
        self.assertTrue(pattern1.match(pattern2))

    def test_no_match_name_pattern(self):
        pattern1 = NamePattern("strata", "module", "id")
        pattern2 = NamePattern("*", "other_module", "id")
        self.assertFalse(pattern1.match(pattern2))

    def test_str_representation(self):
        pattern = NamePattern("strata", "module", "id")
        self.assertEqual(str(pattern), "strata::module::id")


class ModuleNamePatternTest(unittest.TestCase):
    def test_post_init_empty(self):
        with self.assertRaises(ValueError):
            ModuleNamePattern("", "id")
        with self.assertRaises(ValueError):
            ModuleNamePattern("module", "")

    def test_post_init_delimeters(self):
        with self.assertRaises(ValueError):
            ModuleNamePattern("::", "id")
        with self.assertRaises(ValueError):
            ModuleNamePattern("module", "::")

    def test_parse_one_part(self):
        pattern = ModuleNamePattern.parse("id")
        self.assertEqual(pattern.module, "*")
        self.assertEqual(pattern.id, "id")

    def test_parse_two_parts(self):
        pattern = ModuleNamePattern.parse("module::id")
        self.assertEqual(pattern.module, "module")
        self.assertEqual(pattern.id, "id")

    def test_parse_invalid_string(self):
        with self.assertRaises(ValueError):
            ModuleNamePattern.parse("too::many::parts::here")

    def test_parse_empty(self):
        with self.assertRaises(ValueError):
            ModuleNamePattern.parse("")

    def test_to_absolute(self):
        pattern = ModuleNamePattern("module", "id")
        absolute_pattern = pattern.to_absolute("strata")
        self.assertIsInstance(absolute_pattern, NamePattern)
        self.assertEqual(absolute_pattern.strata, "strata")
        self.assertEqual(absolute_pattern.module, "module")
        self.assertEqual(absolute_pattern.id, "id")

    def test_str_representation(self):
        pattern = ModuleNamePattern("module", "id")
        self.assertEqual(str(pattern), "module::id")


T = TypeVar("T")


class _DatabaseTestCase(unittest.TestCase):
    def assert_match(self, expected: T, test: Match[T] | None):
        if test is None:
            self.fail("Expected a match, but it was None.")
        self.assertEqual(expected, test.value)


class DatabaseTest(_DatabaseTestCase):
    def test_basic_usage(self):
        db = Database[int](
            {
                NamePattern("gpm:1", "ipm", "beta"): 1,
                NamePattern("*", "ipm", "delta"): 2,
                NamePattern("*", "*", "gamma"): 3,
                NamePattern("gpm:2", "ipm", "beta"): 4,
            }
        )

        self.assert_match(1, db.query(AbsoluteName("gpm:1", "ipm", "beta")))
        self.assert_match(1, db.query("gpm:1::ipm::beta"))
        self.assert_match(4, db.query("gpm:2::ipm::beta"))
        self.assertIsNone(db.query("gpm:3::ipm::beta"))

        self.assert_match(2, db.query("gpm:1::ipm::delta"))
        self.assert_match(2, db.query("gpm:2::ipm::delta"))
        self.assert_match(2, db.query("gpm:9::ipm::delta"))
        self.assertIsNone(db.query("gpm:1::mm::delta"))

        self.assert_match(3, db.query("gpm:1::ipm::gamma"))
        self.assert_match(3, db.query("gpm:2::ipm::gamma"))
        self.assert_match(3, db.query("gpm:1::mm::gamma"))
        self.assert_match(3, db.query("gpm:1::init::gamma"))

    def test_ambiguous_values(self):
        with self.assertRaises(ValueError) as e:
            Database[int](
                {
                    NamePattern("*", "*", "beta"): 1,
                    NamePattern("gpm:1", "*", "beta"): 2,
                    NamePattern("*", "ipm", "beta"): 3,
                }
            )
        self.assertIn("ambiguous", str(e.exception))


class DatabaseWithFallbackTest(_DatabaseTestCase):
    def test_basic_usage(self):
        fallback = Database[int](
            {
                NamePattern("gpm:1", "ipm", "beta"): 1,
                NamePattern("*", "ipm", "delta"): 2,
                NamePattern("*", "ipm", "gamma"): 3,
                NamePattern("gpm:2", "ipm", "beta"): 4,
                NamePattern("gpm:3", "init", "alpha"): 6,
            }
        )

        db = DatabaseWithFallback(
            {
                NamePattern("gpm:1", "ipm", "beta"): 11,
                NamePattern("gpm:2", "*", "beta"): 44,
                NamePattern("gpm:3", "*", "*"): 55,
            },
            fallback,
        )

        self.assert_match(11, db.query("gpm:1::ipm::beta"))
        self.assert_match(44, db.query("gpm:2::ipm::beta"))

        self.assert_match(55, db.query("gpm:3::ipm::beta"))
        self.assert_match(55, db.query("gpm:3::init::alpha"))
        self.assert_match(55, db.query("gpm:3::foo::bar"))

        self.assert_match(2, db.query("gpm:1::ipm::delta"))
        self.assert_match(2, db.query("gpm:2::ipm::delta"))
        self.assert_match(55, db.query("gpm:3::ipm::delta"))

        self.assertIsNone(db.query("gpm:1::init::alpha"))


class DatabaseWithStrataFallbackTest(_DatabaseTestCase):
    def test_basic_usage(self):
        strata1 = Database[int](
            {
                NamePattern("gpm:1", "ipm", "beta"): 1,
                NamePattern("gpm:1", "ipm", "delta"): 2,
            }
        )

        strata2 = Database[int](
            {
                NamePattern("gpm:2", "ipm", "beta"): 3,
                NamePattern("gpm:2", "ipm", "delta"): 4,
            }
        )

        strata3 = Database[int](
            {
                NamePattern("gpm:3", "ipm", "beta"): 5,
                NamePattern("gpm:3", "ipm", "delta"): 6,
            }
        )

        db = DatabaseWithStrataFallback(
            {
                NamePattern("gpm:1", "ipm", "beta"): 11,
                NamePattern("gpm:2", "ipm", "beta"): 33,
                NamePattern("gpm:3", "*", "*"): 55,
            },
            {
                "gpm:1": strata1,
                "gpm:2": strata2,
                "gpm:3": strata3,
            },
        )

        self.assert_match(11, db.query("gpm:1::ipm::beta"))
        self.assert_match(33, db.query("gpm:2::ipm::beta"))
        self.assert_match(55, db.query("gpm:3::ipm::beta"))

        self.assert_match(2, db.query("gpm:1::ipm::delta"))
        self.assert_match(4, db.query("gpm:2::ipm::delta"))
        self.assert_match(55, db.query("gpm:3::ipm::delta"))

        self.assertIsNone(db.query("gpm1::init::population"))


AD = AttributeDef
AN = AbsoluteName.parse
NP = NamePattern.parse

ParamT = ParamSpec("ParamT")
ReturnT = TypeVar("ReturnT")
FunctionType = Callable[ParamT, ReturnT]


def count_calls(f):
    # NOTE: this is a count per class-method, not instance-method
    # That's preferable here because epymorph clones the instances anyway
    # so the func instance we pass in isn't exactly the one that's used.
    n = 0

    @wraps(f)
    def g(*args, **kwargs):
        nonlocal n
        n += 1
        return f(*args, **kwargs)

    def get_calls():
        nonlocal n
        return n

    setattr(g, "__calls__", get_calls)
    return g


def calls(f):
    if (calls := getattr(f, "__calls__", None)) is not None:
        return calls()
    raise ValueError("This function was not wrapped with @count_calls")


class ParamEvalTest(unittest.TestCase):
    time_frame = TimeFrame.of("2020-01-01", 3)
    scope = MagicMock(spec=GeoScope, nodes=2)

    @property
    def rng(self):
        return np.random.default_rng(1)

    def _to_txn(self, value: float) -> NDArray[np.float64]:
        return np.broadcast_to(value, shape=(self.time_frame.days, self.scope.nodes))

    def test_eval_01(self):
        # Test that a function can be used for multiple strata
        # but will only be evaluated once if all of its dependencies
        # do not vary by strata.
        class F(ParamFunction):
            requirements = (AD("gamma", float, Shapes.TxN),)

            @count_calls
            def evaluate(self):
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

        values = reqs.evaluate(self.scope, self.time_frame, None, None).to_dict(
            simplify_names=True
        )

        # F evaluated once; beta is 1.4 for both strata
        self.assertEqual(1, calls(F.evaluate))
        exp = self._to_txn(1.4)
        np.testing.assert_array_equal(exp, values["gpm:a::ipm::beta"])
        np.testing.assert_array_equal(exp, values["gpm:b::ipm::beta"])

    def test_eval_02(self):
        # Test that a function declared "randomized" will be evaluated
        # every time it's referenced, even if it otherwise wouldn't need to be.
        class F(ParamFunction):
            requirements = (AD("gamma", float, Shapes.TxN),)
            randomized = True

            @count_calls
            def evaluate(self):
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

        values = reqs.evaluate(self.scope, self.time_frame, None, None).to_dict(
            simplify_names=True
        )

        # F evaluated twice, even though it produces the same value each time
        # beta is 1.4 for both strata
        self.assertEqual(2, calls(F.evaluate))
        exp = self._to_txn(1.4)
        np.testing.assert_array_equal(exp, values["gpm:a::ipm::beta"])
        np.testing.assert_array_equal(exp, values["gpm:b::ipm::beta"])

    def test_eval_03(self):
        # Test a single function resolving to different values
        # due to dependencies that differ between strata.
        class F(ParamFunction):
            requirements = (AD("gamma", float, Shapes.TxN),)

            @count_calls
            def evaluate(self):
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

        values = reqs.evaluate(self.scope, self.time_frame, None, None).to_dict(
            simplify_names=True
        )

        # F evaluated twice
        # beta is 0.6 for strata a
        # and 1.4 for strata b
        self.assertEqual(2, calls(F.evaluate))
        np.testing.assert_array_equal(self._to_txn(0.6), values["gpm:a::ipm::beta"])
        np.testing.assert_array_equal(self._to_txn(1.4), values["gpm:b::ipm::beta"])

    def test_eval_04(self):
        # Test a single shared random value when a single instance is used.
        class F(ParamFunction):
            requirements = (AD("gamma", float, Shapes.TxN),)

            @count_calls
            def evaluate(self):
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

        values = reqs.evaluate(self.scope, self.time_frame, None, self.rng).to_dict(
            simplify_names=True
        )

        # F evaluated once
        # beta is random, but the same value is shared between strata
        self.assertEqual(1, calls(F.evaluate))
        np.testing.assert_array_equal(
            values["gpm:a::ipm::beta"],
            values["gpm:b::ipm::beta"],
        )

    def test_eval_05(self):
        # Test unique random values by virtue of providing different instances.
        class F(ParamFunction):
            requirements = (AD("gamma", float, Shapes.TxN),)

            @count_calls
            def evaluate(self):
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

        values = reqs.evaluate(self.scope, self.time_frame, None, self.rng).to_dict(
            simplify_names=True
        )

        # Fs evaluated once each
        # beta is two unique random numbers
        self.assertEqual(2, calls(F.evaluate))
        self.assertFalse(
            np.array_equal(
                values["gpm:a::ipm::beta"],
                values["gpm:b::ipm::beta"],
            )
        )

    def test_eval_06(self):
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

        values = reqs.evaluate(self.scope, self.time_frame, None, None).to_dict(
            simplify_names=True
        )

        T = self.time_frame.days
        N = self.scope.nodes
        self.assertEqual(values["gpm:a::ipm::gamma"], 0.1)
        self.assertEqual(values["gpm:b::ipm::gamma"], 0.1)
        self.assertEqual(values["gpm:a::ipm::beta"].shape, (T, N))
        self.assertEqual(values["gpm:b::ipm::beta"].shape, (T, N))

    def test_eval_07(self):
        # Test when a dependent function is not randomized,
        # it and its parent will only be evaluated once.
        class F(ParamFunction):
            requirements = (AD("gamma", float, Shapes.TxN),)

            @count_calls
            def evaluate(self):
                return 2.0 * self.data("gamma")

        class G(ParamFunction):
            randomized = False

            @count_calls
            def evaluate(self):
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

        values = reqs.evaluate(self.scope, self.time_frame, None, self.rng).to_dict(
            simplify_names=True
        )

        # F and G(randomized=False) evaluated once
        # same random values for both strata
        self.assertEqual(1, calls(F.evaluate))
        self.assertEqual(1, calls(G.evaluate))
        np.testing.assert_array_equal(
            values["gpm:a::ipm::beta"],
            values["gpm:b::ipm::beta"],
        )
        np.testing.assert_array_equal(
            values["gpm:a::ipm::gamma"],
            values["gpm:b::ipm::gamma"],
        )

    def test_eval_08(self):
        # Test when a dependent function is randomized,
        # it and its parent will be evaluated every time.
        class F(ParamFunction):
            requirements = (AD("gamma", float, Shapes.TxN),)

            @count_calls
            def evaluate(self):
                return 2.0 * self.data("gamma")

        class G(ParamFunction):
            randomized = True

            @count_calls
            def evaluate(self):
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

        values = reqs.evaluate(self.scope, self.time_frame, None, self.rng).to_dict(
            simplify_names=True
        )

        # F and G(randomized=True) evaluated twice
        # different random values for the strata
        self.assertEqual(2, calls(F.evaluate))
        self.assertEqual(2, calls(G.evaluate))
        self.assertFalse(
            np.array_equal(
                values["gpm:a::ipm::beta"],
                values["gpm:b::ipm::beta"],
            )
        )
        self.assertFalse(
            np.array_equal(
                values["gpm:a::ipm::gamma"],
                values["gpm:b::ipm::gamma"],
            )
        )

    def test_eval_09(self):
        # Test that different AttributeDefs can specify different shapes
        # and resolve correctly, even when they use the same value,
        # as long as that value can successfully broadcast to both shapes.
        assert_equal = self.assertEqual

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
                assert_equal((t, n), alpha.shape)
                return np.arange(t * n).reshape((t, n)) * alpha

        class G(ParamFunction):
            # G wants a scalar alpha
            requirements = (AD("alpha", float, Shapes.Scalar),)

            def evaluate(self):
                alpha = self.data("alpha")
                assert_equal((), alpha.shape)
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

        data = reqs.evaluate(self.scope, self.time_frame, None, None)

        # alpha should be interpreted differently for F and G:
        beta_a = data.resolve(req_a[0], req_a[1])
        beta_b = data.resolve(req_b[0], req_b[1])

        T = self.time_frame.days
        N = self.scope.nodes
        # (gpm:a) F should get a TxN view of alpha,
        # allowing it to produce a varying result
        self.assertEqual((T, N), beta_a.shape)
        self.assertTrue(np.unique(beta_a).size == T * N)

        # (gpm:b) G should get a scalar view of alpha,
        # producing a constant result over TxN
        self.assertEqual((T, N), beta_b.shape)
        self.assertTrue(np.all(beta_b == beta_b[0]))

    def test_eval_err_01(self):
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
        with self.assertRaises(DataAttributeErrorGroup) as ctx:
            ReqTree.of(
                requirements=requirements,
                params=Database({NP("*::ipm::beta"): F()}),
            ).evaluate(self.scope, self.time_frame, None, None)

        err = "\n".join([str(e).lower() for e in ctx.exception.exceptions])
        self.assertIn(
            "conflicting resolutions for requirement 'gpm:a::ipm::gamma'",
            err,
        )
        self.assertIn(
            "conflicting resolutions for requirement 'gpm:b::ipm::gamma'",
            err,
        )

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
        ).evaluate(self.scope, self.time_frame, None, None)

    def test_eval_err_02(self):
        # Test circular dependency detection.
        class F(ParamFunction):
            requirements = (AD("gamma", float, Shapes.TxN),)

            def evaluate(self):
                return 2.0 * self.data("gamma")

        class G(ParamFunction):
            requirements = (AD("beta", float, Shapes.TxN),)

            def evaluate(self):
                return 3.0 * self.data("beta")

        with self.assertRaises(DataAttributeError) as ctx:
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
            ).evaluate(self.scope, self.time_frame, None, None)

        err = str(ctx.exception).lower()
        self.assertIn("circular dependency", err)
        self.assertIn("gpm:a::ipm::beta", err)


################
# DataResolver #
################


@pytest.fixture
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
