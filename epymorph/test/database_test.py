# pylint: disable=missing-docstring
import unittest
from typing import TypeVar

from epymorph.database import (AbsoluteName, AttributeName, Database,
                               DatabaseWithFallback,
                               DatabaseWithStrataFallback, Match, ModuleName,
                               ModuleNamePattern, ModuleNamespace, NamePattern)


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

    def test_parse_with_defaults_one_part(self):
        name = AbsoluteName.parse_with_defaults(
            "id", "default_strata", "default_module")
        self.assertEqual(name.strata, "default_strata")
        self.assertEqual(name.module, "default_module")
        self.assertEqual(name.id, "id")

    def test_parse_with_defaults_two_parts(self):
        name = AbsoluteName.parse_with_defaults(
            "module::id", "default_strata", "default_module")
        self.assertEqual(name.strata, "default_strata")
        self.assertEqual(name.module, "module")
        self.assertEqual(name.id, "id")

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


T = TypeVar('T')


class _DatabaseTestCase(unittest.TestCase):

    def assert_match(self, expected: T, test: Match[T] | None):
        if test is None:
            self.fail("Expected a match, but it was None.")
        self.assertEqual(expected, test.value)


class DatabaseTest(_DatabaseTestCase):

    def test_basic_usage(self):
        db = Database[int]({
            NamePattern("gpm:1", "ipm", "beta"): 1,
            NamePattern("*", "ipm", "delta"): 2,
            NamePattern("*", "*", "gamma"): 3,
            NamePattern("gpm:2", "ipm", "beta"): 4,
        })

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

    def test_update(self):
        db = Database[int]({
            NamePattern("gpm:1", "ipm", "beta"): 1,
            NamePattern("*", "ipm", "delta"): 2,
            NamePattern("*", "*", "gamma"): 3,
            NamePattern("gpm:2", "ipm", "beta"): 4,
        })

        self.assert_match(1, db.query(AbsoluteName("gpm:1", "ipm", "beta")))
        self.assert_match(1, db.query("gpm:1::ipm::beta"))
        self.assert_match(4, db.query("gpm:2::ipm::beta"))
        self.assertIsNone(db.query("gpm:3::ipm::beta"))

        db.update(NamePattern("gpm:1", "ipm", "beta"), 101)
        db.update(NamePattern("*", "*", "gamma"), 303)

        self.assert_match(101, db.query("gpm:1::ipm::beta"))
        self.assert_match(303, db.query("gpm:1::ipm::gamma"))
        self.assert_match(303, db.query("gpm:2::ipm::gamma"))
        self.assert_match(4, db.query("gpm:2::ipm::beta"))
        self.assertIsNone(db.query("gpm:3::ipm::beta"))

        db.update(NamePattern("gpm:3", "*", "beta"), 505)

        self.assert_match(505, db.query("gpm:3::ipm::beta"))

    def test_ambiguous_values(self):
        with self.assertRaises(ValueError) as e:
            Database[int]({
                NamePattern("*", "*", "beta"): 1,
                NamePattern("gpm:1", "*", "beta"): 2,
                NamePattern("*", "ipm", "beta"): 3,
            })
        self.assertIn("ambiguous", str(e.exception))

    def test_update_ambiguous_values(self):
        db = Database[int]({
            NamePattern("gpm:1", "ipm", "beta"): 1,
            NamePattern("*", "ipm", "delta"): 2,
            NamePattern("*", "*", "gamma"): 3,
            NamePattern("gpm:2", "ipm", "beta"): 4,
        })

        with self.assertRaises(ValueError) as e:
            db.update(NamePattern("gpm:1", "*", "beta"), 777)
        self.assertIn("ambiguous", str(e.exception))


class DatabaseWithFallbackTest(_DatabaseTestCase):

    def test_basic_usage(self):
        fallback = Database[int]({
            NamePattern("gpm:1", "ipm", "beta"): 1,
            NamePattern("*", "ipm", "delta"): 2,
            NamePattern("*", "ipm", "gamma"): 3,
            NamePattern("gpm:2", "ipm", "beta"): 4,
            NamePattern("gpm:3", "init", "alpha"): 6,
        })

        db = DatabaseWithFallback({
            NamePattern("gpm:1", "ipm", "beta"): 11,
            NamePattern("gpm:2", "*", "beta"): 44,
            NamePattern("gpm:3", "*", "*"): 55,
        }, fallback)

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
        strata1 = Database[int]({
            NamePattern("gpm:1", "ipm", "beta"): 1,
            NamePattern("gpm:1", "ipm", "delta"): 2,
        })

        strata2 = Database[int]({
            NamePattern("gpm:2", "ipm", "beta"): 3,
            NamePattern("gpm:2", "ipm", "delta"): 4,
        })

        strata3 = Database[int]({
            NamePattern("gpm:3", "ipm", "beta"): 5,
            NamePattern("gpm:3", "ipm", "delta"): 6,
        })

        db = DatabaseWithStrataFallback({
            NamePattern("gpm:1", "ipm", "beta"): 11,
            NamePattern("gpm:2", "ipm", "beta"): 33,
            NamePattern("gpm:3", "*", "*"): 55,
        }, {
            'gpm:1': strata1,
            'gpm:2': strata2,
            'gpm:3': strata3,
        })

        self.assert_match(11, db.query("gpm:1::ipm::beta"))
        self.assert_match(33, db.query("gpm:2::ipm::beta"))
        self.assert_match(55, db.query("gpm:3::ipm::beta"))

        self.assert_match(2, db.query("gpm:1::ipm::delta"))
        self.assert_match(4, db.query("gpm:2::ipm::delta"))
        self.assert_match(55, db.query("gpm:3::ipm::delta"))

        self.assertIsNone(db.query("gpm1::init::population"))
