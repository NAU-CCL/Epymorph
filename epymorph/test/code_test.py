# pylint: disable=missing-docstring
# ruff: noqa: S105
import ast
import os
import unittest
from typing import Any

from epymorph.code import (
    CodeSecurityException,
    compile_function,
    has_function_structure,
    parse_function,
    scrub_function,
)


class TestHasFunctionStructure(unittest.TestCase):
    def test_valid_function_01(self):
        self.assertTrue(
            has_function_structure("""
            def my_function():
                return 42
        """)
        )

    def test_valid_function_02(self):
        self.assertTrue(
            has_function_structure("""
            def my_function_one():
                return 42
            
            def my_function_two():
                return 84
        """)  # noqa: W293
        )

    def test_valid_function_03(self):
        self.assertTrue(
            has_function_structure("""
            from something import that_thing

            xyz = 999
            
            def this_nifty_function (a: str, b: int, /, some_other_thing):
                return 42
            
            abc = 111
        """)  # noqa: W293
        )

    def test_valid_function_04(self):
        self.assertTrue(has_function_structure("def f():\n    return 'hello world'"))

    def test_invalid_function_01(self):
        self.assertFalse(has_function_structure("42"))

    def test_invalid_function_02(self):
        self.assertFalse(has_function_structure("thisdef is (you): so yeah"))


class TestParseFunction(unittest.TestCase):
    def test_valid_function(self):
        code_string = """
            def my_function():
                return 42
        """
        result = parse_function(code_string)
        self.assertIsInstance(result, ast.FunctionDef)
        self.assertEqual(result.name, "my_function")

    def test_invalid_function_count(self):
        code_string = """
            def function_one():
                return 1

            def function_two():
                return 2
        """
        with self.assertRaises(ValueError):
            parse_function(code_string)

    def test_empty_code_string(self):
        code_string = ""
        with self.assertRaises(ValueError):
            parse_function(code_string)

    def test_array_syntax(self):
        code_string = "[[1,2,3,4],[5,6,7,8]]"
        with self.assertRaises(ValueError):
            parse_function(code_string)

    def test_number_syntax(self):
        code_string = "0.42"
        with self.assertRaises(ValueError):
            parse_function(code_string)

    def test_invalid_syntax(self):
        code_string = "def my_function() return 42"
        with self.assertRaises(SyntaxError):
            parse_function(code_string)

    def test_non_function_definition(self):
        code_string = "x = 42"
        with self.assertRaises(ValueError):
            parse_function(code_string)


class TestScrubFunction(unittest.TestCase):
    def assertAstEqual(self, a0: ast.AST, a1: ast.AST) -> None:
        self.assertEqual(ast.dump(a0), ast.dump(a1))

    def test_scrub_imports(self):
        empty_args = ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        )

        f0 = ast.FunctionDef(
            name="my_function",
            args=empty_args,
            body=[
                ast.Import(names=[ast.alias(name="os")]),
                ast.Return(value=ast.Constant(value=42)),
            ],
            decorator_list=[],
        )

        f1 = ast.FunctionDef(
            name="my_function",
            args=empty_args,
            body=[
                ast.ImportFrom(module="os", names=[ast.alias(name="getenv")], level=0),
                ast.Return(value=ast.Constant(value=42, kind=None)),
            ],
            decorator_list=[],
        )

        exp = ast.FunctionDef(
            name="my_function",
            args=empty_args,
            body=[
                ast.Return(value=ast.Constant(value=42)),
            ],
            decorator_list=[],
        )

        self.assertAstEqual(scrub_function(f0), exp)
        self.assertAstEqual(scrub_function(f1), exp)


class TestCompileFunction(unittest.TestCase):
    def _compile_function(self, code: str, unsafe: bool) -> Any:
        return compile_function(
            parse_function(code, unsafe=unsafe), global_namespace={}
        )()

    def test_access_env(self):
        # Assume we have a juicy secret value defined as an environment variable.
        os.environ["TEST_SECRET_9812398712"] = "42"
        self.assertEqual(os.getenv("TEST_SECRET_9812398712"), "42")

        # This function tries to grab that secret and do something with it.
        evil_function_string = """
            def steal_your_secrets():
                import os
                secret = os.getenv('TEST_SECRET_9812398712')
                # I might exfiltrate the secret over HTTP...
                if secret is not None:
                    return secret
                else:
                    return '?'
        """

        # If the function is run as-is, it can import os and get the secret.
        bad_func_1 = compile_function(
            parse_function(evil_function_string, unsafe=True), {}
        )
        self.assertTrue(bad_func_1(), "42")

        # Or if we accidentally include 'os' in the namespace,
        # it can also access the secret.
        bad_func_2 = compile_function(
            parse_function(evil_function_string, unsafe=True), globals()
        )
        self.assertTrue(bad_func_2(), "42")

        # But with imports scrubbed out and an empty namespace, it will fail to run.
        safe_func = compile_function(
            parse_function(evil_function_string, unsafe=False), {}
        )
        with self.assertRaises(NameError):
            safe_func()

    def test_use_exec(self):
        # Assume we have a juicy secret value defined as an environment variable.
        os.environ["TEST_SECRET_9812398712"] = "42"
        self.assertEqual(os.getenv("TEST_SECRET_9812398712"), "42")

        # If we have eval or exec we can use that to load 'os', and then the secret.
        evil_function_string_1 = """
            def steal_your_secrets():
                glo = dict()
                loc = dict()
                exec("import os\\nsecret = os.getenv('TEST_SECRET_9812398712')", glo, loc)
                secret = loc['secret']
                # I might exfiltrate the secret over HTTP...
                return secret if secret is not None else '?'
        """  # noqa: E501

        # Even if I'm crafty by renaming 'exec' before I call it.
        evil_function_string_2 = """
            def steal_your_secrets():
                glo = dict()
                loc = dict()
                foo = exec
                foo("import os\\nsecret = os.getenv('TEST_SECRET_9812398712')", glo, loc)
                secret = loc['secret']
                # I might exfiltrate the secret over HTTP...
                return secret if secret is not None else '?'
        """  # noqa: E501

        # Even if I use eval to run exec.
        evil_function_string_3 = """
            def steal_your_secrets():
                glo = dict()
                loc = dict()
                eval('''exec("import os\\\\nsecret = os.getenv('TEST_SECRET_9812398712')", globals(), locals())''', glo, loc)
                secret = loc['secret']
                # I might exfiltrate the secret over HTTP...
                return secret if secret is not None else '?'
        """  # noqa: E501

        test = self._compile_function

        # If the function is run as-is, it can import os and get the secret.
        self.assertTrue(test(evil_function_string_1, unsafe=True), "42")
        self.assertTrue(test(evil_function_string_2, unsafe=True), "42")
        self.assertTrue(test(evil_function_string_3, unsafe=True), "42")

        # But with exec detection, it will fail to compile.
        with self.assertRaises(CodeSecurityException):
            test(evil_function_string_1, unsafe=False)

        with self.assertRaises(CodeSecurityException):
            test(evil_function_string_2, unsafe=False)

        with self.assertRaises(CodeSecurityException):
            test(evil_function_string_3, unsafe=False)

    def test_use__import__(self):
        # Assume we have a juicy secret value defined as an environment variable.
        os.environ["TEST_SECRET_9812398712"] = "42"
        self.assertEqual(os.getenv("TEST_SECRET_9812398712"), "42")

        # If we have __import__ we can use that to load 'os', and then the secret.
        evil_function_string_1 = """
            def steal_your_secrets():
                madule_get_it_its_like_module_but_mad = __import__('os')
                secret = madule_get_it_its_like_module_but_mad.getenv('TEST_SECRET_9812398712')
                # I might exfiltrate the secret over HTTP...
                return secret if secret is not None else '?'
        """  # noqa: E501

        test = self._compile_function

        # If the function is run as-is, it can import os and get the secret.
        self.assertTrue(test(evil_function_string_1, unsafe=True), "42")

        # But with __import__ detection, it will fail to compile.
        with self.assertRaises(CodeSecurityException):
            test(evil_function_string_1, unsafe=False)

    def test_use__builtins__(self):
        # Assume we have a juicy secret value defined as an environment variable.
        os.environ["TEST_SECRET_9812398712"] = "42"
        self.assertEqual(os.getenv("TEST_SECRET_9812398712"), "42")

        # In America, first you get the __builtins__,
        # then you get the __import__,
        # then you get os.
        evil_function_string_1 = """
            def steal_your_secrets():
                badule = __builtins__['__import__']('os')
                secret = badule.getenv('TEST_SECRET_9812398712')
                # I might exfiltrate the secret over HTTP...
                return secret if secret is not None else '?'
        """

        test = self._compile_function

        # If the function is run as-is, it can import os and get the secret.
        self.assertTrue(test(evil_function_string_1, unsafe=True), "42")

        # But with __builtins__ detection, it will fail to compile.
        with self.assertRaises(CodeSecurityException):
            test(evil_function_string_1, unsafe=False)

    def test_use__class__(self):
        # Test case inspired by https://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
        # Assume we have a juicy secret value defined as an environment variable.
        os.environ["TEST_SECRET_9812398712"] = "42"
        self.assertEqual(os.getenv("TEST_SECRET_9812398712"), "42")

        evil_function_string_1 = """
            def stealing_secrets():
                bins = next(c for c in ().__class__.__base__.__subclasses__()
                            if c.__name__ == 'catch_warnings')()._module.__builtins__
                secret = bins['__import__']('os').getenv('TEST_SECRET_9812398712')
                # I might exfiltrate the secret over HTTP...
                return secret if secret is not None else '?'
        """

        test = self._compile_function

        # If the function is run as-is, it can import os and get the secret.
        self.assertTrue(test(evil_function_string_1, unsafe=True), "42")

        # But with __class__ detection, it will fail to compile.
        with self.assertRaises(CodeSecurityException):
            test(evil_function_string_1, unsafe=False)

    def test_use_fstrings(self):
        # Assume we have a juicy secret value defined as an environment variable.
        os.environ["TEST_SECRET_9812398712"] = "42"
        self.assertEqual(os.getenv("TEST_SECRET_9812398712"), "42")

        evil_function_string_1 = """
            def stealing_secrets():
                secret = f'''{eval("__builtins__['__import__']('os').getenv('TEST_SECRET_9812398712')")}'''
                # I might exfiltrate the secret over HTTP...
                return secret if secret is not None else '?'
        """  # noqa: E501

        test = self._compile_function

        # If the function is run as-is, it can import os and get the secret.
        self.assertTrue(test(evil_function_string_1, unsafe=True), "42")

        # But malicious code inside the fstring is still detected,
        # so it will fail to compile.
        with self.assertRaises(CodeSecurityException):
            test(evil_function_string_1, unsafe=False)


if __name__ == "__main__":
    unittest.main()
