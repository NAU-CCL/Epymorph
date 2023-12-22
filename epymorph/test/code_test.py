# pylint: disable=missing-docstring
import ast
import os
import unittest
from textwrap import dedent
from typing import Any

from epymorph.code import (CodeSecurityException, compile_function,
                           parse_function, scrub_function)


class TestParseFunction(unittest.TestCase):

    def test_valid_function(self):
        code_string = dedent("""
            def my_function():
                return 42
        """)
        result = parse_function(code_string)
        self.assertIsInstance(result, ast.FunctionDef)
        self.assertEqual(result.name, 'my_function')

    def test_invalid_function_count(self):
        code_string = dedent("""
            def function_one():
                return 1

            def function_two():
                return 2
        """)
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
        f0 = ast.FunctionDef(
            name='my_function',
            body=[
                ast.Import(names=['os']),
                ast.Return(value=ast.Constant(value=42)),
            ],
        )

        f1 = ast.FunctionDef(
            name='my_function',
            body=[
                ast.ImportFrom(module='os', names=['getenv']),
                ast.Return(value=ast.Constant(value=42, kind=None)),
            ],
        )

        exp = ast.FunctionDef(
            name='my_function',
            body=[
                ast.Return(value=ast.Constant(value=42)),
            ],
        )

        self.assertAstEqual(scrub_function(f0), exp)
        self.assertAstEqual(scrub_function(f1), exp)


class TestCompileFunction(unittest.TestCase):

    def test_access_env(self):
        # Assume we have a juicy secret value defined as an environment variable.
        os.environ['TEST_SECRET_9812398712'] = '42'
        self.assertEqual(os.getenv('TEST_SECRET_9812398712'), '42')

        # This function tries to grab that secret and do something with it.
        evil_function_string = dedent("""
            def steal_your_secrets():
                import os
                secret = os.getenv('TEST_SECRET_9812398712')
                # I might exfiltrate the secret over HTTP...
                if secret is not None:
                    return secret
                else:
                    return '?'
        """)

        # If the function is run as-is, it can import os and get the secret.
        bad_func_1 = compile_function(parse_function(
            evil_function_string, unsafe=True), {})
        self.assertTrue(bad_func_1(), '42')

        # Or if we accidentally include 'os' in the namespace, it can also access the secret.
        bad_func_2 = compile_function(parse_function(
            evil_function_string, unsafe=True), globals())
        self.assertTrue(bad_func_2(), '42')

        # But with imports scrubbed out and an empty namespace, it will fail to run.
        safe_func = compile_function(parse_function(
            evil_function_string, unsafe=False), {})
        with self.assertRaises(NameError):
            safe_func()

    def test_use_exec(self):
        # Assume we have a juicy secret value defined as an environment variable.
        os.environ['TEST_SECRET_9812398712'] = '42'
        self.assertEqual(os.getenv('TEST_SECRET_9812398712'), '42')

        # If we have eval or exec we can use that to load 'os', and then the secret.
        evil_function_string_1 = dedent("""
            def steal_your_secrets():
                glo = dict()
                loc = dict()
                exec("import os\\nsecret = os.getenv('TEST_SECRET_9812398712')", glo, loc)
                secret = loc['secret']
                # I might exfiltrate the secret over HTTP...
                return secret if secret is not None else '?'
        """)

        # Even if I'm crafty by renaming 'exec' before I call it.
        evil_function_string_2 = dedent("""
            def steal_your_secrets():
                glo = dict()
                loc = dict()
                foo = exec
                foo("import os\\nsecret = os.getenv('TEST_SECRET_9812398712')", glo, loc)
                secret = loc['secret']
                # I might exfiltrate the secret over HTTP...
                return secret if secret is not None else '?'
        """)

        # Even if I use eval to run exec.
        evil_function_string_3 = dedent("""
            def steal_your_secrets():
                glo = dict()
                loc = dict()
                eval('''exec("import os\\\\nsecret = os.getenv('TEST_SECRET_9812398712')", globals(), locals())''', glo, loc)
                secret = loc['secret']
                # I might exfiltrate the secret over HTTP...
                return secret if secret is not None else '?'
        """)

        def test(code: str, unsafe: bool) -> Any:
            f = compile_function(
                parse_function(code, unsafe=unsafe),
                global_namespace={}
            )
            return f()

        # If the function is run as-is, it can import os and get the secret.
        self.assertTrue(test(evil_function_string_1, unsafe=True), '42')
        self.assertTrue(test(evil_function_string_2, unsafe=True), '42')
        self.assertTrue(test(evil_function_string_3, unsafe=True), '42')

        # But with exec detection, it will fail to compile.
        with self.assertRaises(CodeSecurityException):
            test(evil_function_string_1, unsafe=False)

        with self.assertRaises(CodeSecurityException):
            test(evil_function_string_2, unsafe=False)

        with self.assertRaises(CodeSecurityException):
            test(evil_function_string_3, unsafe=False)

    def test_use__import__(self):
        # Assume we have a juicy secret value defined as an environment variable.
        os.environ['TEST_SECRET_9812398712'] = '42'
        self.assertEqual(os.getenv('TEST_SECRET_9812398712'), '42')

        # If we have __import__ we can use that to load 'os', and then the secret.
        evil_function_string_1 = dedent("""
            def steal_your_secrets():
                madule_get_it_its_like_module_but_mad = __import__('os')
                secret = madule_get_it_its_like_module_but_mad.getenv('TEST_SECRET_9812398712')
                # I might exfiltrate the secret over HTTP...
                return secret if secret is not None else '?'
        """)

        def test(code: str, unsafe: bool) -> Any:
            f = compile_function(
                parse_function(code, unsafe=unsafe),
                global_namespace={}
            )
            return f()

        # If the function is run as-is, it can import os and get the secret.
        self.assertTrue(test(evil_function_string_1, unsafe=True), '42')

        # But with __import__ detection, it will fail to compile.
        with self.assertRaises(CodeSecurityException):
            test(evil_function_string_1, unsafe=False)

    def test_use__builtins__(self):
        # Assume we have a juicy secret value defined as an environment variable.
        os.environ['TEST_SECRET_9812398712'] = '42'
        self.assertEqual(os.getenv('TEST_SECRET_9812398712'), '42')

        # In America, first you get the __builtins__, then you get the __import__, then you get os.
        evil_function_string_1 = dedent("""
            def steal_your_secrets():
                badule = __builtins__['__import__']('os')
                secret = badule.getenv('TEST_SECRET_9812398712')
                # I might exfiltrate the secret over HTTP...
                return secret if secret is not None else '?'
        """)

        def test(code: str, unsafe: bool) -> Any:
            f = compile_function(
                parse_function(code, unsafe=unsafe),
                global_namespace={}
            )
            return f()

        # If the function is run as-is, it can import os and get the secret.
        self.assertTrue(test(evil_function_string_1, unsafe=True), '42')

        # But with __import__ detection, it will fail to compile.
        with self.assertRaises(CodeSecurityException):
            test(evil_function_string_1, unsafe=False)


if __name__ == '__main__':
    unittest.main()
