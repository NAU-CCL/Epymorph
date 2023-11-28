import unittest

import numpy as np

from epymorph.context import evaluate_function


class TestEvaluateFunction(unittest.TestCase):

    def test_case_underscore_underscore(self):
        result = evaluate_function(lambda _, __: 42, 2, 3, dt=np.int64)
        self.assertEqual(result, np.asarray(42, dtype=np.int64))

    def test_case_t_underscore(self):
        result = evaluate_function(lambda t, _: t * 2, 2, 3, dt=np.float64)
        expected_result = np.asarray([0.0, 2.0, 4.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected_result)

    def test_case_underscore_n(self):
        result = evaluate_function(lambda _, n: n + 1, 3, 2, dt=np.int32)
        expected_result = np.asarray([1, 2, 3], dtype=np.int32)
        np.testing.assert_array_equal(result, expected_result)

    def test_case_t_n(self):
        result = evaluate_function(lambda t, n: t + n, 3, 2, dt=np.float32)
        expected_result = np.asarray([[0.0, 1.0, 2.0],
                                      [1.0, 2.0, 3.0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected_result)

    def test_unsupported_signature_too_many_params(self):
        with self.assertRaises(ValueError):
            evaluate_function(lambda a, b, c: a + b + c, 2, 3, dt=np.int64)

    def test_unsupported_signature_invalid_param_names(self):
        with self.assertRaises(ValueError):
            evaluate_function(lambda a, b: a + b, 2, 3, dt=np.int64)


if __name__ == '__main__':
    unittest.main()
