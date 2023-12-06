# pylint: disable=missing-docstring
import unittest

import numpy as np

from epymorph.data_shape import Shapes
from epymorph.engine.context import normalize_params
from epymorph.error import CompilationException
from epymorph.geo.geo import Geo
from epymorph.geo.spec import (LABEL, NO_DURATION, AttribDef, CentroidDType,
                               StaticGeoSpec)
from epymorph.geo.static import StaticGeo
from epymorph.params import _evaluate_param_function


class TestNormalizeParams(unittest.TestCase):
    test_geo: Geo

    def setUp(self) -> None:
        self.test_geo = StaticGeo(
            StaticGeoSpec(
                attributes=[
                    LABEL,
                    AttribDef('geoid', np.str_, Shapes.N),
                    AttribDef('centroid', CentroidDType, Shapes.N),
                    AttribDef('population', np.int64, Shapes.N),
                    AttribDef('commuters', np.int64, Shapes.NxN),
                ],
                time_period=NO_DURATION
            ), {
                'label': np.array(['AZ'], dtype=np.str_),
                'geoid': np.array(['04'], dtype=np.str_),
                'centroid': np.array([(-111.856111, 34.566667)], dtype=CentroidDType),
                'population': np.array([100_000], dtype=np.int64),
                'commuters': np.array([[0]], dtype=np.int64)
            })

    def test_normalize_params_with_empty_data(self):
        """Test that the function normalizes an empty data dictionary."""
        normed = normalize_params({}, self.test_geo, 100)
        self.assertEqual(normed, {})

    def test_normalize_params_with_simple_values(self):
        """Test that the function normalizes simple values to numpy arrays."""
        data = {"a": 1, "b": 2.0, "c": "3"}
        normed = normalize_params(data, self.test_geo, 100)

        self.assertEqual(normed["a"], np.array(1))
        self.assertEqual(normed["b"], np.array(2.0))
        self.assertEqual(normed["c"], np.array("3"))

    def test_normalize_params_with_lists(self):
        """Test that the function normalizes lists to numpy arrays."""
        data = {"a": [1, 2, 3], "b": [2.0, 3.0, 4.0], "c": ["5", "6", "7"]}
        normed = normalize_params(data, self.test_geo, 100, {})

        np.testing.assert_array_equal(normed["a"], np.array([1, 2, 3]))
        np.testing.assert_array_equal(normed["b"], np.array([2.0, 3.0, 4.0]))
        np.testing.assert_array_equal(normed["c"], np.array(["5", "6", "7"]))

    def test_normalize_params_with_functions(self):
        """Test that the function normalizes functions to numpy arrays."""

        def my_function(t, _):
            return t ** 2

        normed = normalize_params({"a": my_function}, self.test_geo, 100, {})

        exp = np.array([my_function(t, '_') for t in range(100)])
        np.testing.assert_array_equal(normed["a"], exp)

    def test_normalize_params_with_function_strings(self):
        """Test that the function normalizes function strings to numpy arrays."""
        def my_function(t, _):
            return t ** 2

        my_function_str = '''\
def cal(t,_):
    return t ** 2
'''
        normed = normalize_params({"a": my_function_str}, self.test_geo, 100)

        exp = np.array([my_function(t, None) for t in range(100)])
        np.testing.assert_array_equal(normed["a"], exp)

    def test_normalize_params_copies_ndarrays(self):
        alpha = np.array([1, 2, 3, 4, 5], dtype=np.int64)

        normed = normalize_params({"alpha": alpha}, self.test_geo, 100)

        alpha_n = normed['alpha']
        self.assertFalse(
            np.shares_memory(alpha, alpha_n),
            "Normalized ndarrays should be distinct copies of the input."
        )


class TestEvaluateFunction(unittest.TestCase):

    def test_case_underscore_underscore(self):
        result = _evaluate_param_function(
            function=lambda _, __: 42,
            nodes=2,
            duration=3,
            dtype=np.int64
        )
        self.assertEqual(result, np.asarray(42, dtype=np.int64))

    def test_case_t_underscore(self):
        result = _evaluate_param_function(
            function=lambda t, _: t * 2,
            nodes=2,
            duration=3,
            dtype=np.float64
        )
        expected_result = np.asarray([0.0, 2.0, 4.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected_result)

    def test_case_underscore_n(self):
        result = _evaluate_param_function(
            function=lambda _, n: n + 1,
            nodes=3,
            duration=2,
            dtype=np.int32
        )
        expected_result = np.asarray([1, 2, 3], dtype=np.int32)
        np.testing.assert_array_equal(result, expected_result)

    def test_case_t_n(self):
        result = _evaluate_param_function(
            function=lambda t, n: t + n,
            nodes=3,
            duration=2,
            dtype=np.float32
        )
        expected_result = np.asarray([[0.0, 1.0, 2.0],
                                      [1.0, 2.0, 3.0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected_result)

    def test_unsupported_signature_too_many_params(self):
        with self.assertRaises(CompilationException):
            _evaluate_param_function(
                function=lambda a, b, c: a + b + c,  # type: ignore
                nodes=2,
                duration=3,
                dtype=np.int64
            )

    def test_unsupported_signature_invalid_param_names(self):
        with self.assertRaises(CompilationException):
            _evaluate_param_function(
                lambda a, b: a + b,
                nodes=2,
                duration=3,
                dtype=np.int64
            )


if __name__ == '__main__':
    unittest.main()
