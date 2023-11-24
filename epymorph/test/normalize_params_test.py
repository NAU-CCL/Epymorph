import unittest
from unittest.mock import MagicMock

import numpy as np

from epymorph.context import evaluate_function, normalize_params
from epymorph.data_shape import Shapes
from epymorph.geo.abstract import _ProxyGeo
from epymorph.geo.geo import Geo
from epymorph.geo.spec import (LABEL, NO_DURATION, AttribDef, CentroidDType,
                               StaticGeoSpec)
from epymorph.geo.static import StaticGeo


class TestNormalizeParams(unittest.TestCase):

    def setUp(self) -> None:
        self.data = {}
        self.duration = 100
        self.dtypes = {}

        def TestGeo() -> Geo:
            """Load the single_pop geo."""
            spec = StaticGeoSpec(
                attributes=[
                    LABEL,
                    AttribDef('geoid', np.str_, Shapes.N),
                    AttribDef('centroid', CentroidDType, Shapes.N),
                    AttribDef('population', np.int64, Shapes.N),
                    AttribDef('commuters', np.int64, Shapes.NxN),
                ],
                time_period=NO_DURATION
            )
            return StaticGeo(spec, {
                'label': np.array(['AZ'], dtype=np.str_),
                'geoid': np.array(['04'], dtype=np.str_),
                'centroid': np.array([(-111.856111, 34.566667)], dtype=CentroidDType),
                'population': np.array([100_000], dtype=np.int64),
                'commuters': np.array([[0]], dtype=np.int64)
            })

        self.testgeo = TestGeo

    def test_normalize_params_with_empty_data(self):
        """
        Test that the function normalizes an empty data dictionary.
        """

        normalized_params = normalize_params(
            self.data, self.testgeo(), self.duration, self.dtypes)

        self.assertEqual(normalized_params, {})

    def test_normalize_params_with_simple_values(self):
        """
        Test that the function normalizes simple values to numpy arrays.
        """

        data = {"a": 1, "b": 2.0, "c": "3"}

        normalized_params = normalize_params(
            data, self.testgeo(), self.duration, self.dtypes)

        self.assertEqual(normalized_params["a"], np.array(1))
        self.assertEqual(normalized_params["b"], np.array(2.0))
        self.assertEqual(normalized_params["c"], np.array("3"))

    def test_normalize_params_with_lists(self):
        """
        Test that the function normalizes lists to numpy arrays.
        """

        data = {"a": [1, 2, 3], "b": [2.0, 3.0, 4.0], "c": ["5", "6", "7"]}

        normalized_params = normalize_params(
            data, self.testgeo(), self.duration, self.dtypes)

        self.assertTrue(np.array_equal(normalized_params["a"], np.array([1, 2, 3])))
        self.assertTrue(np.array_equal(
            normalized_params["b"], np.array([2.0, 3.0, 4.0])))
        self.assertTrue(np.array_equal(
            normalized_params["c"], np.array(["5", "6", "7"])))

    def test_normalize_params_with_functions(self):
        """
        Test that the function normalizes functions to numpy arrays.
        """

        def my_function(t, _):
            return t ** 2

        data = {"a": my_function}

        normalized_params = normalize_params(
            data,  self.testgeo(), self.duration, self.dtypes)

        self.assertTrue(np.array_equal(normalized_params["a"], np.array(
            [my_function(t, '_') for t in range(self.duration)])))

    def test_normalize_params_with_function_strings(self):
        """
        Test that the function normalizes function strings to numpy arrays.
        """
        def my_function(t, _):
            return t ** 2

        data = {
            "a": '''def cal(t,_):
        return t ** 2'''
        }

        normalized_params = normalize_params(
            data,  self.testgeo(), self.duration, self.dtypes)

        self.assertTrue(np.array_equal(normalized_params["a"], np.array(
            [my_function(t, None) for t in range(self.duration)])))


class TestGeoParams(unittest.TestCase):

    def setUp(self) -> None:
        self.data = {}
        self.duration = 100
        self.dtypes = {}

        def TestGeo() -> Geo:
            """Load the single_pop geo."""
            spec = StaticGeoSpec(
                attributes=[
                    LABEL,
                    AttribDef('geoid', np.str_, Shapes.N),
                    AttribDef('centroid', CentroidDType, Shapes.N),
                    AttribDef('population', np.int64, Shapes.N),
                    AttribDef('commuters', np.int64, Shapes.NxN),
                ],
                time_period=NO_DURATION
            )
            return StaticGeo(spec, {
                'label': np.array(['AZ'], dtype=np.str_),
                'geoid': np.array(['04'], dtype=np.str_),
                'centroid': np.array([(-111.856111, 34.566667)], dtype=CentroidDType),
                'population': np.array([100_000], dtype=np.int64),
                'commuters': np.array([[0]], dtype=np.int64)
            })

        self.testgeo = TestGeo


class TestProxyGeo(unittest.TestCase):

    def test_singleton_instance(self):
        proxy1 = _ProxyGeo()
        proxy2 = _ProxyGeo()
        self.assertIs(proxy1, proxy2, "Two instances of ProxyGeo should be the same")

    def test_set_actual_geo(self):
        proxy = _ProxyGeo()
        mock_geo = MagicMock(spec=Geo)
        proxy.set_actual_geo(mock_geo)
        self.assertIs(proxy._actual_geo, mock_geo, "Actual geo should be set correctly")

    def test_get_item(self):
        proxy = _ProxyGeo()
        mock_geo = MagicMock(spec=Geo)
        data = {'key1': 123, 'key2': 456}
        mock_geo.spec = data
        proxy.set_actual_geo(mock_geo)

        self.assertEqual(proxy['key1'], 123,
                         "Accessing data should return the correct value for key1")
        self.assertEqual(proxy['key2'], 456,
                         "Accessing data should return the correct value for key2")

# Unit tests for evaluate_function


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
