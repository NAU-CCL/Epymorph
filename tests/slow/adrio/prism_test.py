import numpy as np
import pytest

from epymorph.adrio import prism, us_tiger
from epymorph.adrio.adrio import ADRIOContextError
from epymorph.kit import *

# Phoenix, Tucson, Flagstaff
_GOOD_CONTEXT = {
    "params": {
        "centroid": np.array(
            [(-112.0777, 33.4482), (-110.9742, 32.2540), (-111.6513, 35.1983)],
            dtype=CentroidDType,
        ),
    },
    "scope": CustomScope(["PHX", "TUS", "FLG"]),
    "time_frame": TimeFrame.range("2020-01-01", "2020-01-07"),
}


def test_unsupported_scopes():
    time_frame = TimeFrame.range("2020-01-01", "2020-01-07")
    centroid = us_tiger.InternalPoint()

    with pytest.raises(ADRIOContextError, match="within the 48 contiguous states"):
        prism.Temperature(temp_var="Minimum").with_context(
            params={"centroid": centroid},
            scope=StateScope.in_states(["HI"], year=2020),
            time_frame=time_frame,
        ).evaluate()

    with pytest.raises(ADRIOContextError, match="within the 48 contiguous states"):
        prism.Temperature(temp_var="Minimum").with_context(
            params={"centroid": centroid},
            scope=CountyScope.in_states(["AK"], year=2020),
            time_frame=time_frame,
        ).evaluate()

    with pytest.raises(ADRIOContextError, match="within the 48 contiguous states"):
        prism.Temperature(temp_var="Minimum").with_context(
            params={"centroid": centroid},
            scope=TractScope.in_states(["PR"], year=2020),
            time_frame=time_frame,
        ).evaluate()


def test_out_of_bounds():
    result = (
        prism.Temperature(temp_var="Minimum")
        .with_context(
            params={
                "centroid": np.array([(0, 0)], dtype=CentroidDType),
            },
            scope=CustomScope(["MiddleOfNowhere"]),
            time_frame=TimeFrame.range("2020-01-01", "2020-01-07"),
        )
        .inspect()
    )
    assert "undefined" in result.issues
    assert np.any(result.issues["undefined"])


def test_over_water():
    result = (
        prism.Temperature(temp_var="Minimum")
        .with_context(
            params={
                "centroid": np.array([(-122.659962, 37.758071)], dtype=CentroidDType),
            },
            scope=CustomScope(["OffTheCoastOfCalifornia"]),
            time_frame=TimeFrame.range("2020-01-01", "2020-01-07"),
        )
        .inspect()
    )
    assert "undefined" in result.issues
    assert np.any(result.issues["undefined"])


def test_bad_dates():
    with pytest.raises(ADRIOContextError, match="date range is out of PRISM scope"):
        prism.Temperature(temp_var="Minimum").with_context(
            **{
                **_GOOD_CONTEXT,
                "time_frame": TimeFrame.range("1970-01-01", "1970-01-07"),
            },
        ).evaluate()

    with pytest.raises(ADRIOContextError, match="date range is out of PRISM scope"):
        prism.Temperature(temp_var="Minimum").with_context(
            **{
                **_GOOD_CONTEXT,
                "time_frame": TimeFrame.range("2199-01-01", "2199-01-07"),
                # warning: test will break; change this date on or before the year 2199
            },
        ).evaluate()


def test_temperature_01():
    # min temp
    expected = np.array(
        [
            [6.49599981, 2.58200002, -13.58399963],
            [4.5619998, 2.32999992, -12.95300007],
            [5.20200014, 2.02900004, -10.0170002],
            [5.19299984, 1.23899996, -11.0539999],
            [5.32800007, 2.47900009, -9.62300014],
            [5.70900011, 2.71000004, -10.90600014],
            [5.57299995, 2.0309999, -12.34200001],
        ],
        dtype=np.float64,
    )
    adrio = prism.Temperature(temp_var="Minimum")
    actual = adrio.with_context(**_GOOD_CONTEXT).evaluate()
    np.testing.assert_allclose(actual, expected)
    assert actual.dtype == expected.dtype


def test_temperature_02():
    # mean temp
    expected = np.array(
        [
            [10.8210001, 9.44200039, -5.92600012],
            [9.71700001, 8.81000042, -3.73300028],
            [11.3210001, 7.79400015, -3.90100026],
            [11.14600086, 8.18400002, -2.55900002],
            [12.31300068, 10.9630003, 1.04400003],
            [13.02000046, 12.25600052, -0.12900001],
            [13.27900028, 10.49400043, -3.17200017],
        ],
        dtype=np.float64,
    )
    adrio = prism.Temperature(temp_var="Mean")
    actual = adrio.with_context(**_GOOD_CONTEXT).evaluate()
    np.testing.assert_allclose(actual, expected)
    assert actual.dtype == expected.dtype


def test_temperature_03():
    # max temp
    expected = np.array(
        [
            [15.14700031, 16.3029995, 1.73099995],
            [14.8739996, 15.29199982, 5.48500013],
            [17.44000053, 13.55900002, 2.21399999],
            [17.10099983, 15.13000011, 5.93499994],
            [19.29899979, 19.4470005, 11.71199989],
            [20.33200073, 21.80200005, 10.64700031],
            [20.98500061, 18.95800018, 5.99700022],
        ],
        dtype=np.float64,
    )
    adrio = prism.Temperature(temp_var="Maximum")
    actual = adrio.with_context(**_GOOD_CONTEXT).evaluate()
    np.testing.assert_allclose(actual, expected)
    assert actual.dtype == expected.dtype


def test_precipitation_01():
    # precip without timeframe override
    expected = np.zeros((7, 3), dtype=np.float64)
    adrio = prism.Precipitation()
    actual = adrio.with_context(**_GOOD_CONTEXT).evaluate()
    np.testing.assert_allclose(actual, expected)
    assert actual.dtype == expected.dtype


def test_precipitation_02():
    # precip with timeframe override
    expected = np.array(
        [
            [0.0, 0.036, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.24600001],
            [0.0, 1.17299998, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    actual = (
        prism.Precipitation()
        .with_context(
            **{
                **_GOOD_CONTEXT,
                "time_frame": TimeFrame.range("2020-08-01", "2020-08-07"),
            }
        )
        .evaluate()
    )
    np.testing.assert_allclose(actual, expected)
    assert actual.dtype == expected.dtype


def test_dew_point_01():
    # precip with timeframe override
    expected = np.array(
        [
            [-7.50000030e-02, 8.99999961e-03, -1.20419998e01],
            [1.87500000e00, 1.98300004e00, -7.48099995e00],
            [1.62899995e00, 8.74000013e-01, -1.30579996e01],
            [-4.99999989e-03, -9.95000005e-01, -1.01759996e01],
            [1.77499998e00, -2.72000015e-01, -8.71800041e00],
            [2.68700004e00, 4.05000001e-01, -1.07830000e01],
            [-7.32999980e-01, -1.68200004e00, -1.09630003e01],
        ],
        dtype=np.float64,
    )
    adrio = prism.DewPoint()
    actual = adrio.with_context(**_GOOD_CONTEXT).evaluate()
    np.testing.assert_allclose(actual, expected)
    assert actual.dtype == expected.dtype


def test_vpd_01():
    # vapor pressure deficit minimum
    expected = np.array(
        [
            [3.13599992, 1.01999998, 0.40200001],
            [1.50600004, 0.80900002, 0.27500001],
            [1.51900005, 0.616, 0.61199999],
            [2.46199989, 0.97500002, 0.33500001],
            [2.19499993, 1.67499995, 0.41600001],
            [1.99399996, 1.08099997, 0.352],
            [1.84899998, 0.98799998, 0.177],
        ],
        dtype=np.float64,
    )
    adrio = prism.VaporPressureDeficit(vpd_var="Minimum")
    actual = adrio.with_context(**_GOOD_CONTEXT).evaluate()
    np.testing.assert_allclose(actual, expected)
    assert actual.dtype == expected.dtype


def test_vpd_02():
    # vapor pressure deficit maximum
    expected = np.array(
        [
            [11.69900036, 12.70600033, 3.91599989],
            [10.64700031, 10.70899963, 5.75400019],
            [14.26000023, 9.17599964, 5.14599991],
            [14.52200031, 12.26299953, 6.23400021],
            [16.36599922, 17.07600021, 10.63300037],
            [17.40399933, 20.74699974, 10.3739996],
            [20.4810009, 17.03199959, 6.671],
        ],
        dtype=np.float64,
    )
    adrio = prism.VaporPressureDeficit(vpd_var="Maximum")
    actual = adrio.with_context(**_GOOD_CONTEXT).evaluate()
    np.testing.assert_allclose(actual, expected)
    assert actual.dtype == expected.dtype
