import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from epymorph.adrio.adrio import ADRIO, InspectResult
from epymorph.adrio.validation import ResultFormat
from epymorph.attribute import NamePattern
from epymorph.data import ipm
from epymorph.data_shape import Shapes
from epymorph.geography.us_census import StateScope
from epymorph.simulation import Context
from epymorph.time import TimeFrame
from epymorph.tools.data import memoize_rume, munge
from tests.fast.tools.conftest import HOME_COMPARTMENT_OFFSET, HOME_EVENT_OFFSET

# See conftest for RUME and Output fixtures


def test_basic_munge(rume, output):
    # Test with a selection in each axis.
    actual = munge(
        output=output,
        geo=rume.scope.select.by_state("04"),
        time=rume.time_frame.select.days(1, 2),
        quantity=rume.ipm.select.compartments(),
    )

    expected = pd.DataFrame(
        {
            "time": np.array([1, 2], dtype=np.int64),
            "geo": ["04", "04"],
            "S": np.array([3200, 3400], dtype=np.int64),
            "I": np.array([2200, 2400], dtype=np.int64),
            "R": np.array([1200, 1400], dtype=np.int64),
        }
    )

    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected)


@pytest.mark.parametrize(
    ("data_mode", "c_offset", "e_offset"),
    [
        ("data_by_visit", 0, 0),
        ("data_by_home", HOME_COMPARTMENT_OFFSET, HOME_EVENT_OFFSET),
    ],
)
def test_basic_munge_data_mode(rume, output, data_mode, c_offset, e_offset):
    actual = munge(
        output=getattr(output, data_mode),
        geo=rume.scope.select.by_state("04"),
        time=rume.time_frame.select.days(1, 2),
        quantity=rume.ipm.select.all(),
    )

    expected = pd.DataFrame(
        {
            "time": np.array([1, 2], dtype=np.int64),
            "geo": ["04", "04"],
            "S": np.array([3200, 3400], dtype=np.int64) + c_offset,
            "I": np.array([2200, 2400], dtype=np.int64) + c_offset,
            "R": np.array([1200, 1400], dtype=np.int64) + c_offset,
            "S → I": np.array([320, 340], dtype=np.int64) + e_offset,
            "I → R": np.array([220, 240], dtype=np.int64) + e_offset,
            "R → S": np.array([120, 140], dtype=np.int64) + e_offset,
        }
    )

    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected)


def test_munge_errors(rume, output):
    # Bad object correlation, geo
    wrong_geo = StateScope.in_states(["08"], year=2021)
    with pytest.raises(ValueError) as err:  # noqa: PT011
        munge(
            output=output,
            geo=wrong_geo.select.all(),
            time=rume.time_frame.select.all(),
            quantity=rume.ipm.select.compartments(),
        )
    assert "same GeoScope instance" in str(err.value)

    # Bad object correlation, time
    wrong_time = TimeFrame.rangex("2021-01-01", "2021-02-01")
    with pytest.raises(ValueError) as err:  # noqa: PT011
        munge(
            output=output,
            geo=rume.scope.select.all(),
            time=wrong_time.select.all(),
            quantity=rume.ipm.select.compartments(),
        )
    assert "same TimeFrame instance" in str(err.value)

    # Bad object correlation, quantity
    wrong_ipm = ipm.SIRH()
    with pytest.raises(ValueError) as err:  # noqa: PT011
        munge(
            output=output,
            geo=rume.scope.select.all(),
            time=rume.time_frame.select.all(),
            quantity=wrong_ipm.select.compartments(),
        )
    assert "same CompartmentModel instance" in str(err.value)


def test_memoize_rume(tmp_path, rume):
    adrio_evaluated = 0

    class TestADRIO(ADRIO[np.int64, np.int64]):
        # An ADRIO that just returns the values it was given,
        # and increments the evaluation counter.
        def __init__(self, values):
            self.values = values

        @property
        def result_format(self):
            return ResultFormat(shape=Shapes.N, dtype=np.int64)

        def validate_context(self, context: Context):
            pass

        def inspect(self):
            return InspectResult(
                adrio=self,
                source=self.values,
                result=self.values,
                shape=Shapes.N,
                dtype=np.int64,
                issues={},
            )

        def evaluate(self):
            nonlocal adrio_evaluated
            adrio_evaluated += 1
            return super().evaluate()

    def rume_with(population):
        pop_key = NamePattern.of("population")
        adrio = TestADRIO(population)
        return replace(rume, params={**rume.params, pop_key: adrio})

    cache_path = tmp_path / "cache.npz"
    pop = NamePattern.of("gpm:all::init::population")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        # First attempt: should evaluate params and save
        rume1 = rume_with(population=np.array([42, 84]))
        cached1 = memoize_rume(cache_path, rume1)
        assert cached1 is not rume1  # returns a clone
        assert cache_path.exists()  # cache file exists
        assert adrio_evaluated == 1  # ADRIO gets evaluated

        adrio_evaluated = 0  # reset counter

        # Second attempt:
        # Should load from cache, not evaluate params.
        # Even though ADRIO values are different this time,
        # the resulting RUME should still use the cached values.
        rume2 = rume_with(population=np.array([11, 22]))
        cached2 = memoize_rume(cache_path, rume2)
        assert adrio_evaluated == 0  # ADRIO not eval'd
        assert pop in cached1.params
        assert pop in cached2.params
        np.testing.assert_array_equal(cached1.params[pop], np.array([42, 84]))
        np.testing.assert_array_equal(cached2.params[pop], np.array([42, 84]))

        adrio_evaluated = 0  # reset counter

        # Third attempt: refresh should overwrite the cached values
        rume3 = rume_with(population=np.array([88, 99]))
        cached3 = memoize_rume(cache_path, rume3, refresh=True)
        assert adrio_evaluated == 1  # ADRIO gets evaluated
        assert pop in cached3.params
        # ensure both the RUME and the cache file have the new values
        np.testing.assert_array_equal(
            cached3.params[pop],
            np.array([88, 99]),
        )
        np.testing.assert_array_equal(
            np.load(cache_path)["gpm:all::init::population"],
            np.array([88, 99]),
        )
