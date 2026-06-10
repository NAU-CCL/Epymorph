from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from epymorph import initializer as init
from epymorph.adrio.adrio import ADRIO, InspectResult
from epymorph.adrio.validation import ResultFormat
from epymorph.attribute import NamePattern
from epymorph.data import ipm, mm
from epymorph.data_shape import Shapes
from epymorph.geography.us_census import StateScope
from epymorph.rume import RUME, SingleStrataRUME
from epymorph.simulation import Context
from epymorph.time import TimeFrame
from epymorph.tools.data import Output, memoize_rume, munge


@pytest.fixture
def rume() -> RUME[StateScope]:
    """A very simple RUME with no external data requirements."""
    return SingleStrataRUME.build(
        ipm=ipm.SIRS(),
        mm=mm.No(),
        init=init.SingleLocation(location=0, seed_size=100),
        scope=StateScope.in_states(["04", "35"], year=2020),
        time_frame=TimeFrame.of("2021-01-01", 4),
        params={
            "beta": 0.4,
            "gamma": 1 / 10,
            "xi": 1 / 90,
            "population": [200_000, 100_000],
        },
    )


@pytest.fixture
def output(rume: RUME[StateScope]) -> Output:
    """A mock output so we don't have to run a sim."""

    @dataclass
    class MockOutput(Output):
        rume: RUME
        data_df: pd.DataFrame

        @property
        def dataframe(self) -> pd.DataFrame:
            return self.data_df

    return MockOutput(
        rume,
        pd.DataFrame(
            {
                "tick": np.repeat(np.arange(0, 4, dtype=np.int64), 2),
                "date": np.repeat(np.arange(date(2021, 1, 1), date(2021, 1, 5)), 2),
                "node": np.tile(["04", "35"], 4),
                "S": np.arange(3000, 3800, 100, dtype=np.int64),
                "I": np.arange(2000, 2800, 100, dtype=np.int64),
                "R": np.arange(1000, 1800, 100, dtype=np.int64),
                "S → I": np.arange(300, 380, 10, dtype=np.int64),
                "I → R": np.arange(200, 280, 10, dtype=np.int64),
                "R → S": np.arange(100, 180, 10, dtype=np.int64),
            }
        ),
    )


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
