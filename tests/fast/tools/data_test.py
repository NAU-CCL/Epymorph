from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import pytest

from epymorph import initializer as init
from epymorph.data import ipm, mm
from epymorph.geography.us_census import StateScope
from epymorph.rume import RUME, SingleStrataRUME
from epymorph.time import TimeFrame
from epymorph.tools.data import Output, munge


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
                "tick": np.repeat(np.arange(0, 4), 2),
                "date": np.repeat(np.arange(date(2021, 1, 1), date(2021, 1, 5)), 2),
                "node": np.tile(["04", "35"], 4),
                "S": np.arange(3000, 3800, 100),
                "I": np.arange(2000, 2800, 100),
                "R": np.arange(1000, 1800, 100),
                "S → I": np.arange(300, 380, 10),
                "I → R": np.arange(200, 280, 10),
                "R → S": np.arange(100, 180, 10),
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
            "time": [1, 2],
            "geo": ["04", "04"],
            "S": [3200, 3400],
            "I": [2200, 2400],
            "R": [1200, 1400],
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
