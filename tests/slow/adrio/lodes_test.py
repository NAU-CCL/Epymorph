import numpy as np
import pytest

from epymorph.adrio import lodes
from epymorph.adrio.adrio import ADRIOContextError
from epymorph.data_usage import AvailableDataEstimate
from epymorph.error import MissingContextError
from epymorph.geography.custom import CustomScope
from epymorph.geography.us_census import (
    BlockGroupScope,
    CountyScope,
    StateScope,
    TractScope,
)
from epymorph.time import TimeFrame


@pytest.fixture
def state_scope():
    return StateScope.in_states(
        ["AZ", "NM", "CO", "NV"],
        year=2020,
    )


@pytest.fixture
def county_scope():
    return CountyScope.in_counties(
        ["04005", "04013", "04025", "35001"],
        year=2020,
    )


@pytest.fixture
def tract_scope():
    return TractScope.in_tracts(
        ["04005001301", "04013817102", "04025000612", "35001000129"],
        year=2020,
    )


@pytest.fixture
def cbg_scope():
    return BlockGroupScope.in_block_groups(
        ["040050013011", "040138171021", "040250006121", "350010001291"],
        year=2020,
    )


def test_commuters_bad_scope_type():
    with pytest.raises(ADRIOContextError, match="Census scope is required"):
        lodes.Commuters(year=2020).with_context(
            scope=CustomScope(["A", "B", "C"]),
        ).evaluate()


def test_commuters_bad_scope_year():
    with pytest.raises(ADRIOContextError, match="Invalid scope year"):
        lodes.Commuters(year=2020).with_context(
            scope=StateScope.in_states(["AZ"], year=2019),
        ).evaluate()


def test_commuters_no_time_frame_no_year():
    with pytest.raises(MissingContextError, match="time_frame"):
        lodes.Commuters().with_context(
            scope=StateScope.in_states(["AZ"], year=2020),
        ).evaluate()


def test_commuters_bad_data_year():
    with pytest.raises(ValueError, match="Invalid year"):
        lodes.Commuters(year=2000).with_context(
            scope=StateScope.in_states(["AZ"], year=2020),
        ).evaluate()
    with pytest.raises(ValueError, match="Invalid year"):
        lodes.Commuters(year=2050).with_context(
            scope=StateScope.in_states(["AZ"], year=2020),
        ).evaluate()


def test_commuters_special_exclusions():
    with pytest.raises(ADRIOContextError, match="Federal job"):
        lodes.Commuters(year=2008, job_type="All Federal Jobs").with_context(
            scope=StateScope.in_states(["AZ"], year=2020),
        ).evaluate()
    with pytest.raises(ADRIOContextError, match="Invalid year for state"):
        lodes.Commuters(year=2002).with_context(
            scope=StateScope.in_states(["05"], year=2020),
        ).evaluate()
    with pytest.raises(ADRIOContextError, match="Invalid year for state"):
        lodes.Commuters(year=2020).with_context(
            scope=StateScope.in_states(["28"], year=2020),
        ).evaluate()


def test_commuters_01(state_scope):
    # Commuters at state granularity
    expected = np.array(
        [
            [2773820, 2636, 14990, 6336],
            [1523, 2491082, 601, 5218],
            [3653, 696, 1230167, 381],
            [6059, 5543, 460, 749077],
        ],
        dtype=np.int64,
    )

    # Specified data year
    actual1 = (
        lodes.Commuters(year=2020)
        .with_context(
            scope=state_scope,
        )
        .evaluate()
    )

    # Equivalent: data year pulled from time_frame
    actual2 = (
        lodes.Commuters(year=2020)
        .with_context(
            scope=state_scope,
            time_frame=TimeFrame.rangex("2020-09-01", "2021-03-01"),
        )
        .evaluate()
    )

    assert expected.dtype == actual1.dtype
    assert np.array_equal(expected, actual1)
    assert np.array_equal(expected, actual2)


def test_commuters_02(county_scope):
    # Commuters at county granularity
    expected = np.array(
        [
            [35481, 8226, 2372, 69],
            [5472, 1787322, 6725, 734],
            [5549, 26227, 45732, 44],
            [33, 432, 26, 242209],
        ],
        dtype=np.int64,
    )

    actual = (
        lodes.Commuters(year=2020)
        .with_context(
            scope=county_scope,
        )
        .evaluate()
    )

    assert expected.dtype == actual.dtype
    assert np.array_equal(expected, actual)


def test_commuters_03(tract_scope):
    # Commuters at tract granularity
    expected = np.array(
        [
            [154, 0, 0, 0],
            [0, 34, 0, 0],
            [2, 0, 9, 0],
            [0, 0, 0, 20],
        ],
        dtype=np.int64,
    )

    actual = (
        lodes.Commuters(year=2020)
        .with_context(
            scope=tract_scope,
        )
        .evaluate()
    )

    assert expected.dtype == actual.dtype
    assert np.array_equal(expected, actual)


def test_commuters_04(cbg_scope):
    # Commuters at cbg granularity
    expected = np.array(
        [
            [12, 0, 0, 0],
            [0, 3, 0, 0],
            [0, 0, 9, 0],
            [0, 0, 0, 2],
        ],
        dtype=np.int64,
    )

    actual = (
        lodes.Commuters(year=2020)
        .with_context(
            scope=cbg_scope,
        )
        .evaluate()
    )

    assert expected.dtype == actual.dtype
    assert np.array_equal(expected, actual)


def test_commuters_05(county_scope):
    # Commuters with a different data year
    expected = np.array(
        [
            [36146, 9194, 2411, 90],
            [7027, 1763023, 7565, 861],
            [5696, 23558, 46123, 75],
            [38, 496, 38, 252458],
        ],
        dtype=np.int64,
    )

    actual = (
        lodes.Commuters(year=2018)
        .with_context(
            scope=county_scope,
        )
        .evaluate()
    )

    assert expected.dtype == actual.dtype
    assert np.array_equal(expected, actual)


def test_commuters_06(county_scope):
    # Commuters with a JobType
    expected = np.array(
        [
            [1339, 45, 43, 2],
            [98, 9307, 49, 11],
            [85, 109, 1171, 4],
            [2, 2, 1, 6159],
        ],
        dtype=np.int64,
    )

    actual = (
        lodes.Commuters(
            year=2020,
            job_type="Federal Primary Jobs",
        )
        .with_context(
            scope=county_scope,
        )
        .evaluate()
    )

    assert expected.dtype == actual.dtype
    assert np.array_equal(expected, actual)


def test_commuters_by_age_01(county_scope):
    # CommutersByAge
    expected = np.array(
        [
            [18983, 4159, 1129, 36],
            [2544, 977880, 3362, 378],
            [2529, 12141, 22647, 16],
            [10, 223, 6, 130168],
        ],
        dtype=np.int64,
    )

    actual = (
        lodes.CommutersByAge(
            year=2020,
            age_range="30_54",
        )
        .with_context(
            scope=county_scope,
        )
        .evaluate()
    )

    assert expected.dtype == actual.dtype
    assert np.array_equal(expected, actual)


def test_commuters_by_age_02(county_scope):
    # CommutersByAge with job type
    expected = np.array(
        [
            [862, 27, 28, 0],
            [47, 5874, 22, 7],
            [44, 62, 618, 2],
            [1, 2, 0, 3777],
        ],
        dtype=np.int64,
    )

    actual = (
        lodes.CommutersByAge(
            year=2020,
            age_range="30_54",
            job_type="Federal Primary Jobs",
        )
        .with_context(
            scope=county_scope,
        )
        .evaluate()
    )

    assert expected.dtype == actual.dtype
    assert np.array_equal(expected, actual)


def test_commuters_by_earnings_01(county_scope):
    # CommutersByEarnings
    expected = np.array(
        [
            [8723, 2140, 669, 19],
            [1962, 351360, 1679, 235],
            [1757, 6886, 10671, 13],
            [13, 87, 10, 56141],
        ],
        dtype=np.int64,
    )

    actual = (
        lodes.CommutersByEarnings(
            year=2020,
            earning_range="$1250 and Under",
        )
        .with_context(
            scope=county_scope,
        )
        .evaluate()
    )

    assert expected.dtype == actual.dtype
    assert np.array_equal(expected, actual)


def test_commuters_by_earnings_02(county_scope):
    # CommutersByEarnings with job type
    expected = np.array(
        [
            [13, 0, 0, 0],
            [2, 55, 0, 0],
            [3, 1, 6, 0],
            [0, 0, 0, 21],
        ],
        dtype=np.int64,
    )

    actual = (
        lodes.CommutersByEarnings(
            year=2020,
            earning_range="$1250 and Under",
            job_type="Federal Primary Jobs",
        )
        .with_context(
            scope=county_scope,
        )
        .evaluate()
    )

    assert expected.dtype == actual.dtype
    assert np.array_equal(expected, actual)


def test_commuters_by_industry_01(county_scope):
    # CommutersByIndustry
    expected = np.array(
        [
            [4092, 2241, 644, 5],
            [957, 344325, 1754, 97],
            [1048, 7541, 6124, 6],
            [3, 117, 5, 38949],
        ],
        dtype=np.int64,
    )

    actual = (
        lodes.CommutersByIndustry(
            year=2020,
            industry="Trade Transport Utility",
        )
        .with_context(
            scope=county_scope,
        )
        .evaluate()
    )

    assert expected.dtype == actual.dtype
    assert np.array_equal(expected, actual)


def test_commuters_by_industry_02(county_scope):
    # CommutersByIndustry with job type
    expected = np.array(
        [
            [4, 1, 0, 0],
            [3, 359, 1, 0],
            [1, 20, 13, 0],
            [0, 0, 0, 319],
        ],
        dtype=np.int64,
    )

    actual = (
        lodes.CommutersByIndustry(
            year=2020,
            industry="Trade Transport Utility",
            job_type="Federal Primary Jobs",
        )
        .with_context(
            scope=county_scope,
        )
        .evaluate()
    )

    assert expected.dtype == actual.dtype
    assert np.array_equal(expected, actual)


def test_estimates(state_scope, county_scope):
    actuals = [
        lodes.Commuters(year=2020).with_context(scope=state_scope).estimate_data(),
        lodes.Commuters(year=2020).with_context(scope=county_scope).estimate_data(),
        lodes.Commuters(year=2020, job_type="Federal Primary Jobs")
        .with_context(scope=state_scope)
        .estimate_data(),
        lodes.Commuters(year=2020, job_type="Federal Primary Jobs")
        .with_context(scope=county_scope)
        .estimate_data(),
        lodes.CommutersByIndustry(year=2020, industry="Trade Transport Utility")
        .with_context(scope=state_scope)
        .estimate_data(),
        lodes.CommutersByIndustry(year=2020, industry="Trade Transport Utility")
        .with_context(scope=county_scope)
        .estimate_data(),
    ]

    expecteds = [
        32792000,
        16046000,
        419600,
        209800,
        32792000,
        16046000,
    ]

    for a, e in zip(actuals, expecteds, strict=True):
        assert isinstance(a, AvailableDataEstimate)
        assert a.total_cache_bytes == e
