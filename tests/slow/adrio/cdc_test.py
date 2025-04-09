import numpy as np
import pytest

from epymorph.adrio import cdc
from epymorph.adrio.processing import RandomFix
from epymorph.geography.us_census import CountyScope
from epymorph.time import TimeFrame

# NOTE: these tests use VCR to record HTTP requests.
# To re-record this test load a census API key into the environment, then:
# uv run pytest tests/slow/adrio/cdc_test.py --record-mode=rewrite


@pytest.fixture(scope="module")
def vcr_config(global_vcr_config):
    return {**global_vcr_config, "filter_headers": ["x-app-token"]}


@pytest.mark.vcr
def test_covid_facility_hospitalization():
    def context():
        return {
            "scope": CountyScope.in_counties(
                ["Maricopa, AZ", "Bernalillo, NM"],
                year=2019,
            ),
            "time_frame": TimeFrame.rangex("2021-04-01", "2021-06-01"),
            "rng": np.random.default_rng(42),
        }

    actual1 = (
        cdc.COVIDFacilityHospitalization(
            age_group="both",
        )
        .with_context(**context())
        .evaluate()
    )

    # fmt: off
    expected1 = np.array([
        [("2021-04-04", 0), ("2021-04-04", 0)],
        [("2021-04-11", 0), ("2021-04-11", 0)],
        [("2021-04-18", 0), ("2021-04-18", 430)],
        [("2021-04-25", 0), ("2021-04-25", 399)],
        [("2021-05-02", 0), ("2021-05-02", 449)],
        [("2021-05-09", 1991), ("2021-05-09", 457)],
        [("2021-05-16", 0), ("2021-05-16", 0)],
        [("2021-05-23", 0), ("2021-05-23", 0)],
        [("2021-05-30", 0), ("2021-05-30", 0)],
    ], dtype=[("date", "datetime64[D]"), ("value", np.int64)])
    expected_mask1 = np.array([
        [(True, True), (True, True)],
        [(True, True), (True, True)],
        [(True, True), (False, False)],
        [(True, True), (False, False)],
        [(True, True), (False, False)],
        [(False, False), (False, False)],
        [(True, True), (True, True)],
        [(True, True), (True, True)],
        [(True, True), (True, True)],
    ], dtype=[("date", np.bool_), ("value", np.bool_)])
    # fmt: on

    assert np.ma.is_masked(actual1["value"])
    np.testing.assert_array_equal(np.ma.getdata(actual1), expected1, strict=True)
    np.testing.assert_array_equal(np.ma.getmask(actual1), expected_mask1, strict=True)

    actual2 = (
        cdc.COVIDFacilityHospitalization(
            age_group="both",
            fix_missing=0,
            fix_redacted=RandomFix.from_range(1, 3),
        )
        .with_context(**context())
        .evaluate()
    )

    # fmt: off
    expected2 = np.array([
        [("2021-04-04", 1609), ("2021-04-04", 372)],
        [("2021-04-11", 1685), ("2021-04-11", 419)],
        [("2021-04-18", 1808), ("2021-04-18", 430)],
        [("2021-04-25", 2044), ("2021-04-25", 399)],
        [("2021-05-02", 1956), ("2021-05-02", 449)],
        [("2021-05-09", 1991), ("2021-05-09", 457)],
        [("2021-05-16", 2027), ("2021-05-16", 473)],
        [("2021-05-23", 1971), ("2021-05-23", 354)],
        [("2021-05-30", 1650), ("2021-05-30", 327)],
    ], dtype=[("date", "datetime64[D]"), ("value", np.int64)])
    # fmt: on

    assert not np.ma.is_masked(actual2["value"])
    np.testing.assert_array_equal(actual2, expected2, strict=True)

    # The unmasked values from the first request
    # should be unchanged in the second request.
    mask = np.ma.getmask(actual2["value"])
    np.testing.assert_array_equal(actual1["value"][mask], actual2["value"][mask])


@pytest.mark.vcr
def test_covid_facility_hospitalization_age_groups():
    def context():
        return {
            "scope": CountyScope.in_counties(
                ["Maricopa, AZ", "Bernalillo, NM"],
                year=2019,
            ),
            "time_frame": TimeFrame.rangex("2021-04-01", "2021-06-01"),
            "rng": np.random.default_rng(42),
        }

    # The pediatric and adult cases should add up to the total,
    # but only if we remove random values.
    actual_both = (
        cdc.COVIDFacilityHospitalization(
            age_group="both",
            fix_missing=0,
            fix_redacted=0,
        )
        .with_context(**context())
        .evaluate()
    )
    actual_adult = (
        cdc.COVIDFacilityHospitalization(
            age_group="adult",
            fix_missing=0,
            fix_redacted=0,
        )
        .with_context(**context())
        .evaluate()
    )
    actual_pediatric = (
        cdc.COVIDFacilityHospitalization(
            age_group="pediatric",
            fix_missing=0,
            fix_redacted=0,
        )
        .with_context(**context())
        .evaluate()
    )
    np.testing.assert_array_equal(
        actual_adult["value"] + actual_pediatric["value"],
        actual_both["value"],
        strict=True,
    )


@pytest.mark.vcr
def test_influenza_facility_hospitalization():
    def context():
        return {
            "scope": CountyScope.in_counties(
                ["Maricopa, AZ", "Bernalillo, NM"],
                year=2019,
            ),
            "time_frame": TimeFrame.rangex("2021-04-01", "2021-06-01"),
            "rng": np.random.default_rng(42),
        }

    actual1 = (
        cdc.InfluenzaFacilityHospitalization().with_context(**context()).evaluate()
    )

    # fmt: off
    expected1 = np.array([
        [("2021-04-04", 53), ("2021-04-04", 0)],
        [("2021-04-11", 58), ("2021-04-11", 0)],
        [("2021-04-18", 56), ("2021-04-18", 0)],
        [("2021-04-25", 59), ("2021-04-25", 0)],
        [("2021-05-02", 56), ("2021-05-02", 0)],
        [("2021-05-09", 49), ("2021-05-09", 0)],
        [("2021-05-16", 65), ("2021-05-16", 0)],
        [("2021-05-23", 67), ("2021-05-23", 0)],
        [("2021-05-30", 53), ("2021-05-30", 0)],
    ], dtype=[("date", "datetime64[D]"), ("value", np.int64)])
    expected_mask1 = np.array([
        [(True, True), (True, True)],
        [(False, False), (True, True)],
        [(False, False), (False, False)],
        [(False, False), (False, False)],
        [(True, True), (False, False)],
        [(True, True), (False, False)],
        [(True, True), (False, False)],
        [(False, False), (True, True)],
        [(True, True), (True, True)],
    ], dtype=[("date", np.bool_), ("value", np.bool_)])
    # fmt: on

    assert np.ma.is_masked(actual1["value"])
    np.testing.assert_array_equal(np.ma.getdata(actual1), expected1, strict=True)
    np.testing.assert_array_equal(np.ma.getmask(actual1), expected_mask1, strict=True)

    actual2 = (
        cdc.InfluenzaFacilityHospitalization(
            fix_missing=0,
            fix_redacted=RandomFix.from_range(1, 3),
        )
        .with_context(**context())
        .evaluate()
    )

    # fmt: off
    expected2 = np.array([
        [("2021-04-04", 54), ("2021-04-04", 3)],
        [("2021-04-11", 58), ("2021-04-11", 2)],
        [("2021-04-18", 56), ("2021-04-18", 0)],
        [("2021-04-25", 59), ("2021-04-25", 0)],
        [("2021-05-02", 58), ("2021-05-02", 0)],
        [("2021-05-09", 54), ("2021-05-09", 0)],
        [("2021-05-16", 66), ("2021-05-16", 0)],
        [("2021-05-23", 67), ("2021-05-23", 3)],
        [("2021-05-30", 57), ("2021-05-30", 3)],
    ], dtype=[("date", "datetime64[D]"), ("value", np.int64)])
    # fmt: on

    assert not np.ma.is_masked(actual2["value"])
    np.testing.assert_array_equal(actual2, expected2, strict=True)

    # The unmasked values from the first request
    # should be unchanged in the second request.
    mask = np.ma.getmask(actual2["value"])
    np.testing.assert_array_equal(actual1["value"][mask], actual2["value"][mask])
