import pytest

from epymorph.error import GeographyError
from epymorph.geography import us_tiger
from epymorph.kit import *


def test_us_tiger_data_source(monkeypatch):
    """
    This canary test fetches geographic data from US TIGER to quickly verify that our
    implementation appears to be valid still. Naturally caching must be disabled
    for the duration of this test.
    """
    monkeypatch.setenv("EPYMORPH_CACHE_DISABLED", "true")

    # Test most recent US TIGER year to make sure it is valid.
    most_recent_year = us_tiger.TIGER_YEARS[-1]
    print(f"{most_recent_year=}")  # noqa: T201

    result = us_tiger.get_states_info(year=most_recent_year)
    expected_columns = {"GEOID", "NAME", "STUSPS", "ALAND", "INTPTLAT", "INTPTLON"}

    assert result.shape[0] == 52
    assert set(result.columns) == expected_columns
    assert "Arizona" in result["NAME"].values


def test_us_tiger_new_data_available(monkeypatch):
    """
    This checks if there is new data available in US TIGER that we haven't added to our
    list of supported years.
    """
    monkeypatch.setenv("EPYMORPH_CACHE_DISABLED", "true")

    try:
        # If this succeeds as loading data,
        # then it's time to update our list of supported years: fail the test!
        next_year = us_tiger.TIGER_YEARS[-1] + 1
        print(f"{next_year=}")  # noqa: T201
        monkeypatch.setattr(us_tiger, "TIGER_YEARS", (next_year,))
        us_tiger.get_states_info(year=next_year)

        pytest.fail(
            f"There is new data available in US TIGER! Add {next_year} "
            "to TIGER_YEARS and the TigerYear type in epymorph.geography.us_tiger"
        )
    except GeographyError as e:  # noqa: PT011
        assert "Unable to retrieve" in str(e)  # noqa: PT017
