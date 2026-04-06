import numpy as np
import pytest

from epymorph.adrio import acs5
from epymorph.kit import *


def test_acs5_data_source(monkeypatch):
    """
    This canary test fetches one data point from ACS5 just to quickly verify that our
    ADRIO implementation appears to be valid still. Naturally caching must be disabled
    for the duration of this test.

    Because this still uses the full ADRIO processing, a test failure may be caused by
    issues not related to the fetching of data. You must rule out other causes by
    running the other ACS5 tests, which focus on our logic's handling of the data.
    (The other tests will use cached data to avoid excess stress on ACS5's servers.)
    """
    monkeypatch.setenv("EPYMORPH_CACHE_DISABLED", "true")

    # Test most recent ACS5 year to make sure it is valid.
    most_recent_year = acs5.ACS5_YEARS[-1]
    print(f"{most_recent_year=}")  # noqa: T201

    result = (
        acs5.Population()
        .with_context(
            scope=StateScope.in_states(["AZ"], year=most_recent_year),
        )
        .evaluate()
    )
    assert result.shape == (1,)
    assert np.issubdtype(result.dtype, np.int64)
    assert result[0] > 0


def test_acs5_new_data_available(monkeypatch):
    """
    This checks if there is new data available in ACS5 that we haven't added to our
    list of supported years.
    """
    monkeypatch.setenv("EPYMORPH_CACHE_DISABLED", "true")

    try:
        # If this succeeds at loading data,
        # then it's time to update our list of supported years: fail the test!
        next_year = acs5.ACS5_YEARS[-1] + 1
        print(f"{next_year=}")  # noqa: T201
        acs5.ACS5Client.get_vars(next_year)

        pytest.fail(
            f"There is new data available in ACS5! Add {next_year} "
            "to ACS5_YEARS and the ACS5Year type in epymorph.adrio.acs5"
        )
    except Exception as e:  # noqa: BLE001 (get_vars should really throw a more specific error, but it doesn't yet)
        assert "Unable to load" in str(e)  # noqa: PT017
