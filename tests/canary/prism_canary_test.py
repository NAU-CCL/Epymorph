import random
from datetime import date, timedelta

import numpy as np

from epymorph.adrio import prism
from epymorph.kit import *


def test_prism_data_source(monkeypatch):
    """
    This canary test fetches one data file from PRISM just to quickly verify that our
    ADRIO implementation appears to be valid still. Naturally caching must be disabled
    for the duration of this test.

    Because this still uses the full ADRIO processing, a test failure may be caused by
    issues not related to the fetching of data. You must rule out other causes by
    running the other PRISM tests, which focus on our logic's handling of the data.
    (The other tests will use cached data to avoid excess stress on PRISM's servers.)
    """
    monkeypatch.setenv("EPYMORPH_CACHE_DISABLED", "true")

    # Pick a random day in 2020 to avoid hitting the same file repeatedly.
    # (2020 was a leap year, so it had 366 days.)
    random_date = date(2020, 1, 1) + timedelta(days=random.randint(0, 365))  # noqa: S311
    print(f"date_loaded: {random_date}")  # noqa: T201

    result = (
        prism.Temperature(temp_var="Minimum")
        .with_context(
            params={
                "centroid": np.array([(-112.0777, 33.4482)], dtype=CentroidDType),
            },
            scope=CustomScope(["PHX"]),
            time_frame=TimeFrame.range(random_date, random_date),
        )
        .evaluate()
    )
    assert result.shape == (1, 1)
    assert np.issubdtype(result.dtype, np.float64)
    assert result[0, 0] > 0
