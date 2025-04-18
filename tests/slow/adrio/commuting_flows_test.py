import numpy as np
import pytest

from epymorph.adrio import commuting_flows
from epymorph.adrio.adrio import ADRIOContextError
from epymorph.geography.us_census import CountyScope


def test_commuters_values():
    # values retrieved manually from ACS commuting flows table1 for 2020
    expected = np.array(
        [
            [14190, 0, 149, 347, 1668],
            [0, 43820, 32, 160, 5],
            [99, 17, 59440, 1160, 525],
            [22, 52, 757, 2059135, 240],
            [706, 14, 1347, 592, 30520],
        ],
        dtype=np.int64,
    )

    actual = (
        commuting_flows.Commuters()
        .with_context(
            scope=CountyScope.in_counties(
                ["04001", "04003", "04005", "04013", "04017"],
                year=2022,
            ),
        )
        .evaluate()
    )

    assert np.dtype(expected.dtype) == np.dtype(actual.dtype)
    assert np.array_equal(expected, actual)


def test_commuters_values_wrong_year():
    with pytest.raises(ADRIOContextError, match="only available for these geo years"):
        (
            commuting_flows.Commuters()
            .with_context(
                scope=CountyScope.in_counties(
                    ["04001", "04003", "04005", "04013", "04017"],
                    year=2020,
                ),
            )
            .evaluate()
        )
