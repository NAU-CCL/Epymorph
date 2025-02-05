import numpy as np

from epymorph.adrio import commuting_flows
from epymorph.geography.us_census import CountyScope
from epymorph.time import TimeFrame
from epymorph.util import match


def test_commuters_values():
    # values retrieved manually from ACS commuting flows table1 for 2020
    expected = [
        [14190, 0, 149, 347, 1668],
        [0, 43820, 32, 160, 5],
        [99, 17, 59440, 1160, 525],
        [22, 52, 757, 2059135, 240],
        [706, 14, 1347, 592, 30520],
    ]

    actual = (
        commuting_flows.Commuters()
        .with_context(
            scope=CountyScope.in_counties(
                ["04001", "04003", "04005", "04013", "04017"],
                year=2020,
            ),
            time_frame=TimeFrame.year(2020),
        )
        .evaluate()
    )

    assert match.dtype(int)(actual.dtype)
    assert np.array_equal(expected, actual)
