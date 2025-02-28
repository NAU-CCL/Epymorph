import numpy as np

from epymorph.adrio.acs5 import ACS5Client, DissimilarityIndex
from epymorph.geography.us_census import (
    BlockGroupScope,
    CountyScope,
    StateScope,
    TractScope,
)

###############
# ACS5 CLIENT #
###############


def test_acs5_client_url():
    assert ACS5Client.url(2010) == "https://api.census.gov/data/2010/acs/acs5"
    assert ACS5Client.url(2021) == "https://api.census.gov/data/2021/acs/acs5"


def test_acs5_client_make_queries_state():
    actual = ACS5Client.make_queries(StateScope.all(year=2020))
    expected = [{"for": "state:*"}]
    assert actual == expected

    actual = ACS5Client.make_queries(
        StateScope.in_states(["AZ", "CO", "NM"], year=2020)
    )
    expected = [{"for": "state:04,08,35"}]
    assert actual == expected


def test_acs5_client_make_queries_county():
    actual = ACS5Client.make_queries(
        CountyScope.in_states(["AZ", "CO", "NM"], year=2019)
    )
    expected = [{"for": "county:*", "in": "state:04,08,35"}]
    assert actual == expected

    actual = ACS5Client.make_queries(
        CountyScope.in_counties(["Maricopa, AZ", "Bernalillo, NM"], year=2018)
    )
    expected = [
        {"for": "county:013", "in": "state:04"},
        {"for": "county:001", "in": "state:35"},
    ]
    assert actual == expected


def test_acs5_client_make_queries_tract():
    actual = ACS5Client.make_queries(TractScope.in_states(["AZ", "NM"], year=2017))
    expected = [{"for": "tract:*", "in": "state:04,35 county:*"}]
    assert actual == expected

    actual = ACS5Client.make_queries(
        TractScope.in_counties(
            ["Maricopa, AZ", "Bernalillo, NM", "Coconino, AZ"],
            year=2016,
        )
    )
    expected = [
        {"for": "tract:*", "in": "state:04 county:005,013"},
        {"for": "tract:*", "in": "state:35 county:001"},
    ]
    assert actual == expected

    actual = ACS5Client.make_queries(
        TractScope.in_tracts(
            ["04013010101", "04013010102", "04005000100", "35001003714"],
            year=2015,
        )
    )
    expected = [
        {"for": "tract:000100", "in": "state:04 county:005"},
        {"for": "tract:010101,010102", "in": "state:04 county:013"},
        {"for": "tract:003714", "in": "state:35 county:001"},
    ]
    assert actual == expected


def test_acs5_client_make_queries_bg():
    actual = ACS5Client.make_queries(BlockGroupScope.in_states(["AZ", "NM"], year=2014))
    expected = [
        {"for": "block group:*", "in": "state:04 county:* tract:*"},
        {"for": "block group:*", "in": "state:35 county:* tract:*"},
    ]
    assert actual == expected

    actual = ACS5Client.make_queries(
        BlockGroupScope.in_counties(
            ["Maricopa, AZ", "Bernalillo, NM", "Coconino, AZ"],
            year=2014,
        )
    )
    expected = [
        {"for": "block group:*", "in": "state:04 county:005,013 tract:*"},
        {"for": "block group:*", "in": "state:35 county:001 tract:*"},
    ]
    assert actual == expected

    actual = ACS5Client.make_queries(
        BlockGroupScope.in_tracts(
            ["04013010101", "04013010102", "04005000100", "35001003714"],
            year=2013,
        )
    )
    expected = [
        {"for": "block group:*", "in": "state:04 county:005 tract:000100"},
        {"for": "block group:*", "in": "state:04 county:013 tract:010101,010102"},
        {"for": "block group:*", "in": "state:35 county:001 tract:003714"},
    ]
    assert actual == expected

    actual = ACS5Client.make_queries(
        BlockGroupScope.in_block_groups(
            [
                "040130101013",
                "040130101011",
                "040130101022",
                "040050001003",
                "350010037142",
                "350010037144",
            ],
            year=2012,
        )
    )
    expected = [
        {"for": "block group:3", "in": "state:04 county:005 tract:000100"},
        {"for": "block group:1,3", "in": "state:04 county:013 tract:010101"},
        {"for": "block group:2", "in": "state:04 county:013 tract:010102"},
        {"for": "block group:2,4", "in": "state:35 county:001 tract:003714"},
    ]
    assert actual == expected


#######################
# DISSIMILARITY INDEX #
#######################


def test_dissimilarity_index_mocked():
    from unittest.mock import MagicMock

    # following the example from:
    # https://coascenters.howard.edu/dissimilarity-index-tutorial

    adrio = DissimilarityIndex(
        majority_pop="White",
        minority_pop="Black",
    ).with_context(
        # Example uses a hypothetical location with five subdivisions;
        # we need to provide a real census scope however.
        # Conveniently, Hawaii has exactly five counties.
        scope=StateScope.in_states(["HI"], year=2021),
    )

    # Example uses these population numbers:
    high_maj = np.array([500])
    high_min = np.array([300])
    low_maj = np.array([10, 40, 100, 200, 150])
    low_min = np.array([50, 200, 10, 30, 10])
    mock_defer = MagicMock(side_effect=[high_maj, high_min, low_maj, low_min])
    adrio.defer = mock_defer  # type: ignore

    np.testing.assert_almost_equal(
        adrio.evaluate(),
        np.array([0.73]),
        decimal=2,
    )
