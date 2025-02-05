"""
Testing our us_census functions for loading canonical sets of IDs for
Census granularities from state to block group for all supported TIGER years
(2000 and 2009-2023).
"""

import numpy as np
import pytest

import epymorph.geography.us_census as c
import epymorph.geography.us_tiger as t


@pytest.mark.parametrize("year", t.TIGER_YEARS)
def test_year(year: int):
    """
    Test that at each level of granularity (above block group) each node contains
    at least one child node. That is every state should contain a county, every county
    a tract, and every tract a block group. Otherwise we know something is missing.
    """

    # 1. test that we have 52 states
    states = t.get_states(year).geoid

    assert len(states) == 52

    # 2. test that each state contains at least one county
    counties = t.get_counties(year).geoid
    counties_by_state = c.STATE.grouped(np.array(counties))

    for x in states:
        assert (
            len(counties_by_state.get(x, [])) > 0
        ), f"State {x} does not have at least one county."

    # 3. test that each county contains at least one tract
    tracts = t.get_tracts(year).geoid
    tracts_by_county = c.COUNTY.grouped(np.array(tracts))

    for x in counties:
        assert (
            len(tracts_by_county.get(x, [])) > 0
        ), f"County {x} does not have at least one tract."

    # 4. test that each tract contains at least one block group
    cbgs = t.get_block_groups(year).geoid
    cbgs_by_tract = c.TRACT.grouped(np.array(cbgs))

    for x in tracts:
        assert (
            len(cbgs_by_tract.get(x, [])) > 0
        ), f"Tract {x} does not have at least one block group."
