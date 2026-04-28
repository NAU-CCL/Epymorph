"""
Test Centroids movement model for limitations.
"""

from datetime import date

import numpy as np
import pytest

from epymorph.data.mm.centroids import CentroidsClause
from epymorph.data_type import CentroidDType
from epymorph.error import DataAttributeError
from epymorph.geography.custom import CustomScope
from epymorph.simulation import Tick

_SCOPE = CustomScope(["Flagstaff", "Phoenix", "Tucson"])
_CENTROIDS = np.array(
    [
        (-111.651, 35.198),  # Flagstaff
        (-112.074, 33.448),  # Phoenix
        (-110.974, 32.222),  # Tucson
    ],
    dtype=CentroidDType,
)
_POPULATION = np.array([10000, 20000, 30000], dtype=np.int64)
_TICK = Tick(sim_index=0, day=0, date=date(2020, 1, 1), step=0, tau=1 / 3)


def _make_clause(phi: float, commuter_proportion: float = 0.1) -> CentroidsClause:
    return CentroidsClause().with_context(
        scope=_SCOPE,
        params={
            "population": _POPULATION,
            "centroid": _CENTROIDS,
            "phi": phi,
            "commuter_proportion": commuter_proportion,
        },
        rng=np.random.default_rng(42),
    )


def test_evaluate():
    commuter_proportion = 0.1
    clause = _make_clause(phi=40.0, commuter_proportion=commuter_proportion)
    result = clause.evaluate(_TICK)

    assert result.shape == (3, 3)
    assert np.all(result >= 0)

    # multinomial distributes exactly n_commuters[i] draws per row i
    expected_movers = np.floor(_POPULATION * commuter_proportion)
    actual_movers = result.sum(axis=1)
    np.testing.assert_array_equal(actual_movers, expected_movers)


def test_phi_zero_error():
    clause = _make_clause(phi=0.0)
    with pytest.raises(DataAttributeError, match="phi"):
        clause.evaluate(_TICK)


def test_phi_negative_error():
    clause = _make_clause(phi=-5.0)
    with pytest.raises(DataAttributeError, match="phi"):
        clause.evaluate(_TICK)


def test_commuter_proportion_negative_error():
    clause = _make_clause(phi=40.0, commuter_proportion=-0.1)
    with pytest.raises(DataAttributeError, match="commuter_proportion"):
        clause.evaluate(_TICK)


def test_small_phi_no_underflow():
    """
    Very small phi causes distance/phi > 700. This can cause np.exp(-x) to underflow,
    raising an error. So instead we clip distance/phi so that the dispersal kernel
    values can be as close as possible to 0 without causing an error.
    """
    clause = _make_clause(phi=0.001)
    kernel = clause.dispersal_kernel

    # No NaNs of Infs
    assert not np.any(np.isnan(kernel)), "kernel contains NaN"
    assert not np.any(np.isinf(kernel)), "kernel contains Inf"

    # No zero-values demonstrates underflow prevention is working
    assert np.all(kernel > 0), (
        "Dispersal kernel should be all > 0 (clip prevented underflow)"
    )

    # Row-sums should equal 1 (row_normalize worked correctly)
    np.testing.assert_allclose(kernel.sum(axis=1), np.ones(3), atol=1e-10)


def test_kernel_decreases_with_distance():
    """Test: dispersal probability decreases as distance increases."""
    clause = _make_clause(phi=40.0)
    kernel = clause.dispersal_kernel

    # pairwise distances for the centroids (in miles):
    #   [  0.   , 123.436, 209.503]  # noqa: ERA001
    #   [123.436,   0.   , 106.200]  # noqa: ERA001
    #   [209.503, 106.200,   0.   ]  # noqa: ERA001

    # From Flagstaff: Phoenix should be preferred over Tucson
    assert kernel[0, 1] > kernel[0, 2]
    # From Phoenix: Tucson should be preferred over Flagstaff
    assert kernel[1, 2] > kernel[1, 0]
    # From Tucson: Phoenix should be preferred over Flagstaff
    assert kernel[2, 1] > kernel[2, 0]
