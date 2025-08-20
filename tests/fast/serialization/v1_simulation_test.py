import numpy as np

from epymorph.adrio import us_tiger
from epymorph.kit import *
from epymorph.serialization import v1 as s


def test_simulation():
    """Round-trip a RUME through serialization and make sure it still works in a sim."""
    rume_before = SingleStrataRUME.build(
        ipm=ipm.SIRS(),
        mm=mm.Centroids(),
        init=init.SingleLocation(location=0, seed_size=10_000),
        scope=CountyScope.in_states(["AZ"], year=2020),
        time_frame=TimeFrame.rangex("2020-01-01", "2020-03-01"),
        params={
            "ipm::beta": 0.4,
            "ipm::gamma": 1 / 10,
            "ipm::xi": 1 / 90,
            "population": np.arange(1, 16) * 100_000,
            "centroid": us_tiger.InternalPoint(),
        },
    )

    # First step: run a sim and check it for basic validity.
    sim1 = BasicSimulator(rume_before)
    out1 = sim1.run(rng_factory=default_rng(42))

    # Compartments can never be less than zero
    assert out1.compartments.min() >= 0
    # Events can never be less than zero
    assert out1.events.min() >= 0
    # All compartments should be greater than zero at some time, in all locations
    assert np.all(out1.compartments.max(axis=0) > 0)
    # All events should be greater than zero at some time, in all locations
    assert np.all(out1.events.max(axis=0) > 0)

    # Second step: serialize and deserialize the RUME,
    # then run a new sim with that RUME and check for consistent results.
    rume_after = s.deserialize(s.serialize(rume_before))

    sim2 = BasicSimulator(rume_after)
    out2 = sim2.run(rng_factory=default_rng(42))

    # Should still produce the same results with the same RNG
    np.testing.assert_array_equal(out1.events, out2.events)
    np.testing.assert_array_equal(out1.compartments, out2.compartments)
