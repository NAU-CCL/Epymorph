from typing import Sequence

import numpy as np

from epymorph.adrio import us_tiger
from epymorph.kit import *
from epymorph.serialization import v1 as s


def test_simulation():
    """
    Round-trip a RUME through serialization and make sure it still works in a sim.
    """
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


def test_simulation_2():
    """
    Round-trip a RUME through serialization and make sure it still works in a sim.
    But this time with a multi-strata RUME.
    """

    def meta_edges(symbols: MultiStrataModelSymbols) -> Sequence[TransitionDef]:
        [sa, ia, _] = symbols.strata_compartments("alpha")
        [sb, ib, _, _] = symbols.strata_compartments("beta")
        [meta_param_a] = symbols.all_meta_requirements
        return [
            edge(sa, ia, rate=meta_param_a / 2),
            edge(sb, ib, rate=meta_param_a / 2),
        ]

    rume_before = MultiStrataRUME.build(
        strata=[
            GPM(
                name="alpha",
                ipm=ipm.SIRS(),
                mm=mm.Centroids(),
                init=init.SingleLocation(0, seed_size=1000),
            ),
            GPM(
                name="beta",
                ipm=ipm.SIRH(),
                mm=mm.Flat(),
                init=init.NoInfection(),
            ),
        ],
        meta_requirements=[AttributeDef("meta_param_a", int, Shapes.N)],
        meta_edges=meta_edges,
        scope=CountyScope.in_states(["AZ"], year=2020),
        time_frame=TimeFrame.rangex("2020-01-01", "2020-03-01"),
        params={
            "gpm:alpha::ipm::beta": 0.4,
            "gpm:alpha::ipm::gamma": 1 / 10,
            "gpm:alpha::ipm::xi": 1 / 90,
            "gpm:beta::ipm::beta": 0.44,
            "gpm:beta::ipm::gamma": 1 / 9,
            "gpm:beta::ipm::xi": 1 / 80,
            "gpm:beta::ipm::hospitalization_prob": 0.1,
            "gpm:beta::ipm::hospitalization_duration": 4,
            "meta::ipm::meta_param_a": 42,
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
