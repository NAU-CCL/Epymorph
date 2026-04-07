import dataclasses

import numpy as np
import pytest

from epymorph.attribute import AttributeDef, NamePattern
from epymorph.compartment_model import CompartmentModel, compartment, edge
from epymorph.data.ipm.pei import Pei as PeiIPM
from epymorph.data.ipm.sirh import SIRH
from epymorph.data.mm.no import No as NoMM
from epymorph.data.mm.pei import Pei as PeiMM
from epymorph.data.pei import pei_commuters, pei_humidity, pei_population, pei_scope
from epymorph.data_shape import Shapes
from epymorph.data_type import SimDType
from epymorph.error import (
    IPMSimInvalidForkError,
    IPMSimLessThanZeroError,
    IPMSimNaNError,
    MMSimError,
)
from epymorph.geography.custom import CustomScope
from epymorph.initializer import SingleLocation
from epymorph.rume import SingleStrataRUME
from epymorph.simulator.basic.fn_simulator import FunctionSimulator
from epymorph.time import TimeFrame


@pytest.fixture
def pei_rume() -> SingleStrataRUME:
    return SingleStrataRUME.build(
        ipm=PeiIPM(),
        mm=PeiMM(),
        init=SingleLocation(location=0, seed_size=10_000),
        scope=pei_scope,
        time_frame=TimeFrame.of("2015-01-01", 10),
        params={
            "ipm::infection_duration": 4,
            "ipm::immunity_duration": 90,
            "mm::move_control": 0.9,
            "mm::theta": 0.1,
            "*::population": pei_population,
            "*::humidity": pei_humidity,
            "*::commuters": pei_commuters,
        },
    )


def test_pei(pei_rume):
    sim = FunctionSimulator(pei_rume)

    out1 = sim.run(np.random.default_rng(42))

    np.testing.assert_array_equal(
        out1.initial[:, 1],
        np.array([10_000, 0, 0, 0, 0, 0], dtype=SimDType),
        "Output should contain accurate initials.",
    )

    assert out1.compartments[:, :, 0].max() > 0, (
        "S compartment should be greater than zero at some point in the sim.",
    )
    assert out1.compartments[:, :, 1].max() > 0, (
        "I compartment should be greater than zero at some point in the sim.",
    )
    assert out1.compartments[:, :, 2].max() > 0, (
        "R compartment should be greater than zero at some point in the sim.",
    )
    assert out1.events[:, :, 0].max() > 0, (
        "S-to-I event should be greater than zero at some point in the sim.",
    )
    assert out1.events[:, :, 1].max() > 0, (
        "I-to-R event should be greater than zero at some point in the sim.",
    )
    assert out1.events[:, :, 2].max() > 0, (
        "R-to-S event should be greater than zero at some point in the sim.",
    )

    assert out1.compartments.min() >= 0, "Compartments can never be less than zero."
    assert out1.events.min() >= 0, "Events can never be less than zero."

    out2 = sim.run(np.random.default_rng(42))

    np.testing.assert_array_equal(
        out1.events,
        out2.events,
        "Running the sim twice with the same RNG should yield the same events.",
    )

    np.testing.assert_array_equal(
        out1.compartments,
        out2.compartments,
        ("Running the sim twice with the same RNG should yield the same compartments."),
    )


def test_less_than_zero_err(pei_rume):
    """
    Test exception handling for a negative rate value due to a negative parameter
    """
    rume = dataclasses.replace(
        pei_rume,
        params={
            **pei_rume.params,
            NamePattern.of("ipm::immunity_duration"): -100,  # negative parameter!
        },
    )

    sim = FunctionSimulator(rume)

    with pytest.raises(IPMSimLessThanZeroError) as e:
        sim.run(np.random.default_rng(42))

    err_msg = str(e.value)
    assert "less than zero" in err_msg
    assert "immunity_duration: -100.0" in err_msg


def test_divide_by_zero_err():
    """Test exception handling for a divide by zero (NaN) error"""

    class Sirs(CompartmentModel):
        compartments = [
            compartment("S"),
            compartment("I"),
            compartment("R"),
        ]

        requirements = [
            AttributeDef("beta", type=float, shape=Shapes.TxN),
            AttributeDef("gamma", type=float, shape=Shapes.TxN),
            AttributeDef("xi", type=float, shape=Shapes.TxN),
        ]

        def edges(self, symbols):
            [S, I, R] = symbols.all_compartments  # noqa: N806
            [β, γ, ξ] = symbols.all_requirements

            # N is NOT protected by Max(1, ...) here
            N = S + I + R  # type: ignore

            return [
                edge(S, I, rate=β * S * I / N),
                edge(I, R, rate=γ * I),
                edge(R, S, rate=ξ * R),
            ]

    rume = SingleStrataRUME.build(
        ipm=Sirs(),
        mm=NoMM(),
        init=SingleLocation(location=1, seed_size=5),
        scope=CustomScope(np.array(["a", "b", "c"])),
        time_frame=TimeFrame.of("2015-01-01", 150),
        params={
            "*::mm::phi": 40.0,
            "*::ipm::beta": 0.4,
            "*::ipm::gamma": 1 / 5,
            "*::ipm::xi": 1 / 100,
            "*::*::population": np.array([0, 10, 20], dtype=np.int64),
        },
    )

    sim = FunctionSimulator(rume)
    with pytest.raises(IPMSimNaNError) as e:
        sim.run(np.random.default_rng(1))

    err_msg = str(e.value)
    assert "transition rate was NaN" in err_msg
    assert "S: 0" in err_msg
    assert "I: 0" in err_msg
    assert "R: 0" in err_msg
    assert "S → I: I*S*beta/(I + R + S)" in err_msg


def test_negative_probs_error(pei_rume):
    """Test for handling negative probability error"""
    rume = SingleStrataRUME.build(
        ipm=SIRH(),
        mm=NoMM(),
        init=SingleLocation(location=1, seed_size=5),
        scope=pei_scope,
        time_frame=TimeFrame.of("2015-01-01", 150),
        params={
            **pei_rume.params,
            "beta": 0.4,
            "gamma": 1 / 5,
            "xi": 1 / 100,
            "hospitalization_prob": -1 / 5,
            "hospitalization_duration": 15,
        },
    )

    sim = FunctionSimulator(rume)
    with pytest.raises(IPMSimInvalidForkError) as e:
        sim.run(np.random.default_rng(1))

    err_msg = str(e.value)
    assert "fork transition is invalid" in err_msg
    assert "hospitalization_prob: -0.2" in err_msg
    assert "hospitalization_duration: 15" in err_msg
    assert "I → (H,R): I*gamma" in err_msg
    assert "Probabilities: hospitalization_prob, 1 - hospitalization_prob" in err_msg


def test_mm_clause_error(pei_rume):
    """Test for handling invalid movement model clause application"""
    rume = SingleStrataRUME.build(
        ipm=PeiIPM(),
        mm=PeiMM(),
        init=SingleLocation(location=1, seed_size=5),
        scope=pei_scope,
        time_frame=TimeFrame.of("2015-01-01", 150),
        params={
            **pei_rume.params,
            "ipm::infection_duration": 40.0,
            "ipm::immunity_duration": 0.4,
            "mm::move_control": 0.4,
            "mm::theta": -5.0,
        },
    )

    sim = FunctionSimulator(rume)
    with pytest.raises(MMSimError, match="Error from applying clause 'Dispersers'"):
        sim.run(np.random.default_rng(1))
