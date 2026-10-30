import numpy as np
import pytest

from epymorph import initializer as init
from epymorph.attribute import NamePattern
from epymorph.data import ipm, mm
from epymorph.data_type import SimDType
from epymorph.forecasting.dynamic_params import BrownianMotion, GaussianPrior
from epymorph.forecasting.pipeline import (
    PipelineConfig,
    PipelineOutput,
    PipelineSimulator,
    UnknownParam,
)
from epymorph.geography.us_census import StateScope
from epymorph.rume import RUME, SingleStrataRUME
from epymorph.time import TimeFrame


@pytest.fixture
def rume() -> RUME[StateScope]:
    """A very simple RUME with no external data requirements."""
    return SingleStrataRUME.build(
        ipm=ipm.SIRS(),
        mm=mm.No(),
        init=init.SingleLocation(location=0, seed_size=100),
        scope=StateScope.in_states(["04", "35"], year=2020),
        time_frame=TimeFrame.of("2021-01-01", 4),
        params={
            "beta": 0.4,
            "gamma": 1 / 10,
            "xi": 1 / 90,
            "population": [200_000, 100_000],
        },
    )


@pytest.fixture
def filter_output(rume):
    """A mock filter output so we don't have to run a filter."""
    R = 3  # noqa: N806
    N = rume.scope.nodes
    C = rume.ipm.num_compartments
    E = rume.ipm.num_events
    S = rume.num_ticks
    T = rume.time_frame.duration_days
    compartments = np.ones((R, S, N, C), dtype=SimDType)
    param_names = [NamePattern.of("test_0")]

    class TestSimulator(PipelineSimulator):
        def run(self, rng):  # type: ignore
            pass

    simulator = TestSimulator(
        config=PipelineConfig(
            rume=rume,
            num_realizations=R,
            initial_values=compartments[:, 0, :, :],  # type: ignore
            unknown_params={
                param_name: UnknownParam(
                    prior=GaussianPrior(mean=np.log(0.2), standard_deviation=0.5),
                    dynamics=BrownianMotion(volatility=0.1),
                )
                for param_name in param_names
            },
        )
    )
    return PipelineOutput(
        simulator=simulator,
        final_compartments=compartments[:, -1, :, :],
        final_params={param_name: np.ones((R, N)) for param_name in param_names},
        compartments=compartments,
        events=np.ones((R, S, N, E), dtype=SimDType),
        initial=compartments[:, 0, :, :],
        estimated_params={param_name: np.ones((R, T, N)) for param_name in param_names},
    )
