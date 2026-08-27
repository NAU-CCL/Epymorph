import numpy as np
import pandas as pd
import pytest

from epymorph import initializer as init
from epymorph.attribute import NamePattern
from epymorph.data import ipm, mm
from epymorph.data.ipm.sirh import SIRH
from epymorph.data.mm.no import No
from epymorph.data_type import SimDType
from epymorph.forecasting.dynamic_params import BrownianMotion, GaussianPrior, Static
from epymorph.forecasting.ensemble_kalman_filter import EnsembleKalmanFilterSimulator
from epymorph.forecasting.filter import ModelLink, Observations
from epymorph.forecasting.likelihood import GaussianLikelihood
from epymorph.forecasting.pipeline import (
    PipelineConfig,
    PipelineOutput,
    PipelineSimulator,
    UnknownParam,
    munge_pipeline_output,
)
from epymorph.geography.us_census import StateScope
from epymorph.initializer import Explicit
from epymorph.rume import RUME, SingleStrataRUME
from epymorph.simulator.basic.basic_simulator import BasicSimulator
from epymorph.time import EveryNDays, TimeFrame
from epymorph.tools.data import munge
from epymorph.util import to_date_value_array


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


def test_basic_pipeline_munge(rume, filter_output):
    # Test with a selection in each axis
    # and for parameters as well as compartments and events
    realization = filter_output.select.all()
    time = rume.time_frame.select.days(0, 1)
    geo = rume.scope.select.all()
    quantity = filter_output.param_select.by_name("test_0")

    # Test using munge on a parameter value
    actual = munge_pipeline_output(filter_output, realization, geo, time, quantity)
    expected = pd.DataFrame(
        {
            "realization": np.repeat(
                np.arange(0, realization.num_realizations, 1), geo.scope.nodes * 2
            ),
            "time": np.tile(
                np.repeat(np.array([0, 1]), geo.scope.nodes),
                realization.num_realizations,
            ),
            "geo": np.tile(geo.scope.node_ids, realization.num_realizations * 2),
            "*::*::test_0": np.ones(geo.scope.nodes * realization.num_realizations * 2),
        }
    )

    # Don't check the dtype due to platform differences. It is relatively
    # inconsequential whether realization indices are int32 or int64.
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)

    # Test using munge on a compartment value
    quantity = rume.ipm.select.compartments("S")
    actual = munge_pipeline_output(filter_output, realization, geo, time, quantity)

    expected = pd.DataFrame(
        {
            "realization": np.repeat(
                np.arange(0, realization.num_realizations, 1), geo.scope.nodes * 2
            ),
            "time": np.tile(
                np.repeat(np.array([0, 1]), geo.scope.nodes),
                realization.num_realizations,
            ),
            "geo": np.tile(geo.scope.node_ids, realization.num_realizations * 2),
            "S": np.ones(geo.scope.nodes * realization.num_realizations * 2),
        }
    )
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


def test_aggregation_pipeline_munge(rume, filter_output):
    # Test specific aggregations across the realization axis

    # Test aggregation for the mean
    # This primarily makes sure the multi-index is there with the correct labels.
    realization = filter_output.select.all().agg(["mean"])
    time = rume.time_frame.select.days(0, 1)
    geo = rume.scope.select.all()
    quantity = filter_output.param_select.by_name("test_0")

    actual = munge_pipeline_output(filter_output, realization, geo, time, quantity)

    data = {
        "time": np.repeat(np.array([0, 1], dtype=np.int64), geo.scope.nodes),
        "geo": np.tile(geo.scope.node_ids, 2),
        "value": np.ones(geo.scope.nodes * 2),
    }

    columns = pd.MultiIndex.from_tuples(
        [("time", ""), ("geo", ""), ("*::*::test_0", "mean")]
    )

    expected = pd.DataFrame(data.values(), index=columns).T
    expected = expected.astype(
        {("time", ""): "int64", ("*::*::test_0", "mean"): "float64"}
    )

    # Don't check the dtype due to platform differences. It is relatively
    # inconsequential whether time ticks are int32 or int64.
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)

    # Test aggregation for the std
    # This primarily makes sure the multi-index is there with the correct labels.
    realization = filter_output.select.all().agg(["std"])
    data["value"] = np.zeros(geo.scope.nodes * 2)

    actual = munge_pipeline_output(filter_output, realization, geo, time, quantity)

    columns = pd.MultiIndex.from_tuples(
        [("time", ""), ("geo", ""), ("*::*::test_0", "std")]
    )

    expected = pd.DataFrame(data.values(), index=columns).T
    expected = expected.astype(
        {("time", ""): "int64", ("*::*::test_0", "std"): "float64"}
    )

    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


def test_ensemble_kalman_filter():
    rume = SingleStrataRUME.build(
        ipm=SIRH(),
        mm=No(),
        scope=StateScope.in_states(["AZ"], year=2020),
        init=Explicit(np.array([[4_500_000, 499_500, 2_000_000, 500]], dtype=np.int64)),
        time_frame=TimeFrame.of(start_date="2021-01-01", duration_days=7),
        params={
            "beta": 0.4,
            "gamma": 1 / 5,
            "xi": 0,
            "hospitalization_prob": 100 / 100_000,
            "hospitalization_duration": 5.0,
        },
    )

    out = BasicSimulator(rume=rume).run(rng_factory=lambda: np.random.default_rng(0))

    geo = rume.scope.select.all()
    time = rume.time_frame.select.all().group(EveryNDays(7)).agg("sum")
    quantity = rume.ipm.select.events("I->H")

    munge_df = munge(out, geo=geo, time=time, quantity=quantity)

    source = to_date_value_array(
        dates=munge_df.iloc[:, 0].to_numpy(),
        values=munge_df.iloc[:, 2].to_numpy().reshape(-1, 1),
    )

    observations = Observations(
        source=source,
        model_link=ModelLink(geo=geo, time=time, quantity=quantity),
        likelihood=GaussianLikelihood(50),
    )

    num_realizations = 9
    initial_beta = np.linspace(0.1, 0.5, num_realizations).reshape(-1, 1)

    filter_sim = EnsembleKalmanFilterSimulator(
        config=PipelineConfig.from_rume(
            rume,
            num_realizations=num_realizations,
            unknown_params={
                "beta": UnknownParam(prior=initial_beta.copy(), dynamics=Static())
            },
        ),
        observations=observations,
    )

    filter_out = filter_sim.run(rng=np.random.default_rng(0))

    prior_infected = filter_out.compartments[:, -1, :, 1]
    posterior_infected = filter_out.final_compartments[:, :, 1]

    # The true beta is in the upper range of the prior, so the posterior infection
    # should increase relative to the prior infection.
    np.testing.assert_array_less(prior_infected.mean(), posterior_infected.mean())

    # The uncertainty of the posterior should always be lower than the prior.
    np.testing.assert_array_less(posterior_infected.std(), prior_infected.std())

    prior_beta = filter_out.estimated_params[NamePattern.of("beta")][:, -1, :]
    posterior_beta = filter_out.final_params[NamePattern.of("beta")]

    # Beta is modeled as a static parameter, so the prior should be equal to the
    # initial value.
    np.testing.assert_array_equal(prior_beta, initial_beta)

    # The true beta is in the upper range of the prior, so the posterior should be
    # larger.
    np.testing.assert_array_less(prior_beta.mean(), posterior_beta.mean())

    # The uncertainty of the posterior should be lower than the prior.
    np.testing.assert_array_less(posterior_beta.std(), prior_beta.std())
