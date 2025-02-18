from pathlib import Path

import numpy as np

from epymorph import *  # noqa: F403
from epymorph.adrio import acs5, csv
from epymorph.data.ipm.sirh import SIRH
from epymorph.data.mm.no import No
from epymorph.geography.us_census import StateScope
from epymorph.initializer import Proportional
from epymorph.parameter_fitting.distribution import Uniform
from epymorph.parameter_fitting.dynamics import GeometricBrownianMotion
from epymorph.parameter_fitting.filter.particle_filter import ParticleFilter
from epymorph.parameter_fitting.likelihood import Poisson
from epymorph.parameter_fitting.output import ParticleFilterOutput
from epymorph.parameter_fitting.particlefilter_simulation import FilterSimulation
from epymorph.parameter_fitting.perturbation import Calvetti
from epymorph.parameter_fitting.utils.observations import ModelLink, Observations
from epymorph.parameter_fitting.utils.parameter_estimation import EstimateParameters
from epymorph.rume import SingleStrataRUME
from epymorph.time import EveryNDays, TimeFrame


class RunPF:
    """Runs the particle filter on some default set of parameters"""

    def __init__(self):
        pass

    def run(self) -> tuple[FilterSimulation, ParticleFilterOutput]:
        duration = 7 * 53 + 1
        t = np.arange(0, duration)
        true_beta = 0.03 * np.cos(t * 2 * np.pi / (365)) + 0.28

        rume = SingleStrataRUME.build(
            ipm=SIRH(),
            mm=No(),
            scope=StateScope.in_states(["AZ"], year=2015),
            init=Proportional(ratios=np.array([9999, 1, 0, 0], dtype=np.int64)),
            time_frame=TimeFrame.of("2022-10-01", 7 * 53 + 1),
            params={
                "beta": true_beta,
                "gamma": 0.25,
                "xi": 1 / 365,  # 0.0111,
                "hospitalization_prob": 0.01,
                "hospitalization_duration": 5.0,
                "population": acs5.Population(),
            },
        )

        csvadrio = csv.CSVTimeSeries(
            file_path=Path("./doc/devlog/data/temp_synthetic_data.csv"),
            time_col=0,
            time_frame=rume.time_frame,
            key_col=1,
            data_col=2,
            data_type=int,
            key_type="geoid",
            skiprows=1,
        )

        quantity_selection = rume.ipm.select.events("I->H")
        time_selection = rume.time_frame.select.all().group(EveryNDays(7)).agg()
        geo_selection = rume.scope.select.all()

        observations = Observations(
            source=csvadrio,
            model_link=ModelLink(
                quantity=quantity_selection,
                time=time_selection,
                geo=geo_selection,
            ),
            likelihood=Poisson(),
        )

        filter_type = ParticleFilter(num_particles=500)

        params_space = {
            "beta": EstimateParameters.TimeVarying(
                distribution=Uniform(a=0.05, b=0.5),
                dynamics=GeometricBrownianMotion(volatility=0.04),
            ),
            "xi": EstimateParameters.Static(
                distribution=Uniform(a=0.001, b=0.01),
                perturbation=Calvetti(a=0.9),
            ),
        }

        sim = FilterSimulation(
            rume=rume,
            observations=observations,
            filter_type=filter_type,
            params_space=params_space,
        )

        rng = np.random.default_rng(seed=1)

        output = sim.run(rng=rng)

        return sim, output
