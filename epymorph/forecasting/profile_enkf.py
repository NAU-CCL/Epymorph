
import numpy as np
import matplotlib.pyplot as plt
from epymorph.forecasting.likelihood import Gaussian
from epymorph.attribute import NamePattern
from epymorph.kit import *

from epymorph.forecasting.pipeline import (
    FromRUME,
    Observations,
    ModelLink,
    UnknownParam
)

from epymorph.simulation import Context
from epymorph.adrio import acs5, us_tiger
from epymorph.forecasting.param_transformations import ExponentialTransform
from epymorph.forecasting.dynamic_params import GaussianPrior, OrnsteinUhlenbeck

from typing_extensions import override
from epymorph.initializer import _POPULATION_ATTR
from epymorph.initializer import *

class RandomLocationsAndRandomSeed(SeededInfection):
    """
    Seed an random number of infected in a number of randomly selected locations.

    Requires "population" as a data attribute.

    Parameters
    ----------
    num_locations :
        The number of locations to choose.
    seed_max :
        The maximum number of individuals to infect.
    initial_compartment :
        Which compartment (by index or name) is "not infected", where most individuals
        start out.
    infection_compartment :
        Which compartment (by index or name) will be seeded as the initial infection.
    """

    requirements = (_POPULATION_ATTR,)

    num_locations: int
    """The number of locations to choose (randomly)."""
    seed_max: int
    """The maximum number of individuals to infect, drawn uniformly on [0,seed_max]."""

    def __init__(
        self,
        num_locations: int,
        seed_max: int,
        initial_compartment: int | str = SeededInfection.DEFAULT_INITIAL,
        infection_compartment: int | str = SeededInfection.DEFAULT_INFECTION,
    ):
        super().__init__(initial_compartment, infection_compartment)
        self.num_locations = num_locations
        self.seed_max = seed_max

    @override
    def evaluate(self) -> SimArray:
        """
        Evaluate the initializer in the current context.

        Returns
        -------
        :
            The initial populations for each node and IPM compartment.
        """
        N = self.scope.nodes
        if not 0 < self.num_locations <= N:
            err = (
                "Initializer argument 'num_locations' must be "
                f"a value from 1 up to the number of locations ({N})."
            )
            raise InitError(err)

        indices = np.arange(N, dtype=np.intp)
        selection = self.rng.choice(indices, self.num_locations)
        seed_size = self.rng.integers(low = 0, high = self.seed_max, endpoint=True)

        sub = IndexedLocations(
            selection=selection,
            seed_size=seed_size,
            initial_compartment=self.initial_compartment,
            infection_compartment=self.infection_compartment,
        )
        return self.defer(sub)


scope = StateScope.all(year=2015)
mm = mm.Centroids()
ipm = ipm.SIRH()
sim_time_frame = TimeFrame.of("2015-01-01", 182)
my_rng = np.random.default_rng(0)

'''Generate a random time dependent beta'''
log_beta_damping = 1/35 * np.ones(scope.nodes)
log_beta_mean = np.log(0.15) * np.ones(scope.nodes)
log_beta_standard_deviation = 0.25 * np.ones(scope.nodes)
initial_log_beta = np.log(0.15) * np.ones(scope.nodes)

delta_t = 1. 

A = np.exp(-log_beta_damping * delta_t)
M = log_beta_mean * (np.exp(-log_beta_damping * delta_t) - 1)
C = log_beta_standard_deviation * np.sqrt(1-np.exp(-2*log_beta_damping * delta_t))

log_beta = np.zeros((scope.nodes,sim_time_frame.duration_days,))
log_beta[:,0] = initial_log_beta

for day in range(1,sim_time_frame.duration_days): 
    log_beta[:,day] = A * log_beta[:,day-1] - M + C * my_rng.normal(size = (scope.nodes,))

beta = np.exp(log_beta)

rume = SingleStrataRUME.build(
    ipm=ipm,
    mm=mm,
    # Describe the geographic scope of our simulation:
    scope= scope,
    # Create a SingleLocation initializer
    init=init.RandomLocations(scope.nodes,
                                seed_size=1000),
    # Set the time-frame to simulate
    time_frame=sim_time_frame,
    # Provide model parameter values
    params={
        "beta": beta.T,
        "gamma":0.1,
        "xi":1/90,
        "phi":10,
        "hospitalization_prob":0.05,
        "hospitalization_duration":5,
        # Geographic data can be loaded using ADRIOs
        "centroid": us_tiger.InternalPoint(),
        "population": acs5.Population(),
        "label":us_tiger.Name()
    },
)


# Construct a simulator for the RUME
sim = BasicSimulator(rume)

# Run inside a sim_messaging context to display a nice progress bar
with sim_messaging():
    # Run and save the simulation Output object for later
    out = sim.run(
        # Use a seeded RNG (for the sake of keeping this notebook's results consistent)
        # This parameter is optional; by default a new RNG is constructed for each run
        # using numpy's default_rng
        rng_factory=lambda : my_rng
    )


# Plot the compartment values in States throughout the simulation.
from epymorph.adrio import csv
from epymorph.tools.data import munge
import seaborn as sns


cases_df = munge(
    out,
    quantity=rume.ipm.select.events("I->H"),
    time=rume.time_frame.select.all().group("day").agg(),
    geo=rume.scope.select.all(),
)

cases_df.columns = ['date','geoid','value']

cases_df.to_csv('./synthetic_data.csv',index=False)

csvadrio = csv.CSVFileAxN(
    file_path='./synthetic_data.csv',
    dtype=np.int64,
    key_col=1,
    key_type="geoid",
    time_col=0,
    data_col=2,
    skiprows=1
)

from epymorph.forecasting.param_transformations import ExponentialTransform
from epymorph.forecasting.pipeline import EnsembleKalmanFilterSimulator


num_realizations = 100

enkf_rume = SingleStrataRUME.build(
    # Load the Pei IPM
    ipm=ipm,
    # Load the Pei MM
    mm=mm,
    # Describe the geographic scope of our simulation:
    scope=scope,
    # Create a SingleLocation initializer
    init=RandomLocationsAndRandomSeed(scope.nodes,1000),
    # Set the time-frame to simulate
    time_frame=sim_time_frame,
    # Provide model parameter values
    params={
        "beta": ExponentialTransform("log_beta"),
        "gamma":0.1,
        "xi":1/90,
        "phi":10,
        "hospitalization_prob":0.05,
        "hospitalization_duration":5,
        # Geographic data can be loaded using ADRIOs
        "centroid": us_tiger.InternalPoint(),
        "population": acs5.Population()
    },
)

my_observations = Observations(
    source=csvadrio,
    model_link=ModelLink(
        geo=enkf_rume.scope.select.all(),
        time=enkf_rume.time_frame.select.all().group("day").agg(),
        quantity=enkf_rume.ipm.select.events("I->H"),
    ),
    likelihood=Gaussian(10.0)
)

my_unknown_params = {
    "log_beta": UnknownParam(
        prior=GaussianPrior(
            mean=log_beta_mean,
            standard_deviation=log_beta_standard_deviation,
        ),
        dynamics=OrnsteinUhlenbeck(
            damping=log_beta_damping,
            mean=log_beta_mean,
            standard_deviation=log_beta_standard_deviation,
        ),
    )
}

enkf_simulator = EnsembleKalmanFilterSimulator(
        config=FromRUME(enkf_rume,num_realizations,unknown_params = my_unknown_params),
        observations=my_observations,
        save_trajectories=True,
    )

enkf_output = enkf_simulator.run(rng=my_rng)
