"""A jupyter notebook detailing the use of the
particle filter to fit a model with a synthetic data set."""

import numpy as np

from epymorph.adrio import acs5, us_tiger
from epymorph.forecasting.likelihood import Poisson
from epymorph.forecasting.pipeline import (
    FromRUME,
    ModelLink,
    Observations,
    ParticleFilterSimulator,
)
from epymorph.kit import *

my_rng = np.random.default_rng(1)

rume = SingleStrataRUME.build(
    # Load the Pei IPM
    ipm=ipm.SIRH(),
    # Load the Pei MM
    mm=mm.Centroids(),
    # Describe the geographic scope of our simulation:
    scope=CountyScope.in_counties(
        ["King, WA", "Whatcom, WA", "Pierce, WA", "Snohomish, WA"], year=2015
    ),
    # Create a SingleLocation initializer
    init=init.SingleLocation(location=0, seed_size=100),
    # Set the time-frame to simulate
    time_frame=TimeFrame.of("2015-01-01", 150),
    # Provide model parameter values
    params={
        "beta": 0.3,
        "gamma": 0.1,
        "xi": 1 / 90,
        "phi": 10,
        "hospitalization_prob": 0.05,
        "hospitalization_duration": 5,
        # Geographic data can be loaded using ADRIOs
        "centroid": us_tiger.InternalPoint(),
        "population": acs5.Population(),
    },
)

# Construct a simulator for the RUME
sim = BasicSimulator(rume)


# Run and save the simulation Output object for later
out = sim.run(
    # Use a seeded RNG (for the sake of keeping this notebook's results consistent)
    # This parameter is optional; by default a new RNG is constructed for each run
    # using numpy's default_rng
    rng_factory=default_rng(1)
)

from epymorph.adrio import csv
from epymorph.tools.data import munge

cases_df = munge(
    out,
    quantity=rume.ipm.select.events("I->H"),
    time=rume.time_frame.select.all().group("day").agg(),
    geo=rume.scope.select.all(),
)

cases_df.columns = ["date", "geoid", "value"]

"""Negative binomial scale parameter, as r->inf NB->Poisson. """
r = 100
cases_df["value"] = my_rng.negative_binomial(p=r / (r + cases_df["value"]), n=r)

cases_df.to_csv("pf_vignettes/synthetic_data.csv", index=False)

csvadrio = csv.CSVFileAxN(
    file_path="pf_vignettes/synthetic_data.csv",
    dtype=np.int64,
    key_col=1,
    key_type="geoid",
    time_col=0,
    data_col=2,
    skiprows=1,
)

num_realizations = 10

pf_scope = CountyScope.in_counties(
    ["King, WA", "Whatcom, WA", "Pierce, WA", "Snohomish, WA"], year=2015
)

pf_rume = SingleStrataRUME.build(
    # Load the Pei IPM
    ipm=ipm.SIRH(),
    # Load the Pei MM
    mm=mm.Centroids(),
    # Describe the geographic scope of our simulation:
    scope=pf_scope,
    # Create a SingleLocation initializer
    init=init.RandomLocationsAndRandomSeed(1, 1000),
    # Set the time-frame to simulate
    time_frame=TimeFrame.of("2015-01-01", 150),
    # Provide model parameter values
    params={
        "beta": 0.3,
        "gamma": 0.1,
        "xi": 1 / 90,
        "phi": 10,
        "hospitalization_prob": 0.05,
        "hospitalization_duration": 5,
        # Geographic data can be loaded using ADRIOs
        "centroid": us_tiger.InternalPoint(),
        "population": acs5.Population(),
    },
)

my_observations = Observations(
    source=csvadrio,
    model_link=ModelLink(
        geo=pf_rume.scope.select.all(),
        time=pf_rume.time_frame.select.all().group("day").agg(),
        quantity=pf_rume.ipm.select.events("I->H"),
    ),
    likelihood=Poisson(),
)

particle_filter_simulator = ParticleFilterSimulator(
    config=FromRUME(pf_rume, num_realizations),
    observations=my_observations,
    save_trajectories=True,
)

from time import perf_counter
time_0 = perf_counter()
particle_filter_output = particle_filter_simulator.run(rng=my_rng)
time_1 = perf_counter()

print(f"Elapsed time: {time_1 - time_0}")
