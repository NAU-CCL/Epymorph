"""
This module provides a simulation class for running particle filters on
epidemiological data.

The `FilterSimulation` class initializes with the necessary parameters,
validates inputs, and runs the particle filter simulation.

Dependencies:
    - numpy: For numerical operations.
    - pandas: For data manipulation.
    - epymorph: Custom package providing filter and likelihood functionalities.
"""

from typing import Dict

import numpy as np

from epymorph import *  # noqa: F403
from epymorph.parameter_fitting.filters.base_filters import BaseFilter
from epymorph.parameter_fitting.likelihood.base_likelihood import Likelihood
from epymorph.parameter_fitting.output import ParticleFilterOutput
from epymorph.parameter_fitting.utils.data_loader import DataLoader
from epymorph.parameter_fitting.utils.observations import ModelLink, Observations
from epymorph.parameter_fitting.utils.parameter_estimation import EstimateParameters
from epymorph.rume import Rume


class FilterSimulation:
    """
    A class to run particle filter simulations on epidemiological data.

    This class is initialized with epidemiological parameters, likelihood function,
    filter type, parameter space, and observational data. It validates these inputs
    and runs the particle filter to produce simulation results.

    Attributes
    ----------
    rume : Rume
        Runtime environment for the simulation,
        containing necessary parameters.
    likelihood_fn : Likelihood
        The likelihood function to be used.
    filter_type : BaseFilter
        Type of filter (e.g., Particle Filter) to be
        used in the simulation.
    params_space : Dict[str, EstimateParameters]
        Parameter estimates
        for the simulation.
    observations : Observations
        Observational data to be used in the simulation.
    dataloader : DataLoader
        A DataLoader instance to handle loading of
        observational data.
    dates : np.ndarray
        The dates associated with the observational data.
    cases : np.ndarray
        The case counts associated with the observational data.
    """

    def __init__(
        self,
        rume: Rume,
        filter_type: BaseFilter,
        params_space: Dict[str, EstimateParameters],
        observations: Observations,
    ):
        """
        Initializes the FilterSimulation class.

        Parameters
        ----------
        rume : Rume
            Runtime environment or configuration for the epidemiological
            model.
        filter_type : BaseFilter
            Type of particle filter to be used.
        params_space : Dict[str, EstimateParameters]
            A dictionary containing
            parameter estimates.
        observations : Observations
            An object containing observational data.

        Raises
        ------
        ValueError
            If any of the input arguments are invalid.
        """
        self.rume = rume
        self.observations = observations
        self.likelihood_fn = observations.likelihood
        self.filter_type = filter_type
        self.params_space = params_space

        # Initialize simulation-related attributes
        self.ipm = self.rume.ipm
        self.model_link: ModelLink = self.observations.model_link
        self.compartments = [compartment.name for compartment in self.ipm.compartments]
        self.events = [e.name.full for e in self.ipm.events]

        # Initialize DataLoader to handle data fetching
        self.dataloader = DataLoader(self.rume)

        # Validate the inputs
        self.validate()

        # Load observational data
        self.dates, self.cases = self.dataloader.load_data(self.observations)

        if len(self.cases.shape) < 2:
            self.cases = self.cases[..., np.newaxis]

    def validate(self) -> None:
        """
        Validates the input parameters for the simulation.

        Ensures that the likelihood function, filter type, model link, and parameter
        estimates are valid.

        Raises
        ------
        ValueError
            If any required field is missing or if the types are incorrect.
        """
        # Validate likelihood function
        if not isinstance(self.likelihood_fn, Likelihood):
            raise ValueError("The 'likelihood_fn' must be an instance of Likelihood.")

        # Validate filter type
        if not isinstance(self.filter_type, BaseFilter):
            raise ValueError("The 'filter_type' must be an instance of BaseFilter.")

        # Validate parameter space
        if not isinstance(self.params_space, dict):
            raise ValueError("The 'params_space' must be a dictionary.")

        for key, value in self.params_space.items():
            if not isinstance(value, EstimateParameters):
                raise ValueError(
                    "Each value in 'params_space' must be an instance of"
                    f" EstimateParameters Invalid entry for '{key}'."
                )

        # Validate observations
        if not isinstance(self.observations, Observations):
            raise ValueError("The 'observations' must be an instance of Observations.")

    def run(self, rng: np.random.Generator | None = None) -> ParticleFilterOutput:
        """
        Runs the particle filter simulation.

        Returns
        -------
        ParticleFilterOutput
            The output of the filter simulation containing
            results.
        """

        if not rng:
            rng = np.random.default_rng()

        output = self.filter_type.run(
            self.rume,
            self.likelihood_fn,
            self.params_space,
            self.model_link,
            self.dates,
            self.cases,
            rng,
        )

        return output

    def plot_data(self):
        import matplotlib.pyplot as plt

        time_axis = range(len(self.cases))

        # Plot the values over time as a bar plot

        plt.bar(time_axis, self.cases[:, 0])
        n_nodes = self.cases.shape[1]
        for i_node in range(n_nodes - 1):
            plt.bar(
                time_axis,
                self.cases[:, i_node + 1],
                bottom=self.cases[:, 0 : i_node + 1].sum(axis=1),
            )

        plt.title("Observations plot over time")
        plt.grid(True)
        plt.show()
