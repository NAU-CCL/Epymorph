"""
This module provides a simulation class for running particle filters on epidemiological data.

The `FilterSimulation` class initializes with the necessary parameters, handles missing values, validates inputs, and runs the filter simulation.

Dependencies:
    - numpy: For numerical operations.
    - pandas: For data manipulation.
    - epymorph: Custom package providing filter and likelihood functionalities.
"""

# ruff: noqa: F405
import shutil
from email import utils
from pathlib import Path
from typing import Any, Dict

import numpy as np

from epymorph import *  # noqa: F403
from epymorph.parameter_fitting.filters.base_filters import BaseFilter
from epymorph.parameter_fitting.likelihoods.base_likelihood import Likelihood
from epymorph.parameter_fitting.utils import utils
from epymorph.parameter_fitting.utils.data_loader import DataLoader
from epymorph.parameter_fitting.utils.parameter_estimation import EstimateParameters


class FilterSimulation:
    def __init__(
        self,
        rume: Any,
        likelihood_fn: Any,
        filter_type: Any,
        params_space: Dict[str, EstimateParameters],
        observations: Any,
    ):
        """
        Initializes the FilterSimulation class.

        Args:
            rume (Dict[str, Any]): Dictionary containing epidemiological parameters and
            settings.
            observations (Dict[str, Any]): Dictionary containing observation data.
            likelihood_fn (Any): Likelihood function to be used in the simulation.
            filter_type (Any): The filter type to be used (should be a subclass of
            BaseFilter).
            p_estimates (Dict[str, list]): Dictionary containing parameter estimates.
            model_link (str): The model link to be used in the simulation.
            fillna (str, optional): Method to handle missing values in observations
            ('ffill', 'bfill', 'interpolate', 'mean', 'median').
        """
        self.rume: Any = rume
        self.observations: Any = observations
        self.likelihood_fn: Any = likelihood_fn
        self.filter_type: Any = filter_type
        self.params_space: Dict[str, EstimateParameters] = params_space
        self.ipm = self.rume.ipm
        self.model_link: str = self.observations.model_link.replace("->", "→")
        self.compartments = [compartment.name for compartment in self.ipm.compartments]
        self.events = [e.name.full for e in self.ipm.events]
        self.is_sum = True  # self.observations["is_sum"]
        self.dataloader = DataLoader(self.rume)

        # Perform validation
        self.validate()

        self.dates, self.cases = self.dataloader.load_data(self.observations)

        if len(self.cases.shape) < 2:
            self.cases = self.cases[..., np.newaxis]

        shutil.rmtree("./epymorph/parameter_fitting/data")

        Path("./epymorph/parameter_fitting/data").mkdir(exist_ok=True)

        utils.save_data(self.cases.tolist(), False)

    def handle_missing_values(self, method: str):
        """
        Handles missing values in the observations using the specified method.

        Args:
            method (str): Method for handling missing values ('ffill', 'bfill', 'interpolate', 'mean', 'median').

        Raises:
            ValueError: If an invalid method is specified.
        """
        if method not in ["ffill", "bfill", "interpolate", "mean", "median"]:
            raise ValueError(
                "Invalid fillna method. Choose from 'ffill', 'bfill', 'interpolate', 'mean' or 'median'."
            )

    def validate(self) -> None:
        """
        Validates the input parameters for the simulation.

        Raises:
            ValueError: If any required field is missing or if types are incorrect.
        """

        # Validate likelihood function
        if not isinstance(self.likelihood_fn, Likelihood):
            raise ValueError(
                "likelihood_fn must be an instance of a subclass of Likelihood."
            )

        # Validate filter type
        if not isinstance(self.filter_type, BaseFilter):
            raise ValueError(
                "filter_type must be an instance of a subclass of BaseFilter."
            )

        if self.model_link not in self.events + self.compartments:
            raise ValueError(
                "Model Link should match with the ipm provided\n"
                "Supported model links: "
                f"{self.events + self.compartments}"
            )

        if self.model_link in self.compartments and self.is_sum:
            print(
                "Warning: Data passed cannot be cummulative over the given "
                f"model link {self.model_link} as it can overcome the total"
                "population\nPlease try a different model link or is_sum False"
                "(non aggregated data)"
            )

        if not isinstance(self.params_space, dict):
            raise ValueError("p_estimates must be a dictionary.")

        # Validate each entry in the p_estimates dictionary
        for key, value in self.params_space.items():
            if not isinstance(value, EstimateParameters):
                raise ValueError(
                    f"Value for '{key}' must be an instance of EstimateParameters."
                )
            if not isinstance(key, str):
                raise ValueError("Keys in p_estimates must be strings.")
            if key not in ["beta"]:
                raise ValueError(
                    "Invalid estimation parameter is passed.\nValid parameters: beta"
                )

    def get_index(self) -> int:
        """
        Retrieves the index of the model link in the IPM or compartment names.

        Returns:
            int: The index of the model link.
        """
        if "->" in self.model_link or "→" in self.model_link:
            index = self.events.index(self.model_link)
        else:
            index = self.compartments.index(self.model_link)

        return index

    def run(self):
        """
        Runs the particle filter simulation.

        Returns:
            dict: The output of the filter simulation.
        """

        index = self.get_index()

        out = self.filter_type.run(
            self.rume,
            self.likelihood_fn,
            self.params_space,
            self.model_link,
            index,
            self.dates,
            self.cases,
        )

        return out

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
