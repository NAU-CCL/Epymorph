from typing import Any, Tuple
from unittest.mock import Mock

import numpy as np
from pandas import read_csv

from epymorph.adrio.cdc import *  # noqa: F403
from epymorph.adrio.csv import CSVTimeSeries
from epymorph.data_shape import SimDimensions
from epymorph.simulation import NamespacedAttributeResolver, SimulationFunctionClass


class DataLoader:
    """
    A class responsible for loading data from various sources such as
    simulations or CSV files.

    Attributes:
        rume (Any): Simulation parameters and configuration.
    """

    def __init__(self, rume: Any) -> None:
        """
        Initializes the DataLoader with simulation parameters.

        Args:
            rume (Any): Simulation runtime environment or configuration object.
        """
        self.rume = rume

    def load_data(self, observations: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads the data from a given source, either a simulation or a CSV file.

        This method handles two types of sources:
        1. SimulationFunctionClass: Data is generated via simulation in the context.
        2. CSVTimeSeries: Data is loaded from a CSV file.

        Args:
            observations (Any): The data source, either a simulation function or
            a CSV time series.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
            - dates: An array of date values.
            - sim_data: Simulated or observed case counts, generated using a
            Poisson distribution.
        """
        data = Mock(spec=NamespacedAttributeResolver)
        dim = Mock(spec=SimDimensions)
        rng = Mock(spec=np.random.Generator)

        flu_sum = observations.source

        # If the source is a simulation function, evaluate it within the simulation
        # context.
        if isinstance(flu_sum, SimulationFunctionClass):
            cases = flu_sum.evaluate_in_context(data, dim, self.rume.scope, rng)  # type: ignore

            # Flatten the multi-dimensional array to extract tuples (date, data)
            cases_flat = cases.reshape(-1)

            # Extract dates and case counts from the flattened array
            dates = cases_flat["date"].flatten()
            cases = cases_flat["data"].astype(int).flatten()

            return dates, cases

        # If the source is a CSV time series, read the data from the CSV file.
        if isinstance(flu_sum, CSVTimeSeries):
            csv_df = read_csv(flu_sum.file_path)
            # dates, cases = (
            #     csv_df.Date.to_numpy(),
            #     csv_df.Cases.to_numpy(),
            # )

            cases = flu_sum.evaluate_in_context(
                data, self.rume.dim, self.rume.scope, rng
            )

            dates = csv_df.Date.to_numpy()

            return dates, cases

        raise ValueError("Unsupported data source provided.")
