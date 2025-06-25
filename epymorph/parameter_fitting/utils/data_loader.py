from typing import Tuple

import numpy as np
from pandas import read_csv

from epymorph.adrio.adrio import ADRIO
from epymorph.adrio.csv import CSVFileTxN
from epymorph.parameter_fitting.utils.observations import Observations
from epymorph.rume import RUME
from epymorph.simulation import Context
from epymorph.util import extract_date_value, is_date_value_dtype


class DataLoader:
    """
    A class responsible for loading data from various sources such as
    simulations or CSV files.

    Attributes
    ----------
    rume : Rume
        Simulation parameters and configuration.
    """

    def __init__(self, rume: RUME) -> None:
        """
        Initializes the DataLoader with simulation parameters.

        Parameters
        ----------
        rume : Rume
            Simulation runtime environment or configuration object.
        """
        self.rume = rume

    def load_data(self, observations: Observations) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads the data from a given source.

        This method handles two types of sources:
        1. CSVTimeSeries: Data is loaded from a CSV file.
        2. Any of the CDC adrios.

        Parameters
        ----------
        observations : Observations
            The data source.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing two numpy arrays:
                - dates: An array of date values.
                - sim_data: Simulated or observed case counts, generated using a Poisson
                distribution.

        Raises
        ------
        ValueError
            Raised if the data source is not supported.
        """
        rng = np.random.default_rng()
        data = self.rume.evaluate_params(rng)
        source = observations.source

        if isinstance(source, CSVFileTxN):
            csv_df = read_csv(source.file_path)

            cases = source.with_context_internal(
                Context.of(data=data, scope=self.rume.scope, rng=rng)
            ).evaluate()

            dates = csv_df.pivot_table(index=csv_df.columns[0]).index.to_numpy()

            return dates, cases

        if isinstance(source, ADRIO) and is_date_value_dtype(
            source.result_format.dtype
        ):
            ctx = Context.of(
                data=data,
                time_frame=self.rume.time_frame,
                scope=self.rume.scope,
                rng=rng,
            )
            result = source.with_context_internal(ctx).evaluate()
            if np.ma.is_masked(result):
                err = (
                    f"Observation data from {source.class_name} contains unresolved "
                    "issues. Use ADRIO constructor options to address all data issues "
                    "as appropriate before execution."
                )
                raise ValueError(err)

            return extract_date_value(result)

        raise ValueError("Unsupported data source provided.")
