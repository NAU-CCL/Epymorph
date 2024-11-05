from typing import Any
from unittest.mock import Mock

import numpy as np
from pandas import read_csv

# noqa: F405
from epymorph.adrio.cdc import *  # noqa: F403
from epymorph.adrio.csv import CSVTimeSeries
from epymorph.data_shape import SimDimensions
from epymorph.simulation import NamespacedAttributeResolver, SimulationFunctionClass


class DataLoader:
    def __init__(self, rume: Any) -> None:
        self.rume = rume

    # @classmethod
    def load_data(self, observations):
        data = Mock(spec=NamespacedAttributeResolver)
        dim = Mock(spec=SimDimensions)
        rng = Mock(spec=np.random.Generator)

        # start_date = self.rume.time_frame.start_date
        # # duration = 7 * self.rume.time_frame.duration_days
        # end_date = start_date + datetime.timedelta(
        #     days=self.rume.time_frame.duration_days
        # )

        # # time_period = TimeFrame.range("2022-01-01", "2022-10-01")
        # time_period = TimeFrame.range(start_date, end_date)

        flu_sum = observations.source

        if isinstance(flu_sum, SimulationFunctionClass):
            cases = flu_sum.evaluate_in_context(data, dim, self.rume.scope, rng)  # type: ignore

            # Flatten the array and extract dates and case counts
            cases_flat = cases.reshape(
                -1
            )  # Remove the extra dimension, keeping the tuples
            dates = cases_flat[
                "date"
            ].flatten()  # Extract dates (first element of each tuple)
            case_counts = (
                cases_flat["data"].astype(int).flatten()
            )  # Extract case counts (second element)

            # # Extracting dates and case counts
            # dates = cases["date"].astype(str).flatten()  # Convert date to string
            # case_counts = cases["data"].astype(int).flatten()  # Convert case
            # counts to int

            rng = np.random.default_rng()
            sim_data = rng.poisson(case_counts)

            return dates, sim_data

        if isinstance(flu_sum, CSVTimeSeries):
            csv_df = read_csv(flu_sum.file_path)
            dates, cases = (
                csv_df.Date.to_numpy().flatten(),
                csv_df.Cases.to_numpy().flatten(),
            )

            # dates = to_datetime(dates)
            rng = np.random.default_rng()
            sim_data = rng.poisson(cases)

            return dates, sim_data
