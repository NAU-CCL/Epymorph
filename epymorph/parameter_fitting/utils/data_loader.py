from typing import Any, Tuple

import numpy as np
from pandas import read_csv

from epymorph.adrio.cdc import *  # noqa: F403
from epymorph.adrio.csv import CSVTimeSeries
from epymorph.parameter_fitting.utils.observations import Observations

_CDC_ADRIOS = [
    CovidCasesPer100k,  # noqa: F405
    CovidHospitalizationsPer100k,  # noqa: F405
    CovidHospitalizationAvgFacility,  # noqa: F405
    CovidHospitalizationSumFacility,  # noqa: F405
    InfluenzaHosptializationAvgFacility,  # noqa: F405
    InfluenzaHospitalizationSumFacility,  # noqa: F405
    CovidHospitalizationAvgState,  # noqa: F405
    CovidHospitalizationSumState,  # noqa: F405
    InfluenzaHospitalizationAvgState,  # noqa: F405
    InfluenzaHospitalizationSumState,  # noqa: F405
    FullCovidVaccinations,  # noqa: F405
    OneDoseCovidVaccinations,  # noqa: F405
    CovidBoosterDoses,  # noqa: F405
    CovidDeathsCounty,  # noqa: F405
    CovidDeathsState,  # noqa: F405
    InfluenzaDeathsState,  # noqa: F405
]


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

    def load_data(self, observations: Observations) -> Tuple[np.ndarray, np.ndarray]:
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
        # data = Mock(spec=NamespacedAttributeResolver)
        # dim = Mock(spec=SimDimensions)
        # rng = Mock(spec=np.random.Generator)

        rng = np.random.default_rng()
        data = self.rume.evaluate_params(rng)

        source = observations.source

        # # If the source is a simulation function, evaluate it within the simulation
        # # context.
        # if isinstance(flu_sum, BaseSimulationFunction):
        #     cases = flu_sum.with_context_internal(
        #         data=data, scope=self.rume.scope, rng=rng
        #     ).evaluate()  # type: ignore

        #     # Flatten the multi-dimensional array to extract tuples (date, data)
        #     cases_flat = cases.reshape(-1)

        #     # Extract dates and case counts from the flattened array
        #     dates = cases_flat["date"].flatten()
        #     cases = cases_flat["data"].astype(int).flatten()

        #     return dates, cases

        # If the source is a CSV time series, read the data from the CSV file.
        if isinstance(source, CSVTimeSeries):
            csv_df = read_csv(source.file_path)
            # dates, cases = (
            #     csv_df.Date.to_numpy(),
            #     csv_df.Cases.to_numpy(),
            # )

            cases = source.with_context_internal(
                data=data, scope=self.rume.scope, rng=rng
            ).evaluate()

            dates = csv_df.pivot_table(index=csv_df.columns[0]).index.to_numpy()

            return dates, cases

        if any([isinstance(source, cdc_adrio) for cdc_adrio in _CDC_ADRIOS]):
            result = np.array(
                source.with_context_internal(
                    data=data,
                    time_frame=self.rume.time_frame,
                    scope=self.rume.scope,
                    rng=rng,
                ).evaluate()
            )

            dates = result["date"][:, 0]
            cases = result["data"]

            # dates = csv_df.pivot_table(index=csv_df.columns[0]).index.to_numpy()

            return dates, cases

        raise ValueError("Unsupported data source provided.")
