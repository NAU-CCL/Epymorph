"""
This module provides utility functions for quantile computation and data saving.

It contains the Utils class with static methods to calculate quantiles and save data to
CSV files.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

# class Utils:
#     """
#     A class providing utility functions for statistical calculations and data
#     management.

#     Methods:
#         quantiles(items: List[float]) -> List[float]:
#             Computes specified quantiles for a list of items.

#         save_data(observations_quantiles: Dict[str, List[float]],
#             observations_values: Dict[str, List[float]]):
#             Saves quantiles and average values of observations to CSV files.
#     """


# @staticmethod
def quantiles(items: np.ndarray) -> List[float]:
    """
    Computes specified quantiles for a list of items.

    Parameters
    ----------
    items : List[float]
        Numerical values to compute quantiles of.

    Returns
    -------
    List[float]
        Quantile values for the provided items.
    """
    qtl_mark = 1.00 * np.array(
        [
            0.010,
            0.025,
            0.050,
            0.100,
            0.150,
            0.200,
            0.250,
            0.300,
            0.350,
            0.400,
            0.450,
            0.500,
            0.550,
            0.600,
            0.650,
            0.700,
            0.750,
            0.800,
            0.850,
            0.900,
            0.950,
            0.975,
            0.990,
        ]
    )
    return np.quantile(items, qtl_mark, axis=0)


# @staticmethod
def save_data(observations: Dict[str, List[float]] | List, quantiles: bool) -> None:
    """
    Saves quantiles and average values of observations to CSV files.

    Parameters
    ----------
    observations_quantiles : Dict[str, List[float]]
        Quantiles of the predicted data.
    observations_values : Dict[str, List[float]]
        Average of the predicted data
    """
    if isinstance(observations, dict):
        for key in observations.keys():
            # Save average values
            if quantiles:
                # Save quantiles
                pd.DataFrame(observations[key]).to_csv(
                    f"./epymorph/parameter_fitting/data/{key}_quantiles.csv"
                )
            else:
                pd.DataFrame(observations[key]).to_csv(
                    f"./epymorph/parameter_fitting/data/{key}_values.csv"
                )
    elif isinstance(observations, List):
        # Assuming the lists are aligned, save quantiles and values by index
        pd.DataFrame(observations).to_csv(
            "./epymorph/parameter_fitting/data/observations.csv", index=False
        )
