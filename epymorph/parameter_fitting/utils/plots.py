"""
This module provides functions for plotting results from the parameter fitting process.

It includes functions to plot quantiles of parameters over time and to generate
histograms of parameter values.

Dependencies:
    - matplotlib: For creating plots.
    - numpy: For numerical operations.
    - pandas: For data manipulation and file I/O.
"""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from epymorph.parameter_fitting.output import ParticleFilterOutput


def params_plot(output: ParticleFilterOutput, parameter: Literal["infection rate"]):
    if parameter == "infection rate":
        key = "beta"

    # Get quantiles directly from the output object
    key_quantiles = np.array(
        output.param_quantiles[key]
    )  # Assuming param_quantiles is a dictionary with 'beta' as a key

    # Check if quantiles have the correct shape (at least 1D)
    if key_quantiles.ndim != 2:
        print("Error: Quantiles data should be 2D (e.g., for different time points).")
        return

    plt.title(f"{parameter} Quantiles Over Time")

    # Plot the quantiles (for example: 25th, 50th, 75th percentiles)
    plt.fill_between(
        np.arange(0, len(key_quantiles)),
        key_quantiles[
            :, 3
        ],  # Adjust the column index based on the quantile you want (e.g., 25th and 75th percentiles)
        key_quantiles[
            :, 22 - 3
        ],  # Adjust as needed based on the quantiles' positions in the data
        facecolor="blue",
        zorder=10,
        alpha=0.2,
        label="Quantile Range (25th to 75th)",
    )
    plt.fill_between(
        np.arange(0, len(key_quantiles)),
        key_quantiles[
            :, 6
        ],  # Adjust for another quantile (e.g., 10th and 90th percentiles)
        key_quantiles[:, 22 - 6],  # Adjust accordingly
        facecolor="blue",
        zorder=11,
        alpha=0.4,
        label="Quantile Range (10th to 90th)",
    )

    # Optionally plot the median or central quantile (e.g., 50th percentile)
    plt.plot(
        np.arange(0, len(key_quantiles)),
        key_quantiles[:, 11],
        color="red",
        zorder=12,
        label="Median (50th Percentile)",
    )

    # Add labels and legend
    plt.xlabel("Time")
    plt.ylabel(f"{parameter} Quantiles")
    plt.legend(loc="upper left")
    plt.grid(True)

    plt.show()


def model_fit(
    output: ParticleFilterOutput,
    quantile_lower: float = 0.25,
    quantile_upper: float = 0.75,
):
    """
    Plot model data, true data, and artificially created quantiles (lower and upper bounds).

    Parameters:
    - model_data: 1D array representing the model values across time.
    - true_data: 1D array representing the actual observed data across time.
    - quantile_lower: The lower quantile (fraction of model data) as a multiplier (e.g., 0.25 for 25% lower bound).
    - quantile_upper: The upper quantile (fraction of model data) as a multiplier (e.g., 0.75 for 75% upper bound).
    """
    # Calculate artificial lower and upper bounds based on the model data
    lower_bound = output.model_data * (1 - quantile_lower)
    upper_bound = output.model_data * (1 + quantile_upper)

    # Plot the model data (central line)
    plt.figure(figsize=(10, 6))
    plt.plot(
        output.model_data,
        label="Model Data",
        color="red",
        linestyle="-",
        zorder=10,
        marker="x",
    )

    # Plot the true data (observed values)
    plt.plot(
        output.true_data,
        label="True Data",
        color="green",
        linestyle="--",
        zorder=9,
        marker="*",
    )

    # Fill between the artificial quantiles (lower and upper bounds)
    time_steps = np.arange(
        len(output.model_data)
    )  # Time steps (indices of the model data)
    plt.fill_between(
        time_steps,
        lower_bound,  # Artificial lower bound (quantile_lower)
        upper_bound,  # Artificial upper bound (quantile_upper)
        color="blue",
        alpha=0.2,
        label="Quantiles",
    )

    # Add labels, title, legend, and grid
    plt.xlabel("Time")
    plt.ylabel("Observation values")
    plt.title("Model Data with Quantiles vs True Data")
    plt.legend(loc="upper right")
    plt.grid(True)

    # Show the plot
    plt.show()
