# epymorph/parameter_fitting/utils/plot.py


import matplotlib.pyplot as plt
import numpy as np

from epymorph.parameter_fitting.utils.data_op import PlotOutput


class PFPlotRenderer:
    """
    Mixin that provides plotting functionality to render parameter plots.
    Can be mixed into any class (e.g., `ParticleFilterOutput`) to add plotting methods.
    """

    output: PlotOutput

    def __init__(self, output: PlotOutput):
        self.output = output

    def params_plot(self, parameter: str, node_index=0):
        """
        Plot the specified parameter (e.g., 'infection rate') and its quantiles over
        time.

        Args:
            parameter (str): The estimated parameter to plot. Example: "beta".
        """

        key = parameter  # Assuming 'infection rate' corresponds to 'beta' in the data

        # Get quantiles directly from the output object
        key_quantiles = np.array(self.output.param_quantiles[key])

        # Ensure the quantiles data has the correct shape (2D: different time points)
        # if key_quantiles.ndim != 2:
        #     print("Error: Quantiles data should be 2D.")
        #     return

        # Plot the quantiles
        # plt.figure(figsize=(10, 6))
        plt.gcf().set_size_inches(6, 4)
        plt.title(f"{parameter} Quantiles Over Time")

        # Plot quantile range and median as in previous example
        plt.fill_between(
            np.arange(0, len(key_quantiles)),
            key_quantiles[:, 3, node_index],
            key_quantiles[:, 22 - 3, node_index],
            facecolor="blue",
            zorder=10,
            alpha=0.2,
            label="Quantile Range (25th to 75th)",
        )

        plt.fill_between(
            np.arange(0, len(key_quantiles)),
            key_quantiles[:, 6, node_index],
            key_quantiles[:, 22 - 6, node_index],
            facecolor="blue",
            zorder=11,
            alpha=0.4,
            label="Quantile Range (10th to 90th)",
        )

        plt.plot(
            np.arange(0, len(key_quantiles)),
            key_quantiles[:, 11, node_index],
            color="red",
            zorder=12,
            label="Median (50th Percentile)",
        )

        plt.xlabel("Time")
        plt.ylabel(f"{parameter} Quantiles")
        plt.legend(loc="upper left")
        plt.grid(True)

        plt.show()

    def model_fit(
        self, quantile_lower: float = 0.25, quantile_upper: float = 0.75, node_index=0
    ):
        """
        Plot model data, true data, and artificial quantiles (lower and upper bounds).

        Parameters:
        - model_data (array-like): Model values across time.
        - true_data (array-like): True observed values across time.
        - quantile_lower (float): The lower quantile (fraction of model data).
        - quantile_upper (float): The upper quantile (fraction of model data).
        """
        # Calculate artificial lower and upper bounds based on the model data
        lower_bound = self.output.model_data[:, node_index] * (1 - quantile_lower)
        upper_bound = self.output.model_data[:, node_index] * (1 + quantile_upper)

        # Plot the model data (central line)
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.output.model_data[:, node_index],
            label="Model Data",
            color="red",
            linestyle="-",
            zorder=10,
            marker="x",
        )

        # Plot the true data (observed values)
        plt.plot(
            self.output.true_data[:, node_index],
            label="True Data",
            color="green",
            linestyle="--",
            zorder=9,
            marker="*",
        )

        # Fill between the artificial quantiles (lower and upper bounds)
        time_steps = np.arange(
            len(self.output.model_data)
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


class PlotWrapper(PlotOutput):
    @property
    def plot(self) -> PFPlotRenderer:
        return PFPlotRenderer(self)
