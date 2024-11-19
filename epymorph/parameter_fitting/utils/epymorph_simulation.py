"""
This module defines the EpymorphSimulation class for initializing and running
simulations using the Epymorph library.

The EpymorphSimulation class is responsible for setting up the simulation environment
and propagating simulations based on provided observations and parameters.
"""

import dataclasses
from typing import Any, Tuple

import numpy as np

from epymorph import BasicSimulator, TimeFrame, init  # noqa: F403


class EpymorphSimulation:
    """
    A class to initialize and run simulations using the Epymorph library.

    Attributes:
        rume (dict): A dictionary containing the model and simulation parameters,
        such as 'rume', 'static_params', 'scope', etc.
        start_date (str): The start date for the simulation in 'YYYY-MM-DD' format.
    """

    def __init__(self, rume: Any, start_date: str):
        """
        Initializes the EpymorphSimulation class with the provided parameters.

        Args:
            rume (dict): A dictionary of parameters required for the simulation,
            including model settings.
            start_date (str): The start date for the simulation in 'YYYY-MM-DD' format.
        """
        self.rume = rume
        self.start_date = start_date

    def propagate(
        self,
        state: np.ndarray,
        parameters: dict,
        rume: Any,
        date: str,
        duration: int,
        is_sum: bool,
        model_link: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagates the simulation for a specified duration and returns the final state.

        Args:
            state (np.ndarray): Initial state of the system (e.g., SIRH compartments)
            to be used in the simulation.
            observations (dict): A dictionary containing the parameter values
            (such as beta, gamma, etc.) to update in the simulation.
            rume (Any): The model configuration to run the simulation.
            date (str): The starting date for the simulation in 'YYYY-MM-DD' format.
            duration (int): The number of days the simulation should be propagated.
            is_sum (bool): Whether to return the sum of the compartments (if true)
            or the final state.
            model_link (str): Specifies which model output to return, either from
            compartments or events.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the final compartment
            state and the processed state (either the sum or last day).
        """

        # Create a copy of the RUME model with updated parameters and time frame
        rume_propagate = dataclasses.replace(
            self.rume,
            time_frame=TimeFrame.of(date, duration),  # Set simulation duration
            strata=[
                dataclasses.replace(
                    g, init=init.Explicit(initials=state)
                )  # Initialize with state values
                for g in rume.strata  # For each stratum, set the initial state
            ],
        )

        # Initialize the simulation using the BasicSimulator from the Epymorph library
        sim = BasicSimulator(rume_propagate)

        # Run the simulation and collect the output based on observations (dynamic params)
        output = sim.run(parameters)

        # Determine whether to return compartments or events based on the model link
        events_state = (
            output.compartments
            if model_link in output.compartment_labels
            else output.events_per_day
        )

        # Sum the state values if the flag `is_sum` is set
        if is_sum:
            events_state = np.array(
                np.sum(events_state, axis=0), dtype=np.int64
            )  # Sum across all compartments
        else:
            events_state = events_state[
                -1, ...
            ]  # Get the last day's state if not summing

        # Return the final propagated state as an integer array
        propagated_x = np.array(output.compartments[-1, ...], dtype=np.int64)

        return propagated_x, events_state
