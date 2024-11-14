"""
This module defines the EpymorphSimulation class for initializing and running
simulations using the Epymorph library.

The EpymorphSimulation class is responsible for setting up the simulation environment
and propagating simulations
based on provided observations and parameters.
"""

import dataclasses
from typing import Any

import numpy as np

# ruff: noqa: F405
from epymorph import *  # noqa: F403


class EpymorphSimulation:
    """
    A class to initialize and run simulations using the Epymorph library.

    Attributes:
        rume (dict): A dictionary containing simulation parameters including
        'rume', 'static_params', 'scope' and
        start_date (str): The start date for the simulation in 'YYYY-MM-DD' format.

    Methods:
        initialize_simulation(initial_particle: tuple) -> StandardSimulation:
            Initializes the simulation with the given initial particle.

        propagate(seird: np.ndarray, observations: dict, sim: StandardSimulation,
        date: str, duration: int) -> np.ndarray:
            Propagates the simulation for a specified duration and returns the output.
    """

    def __init__(self, rume: Any, start_date: str):
        """
        Initializes the EpymorphSimulation class with provided parameters.

        Args:
            rume (dict): A dictionary of parameters required for simulation.
            start_date (str): The start date for the simulation in 'YYYY-MM-DD' format.
        """
        self.rume = rume
        self.start_date = start_date

    def propagate(
        self,
        seird: np.ndarray,
        observations: dict,
        rume: Any,
        date: str,
        duration: int,
        is_sum: bool,
        model_link: str,
    ):
        """
        Propagates the simulation for a specified duration and returns the output.

        Args:
            seird (np.ndarray): Initial state of the system to be used in the simulation.
            observations (dict): A dictionary of parameters to be updated in the simulation.
            sim (StandardSimulation): The simulation object to be used for propagation.
            date (str): The date to start the propagation in 'YYYY-MM-DD' format.
            duration (int): The duration for which the simulation should be propagated.

        Returns:
            np.ndarray: The propagated state of the system as a NumPy array.
        """
        # # Update simulation parameters with current observations
        # Initialize an empty dictionary

        rume_propagate = dataclasses.replace(
            self.rume,
            time_frame=TimeFrame.of(date, duration),
            strata=[
                dataclasses.replace(g, init=init.Explicit(initials=seird))
                for g in rume.strata
            ],
        )

        sim = BasicSimulator(rume_propagate)

        # Run the simulation and return the propagated state
        # print("observations = ", observations)
        output = sim.run(observations)

        # state = np.array([[0]])
        # print("output.prevalence =", output.prevalence
        state = (
            output.compartments
            if model_link in output.compartment_labels
            else output.events_per_day
        )

        # elif model_link in output.event_labels:
        #     state = output.incidence_per_day
        if is_sum:
            state = np.array(np.sum(state, axis=0), dtype=np.int64)

        else:
            # print("stat", state)
            state = state[-1, ...]

        # print("output =", output)
        # print("output.prevalence =", output.prevalence)
        propagated_x = np.array(output.compartments[-1, ...], dtype=np.int64)

        return propagated_x, state
