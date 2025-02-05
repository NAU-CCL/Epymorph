"""
This module defines the EpymorphSimulation class for initializing and running
simulations using the Epymorph library.

The EpymorphSimulation class is responsible for setting up the simulation environment
and propagating simulations based on provided observations and parameters.
"""

import dataclasses
from typing import Tuple

import numpy as np

from epymorph import initializer
from epymorph.parameter_fitting.utils.observations import ModelLink
from epymorph.rume import RUME
from epymorph.simulator.basic.basic_simulator import BasicSimulator
from epymorph.time import TimeFrame
from epymorph.tools.data import munge  # noqa: F403


class EpymorphSimulation:
    """
    A class to initialize and run simulations using the Epymorph library.

    Attributes
    ----------
    rume : Rume
        Contains the model and simulation parameters,
    start_date : str
        The start date for the simulation in 'YYYY-MM-DD' format.
    """

    def __init__(self, rume: RUME, start_date: str):
        """
        Initializes the EpymorphSimulation class with the provided parameters.

        Parameters
        ----------
        rume : Rume
            Parameters required for the simulation, including model settings.
        start_date : str
            The start date for the simulation in 'YYYY-MM-DD' format.
        """
        self.rume = rume
        self.start_date = start_date

    def propagate(
        self,
        state: np.ndarray,
        parameters: dict,
        rume: RUME,
        date: str,
        duration: int,
        model_link: ModelLink,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagates the simulation for a specified duration and returns the final state.

        Parameters
        ----------
        state : np.ndarray
            Initial state of the system (e.g., SIRH compartments)
            to be used in the simulation.
        observations : dict)
            A dictionary containing the parameter values
            (such as beta, gamma, etc.) to update in the simulation.
        rume : Rume
            The model configuration to run the simulation.
        date : str
            The starting date for the simulation in 'YYYY-MM-DD' format.
        duration : int
            The number of days the simulation should be propagated.
        model_link : ModelLink
            Specifies which model output to return, either from
            compartments or events.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing the final compartment
            state and the processed state (either the sum or last day).
        """

        # Create a copy of the RUME model with updated parameters and time frame
        rume_propagate = dataclasses.replace(
            self.rume,
            time_frame=TimeFrame.of(date, duration),  # Set simulation duration
            strata=[
                dataclasses.replace(
                    g, init=initializer.Explicit(initials=state)
                )  # Initialize with state values
                for g in rume.strata  # For each stratum, set the initial state
            ],
        )

        # Initialize the simulation using the BasicSimulator from the Epymorph library
        sim = BasicSimulator(rume_propagate)

        # Run the simulation and collect the output based on observations
        # (dynamic params)
        output = sim.run(parameters, rng_factory=(lambda: rng))

        data_df = munge(
            output,
            geo=model_link.geo,
            time=dataclasses.replace(
                model_link.time, time_frame=rume_propagate.time_frame
            ),
            quantity=model_link.quantity,
        )

        expected_observation = data_df.iloc[-rume.scope.nodes :, -1].to_numpy()

        # Return the final propagated state as an integer array
        propagated_x = np.array(output.compartments[-1, ...], dtype=np.int64)

        return propagated_x, expected_observation
