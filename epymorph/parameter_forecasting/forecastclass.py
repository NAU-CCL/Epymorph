import dataclasses
from types import SimpleNamespace
from typing import Callable, Generic, Optional

import numpy as np

from epymorph import initializer
from epymorph.attribute import NamePattern
from epymorph.parameter_fitting.filter.particle import Particle
from epymorph.simulation import ParamValue
from epymorph.simulator.basic.basic_simulator import BasicSimulator, RUMEType
from epymorph.time import TimeFrame
from epymorph.util import CovariantMapping


class ForecastSimulator(Generic[RUMEType]):
    rume: RUMEType
    """The RUME we will use for the simulation."""

    def __init__(
        self,
        rume: RUMEType,
        initial_realizations: Optional[list[Particle]] = None,
        num_realizations: Optional[int] = None,
        time_frame: Optional[TimeFrame] = None,
    ):
        self.rume = rume

        # Check if initial_particles is None, and generate if num is provided
        if initial_realizations is None:
            self.time_frame = self.rume.time_frame
            if num_realizations is None:
                raise ValueError(
                    "Either 'initial_realizations' or 'num_realizations' must be provided."
                )
            else:
                self.num_realizations = num_realizations
            self.initial_realizations = None
        else:
            if time_frame is None:
                raise ValueError(
                    "Pass a new rume or pass a time frame to run the forecast"
                )
            self.time_frame = time_frame
            self.initial_realizations = initial_realizations
            self.num_realizations = len(self.initial_realizations)

    def run(
        self,
        /,
        params: CovariantMapping[str | NamePattern, ParamValue] | None = None,
        rng_factory: Callable[[], np.random.Generator] | None = None,
    ):
        rume = dataclasses.replace(
            self.rume,
            time_frame=self.time_frame,  # Set simulation duration
        )

        days = self.time_frame.days
        taus = rume.num_tau_steps
        R = self.num_realizations
        S = days * taus
        N = rume.scope.nodes
        C = rume.ipm.num_compartments
        E = rume.ipm.num_events

        # It's my personal preference to preallocate memory.
        initial = np.empty(shape=(R, N, C), dtype=np.int64)
        compartments = np.empty(shape=(R, S, N, C), dtype=np.int64)
        events = np.empty(shape=(R, S, N, E), dtype=np.int64)

        rng = (rng_factory or np.random.default_rng)()

        if params is not None:
            if self.initial_realizations is not None:
                parameters = self.initial_realizations[0].parameters
                updated_params = {}
                for key, value in params.items():  # type: ignore
                    value.initial = parameters.get(  # type: ignore
                        str(key), None
                    )  # Get the initial value from parameters
                    updated_params[key] = value

                dummy_data = rume.evaluate_params(
                    override_params=updated_params, rng=rng
                )
            else:
                dummy_data = rume.evaluate_params(override_params=params, rng=rng)
        else:
            dummy_data = rume.evaluate_params(rng=rng)

        # Initialize a dictionary to store the returned parameters
        return_params = {}

        # Loop through all the keys in the evaluated dummy data
        for key in dummy_data.to_dict().keys():
            # Store the results with the shape of (number of realizations, shape of the raw data)
            return_params[str(key)] = np.empty(
                shape=(self.num_realizations,) + dummy_data.get_raw(key).shape,
                dtype=dummy_data.get_raw(key).dtype,
            )

        for i in range(self.num_realizations):
            # The BasicSimulator does not return the evaluated parameters, so instead we
            # precompute the parameters then pass them into the simulator.
            if self.initial_realizations is None:
                data = rume.evaluate_params(override_params=params, rng=rng)
                evaluated_params = dict(
                    zip(
                        [str(key) for key in data.to_dict().keys()],
                        [data.get_raw(key) for key in data.to_dict().keys()],
                    )
                )

                sim = BasicSimulator(rume)

                out = sim.run(params=evaluated_params, rng_factory=(lambda: rng))
            else:
                rume_propagate = dataclasses.replace(
                    rume,
                    strata=[
                        dataclasses.replace(
                            g,
                            init=initializer.Explicit(
                                initials=self.initial_realizations[i].state
                            ),
                        )  # Initialize with state values
                        for g in rume.strata  # For each stratum, set the initial state
                    ],
                )
                parameters = self.initial_realizations[i].parameters
                updated_params = {}
                for key, value in params.items():  # type: ignore
                    value.initial = parameters.get(  # type: ignore
                        str(key), None
                    )  # Get the initial value from parameters
                    updated_params[key] = value

                data = rume.evaluate_params(override_params=updated_params, rng=rng)
                evaluated_params = dict(
                    zip(
                        [str(key) for key in data.to_dict().keys()],
                        [data.get_raw(key) for key in data.to_dict().keys()],
                    )
                )

                sim = BasicSimulator(rume_propagate)

                # Run the simulation and collect the output based on observations
                # (dynamic params)
                out = sim.run(evaluated_params, rng_factory=(lambda: rng))

            initial[i, ...] = out.initial
            compartments[i, ...] = out.compartments
            events[i, ...] = out.events

            for key in evaluated_params.keys():
                return_params[key][i, ...] = evaluated_params[key]

        return SimpleNamespace(
            rume=self.rume,
            initial=initial,
            compartments=compartments,
            events=events,
            params=return_params,
        )
