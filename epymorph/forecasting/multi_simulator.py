import dataclasses

import numpy as np

from epymorph.attribute import NamePattern
from epymorph.data_type import SimDType
from epymorph.forecasting.multi_output import MultiOutput
from epymorph.initializer import Explicit
from epymorph.simulator.basic.basic_simulator import BasicSimulator


class MultiSimulator:
    """
    A simulator for running multiple realizations of a RUME simulation.
    """

    def run(self, rume, num_realizations, initial_values, param_values, rng_factory):
        """
        A simulator for running multiple realizations of a RUME. Different initial
        values and parameter values can be specified.

        Parameters
        ----------
        rume :
            The base rume to use for simulating each realization.
        num_realizations :
            The number of realizations to use.
        initial_values :
            The initial compartment values used for each realization, given as a numpy
            array. The leading dimension is the number of realizations. If not
            specified the initial values will be taken from the rume initializer.
        param_values :
            The parameter values used for each realization. Overrides any
            corresponding parameter values in the rume.
        rng_factory :
            The random number generator.
        Returns
        -------
        :
        """

        days = rume.time_frame.days
        taus = rume.num_tau_steps
        R = num_realizations
        S = days * taus
        N = rume.scope.nodes
        C = rume.ipm.num_compartments
        E = rume.ipm.num_events
        visit_compartments = np.zeros((R, S, N, C), dtype=SimDType)
        visit_events = np.zeros((R, S, N, E), dtype=SimDType)
        home_compartments = np.zeros((R, S, N, C), dtype=SimDType)
        home_events = np.zeros((R, S, N, E), dtype=SimDType)

        for i_realization in range(num_realizations):
            rume_temp = rume
            if initial_values is not None:
                initial = initial_values[i_realization, ...]
                strata_temp = [
                    dataclasses.replace(
                        stratum,
                        init=Explicit(
                            initials=initial[..., rume.compartment_mask[stratum.name]]
                        ),
                    )
                    for stratum in rume.strata
                ]
                rume_temp = dataclasses.replace(rume_temp, strata=strata_temp)
            if param_values is not None:
                params_temp = dict(rume.params).copy()
                for name in param_values.keys():
                    pattern = NamePattern.of(name)
                    params_temp[pattern] = param_values[name][i_realization, ...]
                rume_temp = dataclasses.replace(rume_temp, params=params_temp)
            sim = BasicSimulator(rume_temp)
            out_temp = sim.run(rng_factory=rng_factory)
            visit_compartments[i_realization, ...] = out_temp.visit_compartments
            visit_events[i_realization, ...] = out_temp.visit_events

        return MultiOutput(
            rume,
            initial_values,
            visit_compartments,
            visit_events,
            home_compartments,
            home_events,
        )
