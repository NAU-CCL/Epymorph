"""Defines a compartmental IPM with one compartment and no transitions."""

import warnings

from epymorph.compartment_model import CompartmentModel, compartment

# IPM validation is going to complain about the lack of transitions,
# but in this case it's intentional. Suppress warnings.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", module="epymorph.compartment_model")

    class No(CompartmentModel):
        """The 'no' IPM: a single compartment with no transitions."""

        compartments = (compartment("P"),)

        def edges(self, symbols):
            return []
