"""Defines a compartmental IPM with one compartment and no transitions."""

from epymorph.compartment_model import CompartmentModel, compartment
from epymorph.data import registry


@registry.ipm("no")
class No(CompartmentModel):
    """The 'no' IPM: a single compartment with no transitions."""

    compartments = (compartment("P"),)

    def edges(self, symbols):
        return []
