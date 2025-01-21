"""Defines a compartmental IPM with one compartment and no transitions."""

from epymorph.compartment_model import CompartmentModel, compartment


class No(CompartmentModel):
    """The 'no' IPM: a single compartment with no transitions."""

    _suppress_ipm_validation_warnings = True

    compartments = (compartment("P"),)

    def edges(self, symbols):
        return []
