"""Defines a compartmental IPM with one compartment and no transitions."""
from epymorph.compartment_model import (CompartmentModel, compartment,
                                        create_model, create_symbols)
from epymorph.data import registry


@registry.ipm('no')
def load() -> CompartmentModel:
    """Load the 'no' IPM."""
    return create_model(
        symbols=create_symbols(
            compartments=[compartment('P')],
            attributes=[]
        ),
        transitions=[]
    )
