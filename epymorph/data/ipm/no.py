"""Defines a compartmental IPM with one compartment and no transitions."""
from __future__ import annotations

from epymorph.compartment_model import (CompartmentModel, compartment,
                                        create_model, create_symbols)


def load() -> CompartmentModel:
    """Load the 'no' IPM."""
    return create_model(
        symbols=create_symbols(
            compartments=[compartment('P')],
            attributes=[]
        ),
        transitions=[]
    )
