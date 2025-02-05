"""Defines a compartmental IPM for a generic SIRS model."""

from sympy import Max

from epymorph.attribute import AttributeDef
from epymorph.compartment_model import CompartmentModel, compartment, edge
from epymorph.data_shape import Shapes


class SIRS(CompartmentModel):
    """A basic SIRS model."""

    compartments = [
        compartment("S"),
        compartment("I"),
        compartment("R"),
    ]

    requirements = [
        AttributeDef("beta", type=float, shape=Shapes.TxN, comment="infectivity"),
        AttributeDef(
            "gamma",
            type=float,
            shape=Shapes.TxN,
            comment="progression from infected to recovered",
        ),
        AttributeDef(
            "xi",
            type=float,
            shape=Shapes.TxN,
            comment="progression from recovered to susceptible",
        ),
    ]

    def edges(self, symbols):
        [S, I, R] = symbols.all_compartments
        [β, γ, ξ] = symbols.all_requirements

        # formulate N so as to avoid dividing by zero;
        # this is safe in this instance because if the denominator is zero,
        # the numerator must also be zero
        N = Max(1, S + I + R)

        return [
            edge(S, I, rate=β * S * I / N),
            edge(I, R, rate=γ * I),
            edge(R, S, rate=ξ * R),
        ]
