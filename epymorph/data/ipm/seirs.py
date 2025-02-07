from sympy import Max

from epymorph.attribute import AttributeDef
from epymorph.compartment_model import CompartmentModel, compartment, edge
from epymorph.data_shape import Shapes


class SEIRS(CompartmentModel):
    """A basic SEIRS model."""

    compartments = [
        compartment("S"),
        compartment("E"),
        compartment("I"),
        compartment("R"),
    ]
    requirements = [
        AttributeDef("beta", type=float, shape=Shapes.TxN, comment="infectivity"),
        AttributeDef(
            "sigma",
            type=float,
            shape=Shapes.TxN,
            comment="progression from exposed to infected",
        ),
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
        [S, E, I, R] = symbols.all_compartments
        [β, σ, γ, ξ] = symbols.all_requirements

        N = Max(1, S + E + I + R)

        return [
            edge(S, E, rate=β * S * I / N),
            edge(E, I, rate=σ * E),
            edge(I, R, rate=γ * I),
            edge(R, S, rate=ξ * R),
        ]
