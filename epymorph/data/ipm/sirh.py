"""Defines a compartmental IPM for a generic SIRH model."""

from sympy import Max

from epymorph.attribute import AttributeDef
from epymorph.compartment_model import CompartmentModel, compartment, edge, fork
from epymorph.data_shape import Shapes


class SIRH(CompartmentModel):
    """A basic SIRH model."""

    compartments = [
        compartment("S"),
        compartment("I"),
        compartment("R"),
        compartment("H", tags=["immobile"]),
    ]

    requirements = [
        AttributeDef("beta", type=float, shape=Shapes.TxN, comment="infectivity"),
        AttributeDef("gamma", type=float, shape=Shapes.TxN, comment="recovery rate"),
        AttributeDef("xi", type=float, shape=Shapes.TxN, comment="immune waning rate"),
        AttributeDef(
            "hospitalization_prob",
            type=float,
            shape=Shapes.TxN,
            comment="a ratio of cases which are expected to require hospitalization",
        ),
        AttributeDef(
            "hospitalization_duration",
            type=float,
            shape=Shapes.TxN,
            comment="the mean duration of hospitalization, in days",
        ),
    ]

    def edges(self, symbols):
        [S, I, R, H] = symbols.all_compartments
        [β, γ, ξ, h_prob, h_dur] = symbols.all_requirements

        # formulate N so as to avoid dividing by zero;
        # this is safe in this instance because if the denominator is zero,
        # the numerator must also be zero
        N = Max(1, S + I + R + H)

        return [
            edge(S, I, rate=β * S * I / N),
            fork(
                edge(I, H, rate=γ * I * h_prob),
                edge(I, R, rate=γ * I * (1 - h_prob)),
            ),
            edge(H, R, rate=H / h_dur),
            edge(R, S, rate=ξ * R),
        ]
