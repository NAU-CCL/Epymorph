To create a custom IPM, you will write an implementation of this class:

``` python
from epymorph.kit import *
from sympy import Max

class MyIPM(CompartmentModel):
    compartments = (
        compartment("S"),
        compartment("I"),
        compartment("R"),
    )

    requirements = [
        AttributeDef(
            "beta",
            type=float,
            shape=Shapes.TxN,
            comment="infection rate",
        ),
        AttributeDef(
            "gamma",
            type=float,
            shape=Shapes.TxN,
            comment="recovery rate",
        ),
    ]

    def edges(self, symbols):
        [S, I, R] = symbols.all_compartments
        [beta, gamma] = symbols.all_requirements
        N = Max(1, S + I + R)
        return [
            edge(S, I, rate=beta * S * I / N),
            edge(I, R, rate=gamma * I),
        ]
```
