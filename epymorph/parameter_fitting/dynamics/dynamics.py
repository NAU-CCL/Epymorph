from abc import ABC


class Dynamics(ABC): ...


class GeometricBrownianMotion(Dynamics):
    def __init__(self, volatility=0.1) -> None:
        self.volatility = volatility


class Calvetti(Dynamics):
    def __init__(self, a=0.9):
        self.a = a
