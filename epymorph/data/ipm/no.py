from __future__ import annotations

import numpy as np

from epymorph.clock import Tick
from epymorph.context import SimContext
from epymorph.epi import Ipm, IpmBuilder
from epymorph.util import Compartments, Events
from epymorph.world import Location


def load() -> IpmBuilder:
    return NoModelBuilder()


class NoModelBuilder(IpmBuilder):
    def __init__(self):
        super().__init__(1, 0)
    
    def verify(self, ctx: SimContext) -> None:
        if 'population' not in ctx.geo:
            raise Exception("geo missing population")
    
    def initialize_compartments(self, ctx: SimContext) -> list[Compartments]:
        population = ctx.geo['population']
        num_nodes = len(population)
        cs = [self.compartment_array() for _ in range(num_nodes)]
        for i in range(num_nodes):
            cs[i][0] = population[i]
        return cs
    
    def build(self, ctx: SimContext) -> Ipm:
        return NoModel(ctx)


class NoModel(Ipm):
    """A 'null' model with only one compartment per population and no event transitions."""

    def __init__(self, ctx: SimContext):
        super().__init__(ctx)

    def events(self, loc: Location, tick: Tick) -> Events:
        return np.zeros(0, dtype=np.int_)
    
    def apply_events(self, loc: Location, es: Events) -> None:
        pass
