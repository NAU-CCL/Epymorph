from __future__ import annotations

import numpy as np

from epymorph.clock import Tick
from epymorph.context import Events, SimContext, SimDType
from epymorph.ipm.ipm import Ipm, IpmBuilder
from epymorph.movement.world import Location


def load() -> IpmBuilder:
    return NoModelBuilder()


class NoModelBuilder(IpmBuilder):
    def __init__(self):
        super().__init__(1, 0)

    def verify(self, ctx: SimContext) -> None:
        if 'population' not in ctx.geo:
            raise Exception("geo missing population")

    def build(self, ctx: SimContext) -> Ipm:
        return NoModel(ctx)


class NoModel(Ipm):
    """A 'null' model with only one compartment per population and no event transitions."""

    def __init__(self, ctx: SimContext):
        super().__init__(ctx)

    def events(self, loc: Location, tick: Tick) -> Events:
        return np.zeros(0, dtype=SimDType)

    def apply_events(self, loc: Location, es: Events) -> None:
        pass
