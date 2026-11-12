import numpy as np
import pytest

from epymorph import initializer as init
from epymorph.data import ipm, mm
from epymorph.geography.us_census import StateScope
from epymorph.rume import RUME, SingleStrataRUME
from epymorph.simulator.basic.output import Output
from epymorph.time import TimeFrame


@pytest.fixture
def rume() -> RUME[StateScope]:
    """A very simple RUME with no external data requirements."""
    return SingleStrataRUME.build(
        ipm=ipm.SIRS(),
        mm=mm.No(),
        init=init.SingleLocation(location=0, seed_size=100),
        scope=StateScope.in_states(["04", "35"], year=2020),
        time_frame=TimeFrame.of("2021-01-01", 4),
        params={
            "beta": 0.4,
            "gamma": 1 / 10,
            "xi": 1 / 90,
            "population": [200_000, 100_000],
        },
    )


HOME_COMPARTMENT_OFFSET = 10_000
HOME_EVENT_OFFSET = 1_000


@pytest.fixture
def output(rume: RUME[StateScope]) -> Output:
    """Construct an output without running a simulation."""
    compartments = np.stack(
        [np.arange(start, start + 800, 100) for start in (3000, 2000, 1000)],
        axis=-1,
    ).reshape(4, 2, 3)
    events = np.stack(
        [np.arange(start, start + 80, 10) for start in (300, 200, 100)],
        axis=-1,
    ).reshape(4, 2, 3)
    return Output(
        rume=rume,
        initial=compartments[0],
        visit_compartments=compartments,
        visit_events=events,
        # add an unrealistically large offset to home distinguish from visit data
        home_compartments=compartments + HOME_COMPARTMENT_OFFSET,
        home_events=events + HOME_EVENT_OFFSET,
    )
