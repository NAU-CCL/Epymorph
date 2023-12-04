"""epymorph's main package and main exports"""

import epymorph.compartment_model as IPM
from epymorph.data import geo_library, ipm_library, mm_library
from epymorph.engine.standard_sim import StandardSimulation
from epymorph.plots import plot_event, plot_pop
from epymorph.simulation import SimDType, TimeFrame, default_rng, sim_messaging

__all__ = [
    'IPM',
    'ipm_library',
    'mm_library',
    'geo_library',
    'StandardSimulation',
    'plot_event',
    'plot_pop',
    'SimDType',
    'TimeFrame',
    'default_rng',
    'sim_messaging',
]
