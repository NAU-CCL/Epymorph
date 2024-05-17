"""epymorph's main package and main exports"""

from numpy import seterr

import epymorph.compartment_model as IPM
from epymorph.data import geo_library, ipm_library, mm_library
from epymorph.data_shape import Shapes
from epymorph.data_type import CentroidDType, SimDType
from epymorph.draw import render, render_and_save
from epymorph.engine.standard_sim import StandardSimulation
from epymorph.log.messaging import sim_messaging
from epymorph.plots import plot_event, plot_pop
from epymorph.proxy import dim, geo
from epymorph.simulation import TimeFrame, default_rng

# set numpy errors to raise exceptions instead of warnings, useful for catching
# simulation errrors
seterr(all='raise')

__all__ = [
    'IPM',
    'ipm_library',
    'mm_library',
    'geo_library',
    'Shapes',
    'StandardSimulation',
    # TODO: the names 'geo' and 'dim' are so widely used
    # that this export winds up causing problems.
    # We should require the user to import proxies intentionally.
    'geo',
    'dim',
    'plot_event',
    'plot_pop',
    'SimDType',
    'CentroidDType',
    'TimeFrame',
    'default_rng',
    'sim_messaging',
    'render',
    'render_and_save'
]
