"""epymorph's main package and main exports"""

from numpy import seterr

import epymorph.compartment_model as IPM
import epymorph.initializer as init
from epymorph.data import geo_library, ipm_library, mm_library
from epymorph.data_shape import Shapes
from epymorph.data_type import CentroidType, SimDType
from epymorph.draw import render, render_and_save
from epymorph.log.messaging import sim_messaging
from epymorph.plots import plot_event, plot_pop
from epymorph.rume import Gpm, Rume, RumeSymbols
from epymorph.simulation import AttributeDef, TimeFrame, default_rng
from epymorph.simulator.basic.basic_simulator import BasicSimulator

# set numpy errors to raise exceptions instead of warnings;
# useful for catching simulation errors
seterr(all='raise')

__all__ = [
    'IPM',
    'ipm_library',
    'mm_library',
    'geo_library',
    'SimDType',
    'CentroidType',
    'Shapes',
    'TimeFrame',
    'AttributeDef',
    'init',
    'Rume',
    'Gpm',
    'RumeSymbols',
    'BasicSimulator',
    'sim_messaging',
    'plot_event',
    'plot_pop',
    'default_rng',
    'render',
    'render_and_save'
]
