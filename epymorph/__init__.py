"""epymorph's main package and main exports"""

from numpy import seterr

import epymorph.compartment_model as IPM
import epymorph.initializer as init
from epymorph.data import ipm_library, mm_library
from epymorph.data_shape import Shapes
from epymorph.data_type import CentroidType, SimDType
from epymorph.draw import render, render_and_save
from epymorph.log.messaging import sim_messaging
from epymorph.plots import plot_event, plot_pop
from epymorph.rume import Gpm, MultistrataRume, Rume, SingleStrataRume
from epymorph.simulation import AttributeDef, default_rng
from epymorph.simulator.basic.basic_simulator import BasicSimulator
from epymorph.time import TimeFrame

# set numpy errors to raise exceptions instead of warnings;
# useful for catching simulation errors
seterr(all="raise")

__all__ = [
    "IPM",
    "ipm_library",
    "mm_library",
    "SimDType",
    "CentroidType",
    "Shapes",
    "TimeFrame",
    "AttributeDef",
    "init",
    "Rume",
    "Gpm",
    "SingleStrataRume",
    "MultistrataRume",
    "BasicSimulator",
    "sim_messaging",
    "plot_event",
    "plot_pop",
    "default_rng",
    "render",
    "render_and_save",
]
