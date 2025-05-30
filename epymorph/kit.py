"""
A convenient star-import for the most-commonly-used features in epymorph.

Most of the included symbols are classes or functions which you can find by searching
their names. The three exceptions are `init`, `ipm`, and `mm` which are whole Python
modules or packages representing our built-in component libraries.

- `init` is the `epymorph.initializer` module
- `ipm` is the `epymorph.data.ipm` package
- `mm` is the `epymorph.data.mm` package

Included this way, for instance, you can instantiate the built-in SIRS IPM like:

```
my_ipm = ipm.SIRS()
```
"""

import epymorph.data.ipm as ipm
import epymorph.data.mm as mm
import epymorph.initializer as init
from epymorph.attribute import AttributeDef
from epymorph.compartment_model import (
    BIRTH,
    DEATH,
    CompartmentModel,
    ModelSymbols,
    MultiStrataModelSymbols,
    TransitionDef,
    compartment,
    edge,
    fork,
)
from epymorph.data_shape import Shapes
from epymorph.data_type import CentroidDType, CentroidType, SimDType
from epymorph.geography.custom import CustomScope
from epymorph.geography.us_census import (
    BlockGroupScope,
    CountyScope,
    StateScope,
    TractScope,
)
from epymorph.log.messaging import sim_messaging
from epymorph.params import (
    ParamFunction,
    ParamFunctionNode,
    ParamFunctionNodeAndCompartment,
    ParamFunctionNodeAndNode,
    ParamFunctionNumpy,
    ParamFunctionScalar,
    ParamFunctionTime,
    ParamFunctionTimeAndNode,
)
from epymorph.rume import GPM, MultiStrataRUME, MultiStrataRUMEBuilder, SingleStrataRUME
from epymorph.simulation import default_rng
from epymorph.simulator.basic.basic_simulator import BasicSimulator
from epymorph.time import TimeFrame

__all__ = [
    # Built-in Components
    "init",
    "ipm",
    "mm",
    # Compartment Models
    "CompartmentModel",
    "ModelSymbols",
    "TransitionDef",
    "compartment",
    "edge",
    "fork",
    "BIRTH",
    "DEATH",
    # Single Strata RUME
    "SingleStrataRUME",
    # Multi-strata RUME
    "MultiStrataRUME",
    "MultiStrataRUMEBuilder",
    "GPM",
    "MultiStrataModelSymbols",
    # Simulator
    "BasicSimulator",
    "sim_messaging",
    "default_rng",
    # Scopes
    "CustomScope",
    "StateScope",
    "CountyScope",
    "TractScope",
    "BlockGroupScope",
    # ParamFunctions
    "ParamFunction",
    "ParamFunctionNumpy",
    "ParamFunctionScalar",
    "ParamFunctionTime",
    "ParamFunctionNode",
    "ParamFunctionNodeAndNode",
    "ParamFunctionNodeAndCompartment",
    "ParamFunctionTimeAndNode",
    # Utilities and Data Types
    "TimeFrame",
    "AttributeDef",
    "Shapes",
    "SimDType",
    "CentroidType",
    "CentroidDType",
]
"""When you do `from epymorph.kit import *` you get all of these symbols."""
