"""Functions for managing simulation data."""

import numpy as np

from epymorph.data_shape import (
    DataShapeMatcher,
    Shapes,
    SimDimensions,
)
from epymorph.data_type import (
    SimArray,
    SimDType,
)
from epymorph.database import (
    DataResolver,
    ModuleNamespace,
)
from epymorph.error import InitException
from epymorph.rume import Gpm, Rume
from epymorph.simulation import (
    gpm_strata,
)
from epymorph.util import (
    NumpyTypeError,
    check_ndarray,
    match,
)


# TODO: see if we can put this somewhere better...
def initialize_rume(
    rume: Rume,
    rng: np.random.Generator,
    data: DataResolver,
) -> SimArray:
    """
    Executes Initializers for a multi-strata simulation by running each strata's
    Initializer and combining the results. Raises InitException if anything goes wrong.
    """

    def init_strata(gpm: Gpm) -> SimArray:
        namespace = ModuleNamespace(gpm_strata(gpm.name), "init")
        strata_dim = SimDimensions.build(
            rume.dim.tau_step_lengths,
            rume.dim.start_date,
            rume.dim.days,
            rume.dim.nodes,
            gpm.ipm.num_compartments,
            gpm.ipm.num_events,
        )

        result = gpm.init.with_context_internal(
            namespace,
            data,
            strata_dim,
            rume.scope,
            rng,
        ).evaluate()

        # Result validation: it must be an NxC array of integers
        # where no value is less than zero.
        try:
            check_ndarray(
                result,
                dtype=match.dtype(SimDType),
                shape=DataShapeMatcher(Shapes.NxC, strata_dim, allow_broadcast=False),
            )
        except NumpyTypeError as e:
            err = f"Invalid return type from '{gpm.init.__class__.__name__}'"
            raise InitException(err) from e

        if np.min(result) < 0:
            err = (
                f"Initializer '{gpm.init.__class__.__name__}' returned "
                "values less than zero"
            )
            raise InitException(err)

        return result

    try:
        return np.column_stack([init_strata(gpm) for gpm in rume.strata])
    except InitException as e:
        raise e
    except Exception as e:
        raise InitException("Initializer failed during execution.") from e
