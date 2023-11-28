"""Initialization logic."""
import inspect
from functools import partial
from typing import cast

from numpy.typing import NDArray

from epymorph.engine.context import RumeContext
from epymorph.error import InitException
from epymorph.initializer import InitContext, Initializer
from epymorph.simulation import SimDType
from epymorph.util import NumpyTypeError, check_ndarray


def initialize(init: Initializer, ctx: RumeContext) -> NDArray[SimDType]:
    """Executes an initializer, attempting to auto-wire arguments from the context."""
    # NOTE: initializers and partials generated from them should exclusively use keyword arguments

    partial_kwargs = set[str]()
    if isinstance(init, partial):
        # partial funcs require a bit of extra massaging
        init_name = init.func.__name__
        partial_kwargs = set(init.keywords.keys())
        init = cast(Initializer, init)
    else:
        init_name = cast(str, getattr(init, '__name__', 'UNKNOWN'))

    # get list of args for function
    sig = inspect.signature(init)

    # Build up the arguments dict.
    kwargs = {}
    for p in sig.parameters.values():
        if p.kind in ['POSITIONAL_ONLY', 'VAR_POSITIONAL', 'VAR_KEYWORD']:
            # var-args not supported
            msg = f"'{init_name}' requires an argument of an unsupported kind: {p.name} is a {p.kind} parameter"
            raise InitException(msg)

        if p.name in partial_kwargs:
            # Skip any args in partial_kwargs, otherwise we're just repeating them.
            continue
        elif p.name == 'ctx' or p.annotation == InitContext:
            # If context needed, supply context!
            kwargs[p.name] = ctx
        elif p.name in ctx.geo.spec.attribute_map:
            # If name is in geo, use that.
            kwargs[p.name] = ctx.geo[p.name]
        elif p.name in ctx.params:
            # If name is in params, use that.
            kwargs[p.name] = ctx.params[p.name]
        elif p.default is not inspect.Parameter.empty:
            # If arg has a default, use that.
            kwargs[p.name] = p.default
        else:
            # Unable to auto-wire the arg!
            msg = f"'{init_name}' requires an argument that we couldn't auto-wire: {p.name} ({p.annotation})"
            raise InitException(msg)

    # Execute function with matched args.
    try:
        result = init(**kwargs)
    except InitException as e:
        raise e
    except Exception as e:
        raise InitException('Initializer failed during execution.') from e

    try:
        _, N, C, _ = ctx.dim.TNCE
        check_ndarray(result, SimDType, (N, C))
    except NumpyTypeError as e:
        raise InitException(f"Invalid return type from '{init_name}'") from e
    return result
