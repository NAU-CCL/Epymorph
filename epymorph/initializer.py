"""
Initializers are responsible for setting up the starting populations at each node,
and which disease compartments they belong to. There are potentially many ways to 
do this, driven by source data from the geo or simulation parameters, so this 
module provides a uniform interface to accomplish the task, as well as a few
common implementations.
"""
import inspect
from functools import partial
from typing import Callable, cast

import numpy as np
from numpy.typing import NDArray

from epymorph.context import Compartments, SimContext, SimDType
from epymorph.util import NumpyTypeError, check_ndarray

Initializer = Callable[..., Compartments]
"""
An Initializer is just a function that takes any number of arguments
and returns a Compartments array.
"""


class InitializerException(Exception):
    """Unable to initialize the populations by compartment for a simulation."""


def initialize(init: Initializer, ctx: SimContext) -> Compartments:
    """Executes an initializer, attempting to fill arguments from the context."""
    # NOTE: initializers and partials generated from them should exclusively use keyword arguments

    partial_kwargs = set()
    if isinstance(init, partial):
        # partial funcs require a bit of extra massaging
        init_name = init.func.__name__
        partial_kwargs = set(init.keywords.keys())
        init = cast(Initializer, init)
    else:
        init_name = getattr(init, '__name__', 'UNKNOWN')

    # get list of args for function
    sig = inspect.signature(init)

    # Build up the arguments dict.
    kwargs = {}
    for p in sig.parameters.values():
        if p.kind in ['POSITIONAL_ONLY', 'VAR_POSITIONAL', 'VAR_KEYWORD']:
            # var-args not supported
            msg = f"'{init_name}' requires an argument of an unsupported kind: {p.name} is a {p.kind} parameter"
            raise InitializerException(msg)

        if p.name in partial_kwargs:
            # Skip any args in partial_kwargs, otherwise we're just repeating them.
            continue
        elif p.annotation == SimContext:
            # If context needed, supply context!
            kwargs[p.name] = ctx
        elif p.name in ctx.geo.attributes:
            # If name is in geo, use that.
            kwargs[p.name] = ctx.geo[p.name]
        elif p.name in ctx.param:
            # If name is in params, use that.
            kwargs[p.name] = ctx.param[p.name]
        elif p.default is not inspect.Parameter.empty:
            # If arg has a default, use that.
            kwargs[p.name] = p.default
        else:
            # Unable to auto-wire the arg!
            msg = f"'{init_name}' requires an argument that we couldn't auto-wire: {p.name} ({p.annotation})"
            raise InitializerException(msg)

    # Execute function with matched args.
    result = init(**kwargs)
    try:
        _, N, C, _ = ctx.TNCE
        check_ndarray(result, SimDType, (N, C))
    except NumpyTypeError as e:
        raise InitializerException(f"Invalid return type from '{init_name}'") from e
    return result


def _default_compartments(ctx: SimContext) -> Compartments:
    """
    Create a default compartments by population array (N,C).
    This places everyone from geo['population'] in the first
    compartment, and zero everywhere else.
    """
    _, N, C, _ = ctx.TNCE
    cs = np.zeros((N, C), dtype=SimDType)
    cs[:, 0] = ctx.geo['population']
    return cs


def _as_arg_exception(name: str, exception_or_message: Exception | str) -> InitializerException:
    """Uniform error messaging for invalid initializer arguments."""
    if isinstance(exception_or_message, Exception):
        root_cause = f"{exception_or_message.__class__.__name__}: {str(exception_or_message)}"
    else:
        root_cause = exception_or_message
    return InitializerException(f"Invalid argument '{name}'\n{root_cause}")


# Pre-baked initializer implementations


def explicit(ctx: SimContext, initials: NDArray[SimDType]) -> Compartments:
    """
    Set all compartments explicitly. Parameters:
    - `initials` a (N,C) numpy array of SimDType

    Note: this does not account for the geo's population numbers.
    """
    try:
        _, N, C, _ = ctx.TNCE
        check_ndarray(initials, SimDType, (N, C))
    except NumpyTypeError as e:
        raise _as_arg_exception('initials', e) from None
    return initials.copy()


def proportional(ctx: SimContext, ratios: NDArray[np.integer | np.floating]) -> Compartments:
    """
    Set all compartments as a proportion of their population according to the geo.
    The parameter array provided to this initializer will be normalized, then multiplied
    element-wise by the geo population. So you're free to express the proportions in any way,
    (as long as no row is zero). Parameters:
    - `ratios` a (C,) or (N,C) numpy array describing the ratios for each compartment
    """
    try:
        _, N, C, _ = ctx.TNCE
        check_ndarray(ratios, dtype=[np.integer, np.floating], shape=[(C,), (N, C)])
    except NumpyTypeError as e:
        raise _as_arg_exception('ratios', e) from None

    ratios = np.broadcast_to(ratios, (N, C))
    row_sums = cast(NDArray[np.float64], np.sum(ratios, axis=1, dtype=np.float64))
    if np.any(row_sums <= 0):
        raise _as_arg_exception('ratios', "One or more rows sum to zero or less.")

    result = ctx.geo['population'][:, np.newaxis] * (ratios / row_sums[:, np.newaxis])
    return result.round().astype(SimDType, copy=True)


def indexed_locations(ctx: SimContext, selection: NDArray[np.intp], seed_size: int) -> Compartments:
    """
    Infect a fixed number of people distributed (proportional to their population) across a
    selection of nodes. A multivariate hypergeometric draw using the available populations is
    used to distribute infections. Parameters:
    - `selection` a one-dimensional array of indices; all values must be in range (-N,+N)
    - `seed_size` the number of individuals to infect in total
    """
    try:
        check_ndarray(selection, np.intp, dimensions=1)
    except NumpyTypeError as e:
        raise _as_arg_exception('selection', e) from None
    if not np.all((-ctx.nodes < selection) & (selection < ctx.nodes)):
        raise _as_arg_exception(
            'selection', f"Some indices are out of range ({-ctx.nodes}, {ctx.nodes})")

    if not isinstance(seed_size, int) or seed_size < 0:
        raise _as_arg_exception('seed_size', "Must be a non-negative integer value.")

    selected = ctx.geo['population'][selection]
    available = selected.sum()
    if available < seed_size:
        msg = f"Attempted to infect {seed_size} individuals but only had {available} available."
        raise InitializerException(msg)

    # Randomly select individuals from each of the selected locations.
    infected = ctx.rng.multivariate_hypergeometric(selected, seed_size)

    result = _default_compartments(ctx)

    # Special case: the "no" IPM has only one compartment!
    # Technically it would be more "correct" to choose a different initializer,
    # but it's convenient to allow this special case for ease of testing.
    if ctx.compartments == 1:
        return result

    for i, n in zip(selection, infected):
        result[i][0] -= n
        result[i][1] += n
    return result


def labeled_locations(ctx: SimContext, labels: NDArray[np.str_], seed_size: int) -> Compartments:
    """
    Infect a fixed number of people across a specified selection of locations (by label). Parameters:
    - `labels` the labels of the locations to select for infection
    - `seed_size` the number of individuals to infect in total
    """
    try:
        check_ndarray(labels, np.str_)
    except NumpyTypeError as e:
        raise _as_arg_exception('label', e) from None

    geo_labels = ctx.geo['label']
    if not np.all(np.isin(labels, geo_labels)):
        raise _as_arg_exception(
            'label', "Some labels are not in the geo model.")

    (selection,) = np.isin(geo_labels, labels).nonzero()
    return indexed_locations(ctx, selection, seed_size)


def single_location(ctx: SimContext, location: int, seed_size: int) -> Compartments:
    """
    Infect a fixed number of people at a single location (by index).
    (This is the default initialization logic, mostly due to legacy reasons.)
    """
    if not isinstance(location, int) or not -ctx.nodes < location < ctx.nodes:
        raise _as_arg_exception('location',
                                f"Must be a valid index to an array of {ctx.nodes} populations.")

    selection = np.array([location], dtype=np.intp)
    return indexed_locations(ctx, selection, seed_size)


def random_locations(ctx: SimContext, num_locations: int, seed_size: int) -> Compartments:
    """
    Infect a total of `seed_size` individuals spread randomly across a random selection of locations.
    """
    if not isinstance(num_locations, int) or not 0 < num_locations <= ctx.nodes:
        raise _as_arg_exception('num_locations',
                                f"Must be an integer value between 1 and the number of locations ({ctx.nodes}).")
    if not isinstance(seed_size, int) or not 0 < seed_size:
        raise _as_arg_exception('seed_size',
                                "Must be greater than 0.")

    indices = np.arange(ctx.nodes, dtype=np.intp)
    selection = ctx.rng.choice(indices, num_locations)
    return indexed_locations(ctx, selection, seed_size)


def top_locations(ctx: SimContext, attribute: str, num_locations: int, seed_size: int) -> Compartments:
    """
    Infect a fixed number of people across a fixed number of locations,
    selecting the top locations as measured by a given geo attribute.
    """
    if not isinstance(attribute, str) or not attribute in ctx.geo.attributes:
        raise _as_arg_exception('attribute',
                                "Must name an existing attribute in the geo.")
    if not isinstance(num_locations, int) or not 0 < num_locations <= ctx.nodes:
        raise _as_arg_exception('num_locations',
                                f"Must be an integer value between 1 and the number of locations ({ctx.nodes}).")

    # `argpartition` chops an array in two halves (yielding indices of the original array):
    # all indices whose values are smaller than the kth element,
    # followed by the index of the kth element,
    # followed by all indices whose values are larger.
    # So by using -k we create a partition of the largest k elements at the end of the array,
    # then slice using [-k:] to get just those indices.
    # This should be O(k log k) and saves us from copying+sorting (or some-such)
    arr = ctx.geo[attribute]
    selection = np.argpartition(arr, -num_locations)[-num_locations:]
    return indexed_locations(ctx, selection, seed_size)


def bottom_locations(ctx: SimContext, attribute: str, num_locations: int, seed_size: int) -> Compartments:
    """
    Infect a fixed number of people across a fixed number of locations,
    selecting the bottom locations as measured by a given geo attribute.
    """
    if not isinstance(attribute, str) or not attribute in ctx.geo.attributes:
        raise _as_arg_exception('attribute',
                                "Must name an existing attribute in the geo.")
    if not isinstance(num_locations, int) or not 0 < num_locations <= ctx.nodes:
        raise _as_arg_exception('num_locations',
                                f"Must be an integer value between 1 and the number of locations ({ctx.nodes}).")

    # `argpartition` chops an array in two halves (yielding indices of the original array):
    # all indices whose values are smaller than the kth element,
    # followed by the index of the kth element,
    # followed by all indices whose values are larger.
    # So by using k we create a partition of the smallest k elements at the start of the array,
    # then slice using [:k] to get just those indices.
    # This should be O(k log k) and saves us from copying+sorting (or some-such)
    arr = ctx.geo[attribute]
    selection = np.argpartition(arr, num_locations)[:num_locations]
    return indexed_locations(ctx, selection, seed_size)


DEFAULT_INITIALIZER = single_location

initializer_library: dict[str, Initializer] = {
    'explicit': explicit,
    'proportional': proportional,
    'indexed_locations': indexed_locations,
    'labeled_locations': labeled_locations,
    'single_location': single_location,
    'random_locations': random_locations,
    'top_locations': top_locations,
    'bottom_locations': bottom_locations,
}
"""A library for the built-in initializer functions."""
