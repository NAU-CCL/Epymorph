"""
Initializers are responsible for setting up the initial conditions for the simulation:
the populations at each node, and which disease compartments they belong to.
There are potentially many ways to do this, driven by source data from
the geo or simulation parameters, so this module provides a uniform interface
to accomplish the task, as well as a few common implementations.
"""
import inspect
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Mapping, cast

import numpy as np
from numpy.typing import NDArray

from epymorph.data_shape import SimDimensions
from epymorph.data_type import RawParam, SimDType
from epymorph.error import InitException
from epymorph.params import normalize_params
from epymorph.simulation import GeoData, ParamsData
from epymorph.util import NumpyTypeError, check_ndarray

Initializer = Callable[..., NDArray[SimDType]]
"""
An Initializer is a function that takes an arbitrary set of parameters
and returns the initial conditions for a simulation, appropriate for the GEO and IPM
being simulated.

It answers the question: what are the initial populations of every node,
and what disease compartments are they in, at time-zero? (There are many ways
to answer this question depending on your goals, and hence many initializers!)

The expected return type is a numpy array of integers (compatible with SimDType)
whose shape is two-dimensions -- (N,C) -- with N being the number of GEO nodes
and C being the number of IPM compartments.

Initializer implementations can specify a variety of parameters that they need
to fulfill their function. These parameters will be auto-wired from all of the
available simulation data when the simulation starts. Data sources include
geo attributes, parameter attributes, and the initializer block of the toml file
(if running via the CLI). Matches from these sources happen by name -- so
if you specify an initializer parameter called 'population' and there's a geo
attribute called 'population', those can be auto-wired together. (If a geo attribute
and params attribute happened to use the same name, geo wins.) There is also a special
value to get the simulation's context info: the name 'ctx' and/or any parameter with
InitContext as its type annotation will be auto-wired with the context.
This gives access to useful things like the simulation dimensions and random number generator.
Of course you can always use partial function application to directly specify initializer arguments,
if that's your style.
"""


@dataclass(frozen=True)
class InitContext:
    """The subset of the context that initialization might need."""
    # This machine avoids circular deps.
    dim: SimDimensions
    geo: GeoData
    params: ParamsData
    rng: np.random.Generator


def _get_population(ctx: InitContext) -> NDArray[np.int64]:
    try:
        pop = ctx.geo['population']
        check_ndarray(pop, dtype=[np.int64], shape=(ctx.dim.nodes,))
        return cast(NDArray[np.int64], pop)
    except KeyError as e:
        raise InitException.for_arg('population', 'No such value in the geo.') from e
    except NumpyTypeError as e:
        raise InitException.for_arg('population') from e


# Pre-baked initializer implementations


def no_infection(ctx: InitContext) -> NDArray[SimDType]:
    """
    Create a default compartments by population array (N,C).
    This places everyone from geo['population'] in the first
    compartment, and zero everywhere else.
    """
    _, N, C, _ = ctx.dim.TNCE
    cs = np.zeros((N, C), dtype=SimDType)
    cs[:, 0] = _get_population(ctx)
    return cs


def explicit(ctx: InitContext, initials: NDArray[SimDType]) -> NDArray[SimDType]:
    """
    Set all compartments explicitly. Parameters:
    - `initials` a (N,C) numpy array of SimDType

    Note: this does not account for the geo's population numbers.
    """
    _, N, C, _ = ctx.dim.TNCE
    try:
        check_ndarray(initials, [SimDType], (N, C))
    except NumpyTypeError as e:
        raise InitException.for_arg('initials') from e
    return initials.copy()


def proportional(ctx: InitContext, ratios: NDArray[np.int64 | np.float64]) -> NDArray[SimDType]:
    """
    Set all compartments as a proportion of their population according to the geo.
    The parameter array provided to this initializer will be normalized, then multiplied
    element-wise by the geo population. So you're free to express the proportions in any way,
    (as long as no row is zero). Parameters:
    - `ratios` a (C,) or (N,C) numpy array describing the ratios for each compartment
    """
    _, N, C, _ = ctx.dim.TNCE
    try:
        check_ndarray(ratios, dtype=[np.int64, np.float64], shape=[(C,), (N, C)])
    except NumpyTypeError as e:
        raise InitException.for_arg('ratios') from e

    ratios = np.broadcast_to(ratios, (N, C)).astype(np.float64, copy=False)
    row_sums = cast(NDArray[np.float64], np.sum(ratios, axis=1, dtype=np.float64))
    if np.any(row_sums <= 0):
        raise InitException.for_arg('ratios', 'One or more rows sum to zero or less.')

    pop = _get_population(ctx)
    result = pop[:, np.newaxis] * (ratios / row_sums[:, np.newaxis])
    return result.round().astype(SimDType, copy=True)


def indexed_locations(ctx: InitContext, selection: NDArray[np.intp], seed_size: int) -> NDArray[SimDType]:
    """
    Infect a fixed number of people distributed (proportional to their population) across a
    selection of nodes. A multivariate hypergeometric draw using the available populations is
    used to distribute infections. Parameters:
    - `selection` a one-dimensional array of indices; all values must be in range (-N,+N)
    - `seed_size` the number of individuals to infect in total
    """
    N = ctx.dim.nodes

    try:
        check_ndarray(selection, [np.intp], dimensions=1)
    except NumpyTypeError as e:
        raise InitException.for_arg('selection') from e
    if not np.all((-N < selection) & (selection < N)):
        raise InitException.for_arg(
            'selection', f"Some indices are out of range ({-N}, {N})")

    if not isinstance(seed_size, int) or seed_size < 0:
        raise InitException.for_arg(
            'seed_size', 'Must be a non-negative integer value.')

    selected = _get_population(ctx)[selection]
    available = selected.sum()
    if available < seed_size:
        msg = f"Attempted to infect {seed_size} individuals but only had {available} available."
        raise InitException(msg)

    # Randomly select individuals from each of the selected locations.
    infected = ctx.rng.multivariate_hypergeometric(selected, seed_size)

    result = no_infection(ctx)

    # Special case: the "no" IPM has only one compartment!
    # Technically it would be more "correct" to choose a different initializer,
    # but it's convenient to allow this special case for ease of testing.
    if ctx.dim.compartments == 1:
        return result

    # TODO: we should probably have some way for the user to configure which compartment
    # is the "default" and which compartment is the "seed". Right now those are hard-coded
    # to compartments 0 and 1 respectively, but these initializers could support more models
    # if there were a setting for it.
    for i, n in zip(selection, infected):
        result[i][0] -= n
        result[i][1] += n
    return result


def labeled_locations(ctx: InitContext, labels: NDArray[np.str_], seed_size: int) -> NDArray[SimDType]:
    """
    Infect a fixed number of people across a specified selection of locations (by label). Parameters:
    - `labels` the labels of the locations to select for infection
    - `seed_size` the number of individuals to infect in total
    """
    try:
        check_ndarray(labels, [np.str_])
    except NumpyTypeError as e:
        raise InitException.for_arg('label') from e

    # 'label' is a required geo attribute; no need to check it here
    geo_labels = ctx.geo['label']
    if not np.all(np.isin(labels, geo_labels)):
        raise InitException.for_arg('label', 'Some labels are not in the geo model.')

    (selection,) = np.isin(geo_labels, labels).nonzero()
    return indexed_locations(ctx, selection, seed_size)


def single_location(ctx: InitContext, location: int, seed_size: int) -> NDArray[SimDType]:
    """
    Infect a fixed number of people at a single location (by index).
    (This is the default initialization logic, mostly due to legacy reasons.)
    """
    N = ctx.dim.nodes
    if not isinstance(location, int) or not -N < location < N:
        raise InitException.for_arg(
            'location', f"Must be a valid index to an array of {N} populations.")

    selection = np.array([location], dtype=np.intp)
    return indexed_locations(ctx, selection, seed_size)


def random_locations(ctx: InitContext, num_locations: int, seed_size: int) -> NDArray[SimDType]:
    """
    Infect a total of `seed_size` individuals spread randomly across a random selection of locations.
    """
    N = ctx.dim.nodes
    if not isinstance(num_locations, int) or not 0 < num_locations <= N:
        raise InitException.for_arg(
            'num_locations', f"Must be an integer value between 1 and the number of locations ({N}).")
    if not isinstance(seed_size, int) or not 0 < seed_size:
        raise InitException.for_arg('seed_size', 'Must be greater than 0.')

    indices = np.arange(N, dtype=np.intp)
    selection = ctx.rng.choice(indices, num_locations)
    return indexed_locations(ctx, selection, seed_size)


def top_locations(ctx: InitContext, attribute: str, num_locations: int, seed_size: int) -> NDArray[SimDType]:
    """
    Infect a fixed number of people across a fixed number of locations,
    selecting the top locations as measured by a given geo attribute.
    """
    N = ctx.dim.nodes
    if not isinstance(attribute, str) or not attribute in ctx.geo:
        raise InitException.for_arg(
            'attribute', 'Must name an existing attribute in the geo.')
    if not isinstance(num_locations, int) or not 0 < num_locations <= N:
        raise InitException.for_arg(
            'num_locations', f"Must be an integer value between 1 and the number of locations ({N}).")

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


def bottom_locations(ctx: InitContext, attribute: str, num_locations: int, seed_size: int) -> NDArray[SimDType]:
    """
    Infect a fixed number of people across a fixed number of locations,
    selecting the bottom locations as measured by a given geo attribute.
    """
    N = ctx.dim.nodes
    if not isinstance(attribute, str) or not attribute in ctx.geo:
        raise InitException.for_arg(
            'attribute', 'Must name an existing attribute in the geo.')
    if not isinstance(num_locations, int) or not 0 < num_locations <= N:
        raise InitException.for_arg(
            'num_locations', f"Must be an integer value between 1 and the number of locations ({N}).")

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
    'no_infection': no_infection,
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


def normalize_init_params(data: Mapping[str, RawParam]) -> dict[str, Any]:
    """Normalize parameters for initializers."""
    return {
        # Replace list values with numpy arrays.
        # NOTE: this is necessary for now, since TOML file input doesn't give us any way
        # to supply numpy arrays as input. However we can't array-ify _everything_ like we do
        # with params, because it makes sense to allow initializers to use other types for arguments.
        # A definite point of akwardness when it comes to auto-wiring initializer arguments from the
        # simulation parameters. We have to duplicate this normalization there, too, for it to be consistent.
        # In any case, we'll need come up with something smarter than this, but this is okay for now.
        key: np.asarray(value) if isinstance(value, list) else value
        for key, value in data.items()
    }


def initialize(
    init: Initializer,
    dim: SimDimensions,
    geo: GeoData,
    raw_params: Mapping[str, RawParam],
    rng: np.random.Generator
) -> NDArray[SimDType]:
    """
    Executes an initialization function, auto-wiring from available data where necessary.
    Auto-wiring initializer function parameters can provide an InitContext instance,
    a geo attribute, a params attribute, or a default argument -- if they have
    not already been provided using partial.
    """
    partial_kwargs = set[str]()
    if isinstance(init, partial):
        # partial funcs require a bit of extra massaging
        init_name = init.func.__name__
        partial_kwargs = set(init.keywords)
        init = cast(Initializer, init)
    else:
        init_name = cast(str, getattr(init, '__name__', 'UNKNOWN'))

    norm_params = normalize_params(raw_params, geo, dim)
    ctx = InitContext(dim, geo, norm_params, rng)
    init_params = normalize_init_params(raw_params)

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
        elif p.name == 'ctx' or p.annotation in [InitContext]:
            # If context needed, supply context!
            kwargs[p.name] = ctx
        elif p.name in geo:
            # If name is in geo, use that.
            kwargs[p.name] = geo[p.name]
        elif p.name in init_params:
            # If name is in params, use that.
            kwargs[p.name] = init_params[p.name]
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

    # NOTE: I'm boxing the np.min result as an int to convince Pylance/Pyright that the < operator
    # works. For some reason, it seems to be having trouble discovering operators in library code.
    # I'm not able to duplicate this in a sandbox environment, so it must be an obscure peculiarity
    # of our project. For now, better to work around it; this isn't in a performance-critical code path.
    if int(np.min(result)) < 0:
        raise InitException(f"Initializer '{init_name}' returned values less than zero")

    try:
        _, N, C, _ = dim.TNCE
        check_ndarray(result, [SimDType], (N, C))
    except NumpyTypeError as e:
        raise InitException(f"Invalid return type from '{init_name}'") from e
    return result
