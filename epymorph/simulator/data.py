"""Functions for managing simulation data."""
from typing import Callable, Generator, Mapping, Sequence, TypeVar

import numpy as np
from sympy import Expr

from epymorph.data_shape import DataShapeMatcher, SimDimensions
from epymorph.data_type import AttributeArray, SimArray
from epymorph.database import (AbsoluteName, Database, DatabaseWithFallback,
                               DatabaseWithStrataFallback, ModuleNamespace,
                               NamePattern)
from epymorph.error import AttributeException, InitException
from epymorph.params import (ParamExpressionTimeAndNode, ParamFunction,
                             ParamValue)
from epymorph.rume import GEO_LABELS, Gpm, Rume
from epymorph.simulation import NamespacedAttributeResolver
from epymorph.util import NumpyTypeError, check_ndarray, match

_EvalFunction = Callable[
    [AbsoluteName, list[AbsoluteName], ParamValue | None],
    AttributeArray
]

ParamValueT = TypeVar('ParamValueT')


def _evaluation_context(
    rume: Rume,
    override_params: Mapping[NamePattern, ParamValue],
    rng: np.random.Generator,
) -> tuple[Database[AttributeArray], _EvalFunction]:
    """Constructs a function for evaluating parameters while accumulating results to the returned Database."""
    format_name = rume.name_display_formatter()

    # vals_db stores the raw values as provided by the user.
    vals_db = DatabaseWithFallback(
        # Simulation overrides...
        dict(override_params.items()),
        # falls back to RUME params...
        DatabaseWithStrataFallback(
            data=dict(rume.params.items()),
            children={
                # which falls back to GPM params, as scoped to that GPM
                f"gpm:{strata}": Database[ParamValue]({
                    k.to_absolute(f"gpm:{strata}"): v
                    for k, v in gpm.params.items()
                })
                for strata, gpm in rume.original_gpms.items()
            },
        ),
    )

    # This is the database of parameter evaluation results.
    # It is mutable so that we can add values as we go.
    # We basically need a post-order tree traversal to support nested parameters.
    attr_db = Database[AttributeArray]({})

    # Evaluation caching:
    # pylint: disable=dangerous-default-value
    def use_eval_cache(
        name: NamePattern,
        raw_value: ParamValueT,
        or_else: Callable[[ParamValueT], AttributeArray],
        *, cache: dict[NamePattern, AttributeArray] = {},
        # WARNING: I am deliberately (ab)using a dict value as a default parameter here,
        # which will be the same instance for every function call. This is not typical python.
        # However in this case it has the nice effect of limiting `cache`'s scope so it can
        # only be accessed in this function. And, since it's a locally-scoped function, is acceptable.
    ) -> AttributeArray:
        """
        Literal values (scalars, arrays) can be safely re-used in every namespace to which they apply.
        For such values, it's advantageous to keep a single copy in memory. This function sees to that.
        When a value can be cached, use this function to evaluate it.
        """
        # NOTE: the critical distinction for which values can be cached and which cannot lies in
        # whether or not they are sensitive to the namespace in which they're evaluated.
        # Our Function types can be sensitive to their namespace, and so evaluation results must be isolated
        # to the most-specific scope -- e.g., "gpm:1::ipm::alpha" -- even if they were provided for a more-general
        # scope -- e.g., "*::*::alpha".
        if (value := cache.get(name)) is not None:
            return value
        value = or_else(raw_value)
        cache[name] = value
        return value

    def evaluate(name: AbsoluteName, chain: list[AbsoluteName], default_value: ParamValue | None) -> AttributeArray:
        # Evaluate a parameter and store its value to `attr_db`

        # `chain` tracks the path of nested attribute resolutions involved.
        # If we are currently trying to resolve an attribute that already occurred earlier in the chain,
        # we've found a circular dependency!
        if name in chain:
            msg = f"Circular dependency in evaluation of parameter {name}"
            raise AttributeException(msg)

        # If we already have a value for this name; skip!
        # NOTE: this is slightly different from the evaluation cache
        # which indexes on the supplied parameter name;
        # _this_ indexes on the fully-qualified attribute name.
        if (attr_match := attr_db.query(name)) is not None:
            return attr_match.value

        if (val_match := vals_db.query(name)) is not None:
            # User provided a parameter value to use.
            raw_value = val_match.value
            match_pattern = val_match.pattern
        elif default_value is not None:
            # User did not provide a value, but we have a default.
            raw_value = default_value
            match_pattern = name.to_pattern()
        else:
            # No value!
            msg = f"Missing required parameter: '{format_name(name)}'"
            raise AttributeException(msg)

        # Raw value conversions:
        if isinstance(raw_value, Expr):
            # Automatically convert sympy expressions into a ParamFunction instance.
            try:
                raw_value = ParamExpressionTimeAndNode(raw_value)
            except ValueError as e:
                msg = f"'{format_name(match_pattern)}' {e}"
                raise AttributeException(msg) from None

        # Otherwise, evaluate and store the parameter based on its type.
        if isinstance(raw_value, ParamFunction):
            # ParamFunction: first evaluate all dependencies of this function (recursively),
            # then evaluate the function itself.
            namespace = name.to_namespace()
            for dependency in raw_value.attributes:
                dep_name = namespace.to_absolute(dependency.name)
                evaluate(dep_name, [*chain, name], dependency.default_value)
            data = NamespacedAttributeResolver(attr_db, rume.dim, namespace)
            value = raw_value(data, rume.dim, rng)
        elif isinstance(raw_value, type) and issubclass(raw_value, ParamFunction):
            msg = f"Invalid parameter: '{format_name(match_pattern)}' "\
                "is a ParamFunction class instead of an instance."
            raise AttributeException(msg)

        # elif isinstance(param, Adrio): # TODO: adrios as param values!

        elif isinstance(raw_value, np.ndarray):
            # numpy array: make a copy so we don't risk unexpected mutations
            value = use_eval_cache(match_pattern, raw_value,
                                   lambda x: x.copy())

        elif isinstance(raw_value, int | float | str | tuple | Sequence):
            # scalar value or python collection: re-pack it as a numpy array
            value = use_eval_cache(match_pattern, raw_value,
                                   lambda x: np.asarray(x, dtype=None))

        else:
            # bad value!
            msg = f"Invalid parameter: '{format_name(match_pattern)}' "\
                f"is not a supported type (found: {type(raw_value)})"
            raise AttributeException(msg)

        # Store the value to our database and the eval cache.
        attr_db.update(name.to_pattern(), value)
        return value

    return attr_db, evaluate


def evaluate_params(
    rume: Rume,
    override_params: Mapping[NamePattern, ParamValue],
    rng: np.random.Generator,
) -> Database[AttributeArray]:
    """
    Evaluates the attributes for a RUME, combining all of its included parameter values
    with any given override parameters. Returns a Database which is fully resolved and normalized
    for the needs of the RUME.
    """
    # Collect all attribute errors to raise as a group.
    errors = list[AttributeException]()

    # Evaluate every attribute required by the RUME.
    attr_db, evaluate = _evaluation_context(rume, override_params, rng)

    rume_attributes: list[tuple[AbsoluteName, ParamValue | None]] = [
        *((name, attr.default_value) for name, attr in rume.attributes.items()),
        # Artificially require the special geo labels attribute.
        (GEO_LABELS, rume.scope.get_node_ids()),
    ]

    for name, default_value in rume_attributes:
        try:
            evaluate(name, [], default_value)
        except AttributeException as e:
            errors.append(e)

    if len(errors) > 0:
        raise ExceptionGroup("Errors found accessing attributes.", errors)

    return attr_db


def evaluate_param(
    rume: Rume,
    param: str | AbsoluteName,
    rng: np.random.Generator | None = None,
) -> AttributeArray:
    """
    A utility to evaluate a single named parameter in the context of a RUME, perhaps for debug or display purposes.
    `param` can be an absolute name or a string which will be parsed as such, using `AbsoluteName.parse_with_defaults()`
    to allow for partial names.
    """
    if not isinstance(param, AbsoluteName):
        param = AbsoluteName.parse_with_defaults(
            param,
            strata="(unspecified)",
            module="(unspecified)",
        )
    rng = rng or np.random.default_rng()
    _, evaluate = _evaluation_context(rume, {}, rng)
    return evaluate(param, [], None)


def validate_attributes(
    rume: Rume,
    data: Database[AttributeArray],
) -> None:
    """
    Validate all attributes in a RUME; raises an ExceptionGroup containing all errors.
    """
    def validate() -> Generator[AttributeException, None, None]:
        for name, attr in rume.attributes.items():
            attr_match = data.query(name)
            if attr_match is None:
                msg = f"Missing required parameter: '{name}'"
                yield AttributeException(msg)
            else:
                try:
                    check_ndarray(
                        attr_match.value,
                        dtype=match.dtype_cast(attr.dtype),
                        shape=DataShapeMatcher(
                            attr.shape, rume.dim, allow_broadcast=True),
                    )
                except NumpyTypeError as e:
                    msg = f"Attribute '{attr_match.pattern}' is not properly specified. {e}"
                    yield AttributeException(msg)

    # Collect all attribute errors to raise as a group.
    errors = list(validate())
    if len(errors) > 0:
        raise ExceptionGroup("Errors found accessing attributes.", errors)


def initialize_rume(
    rume: Rume,
    rng: np.random.Generator,
    data: Database[AttributeArray],
) -> SimArray:
    """
    Executes Initializers for a multi-strata simulation by running each strata's
    Initializer and combining the results. Raises InitException if anything goes wrong.
    """
    def init_strata(strata: str, gpm: Gpm) -> SimArray:
        namespace = ModuleNamespace(f"gpm:{strata}", "init")
        strata_dim = SimDimensions.build(
            rume.dim.tau_step_lengths,
            rume.dim.start_date,
            rume.dim.days,
            rume.dim.nodes,
            gpm.ipm.num_compartments,
            gpm.ipm.num_events,
        )
        strata_data = NamespacedAttributeResolver(data, strata_dim, namespace)
        return gpm.init(strata_data, strata_dim, rng)

    try:
        return np.column_stack([
            init_strata(strata, gpm)
            for strata, gpm in rume.original_gpms.items()
        ])
    except InitException as e:
        raise e
    except Exception as e:
        raise InitException("Initializer failed during execution.") from e
