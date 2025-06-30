"""Provide typed shims for the sympy functions we're using."""

from typing import Any, Callable, Mapping, cast

import sympy as sy


def to_symbol(name: str) -> sy.Symbol:
    """
    Create a symbol from the given string.

    To be consistent with the typing given, only include a single symbol.
    (But this is not checked.)
    """
    return cast(sy.Symbol, sy.symbols(name))


def simplify(expr: sy.Expr) -> sy.Expr:
    """Simplify the given expression."""
    return cast(sy.Expr, sy.simplify(expr))


def simplify_sum(exprs: list[sy.Expr]) -> sy.Expr:
    """Simplify the sum of the given expressions."""
    return cast(sy.Expr, sy.simplify(sy.Add(*exprs)))


SympyLambda = Callable[[list[Any]], Any]
"""(Vaguely) describes the result of lambdifying an expression."""


def lambdify(params: list[Any], expr: sy.Expr) -> SympyLambda:
    """
    Create a lambda function from the given expression,
    taking the given parameters and returning a single result.
    """
    # Note: calling lambdify with `[params]` means we will call it with a list of
    # arguments later (rather than having to spread the list)
    # i.e., f([1,2,3]) rather than f(*[1,2,3])
    # This is better because we have to construct the arguments at run-time as lists.
    return cast(SympyLambda, sy.lambdify([params], expr))


def lambdify_list(params: list[Any], exprs: list[sy.Expr]) -> SympyLambda:
    """
    Create a lambda function from the given expressions,
    taking the given parameters and returning a list of results.
    """
    return cast(SympyLambda, sy.lambdify([params], exprs))


def substitute(expr: sy.Expr, symbol_mapping: Mapping[sy.Symbol, sy.Symbol]) -> sy.Expr:
    """
    Substitute symbols in one expression according to the given mapping,
    and return a new expression.
    """
    return cast(sy.Expr, expr.subs(symbol_mapping))
