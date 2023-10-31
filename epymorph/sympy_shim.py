"""Provide typed shims for the sympy functions we're using."""

from typing import Any, Callable, cast

import sympy

Expr = sympy.Expr
Symbol = sympy.Symbol


def to_symbol(name: str) -> Symbol:
    """
    Create a symbol from the given string. To be consistent with the typing given, only include a single symbol.
    (But this is not checked.)
    """
    return cast(Symbol, sympy.symbols(name))


def simplify(expr: Expr) -> Expr:
    """Simplify the given expression."""
    return cast(Expr, sympy.simplify(expr))


def simplify_sum(exprs: list[Expr]) -> Expr:
    """Simplify the sum of the given expressions."""
    return cast(Expr, sympy.simplify(sympy.Add(*exprs)))


SympyLambda = Callable[[list[Any]], Any]
"""(Vaguely) describes the result of lambdifying an expression."""


def lambdify(params: list[Any], expr: Expr) -> SympyLambda:
    """Create a lambda function from the given expression, taking the given parameters and returning a single result."""
    # Note: calling lambdify with `[params]` means we will call it with a list of arguments later (rather than having to spread the list)
    # i.e., f([1,2,3]) rather than f(*[1,2,3])
    # This is better because we have to construct the arguments at run-time as lists.
    return cast(SympyLambda, sympy.lambdify([params], expr))


def lambdify_list(params: list[Any], exprs: list[Expr]) -> SympyLambda:
    """Create a lambda function from the given expressions, taking the given parameters and returning a list of results."""
    return cast(SympyLambda, sympy.lambdify([params], exprs))
