"""Utilities for epymorph's strata system."""

DEFAULT_STRATA = "all"
"""The strata name used as the default, primarily for single-strata simulations."""

META_STRATA = "meta"
"""A strata for information that concerns the other strata."""


def gpm_strata(strata_name: str) -> str:
    """
    Format a strata (GPM) name to its internal format.

    Parameters
    ----------
    strata_name :
        The human-readable strata name.

    Returns
    -------
    :
        The internal format of the strata's name.
    """
    return f"gpm:{strata_name}"
