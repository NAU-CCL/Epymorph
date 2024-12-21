"""Utilities for epymorph's strata system."""

DEFAULT_STRATA = "all"
"""The strata name used as the default, primarily for single-strata simulations."""

META_STRATA = "meta"
"""A strata for information that concerns the other strata."""


def gpm_strata(strata_name: str) -> str:
    """The strata name for a GPM in a multistrata RUME."""
    return f"gpm:{strata_name}"
