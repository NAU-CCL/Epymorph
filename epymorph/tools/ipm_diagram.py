"""
Produce graphical diagrams of compartmental disease models (IPMs).

Usage of these features requires separate installation of a latex converter and
graphviz.
"""

from abc import abstractmethod
from collections.abc import Iterable
from contextlib import contextmanager
from functools import lru_cache, reduce
from io import BytesIO
from itertools import groupby
from pathlib import Path
from shutil import which
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Iterator, Protocol, Sequence

import matplotlib.pyplot as plt
from graphviz import Digraph
from matplotlib.image import imread
from sympy import Expr, preview

from epymorph.error import ExternalDependencyError


@lru_cache(maxsize=1)
def _dependency_issues() -> list[tuple[str, str]]:
    """
    Return a record of dependency issues for the IPM diagram feature.
    Cached so we only compute this once.

    If there are no issues, an empty list is returned.
    If there are issues, they are returned as (package name, issue description) tuples.
    """
    issues = []

    if which("latex") is None:
        msg = (
            "- Unable to find LaTeX converter 'latex'.\n"
            "  We recommend MiKTeX (https://miktex.org/download) "
            "or TexLive (https://tug.org/texlive/)"
        )
        issues.append(("latex", msg))

    if which("dvipng") is None:
        msg = (
            "- Unable to find png converter 'dvipng'.\n"
            "  You may need to install this from your system's package manager or "
            "alternatively your latex library might be able to install this as an "
            "option."
        )
        issues.append(("dvipng", msg))

    if which("dot") is None:
        msg = (
            "- Unable to find Graphviz renderer 'dot'.\n"
            "  See installation instructions (https://graphviz.org/download/)"
        )
        issues.append(("dot", msg))

    return issues


def check_dependencies() -> None:
    """
    Check if the external requirements for drawing diagrams are installed.

    The results of this check are memorized so it can be called repeatedly
    without adding much overhead.

    Raises
    ------
    ExternalDependencyError
        If some dependencies are missing.
    """
    if issues := _dependency_issues():
        err = [
            "Rendering IPM diagrams requires you to install some additional programs:",
            *(msg for _, msg in issues),
        ]
        raise ExternalDependencyError("\n".join(err), [pkg for pkg, _ in issues])


# NOTE: the following Protocol-ized CompartmentModel bits are to avoid
# circular dependency issues.


class CompartmentName(Protocol):
    """A simplified `CompartmentName` interface."""

    @abstractmethod
    def __str__(self) -> str:
        """The compartment's full name."""


class CompartmentDef(Protocol):
    """A simplified `CompartmentDef` interface."""

    @property
    @abstractmethod
    def name(self) -> CompartmentName:
        """The compartment name."""


class EdgeDef(Protocol):
    """A simplified `EdgeDef` interface."""

    @property
    @abstractmethod
    def rate(self) -> Expr:
        """The rate of flow along this edge."""

    @property
    @abstractmethod
    def tuple(self) -> tuple[str, str]:
        """The edge in tuple form: `(from_name, to_name)`."""


class CompartmentModel(Protocol):
    """A simplified `CompartmentModel` interface."""

    @property
    @abstractmethod
    def compartments(self) -> Sequence[CompartmentDef]:
        """The unique compartments in the model."""

    @property
    @abstractmethod
    def events(self) -> Sequence[EdgeDef]:
        """The unique transition events in the model."""


@contextmanager
def construct_digraph(ipm: CompartmentModel) -> Iterator[Digraph]:
    """
    Construct a graphviz object (`Digraph`) for the compartment model's diagram.

    The `Digraph` instance is only valid within the managed context because we rely on
    temporary files to render latex expressions.

    Yields
    ------
    :
        The `Digraph` instance, which is valid as soon as the context opens but only
        until the context is closed.
    """
    check_dependencies()

    def expr_sum(exprs: Iterable[Expr]) -> Expr:
        """Sum a bunch of expressions together."""
        return reduce(lambda a, b: a + b, exprs)

    def edges_group_sum(edges: Iterable[EdgeDef]) -> list[tuple[str, str, Expr]]:
        """Compute the sum of rate expressions by unique (src,dst) pair."""
        return [
            (src, dst, expr_sum(e.rate for e in group_edges))
            for (src, dst), group_edges in groupby(edges, key=lambda e: e.tuple)
        ]

    with TemporaryDirectory() as tmp_dir:
        graph = Digraph(
            graph_attr={"rankdir": "LR", "latex": "true"},
            node_attr={"shape": "square", "width": ".9", "height": ".8"},
            edge_attr={"minlen": "2.0"},
        )

        for c in ipm.compartments:
            # Explicitly declare nodes for all compartments, in case there are
            # compartments which are neither the source nor destination of a
            # transition edge. Otherwise this would be covered when declaring edges,
            # but it doesn't hurt to do it now.
            graph.node(str(c.name))

        for src, dst, rate in edges_group_sum(ipm.events):
            # Each rate is a sympy expression which we want to render as latex
            # to do so, we have to `preview` them into temp files
            # then use HTML table markup to place them on the graph.
            with NamedTemporaryFile(suffix=".png", dir=tmp_dir, delete=False) as f:
                preview(rate, viewer="file", filename=f.name, euler=False)
                label = f'<<TABLE border="0"><TR><TD><IMG SRC="{f.name}"/></TD></TR></TABLE>>'  # noqa: E501
                graph.edge(src, dst, label=label)
        yield graph


def render_diagram_to_bytes(ipm: CompartmentModel) -> BytesIO:
    """
    Render a diagram of the given compartment model and return it as the bytes of a
    png-formatted image.

    Parameters
    ----------
    ipm :
        The compartment model to render.

    Returns
    -------
    :
        The bytes of the image.
    """
    check_dependencies()
    with construct_digraph(ipm) as graph:
        return BytesIO(graph.pipe(format="png"))


def render_diagram(
    ipm: CompartmentModel,
    *,
    file: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
) -> None:
    """
    Render a diagram of the given compartment model and either show it with matplotlib
    (default) or save it to `file` as a png image.

    Parameters
    ----------
    ipm :
        The compartment model to render.
    file :
        Provide a file path to save a png image of the diagram to this path.
        If `file` is None, we will instead use matplotlib to show the diagram.
    figsize :
        The matplotlib figure size to use when displaying the diagram.
        Only used if `file` is not provided.
    """
    check_dependencies()
    image = render_diagram_to_bytes(ipm)
    if file is not None:
        # Save to file.
        with Path(file).open("wb") as f:
            f.write(image.getvalue())
    else:
        # Display using matplotlib.
        plt.figure(figsize=figsize or (10, 6))
        plt.imshow(imread(image))
        plt.axis("off")
        plt.show()
