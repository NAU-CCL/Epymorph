from abc import abstractmethod
from contextlib import contextmanager
from functools import reduce
from io import BytesIO
from itertools import groupby
from pathlib import Path
from shutil import which
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Iterator, Protocol, Sequence

import matplotlib.pyplot as plt
from graphviz import Digraph
from matplotlib.image import imread
from sympy import Expr, Symbol, preview

from epymorph.error import ExternalDependencyError

_dependencies_checked = False
"""True if we have already passed the dependency inspection."""


def check_dependencies() -> None:
    """Checks if the external requirements for drawing diagrams are installed.
    Raises ExternalDependencyError if not."""
    global _dependencies_checked
    if not _dependencies_checked:
        missing = []
        messages = []
        if which("latex") is None:
            missing.append("latex")
            messages.append(
                "- Unable to find LaTeX converter 'latex'.\n"
                "  We recommend MiKTeX (https://miktex.org/download) "
                "or TexLive (https://tug.org/texlive/)"
            )

        # print errors if needed for graphviz check
        if which("dot") is None:
            missing.append("dot")
            messages.append(
                "- Unable to find Graphviz renderer 'dot'.\n"
                "  See installation instructions (https://graphviz.org/download/)"
            )

        if len(missing) == 0:
            _dependencies_checked = True
        else:
            prologue = (
                "Rendering IPM diagrams requires you to install some "
                "additional programs:"
            )
            messages.insert(0, prologue)
            raise ExternalDependencyError("\n".join(messages), missing)


# NOTE: Protocol-ize CompartmentModels (etc) to avoid circular dependency issues.


class EdgeDef(Protocol):
    """A simplified EdgeDef interface."""

    @property
    @abstractmethod
    def rate(self) -> Expr:
        pass

    @property
    @abstractmethod
    def compartment_from(self) -> Symbol:
        pass

    @property
    @abstractmethod
    def compartment_to(self) -> Symbol:
        pass


class CompartmentModel(Protocol):
    """A simplified CompartmentModel interface."""

    @property
    @abstractmethod
    def events(self) -> Sequence[EdgeDef]:
        pass


def merge_edges(ipm: CompartmentModel) -> list[tuple[str, str, Expr]]:
    """Combines (by addition) the rate expressions of all edges with
    the same source and destination compartment."""
    return [
        (
            src,
            dst,
            reduce(lambda a, b: a + b, (e.rate for e in group)),
        )
        for (src, dst), group in groupby(
            ipm.events,
            key=lambda e: (e.compartment_from.name, e.compartment_to.name),
        )
    ]


@contextmanager
def construct_digraph(ipm: CompartmentModel) -> Iterator[Digraph]:
    """
    Constructs a Digraph from the given compartment model.
    The Digraph is only valid as long as the context is open,
    because we rely on temporary files to render latex expressions.
    """
    with TemporaryDirectory() as tmp_dir:
        graph = Digraph(
            graph_attr={"rankdir": "LR", "latex": "true"},
            node_attr={"shape": "square", "width": ".9", "height": ".8"},
            edge_attr={"minlen": "2.0"},
        )
        for src, dst, rate in merge_edges(ipm):
            # each rate is a sympy expression which we want to render as latex
            # to do so, we have to `preview` them into temp files
            # then use HTML table markup to place them on the graph
            with NamedTemporaryFile(suffix=".png", dir=tmp_dir, delete=False) as f:
                preview(rate, viewer="file", filename=f.name, euler=False)
                label = f'<<TABLE border="0"><TR><TD><IMG SRC="{f.name}"/></TD></TR></TABLE>>'  # noqa: E501
                graph.edge(src, dst, label=label)
        yield graph


def digraph_png(ipm: CompartmentModel) -> BytesIO:
    """Constructs a Digraph from the given compartment model and renders it to PNG.
    Returns the PNG file content in a BytesIO."""
    with construct_digraph(ipm) as graph:
        return BytesIO(graph.pipe(format="png"))


def render_diagram(
    ipm: CompartmentModel,
    *,
    file: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Render a diagram of the given IPM, either by showing it with matplotlib (default)
    or by saving it to `file` as a png image.

    Parameters
    ----------
    ipm : CompartmentModel
        The IPM to render.
    file : str | Path, optional
        Provide a file path to save a png image of the diagram to this path.
        If `file` is None, we will instead use matplotlib to show the diagram.
    figsize : tuple[float, float], optional
        The matplotlib figure size to use when displaying the diagram.
        Only used if `file` is not provided.
    """
    check_dependencies()
    image = digraph_png(ipm)
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
