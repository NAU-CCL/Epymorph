from io import BytesIO
from pathlib import Path
from shutil import which
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
from graphviz import Digraph
from IPython import display
from matplotlib.image import imread
from sympy import Expr, preview

from epymorph.compartment_model import BaseCompartmentModel


class EdgeTracker:
    """class for keeping track of the edges added to the visualization"""

    def __init__(self):
        self.edge_dict = {}
        """
        dictionary for tracking edges, key = (head, tail)
        value = edge label
        """

    def track_edge(self, head: str, tail: str, label: Expr) -> None:
        """
        given a head, tail, and label for an edge, tracks it and updates the
        edge label (a sympy expr) if necessary
        """

        # check if edge already exists
        if (head, tail) in self.edge_dict:
            # update label by appending given label, adding the expressions
            self.edge_dict[(head, tail)] += label

        # edge doesn't exist
        else:
            # add edge w/ label to edge dict
            self.edge_dict[(head, tail)] = label


def check_draw_requirements() -> bool:
    """
    checks if the requirements necessary for draw module are installed, if not,
    displays messages to help guide installation
    """
    # check for latex installation
    latex_check = which("latex")

    # check for graphviz installation
    graphviz_check = which("dot")

    # print errors if needed for latex check
    if latex_check is None:
        print(
            "ERROR: No LaTeX converter found for IPM visualization.\n"
            "We recommend MiKTeX, found at https://miktex.org/download, or TexLive, "
            "found at https://tug.org/texlive/.\n"
            "These distributions are recommended by SymPy, a package we use "
            "for mathematical expressions."
        )

    # print errors if needed for graphviz check
    if graphviz_check is None:
        print(
            "ERROR: Graphviz not found for IPM visualization. Installation guides "
            "can be found at https://graphviz.org/download/"
        )

    return latex_check is not None and graphviz_check is not None


def build_ipm_edge_set(ipm: BaseCompartmentModel) -> EdgeTracker:
    """
    given an ipm, creates an edge tracker object that converts the transitions
    of the ipm into a set of adjacencies.
    """
    # init a tracker to be used for tacking edges and edge labels
    tracker = EdgeTracker()

    # fetch ipm event data
    ipm_events = ipm.events

    # init graph for model visualization to save to png, strict flag makes
    # it so repeated edges are merged

    # render edges
    for event in ipm_events:
        # get the current head and tail of the edge
        curr_head, curr_tail = str(event.compartment_from), str(event.compartment_to)

        # add edge to tracker, using the rate as the label
        tracker.track_edge(curr_head, curr_tail, event.rate)

    # return set of nodes and edges
    return tracker


def edge_to_graphviz(edge_set: EdgeTracker) -> Digraph:
    """
    given a set of edges from an edge tracker, converts into a graphviz directed
    graph for visualization purposes
    """

    def png_to_label(png_filepath: str) -> str:
        """
        helper function for displaying an image label using graphviz, requires
        the image to be in a table
        """

        return (
            f'<<TABLE border="0"><TR><TD><IMG SRC="{png_filepath}"/>'
            "</TD></TR></TABLE>>"
        )

    # init a graph viz directed graph for visualization
    model_viz = Digraph(
        format="png",
        strict=True,
        graph_attr={"rankdir": "LR"},
        node_attr={"shape": "square", "width": ".9", "height": ".8"},
        edge_attr={"minlen": "2.0"},
    )

    # iterate through edges in tracker
    for edge, label in edge_set.edge_dict.items():
        # get the head and tail for the edge
        head, tail = edge

        # create a temporary png file to render LaTeX edge label
        with NamedTemporaryFile(suffix=".png", delete=False) as temp_png:
            # load label as LaTeX png into temp file
            preview(label, viewer="file", filename=temp_png.name, euler=False)

            # render edge
            model_viz.edge(head, tail, label=png_to_label(temp_png.name))

    # return created graphviz model
    return model_viz


def draw_jupyter(graph: Digraph):
    """draws the graph in a jupyter notebook"""

    display.display_png(graph)


def draw_console(graph: Digraph):
    """draws graph to console"""

    # render png of graph
    model_bytes = graph.pipe(format="png")

    # convert png bytes to bytesio object
    model_bytes = BytesIO(model_bytes)

    # read the png file for matplotlib visualization
    ipm_png = imread(model_bytes)

    # mark the model png as the plot to show
    plt.imshow(ipm_png)

    # turn of axes
    plt.axis("off")

    # show the model png
    plt.show()


def draw_and_return(ipm: BaseCompartmentModel, console: bool) -> Digraph | None:
    """
    main function for converting an ipm into a visual model to be displayed
    by default in jupyter notebook, but optionally to console.
    returns model for potential further processing, no model if failed
    """
    # init ipm graph
    ipm_graph = None

    # check for installed software for drawing
    if check_draw_requirements():
        # convert events in ipm to a set of edges
        ipm_edge_set = build_ipm_edge_set(ipm)

        # convert set of edges into a visualization using graphviz
        ipm_graph = edge_to_graphviz(ipm_edge_set)

        # check to draw to console
        if console:
            draw_console(ipm_graph)

        # otherwise draw to jupyter
        else:
            draw_jupyter(ipm_graph)

    # return graph result for potential further processing
    return ipm_graph


def render(ipm: BaseCompartmentModel, console: bool = False) -> None:
    """
    default render function, draws to jupyter by default
    """
    draw_and_return(ipm, console)


def render_and_save(
    ipm: BaseCompartmentModel, file_path: str, console: bool = False
) -> None:
    """
    render function that saves to file system, draws jupyter by default
    """

    # draw ipm, get result back
    ipm_graph = draw_and_return(ipm, console)

    # save to file system if graph not empty
    if ipm_graph:
        save_model(ipm_graph, file_path)


def save_model(ipm_graph: Digraph, filepath: str) -> bool:
    """
    function that saves a given graphviz ipm digraph to a png in the
    given file path. Returns true upon save success, false upon save failure
    """

    # ensure filepath not empty
    if filepath:
        # get the directory and filename
        fp = Path(filepath)
        directory, filename = fp.parent, fp.name

        # ensure directory exists
        if directory.exists():
            # render and save png
            ipm_graph.render(filename=filename, directory=directory, cleanup=True)

            # display save succes
            print(f"Model saved successfully at {filepath}")
            return True

    # file name is empty, print err message
    print("ERR: invalid file path, could not save model")

    return False
