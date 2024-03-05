from graphviz import Digraph
from sympy import Expr, Symbol, preview
from epymorph import ipm_library
from epymorph.compartment_model import EdgeDef, ForkDef, CompartmentModel 
from typing import List, Union
from IPython import display
from tempfile import NamedTemporaryFile
from os import path, makedirs
from io import BytesIO
from matplotlib.image import imread
import matplotlib.pyplot as plt

class ipmDraw():
    """class for functions related to drawing a graphviz ipm graph"""

    def jupyter(graph: Digraph):
        """draws the graph in a jupyter notebook"""

        display.display_png(graph)

    def console(graph: Digraph):
        """draws graph to console"""

        # render png of graph
        model_bytes = graph.pipe(format='png')

        # convert png bytes to bytesio object
        model_bytes = BytesIO(model_bytes)

        # read the png file for matplotlib visualization
        ipm_png = imread(model_bytes)

        # mark the model png as the plot to show
        plt.imshow(ipm_png)

        # turn of axes
        plt.axis('off')

        # show the model png
        plt.show()       

class EdgeTracker():
    """ class for keeping track of the edges added to the visualization """
    
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


    def get_edge_label(self, head:str, tail:str) -> str:
        """ given a head and tail for an edge, return its label """

        # ensure label exists
        if (head, tail) in self.edge_dict:

            # return label for given edge
            return self.edge_dict[(head, tail)]

        # not in list, return empty expression
        return None


def build_ipm_graph(ipm: CompartmentModel) -> Digraph:
    """
    primary function for creating a model visualization, given an ipm label 
    that exists within the ipm library
    """
    # init a tracker to be used for tacking edges and edge labels
    tracker = EdgeTracker()

    # fetch ipm event data
    ipm_events = ipm.events
    
    # init graph for model visualization to save to png, strict flag makes
    # it so repeated edges are merged
    model_viz = Digraph(format = 'png', strict=True, 
                        graph_attr = {'rankdir': 'LR'},
                        node_attr = {'shape': 'square',
                                     'width': '.9',
                                     'height': '.8'},
                        edge_attr = {'minlen': '2.0'})

    # render edges
    for event in ipm_events:

        # get the current head and tail of the edge
        curr_head, curr_tail = str(event.compartment_from), \
                                         str(event.compartment_to)
        
        # add edge to tracker, using the rate as the label
        tracker.track_edge(curr_head, curr_tail, event.rate)

        # get santized edge label from newly tracked edge
        label_expr = tracker.get_edge_label(curr_head, curr_tail)

        # create a temporary png file to render LaTeX edge label
        with NamedTemporaryFile(suffix='.png', 
                                             delete=False) as temp_png:

            # load label as LaTeX png into temp file
            preview(label_expr, viewer='file', filename=temp_png.name, 
                                                           euler=False)

            # render edge
            model_viz.edge(curr_head, curr_tail, 
                                     label=png_to_label(temp_png.name))

    # return created visualization graph
    return model_viz

def render(ipm: CompartmentModel, save: bool = False, filename: str = "",
                                                      console: bool = False) \
                                                                       -> None:
    """
    main function for converting an ipm into a visual model
    ipm: the model to be converted
    save: inidicates if the model should be saved
    filename: file name to save to, if model is to be saved
    console: flag for printing to console vs the standard output to jupyter
    """
    ipm_graph = build_ipm_graph(ipm)

    if console:
        ipmDraw.console(ipm_graph)
    
    else:
        ipmDraw.jupyter(ipm_graph)
    
    if save:
        save_model(ipm_graph, filename)

def save_model(ipm_graph: Digraph, filename: str) -> None:
    """
    function that saves a given graphviz ipm digraph to a png in the
    'model_pngs' folder with the given file name. Creates the folder if it 
    does not exist
    """

    # ensure filename not empty
    if filename:

        # create visualization directory if it doesn't exist
        if not path.exists('model_pngs'):

            # doesn't exist, create the directory
            makedirs('model_pngs')

        # render and save png
        ipm_graph.render(filename, 'model_pngs', cleanup=True)
        

    # file name is empty, print err message
    else:
        print("ERR: no file name provided, could not save model")

def png_to_label(png_filepath: str) -> str:
    """
    helper function for displaying an image label using graphvz, requires the
    image to be in a table
    """

    return (
        f'<<TABLE border="0"><TR><TD><IMG SRC="{png_filepath}"/>' +
                                                          '</TD></TR></TABLE>>'
    )
