from graphviz import Digraph
from sympy import Expr, Symbol, init_printing, preview
from epymorph import ipm_library
from epymorph.compartment_model import EdgeDef, ForkDef, CompartmentModel 
from typing import List, Union
from tempfile import NamedTemporaryFile
from matplotlib.image import imread
from os import path, makedirs
from shutil import copy
import matplotlib.pyplot as plt

""" class for keeping track of the edges added to the visualization """
class EdgeTracker():

    """ dictionary for tracking edges, key = (head, tail) value = edge label"""
    edge_dict = {}

    """ 
    given a head, tail, and label for an edge, tracks it and updates the 
    edge label (a sympy expr) if necessary
    """
    def track_edge(self, head: str, tail: str, label: Expr) -> None:

        # check if edge already exists
        if (head, tail) in self.edge_dict:

            # update label by appending given label, adding the expressions
            self.edge_dict[(head, tail)] += label

        # edge doesn't exist
        else:

            # add edge w/ label to edge dict
            self.edge_dict[(head, tail)] = label


    """ given a head and tail for an edge, return its label """
    def get_edge_label(self, head:str, tail:str) -> str:

        # ensure label exists
        if (head, tail) in self.edge_dict:

            # return label for given edge
            return self.edge_dict[(head, tail)]

        # not in list, return empty expression
        return None

    """ clears all tracked edges """
    def clear(self):
        self.edge_dict.clear()



"""
primary function for creating a model visualization, given an ipm label that 
exists within the ipm library
"""
def render_model(ipm: CompartmentModel, save: bool = False, 
                                                   filename: str = "") -> None:

    # render model as a temp file
    with NamedTemporaryFile(suffix = '.gv', delete=False) as temp_gv:

        # init a tracker to be used for tacking edges and edge labels
        tracker = EdgeTracker()

        # fetch ipm transition data
        ipm_transitions = ipm.transitions
        
        # init graph for model visualization to save to png, strict flag makes
        # it so repeated edges are merged
        model_viz = Digraph(format = 'png', strict=True,
                            graph_attr = {'rankdir': 'LR'},
                            node_attr = {'shape': 'square',
                                         'width': '.9',
                                         'height': '.8'},
                            edge_attr = {'minlen': '2.0'})

        # clear graph so repeated calls of render model do not repeat edges
        model_viz.clear(keep_attrs=True)

        # clear tracker, see above
        tracker.clear()


        # render edges
        for transition in ipm_transitions:

            # check for fork
            if isinstance(transition, ForkDef):

                # add fork transitions to list
                ipm_transitions += transition.edges

            # transition is an edge
            else:

                # get the current head and tail of the edge
                curr_head, curr_tail = str(transition.compartment_from), \
                                                 str(transition.compartment_to)
                
                # add edge to tracker, using the rate as the label
                tracker.track_edge(curr_head, curr_tail, transition.rate)

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

        # render png of graph
        model_viz.render(temp_gv.name)

        # get the optimal size for the rendered graph
        plt.figure(dpi=find_optimal_dpi(f'{temp_gv.name}.png'))


        # show the png 
        ipm_png = imread(f'{temp_gv.name}.png')
        plt.imshow(ipm_png)
        plt.axis('off')
        plt.show()

        # check if image should be saved
        if save:
            
            # ensure filename not empty
            if filename:

                # create visualization directory if it doesn't exist
                if not path.exists('model_pngs'):

                    # doesn't exist, create the directory
                    makedirs('model_pngs')

                # copy temp file to png directory to save it
                copy(f'{temp_gv.name}.png', f'model_pngs/{filename}.png')

            # file name is empty, print err message
            else:

                print("ERR: no file name provided, could not save model")



"""
helper function for displaying an image label using graphvz, requires the image
to be in a table
"""
def png_to_label(png_filepath: str) -> str:
    return (
        f'<<TABLE border="0"><TR><TD><IMG SRC="{png_filepath}"/>' +
                                                          '</TD></TR></TABLE>>'
    )

"""
helper function for determining the best dpi (which determines size) for 
displaying the given ipm model .png using motplotlib
"""
def find_optimal_dpi(ipm_png_filename: str) -> int:
    # init dpi to start at 100
    opt_dpi = 100

    # get the size of the png file
    png_size = path.getsize(ipm_png_filename)

    # for every 10k bytes, increase the dpi
    while (png_size > 10000):

        # increase the dpi by 50
        opt_dpi += 50

        # remove 10 kbytes from the size for loop control
        png_size -= 10000

    # return the determined dpi
    return opt_dpi

