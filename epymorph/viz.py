from graphviz import Digraph
from sympy import Expr, Symbol
from epymorph import ipm_library
from epymorph.compartment_model import EdgeDef, ForkDef
from typing import List, Union

""" class for keeping track of the edges added to the visualization """
class EdgeTracker():

    """ dictionary for tracking edges, key = (head, tail) value = edge label"""
    edge_dict = {}

    """ 
    given a head, tail, and label for an edge, tracks it and updates the 
    edge label if necessary
    """
    def track_edge(self, head: str, tail: str, label: str) -> None:

        # check if edge already exists
        if (head, tail) in self.edge_dict:

            # update label by appending given label, adding the expressions
            self.edge_dict[(head, tail)] += f" + {label}"

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

        # not in list, return blank string
        return ""



"""
primary function for creating a model visualization,  given an ipm label that 
exists within the ipm library
"""
def display_model(ipm_name: str) -> None:

    # fetch ipm transition data
    ipm_transitions = ipm_library[ipm_name]().transitions 
    
    # init graph for model visualization to save to png, strict flag makes it
    # so repeated edges are merged
    model_viz = Digraph(comment = ipm_name, format = 'png', strict=True)

    # set graph type to left-to-right
    model_viz.attr(rankdir = 'LR')

    # set node shape to square
    model_viz.attr('node', shape='square')

    # set a minimum edge length
    model_viz.attr('edge')

    # add ipm edges to graph
    add_ipm_edges(model_viz, ipm_transitions)

    # render png of graph
    model_viz.render(f"{ipm_name}.gv")


"""
given a graphviz graph and a set of ipm transitions, maps transitions to graph
"""
def add_ipm_edges(graph: Digraph, 
                           transitions: List[Union[EdgeDef, ForkDef]]) -> None:
    # init tracker to keep track of edge data
    tracker = EdgeTracker()

    # iterate through transition list
    for transition in transitions:
        
        # handle a fork
        if (type(transition) == ForkDef):

            # add fork edges to list
            transitions += transition.edges

        # transition is a normal edge
        else:

            # get current edge head and tail
            curr_head = str(transition.compartment_from)
            curr_tail = str(transition.compartment_to)

            # add edge to tracker
            tracker.track_edge(curr_head, curr_tail, str(transition.rate))

            # add edge to graphviz graph with label from tracker
            graph.edge(curr_head, curr_tail, 
                                  tracker.get_edge_label(curr_head, curr_tail))

ipm = ipm_library['sirh']()

print(ipm)

