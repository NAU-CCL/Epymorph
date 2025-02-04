from epymorph.compartment_model import QuantityAggregation, QuantitySelection
from epymorph.geography.scope import GeoAggregation, GeoSelection
from epymorph.parameter_fitting.likelihood import Likelihood
from epymorph.simulation import BaseSimulationFunction
from epymorph.time import TimeAggregation, TimeSelection


class ModelLink:
    """
    Contains the information needed to compute the expected observation from a
    simulation output.
    """

    def __init__(
        self,
        time: TimeSelection | TimeAggregation,
        geo: GeoSelection | GeoAggregation,
        quantity: QuantitySelection | QuantityAggregation,
    ):
        self.time = time
        self.geo = geo
        self.quantity = quantity


class Observations:
    """
    The `Observations` class is used to handle observational data for the simulation.

    This class initializes with a data source (which can be any object) and a model link
    that represents the connection between the observational data and the corresponding
    model compartments or events.

    Attributes
    ----------
    source : BaseSimulationFunction
        The data source containing the observational data.
    model_link : ModelLink
        The link that maps the observations to specific
        compartments or events in the model.
    """

    def __init__(
        self,
        source: BaseSimulationFunction,
        model_link: ModelLink,
        likelihood: Likelihood,
    ):
        """
        Initializes the Observations class.

        Parameters
        ----------
        source : BaseSimulationFunction
            The data source.
        model_link : ModelLink
            Represents the connection between the observational data and the model's
            compartment or event.
        """
        self.source = source
        self.model_link = model_link
        self.likelihood = likelihood
