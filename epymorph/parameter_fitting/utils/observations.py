from epymorph.compartment_model import QuantityAggregation, QuantitySelection
from epymorph.geography.scope import GeoAggregation, GeoSelection
from epymorph.parameter_fitting.likelihoods.base_likelihood import Likelihood
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

    Attributes:
        source (object): The data source containing the observational data.
        model_link (ModelLink): The link that maps the observations to specific
        compartments or events in the model.
    """

    def __init__(self, source: object, model_link: ModelLink, likelihood: Likelihood):
        """
        Initializes the Observations class.

        Args:
            source (object): The data source, which could be any object type containing
                             the observation data (e.g., a DataFrame, list, etc.).
            model_link (ModelLink): Represents the connection between the observational
                                    data and the model's compartment or event.

        Attributes:
            source (object): The observational data source provided at initialization.
            model_link (ModelLink): Links the data to the model's compartments
            or events.
        """
        self.source = source
        self.model_link = model_link
        self.likelihood = likelihood
