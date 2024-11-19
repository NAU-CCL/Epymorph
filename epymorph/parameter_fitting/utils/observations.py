"""
The `Observations` class is used to handle observational data for the simulation.

This class initializes with a data source (which can be any object) and a model link
that represents the connection between the observational data and the corresponding
model compartments or events.

Attributes:
    source (object): The data source containing the observational data.
    model_link (str): The link that maps the observations to specific compartments
    or events in the model.
"""


class Observations:
    def __init__(self, source: object, model_link: str):
        """
        Initializes the Observations class.

        Args:
            source (object): The data source, which could be any object type containing
                             the observation data (e.g., a DataFrame, list, etc.).
            model_link (str): The string representing the connection between the
                              observational data and the model's compartment or event.

        Attributes:
            source (object): The observational data source provided at initialization.
            model_link (str): The string linking the data to the model's compartments
            or events.
        """
        self.source = source
        self.model_link = model_link
