import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.adrio.adrio import Adrio
from epymorph.data_shape import Shapes
from epymorph.simulation import AttributeDef


class AbsoluteHumidity(Adrio[np.float64]):
    """
    Creates a TxN matrix of floats representing absolute humidity in kilograms per cubic
    meter calculated from a relative humidity, which is calculated from a given
    temperature and dew point temperature, both in degrees Celsius.
    """

    requirements = [
        AttributeDef("temperature", type=float, shape=Shapes.TxN),
        AttributeDef("dewpoint", type=float, shape=Shapes.TxN),
    ]

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        temperature = self.data("temperature")
        relH = self.defer(RelativeHumidity())
        npHumidity = []

        # equation from relative humidity to absolute humidity provided by following url
        # https://carnotcycle.wordpress.com/2012/08/04/how-to-convert-relative-humidity-to-absolute-humidity/
        npHumidity = (
            (
                6.112
                * np.exp((17.67 * temperature) / (temperature + 243.5))
                * relH
                * 2.1674
            )
            / (273.15 + temperature)
        ) / 1000

        return npHumidity


class RelativeHumidity(Adrio[np.float64]):
    """
    Creates a TxN matrix of floats representing relative humidity as a percentage
    which is calculated from a given temperature and dew point temperature, both in
    degrees Celsius.
    """

    requirements = [
        AttributeDef("temperature", type=float, shape=Shapes.TxN),
        AttributeDef("dewpoint", type=float, shape=Shapes.TxN),
    ]

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        temperature = self.data("temperature")
        dewpoint = self.data("dewpoint")
        npHumidity = []

        # equation for calculating relative humidity provided by following url
        # https://qed.epa.gov/hms/meteorology/humidity/algorithms/#:~:text=Relative%20humidity%20is%20calculated%20using,is%20air%20temperature%20(celsius).
        npHumidity = 100 * (
            np.exp((17.625 * dewpoint) / (243.04 + dewpoint))
            / np.exp((17.625 * temperature) / (243.04 + temperature))
        )

        return np.array(npHumidity, dtype=np.float64)
