from warnings import warn

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.adrio.adrio import ADRIOLegacy, adrio_legacy_cache
from epymorph.attribute import AttributeDef
from epymorph.data_shape import Shapes


@adrio_legacy_cache
class AbsoluteHumidity(ADRIOLegacy[np.float64]):
    """
    Creates a TxN matrix of floats representing absolute humidity in kilograms per cubic
    meter calculated from a relative humidity, which is calculated from a given
    temperature and dew point temperature, both in degrees Celsius.
    """

    requirements = [
        AttributeDef("temperature", type=float, shape=Shapes.TxN),
        AttributeDef("dewpoint", type=float, shape=Shapes.TxN),
    ]
    """The temperature and dew point temperature from the PRISM ADRIO are required to
    calculate the absolute humidity."""

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        temperature = self.data("temperature")
        rel_h = self.defer(RelativeHumidity())
        np_humidity = []

        # equation from relative humidity to absolute humidity provided by following url
        # https://carnotcycle.wordpress.com/2012/08/04/how-to-convert-relative-humidity-to-absolute-humidity/
        # values 17.67 and 243.5 are changed to 17.625 and 243.04 respectively to cover
        # a larger range of temperature values with a smaller margin of error
        # (Alduchov and Eskridge 1996)
        constants = 6.112 * 2.16679
        np_humidity = (
            (
                constants
                * np.exp((17.625 * temperature) / (temperature + 243.04))
                * (rel_h)
            )
            / (273.15 + temperature)
            / 1000  # convert to kilograms
        )

        return np_humidity


@adrio_legacy_cache
class RelativeHumidity(ADRIOLegacy[np.float64]):
    """
    Creates a TxN matrix of floats representing relative humidity as a percentage
    which is calculated from a given temperature and dew point temperature, both in
    degrees Celsius.
    """

    requirements = [
        AttributeDef("temperature", type=float, shape=Shapes.TxN),
        AttributeDef("dewpoint", type=float, shape=Shapes.TxN),
    ]
    """The temperature and dew point temperature from the PRISM ADRIO are required to
    calculate the relative humidity."""

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        temperature = self.data("temperature")
        dewpoint = self.data("dewpoint")
        np_humidity = []

        # equation for calculating relative humidity provided by following url
        # https://qed.epa.gov/hms/meteorology/humidity/algorithms/#:~:text=Relative%20humidity%20is%20calculated%20using,is%20air%20temperature%20(celsius).
        np_humidity = 100 * (
            np.exp((17.625 * dewpoint) / (243.04 + dewpoint))
            / np.exp((17.625 * temperature) / (243.04 + temperature))
        )

        # if the humidity exceeds 100%, warn users
        if (np_humidity > 100).any():
            warn("At least one value of relative humidity exceeds 100%.")

        return np.array(np_humidity, dtype=np.float64)
