"""ADRIOs for caculating absolute or relative humidity."""

import numpy as np
from numpy.core.records import fromarrays
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.adrio.adrio import (
    ADRIO,
    ADRIOContextError,
    InspectResult,
    adrio_cache,
)
from epymorph.adrio.validation import ResultFormat
from epymorph.attribute import AttributeDef
from epymorph.data_shape import Shapes
from epymorph.error import MissingContextError
from epymorph.simulation import Context


def calculate_relative_humidity(
    temperature: NDArray[np.float64],
    dewpoint: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculate relative humidity in percent (0 to 100). The calculation is designed to
    be vectorized over the input arrays, so they must be compatible shapes.

    Note: results may exceed 100% if temperature and dew point close in value.
    There are atmospheric conditions where this can be the case, but in typical usage
    might be more accurately thought of as data inconsistencies. Thus models that use
    humidity may wish to cap values in the model.

    Parameters
    ----------
    temperature :
        Air temperature in degrees Celsius.
    dewpoint :
        Dew point temperature in degrees Celsius.

    Returns
    -------
    :
        Relative humidity according to the element-wise pairs of the temperature and
        dew point inputs. The size of the result array is governed by numpy's
        broadcasting rules.
    """
    # equation for calculating relative humidity provided by following url
    # https://qed.epa.gov/hms/meteorology/humidity/algorithms/#:~:text=Relative%20humidity%20is%20calculated%20using,is%20air%20temperature%20(celsius).
    return 100 * (
        np.exp((17.625 * dewpoint) / (243.04 + dewpoint))
        / np.exp((17.625 * temperature) / (243.04 + temperature))
    )


_AH_CONSTANTS = 6.112 * 2.16679


def calculate_absolute_humidity(
    temperature: NDArray[np.float64],
    dewpoint: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculate absolute humidity in kilograms per cubic meter. The calculation is
    designed to be vectorized over the input arrays, so they must be compatible shapes.

    Parameters
    ----------
    temperature :
        Air temperature in degrees Celsius.
    dewpoint :
        Dew point temperature in degrees Celsius.

    Returns
    -------
    :
        Absolute humidity according to the element-wise pairs of the temperature and
        dew point inputs. The size of the result array is governed by numpy's
        broadcasting rules.
    """
    # equation from relative humidity to absolute humidity provided by following url
    # https://carnotcycle.wordpress.com/2012/08/04/how-to-convert-relative-humidity-to-absolute-humidity/
    # values 17.67 and 243.5 are changed to 17.625 and 243.04 respectively to cover
    # a larger range of temperature values with a smaller margin of error
    # (Alduchov and Eskridge 1996)
    return (
        (
            _AH_CONSTANTS
            * np.exp((17.625 * temperature) / (temperature + 243.04))
            * calculate_relative_humidity(temperature, dewpoint)
        )
        / (273.15 + temperature)
        / 1000  # convert to kilograms
    )


_TEMPERATURE = AttributeDef("temperature", type=float, shape=Shapes.TxN)
_DEWPOINT = AttributeDef("dewpoint", type=float, shape=Shapes.TxN)


@adrio_cache
class AbsoluteHumidity(ADRIO[np.float64, np.float64]):
    """
    Calculates absolute humidity (in kilograms per cubic meter) calculated from
    from air temperature and dew point temperature.

    This ADRIO requires two data attributes:

    - "temperature": the air temperature in degrees Celsius
    - "dewpoint": the dew-point temperature in degress Celsius
    """

    requirements = (_TEMPERATURE, _DEWPOINT)

    @property
    @override
    def result_format(self) -> ResultFormat:
        return ResultFormat(shape=Shapes.TxN, dtype=np.float64)

    @override
    def validate_context(self, context: Context) -> None:
        # need scope and time_frame in order to validate result shape
        try:
            context.scope
            context.time_frame
        except MissingContextError as e:
            raise ADRIOContextError(self, self.context, str(e))

    @override
    def inspect(self) -> InspectResult[np.float64, np.float64]:
        self.validate_context(self.context)
        temperature = self.data(_TEMPERATURE)
        dewpoint = self.data(_DEWPOINT)
        issues = {}
        if np.any(temperature_mask := np.ma.getmaskarray(temperature)):
            issues["temperature_masked", temperature_mask]
        if np.any(dewpoint_mask := np.ma.getmaskarray(dewpoint)):
            issues["dewpoint_masked", dewpoint_mask]

        result = calculate_absolute_humidity(temperature, dewpoint)
        self.validate_result(self.context, result)
        return InspectResult(
            adrio=self,
            source=fromarrays(
                [temperature, dewpoint],
                names=["temperature", "dewpoint"],  # type: ignore
            ),
            result=result,
            dtype=self.result_format.dtype.type,
            shape=self.result_format.shape,
            issues=issues,
        )


@adrio_cache
class RelativeHumidity(ADRIO[np.float64, np.float64]):
    """
    Calculates relative humidity as a percentage from air temperature and
    dew point temperature.

    This ADRIO requires two data attributes:

    - "temperature": the air temperature in degrees Celsius
    - "dewpoint": the dew-point temperature in degress Celsius
    """

    requirements = (_TEMPERATURE, _DEWPOINT)

    @property
    @override
    def result_format(self) -> ResultFormat:
        return ResultFormat(shape=Shapes.TxN, dtype=np.float64)

    @override
    def validate_context(self, context: Context) -> None:
        # need scope and time_frame in order to validate result shape
        try:
            context.scope
            context.time_frame
        except MissingContextError as e:
            raise ADRIOContextError(self, self.context, str(e))

    @override
    def inspect(self) -> InspectResult[np.float64, np.float64]:
        self.validate_context(self.context)
        temperature = self.data(_TEMPERATURE)
        dewpoint = self.data(_DEWPOINT)
        issues = {}
        if np.any(temperature_mask := np.ma.getmaskarray(temperature)):
            issues["temperature_masked", temperature_mask]
        if np.any(dewpoint_mask := np.ma.getmaskarray(dewpoint)):
            issues["dewpoint_masked", dewpoint_mask]

        result = calculate_relative_humidity(temperature, dewpoint)
        self.validate_result(self.context, result)
        return InspectResult(
            adrio=self,
            source=fromarrays(
                [temperature, dewpoint],
                names=["temperature", "dewpoint"],  # type: ignore
            ),
            result=result,
            dtype=self.result_format.dtype.type,
            shape=self.result_format.shape,
            issues=issues,
        )
