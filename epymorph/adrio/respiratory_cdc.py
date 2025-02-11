from datetime import date
from typing import Literal
from warnings import warn

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.adrio.adrio import ADRIO, ProgressCallback, adrio_cache
from epymorph.adrio.cdc import DataSource, _api_query, _validate_scope
from epymorph.error import DataResourceError
from epymorph.geography.us_census import CensusScope
from epymorph.geography.us_tiger import get_states
from epymorph.time import TimeFrame

DiseaseType = Literal["Covid", "Influenza", "RSV"]

_DISEASE_VARIABLES: dict[DiseaseType, str] = {
    "Covid": "c19",
    "Influenza": "flu",
    "RSV": "rsv",
}


def _fetch_respiratory(
    attrib_name: str,
    scope: CensusScope,
    time_frame: TimeFrame,
    progress: ProgressCallback,
) -> NDArray[np.float64]:
    """
    Fetches data from CDC dataset reporting weekly hospital data and metrics from rsv
    and other respiratory illnesses during manditory and voluntary
    reporting periods.
    Available from 8/8/2020 to present at state granularity.
    https://data.cdc.gov/Public-Health-Surveillance/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/mpgq-jmmr/about_data
    """

    source = DataSource(
        url_base="https://data.cdc.gov/resource/mpgq-jmmr.csv?",
        date_col="weekendingdate",
        fips_col="jurisdiction",
        data_col=attrib_name,
        granularity="state",
        replace_sentinel=None,
        map_geo_ids=get_states(scope.year).state_fips_to_code,
    )

    return _api_query(source, scope, time_frame, progress)


class _RespiratoryADRIO(ADRIO[np.float64]):
    _override_time_frame: TimeFrame | None
    """The time period the data encompasses."""

    disease_name: DiseaseType

    def __init__(
        self,
        disease_name: DiseaseType,
        voluntary_reporting=True,
        time_frame: TimeFrame | None = None,
    ):
        self.disease_name = disease_name
        self._override_time_frame = time_frame
        self.voluntary_reporting = voluntary_reporting

    @property
    def data_time_frame(self) -> TimeFrame:
        """The time frame for which to fetch data."""
        return self._override_time_frame or self.time_frame

    def _validate_dates_(self):
        dataset_start = date(2020, 8, 8)
        first_mandate_end = date(2024, 4, 30)
        second_mandate_start = date(2024, 11, 1)
        no_mandate_range = TimeFrame.rangex(first_mandate_end, second_mandate_start)

        covid_flu_voluntary_msg = (
            "The dates you entered take place during a voluntary reporting "
            "period.\nEnter dates between August 8th, 2020 through April"
            " 30th, 2024 or from November 1st, 2024 to the present day for data "
            "captured during a mandatory reporting period."
        )
        rsv_voluntary_msg = (
            "All data and metrics reported before November 1st, 2024 for RSV"
            " were reported voluntarily.\nEnter a date on or after"
            " 11/01/2024 for data captured during a mandatory reporting period."
        )

        # check if the dates are before August 8th, 2020
        if self.data_time_frame.start_date < dataset_start:
            raise DataResourceError(
                "The Weekly Hospital Respiratory dataset provides metrics starting"
                " August 8th, 2020.\nPlease enter a time frame starting on or after "
                "08/08/2020."
            )

        # check for the voluntary reporting period for Covid and Influenza
        voluntary_covid_flu = (
            self.disease_name != "RSV"
            and self.data_time_frame.is_subset(no_mandate_range)
        )
        # check for the voluntary reporting period for RSV
        voluntary_rsv = (
            self.disease_name == "RSV"
            and self.data_time_frame.start_date < second_mandate_start
        )

        # warn or raise the error
        if voluntary_covid_flu or voluntary_rsv:
            msg = covid_flu_voluntary_msg if voluntary_covid_flu else rsv_voluntary_msg
            if not self.voluntary_reporting:
                raise DataResourceError(msg)
            warn(msg)

        return self.data_time_frame


class DiseaseHospitalizations(_RespiratoryADRIO):
    """
    Creates a TxN matrix of tuples, containing a date and a float representing the
    number of patients hospitalized with a confirmed disease for that week. May be
    specified for the total number of patients, the number of adult patients, or the
    number of pediatric patients.
    """

    AmountType = Literal["Total", "Adult", "Pediatric"]

    amount_variables: dict[AmountType, str] = {
        "Total": "total",
        "Adult": "adult",
        "Pediatric": "ped",
    }

    amount_type: AmountType

    def __init__(
        self,
        disease_name: DiseaseType,
        amount_type: AmountType = "Total",
        voluntary_reporting=True,
        time_frame: TimeFrame | None = None,
    ):
        """
        Creates an ADRIO of the confirmed hospitalizations for a disease.

        Parameters
        ----------
        disease_name: DiseaseType
            The name of the disease that is desired to be fetched for (options: 'RSV',
            'Influenza', 'Covid').
        amount_type : AmountType
            The category of hospitalized patient sums to fetch for.
            - `'Total'`: Displays the total amount of patients that have been
            hospitalized with a disease (default).
            - `'Adult'`: Displays the number of adults, starting from age 18 and beyond,
            who have been hospitalized with a disease.
            - `'Pediatric'`: Displays the number of pediatric patients, from ages 0 to
            17, who have been hospitalized with a disease.
        voluntary_reporting: bool, optional
            The flag that indicates whether the user would like the ADRIO to warn or
            error when the timeframe is within a voluntary reporting period.
            If True, all available data is returned with a warning about the
            timeframe. (default)
            If False, the ADRIO will error and will not return any requested data.
        time_frame : TimeFrame, optional
            The range of dates to fetch hospital metric data for.
            Default: the simulation time frame.
        """
        super().__init__(disease_name, voluntary_reporting, time_frame)
        self.amount_type = amount_type

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        time_frame = self._validate_dates_()
        scope = _validate_scope(self.scope)
        amount_var = self.amount_variables[self.amount_type]
        disease_var = _DISEASE_VARIABLES[self.disease_name]
        if amount_var == "total":
            hosp_var = f"totalconf{disease_var}hosppats"
        else:
            hosp_var = f"numconf{disease_var}hosppats{amount_var}"
        return _fetch_respiratory(
            hosp_var,
            scope,
            time_frame,
            self.progress,
        )


class DiseaseAdmissions(_RespiratoryADRIO):
    """
    Creates a TxN matrix of tuples, containing a date and a float representing the
    number of new admissions for a confirmed disease for that week. May be
    specified for the total number of patients, the total number of adult patients, the
    total number of pediatric patients, or any of the specified age ranges.
    """

    AmountType = Literal[
        "0 to 4",
        "5 to 17",
        "18 to 49",
        "50 to 64",
        "65 to 74",
        "75 and above",
        "Unknown",
        "Adult",
        "Pediatric",
        "Total",
    ]

    amount_variables: dict[AmountType, str] = {
        "0 to 4": "0to4",
        "5 to 17": "5to17",
        "18 to 49": "18to49",
        "50 to 64": "50to64",
        "65 to 74": "65to74",
        "75 and above": "75plus",
        "Unknown": "unk",
        "Adult": "adult",
        "Pediatric": "ped",
        "Total": "total",
    }

    amount_type: AmountType

    def __init__(
        self,
        disease_name: DiseaseType,
        amount_type: AmountType = "Total",
        voluntary_reporting=True,
        time_frame: TimeFrame | None = None,
    ):
        """
        Creates an ADRIO of the confirmed admissions for a confirmed disease.

        Parameters
        ----------
        disease_name: DiseaseType
            The name of the disease that is desired to be fetched for (options: 'RSV',
            'Influenza', 'Covid').
        amount_type : AmountType
            The category of the disease of hospitalized patient sums to fetch for.
            The parameters for 'Total', 'Adult', and 'Pediatric' start from
            November 25th, 2023 to present. For any numerical age range parameter,
            the starting date is October 12th, 2024.
            - `'Total'`: Displays the total amount of patient admissions with the
            confirmed disease (default).
            - `'Adult'`: Displays the number of adult patient admissions, starting from
            age 18 and beyond, confirmed with the disease.
            - `'Pediatric'`: Displays the number of pediatric patient admissions, from
            ages 0 to 17, confirmed with the disease.
            - `'0 to 4'`: Displays the number of patient admissions, ages 0 to 4, with
            the confirmed disease.
            - `'5 to 17'`: Displays the number of patient admissions, ages 5 to 17, with
            the confirmed disease.
            - `'18 to 49'`: Displays the number of patient admissions, ages 18 to 49,
            with the confirmed disease.
            - `'50 to 64'`: Displays the number of patient admissions, ages 50 to 64,
            with the confirmed disease.
            - `'65 to 74'`: Displays the number of patient admissions, ages 65 to 74,
            with the confirmed disease.
            - `'75 and above'`: Displays the number of patient admissions, from ages 75
            and beyond, with the confirmed disease.
            - `'Unknown'`: Displays the number of patient admissions with an unkown age
        voluntary_reporting: bool, optional
            The flag that indicates whether the user would like the ADRIO to warn or
            error when the timeframe is within a voluntary reporting period.
            If True, all available data is returned with a warning about the
            timeframe. (default)
            If False, the ADRIO will error and will not return any requested data.
        time_frame : TimeFrame, optional
            The range of dates to fetch hospital metric data for.
            Default: the simulation time frame.
        """
        super().__init__(disease_name, voluntary_reporting, time_frame)
        self.amount_type = amount_type

    def evaluate_adrio(self) -> NDArray[np.float64]:
        """
        Allow admissions classes to set up their queries.
        """
        time_frame = self._validate_dates_()
        scope = _validate_scope(self.scope)
        amount_var = self.amount_variables[self.amount_type]
        disease_var = _DISEASE_VARIABLES[self.disease_name]
        age_type_var = ""
        if amount_var == "total":
            adm_var = f"totalconf{disease_var}newadm"
        elif amount_var in ["adult", "ped"]:
            adm_var = f"totalconf{disease_var}newadm{amount_var}"
        else:
            if amount_var in ["0to4", "5to17"]:
                age_type_var = "ped"
            elif amount_var == "unk":
                age_type_var = ""
            else:
                age_type_var = "adult"
            adm_var = f"numconf{self.disease_name}newadm{age_type_var}{amount_var}"
        return _fetch_respiratory(
            adm_var,
            scope,
            time_frame,
            self.progress,
        )


@adrio_cache
class AdmissionsPer100k(_RespiratoryADRIO):
    """
    Creates a TxN matrix of tuples, containing a date and a float representing the
    number of new admissions for a confirmed disease for that week per
    100k population. May be specified for the total number of patients, the total number
    of adult patients, the total number of pediatric patients, or any of the specified
    age ranges.
    """

    AmountType = Literal[
        "0 to 4",
        "5 to 17",
        "18 to 49",
        "50 to 64",
        "65 to 74",
        "75 and above",
        "Adult",
        "Pediatric",
        "Total",
    ]

    amount_variables: dict[AmountType, str] = {
        "0 to 4": "0to4",
        "5 to 17": "5to17",
        "18 to 49": "18to49",
        "50 to 64": "50to64",
        "65 to 74": "65to74",
        "75 and above": "75plus",
        "Adult": "adult",
        "Pediatric": "ped",
        "Total": "total",
    }

    amount_type: AmountType

    def __init__(
        self,
        disease_name: DiseaseType,
        amount_type: AmountType = "Total",
        voluntary_reporting=True,
        time_frame: TimeFrame | None = None,
    ):
        """
        Creates an ADRIO of the admissions for a confirmed disease per 100k
        population.

        Parameters
        ----------
        disease_name: DiseaseType
            The name of the disease that is desired to be fetched for (options: 'RSV',
            'Influenza', 'Covid').
        amount_type : AmountType
            The category of hospitalized patient sums to fetch for.
            - `'Total'`: Displays the total amount of patient admissions with
            the confirmed disease (default).
            - `'Adult'`: Displays the number of adult patient admissions, starting from
            age 18 and beyond, confirmed with the confirmed disease.
            - `'Pediatric'`: Displays the number of pediatric patient admissions, from
            ages 0 to 17, confirmed with the confirmed disease.
            - `'0 to 4'`: Displays the number of patient admissions, ages 0 to 4, with
            the confirmed disease.
            - `'5 to 17'`: Displays the number of patient admissions, ages 5 to 17, with
            the confirmed disease.
            - `'18 to 49'`: Displays the number of patient admissions, ages 18 to 49,
            with the confirmed disease.
            - `'50 to 64'`: Displays the number of patient admissions, ages 50 to 64,
            with the confirmed disease.
            - `'65 to 74'`: Displays the number of patient admissions, ages 65 to 74,
            with the confirmed disease.
            - `'75 and above'`: Displays the number of patient admissions, from ages 75
            and beyond, with the confirmed disease.
        voluntary_reporting: bool, optional
            The flag that indicates whether the user would like the ADRIO to warn or
            error when the timeframe is within a voluntary reporting period.
            If True, all available data is returned with a warning about the
            timeframe. (default)
            If False, the ADRIO will error and will not return any requested data.
        time_frame : TimeFrame, optional
            The range of dates to fetch hospital metric data for.
            Default: the simulation time frame.
        """
        super().__init__(disease_name, voluntary_reporting, time_frame)
        self.amount_type = amount_type

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        time_frame = self._validate_dates_()
        scope = _validate_scope(self.scope)
        amount_var = self.amount_variables[self.amount_type]
        disease_var = _DISEASE_VARIABLES[self.disease_name]
        age_type_var = ""
        if amount_var == "total":
            adm_var = f"totalconf{disease_var}newadmper100k"
        elif amount_var in ["adult", "ped"]:
            adm_var = f"totalconf{disease_var}newadm{amount_var}per100k"
        else:
            if amount_var in ["0to4", "5to17"]:
                age_type_var = "ped"
            else:
                age_type_var = "adult"
            adm_var = f"numconf{disease_var}newadm{age_type_var}{amount_var}per100k"
        return _fetch_respiratory(
            adm_var,
            scope,
            time_frame,
            self.progress,
        )


@adrio_cache
class HospitalizationsICU(_RespiratoryADRIO):
    """
    Creates a TxN matrix of tuples, containing a date and a float representing the
    number of ICU patients hospitalized with a confirmed disease for that week. May be
    specified for the total number of patients, the number of adult patients, or the
    number of pediatric patients.
    """

    AmountType = Literal["Total", "Adult", "Pediatric"]

    amount_variables: dict[AmountType, str] = {
        "Total": "total",
        "Adult": "adult",
        "Pediatric": "ped",
    }

    amount_type: AmountType

    def __init__(
        self,
        disease_name: DiseaseType,
        amount_type: AmountType = "Total",
        voluntary_reporting=True,
        time_frame: TimeFrame | None = None,
    ):
        """
        Creates an ADRIO of the confirmed hospitalizations for a confirmed disease.

        Parameters
        ----------
        disease_name: DiseaseType
            The name of the disease that is desired to be fetched for (options: 'RSV',
            'Influenza', 'Covid').
        amount_type : AmountType
            The category of hospitalized patient sums to fetch for.
            - `'Total'`: Displays the total amount of patients that have been
            hospitalized with a confirmed disease (default).
            - `'Adult'`: Displays the number of adults, starting from age 18 and beyond,
            who have been hospitalized with a confirmed disease.
            - `'Pediatric'`: Displays the number of pediatric patients, from ages 0 to
            17, who have been hospitalized with a confirmed disease.
        voluntary_reporting: bool, optional
            The flag that indicates whether the user would like the ADRIO to warn or
            error when the timeframe is within a voluntary reporting period.
            If True, all available data is returned with a warning about the
            timeframe. (default)
            If False, the ADRIO will error and will not return any requested data.
        time_frame : TimeFrame, optional
            The range of dates to fetch hospital metric data for.
            Default: the simulation time frame.
        """
        super().__init__(disease_name, voluntary_reporting, time_frame)
        self.amount_type = amount_type

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
        time_frame = self._validate_dates_()
        scope = _validate_scope(self.scope)
        amount_var = self.amount_variables[self.amount_type]
        disease_var = _DISEASE_VARIABLES[self.disease_name]
        if amount_var == "total":
            hosp_var = f"totalconf{disease_var}icupats"
        else:
            hosp_var = f"numconf{disease_var}icupats{amount_var}"
        return _fetch_respiratory(
            hosp_var,
            scope,
            time_frame,
            self.progress,
        )
