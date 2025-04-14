from dataclasses import dataclass
from datetime import date
from typing import Literal, Mapping
from urllib.parse import quote, urlencode
from warnings import warn

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.adrio.adrio import ADRIOLegacy, ProgressCallback, adrio_legacy_cache
from epymorph.error import DataResourceError
from epymorph.geography.scope import GeoScope
from epymorph.geography.us_census import CensusScope
from epymorph.geography.us_geography import STATE, CensusGranularityName
from epymorph.geography.us_tiger import get_states
from epymorph.time import TimeFrame


@dataclass(frozen=True)
class DataSource:
    url_base: str
    date_col: str
    fips_col: str
    data_col: str
    granularity: CensusGranularityName
    """The geographic granularity of the source data."""
    replace_sentinel: float | None
    """If None, ignore sentinel values (-999999); otherwise, replace them with
    the given value."""
    map_geo_ids: Mapping[str, str] | None = None
    """If None, use the scope node IDs as they are, otherwise use this mapping
    to map them."""


def _query_location(
    info: DataSource, loc_clause: str, date_clause: str
) -> pd.DataFrame:
    """
    Helper function for _api_query() that builds and sends queries for
    individual locations.
    """
    current_return = 10000
    total_returned = 0
    cdc_df = pd.DataFrame()

    group_col = ""
    group_filter = ""
    group_filter = (
        "AND `group`='By Week'"
        if info.url_base == "https://data.cdc.gov/resource/r8kw-7aab.csv?"
        else ""
    )

    while current_return == 10000:
        url = info.url_base + urlencode(
            quote_via=quote,
            safe=",()'$:",
            query={
                "$select": (
                    f"{info.date_col},{info.fips_col},{info.data_col}{group_col}"
                ),
                "$where": f"{loc_clause} AND {date_clause}{group_filter}",
                "$limit": 10000,
                "$offset": total_returned,
            },
        )

        cdc_df = pd.concat([cdc_df, pd.read_csv(url, dtype={info.fips_col: str})])

        current_return = len(cdc_df.index) - total_returned
        total_returned += current_return

    return cdc_df


_SENTINEL = -999999
"""A common sentinel value which represents values which have been redacted
for privacy because there were less than 4 individuals in that data point."""


def _api_query(
    source: DataSource,
    scope: CensusScope,
    time_frame: TimeFrame,
    progress: ProgressCallback,
) -> NDArray[np.float64]:
    """
    Composes URLs to query API and sends query requests. Limits each query to
    10000 rows, combining several query results if this number is exceeded.
    Returns Dataframe containing requested data sorted by date and location fips.
    """
    node_ids = (
        [source.map_geo_ids[x] for x in scope.node_ids]
        if source.map_geo_ids is not None
        else scope.node_ids
    )
    if scope.granularity == "state" and source.granularity != "state":
        # query county level data with state fips codes
        location_clauses = [f"starts_with({source.fips_col}, '{x}')" for x in node_ids]
    else:
        # query with full fips codes
        formatted_fips = ",".join(f"'{node}'" for node in node_ids)
        location_clauses = [f"{source.fips_col} in ({formatted_fips})"]

    date_clause = (
        f"{source.date_col} "
        f"between '{time_frame.start_date}T00:00:00' "
        f"and '{time_frame.end_date}T00:00:00'"
    )

    processing_steps = len(location_clauses) + 1

    def query_step(index, loc_clause) -> pd.DataFrame:
        step_result = _query_location(source, loc_clause, date_clause)
        progress((index + 1) / processing_steps, None)
        return step_result

    cdc_df = pd.concat(
        [query_step(i, loc_clause) for i, loc_clause in enumerate(location_clauses)]
    )

    if source.replace_sentinel is not None:
        num_sentinel = (cdc_df[source.data_col] == _SENTINEL).sum()
        if num_sentinel > 0:
            cdc_df = cdc_df.replace(_SENTINEL, source.replace_sentinel)
            warn(
                f"{num_sentinel} values < 4 were replaced with "
                f"{source.replace_sentinel} in returned data."
            )

    if scope.granularity == "state" and source.granularity != "state":
        # aggregate county data to state level
        cdc_df[source.fips_col] = cdc_df[source.fips_col].map(STATE.truncate)
        cdc_df = cdc_df.groupby([source.fips_col, source.date_col]).sum().reset_index()

    return _as_numpy(
        cdc_df.sort_values(by=[source.date_col, source.fips_col]).pivot_table(
            index=source.date_col,
            columns=source.fips_col,
            values=source.data_col,
        )
    )


def _as_numpy(data_df: pd.DataFrame) -> NDArray[np.float64]:
    """Convert a DataFrame to a time-series by node numpy array where each value is a
    tuple of date and data value. Note: this time-series is not necessarily the same
    length as simulation T, because not all ADRIOs produce a daily value."""
    dates = data_df.index.to_numpy(dtype="datetime64[D]")
    return np.array(
        [list(zip(dates, data_df[col], strict=True)) for col in data_df.columns],
        dtype=[("date", "datetime64[D]"), ("data", np.float64)],
    ).T


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


class _RespiratoryADRIO(ADRIOLegacy[np.float64]):
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


def _validate_scope(scope: GeoScope) -> CensusScope:
    if not isinstance(scope, CensusScope):
        msg = "Census scope is required for CDC attributes."
        raise DataResourceError(msg)
    return scope


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


@adrio_legacy_cache
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


@adrio_legacy_cache
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
