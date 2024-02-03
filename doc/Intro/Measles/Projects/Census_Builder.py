import os
import random
from datetime import date
from typing import cast

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygris
from census import Census
from IPython.display import Image
from matplotlib.patches import ConnectionPatch
from PIL import Image
from pygris import (block_groups, counties, primary_secondary_roads,
                    school_districts, tracts)
from pygris.geocode import geocode
from shapely.geometry import Point, mapping

# Global variables
census_df = None

def build_census_df():
    global census_df

    STATE = 'AZ'
    COUNTY = 'Maricopa'
    YEAR = 2019
    crs_to_use = 'EPSG:4269'

    dir = os.path.expanduser('~/Desktop/Github/EpiMoRPH-Modeling/social-determinate-of-health/scratch/Movement_model')
    powerpoint = os.path.expanduser('~/Desktop/Github/EpiMoRPH-Modeling/social-determinate-of-health/scratch/powerpoints/School_movement_01-22-24')

    census = Census('98a3a86a5708b65b92a50cc76d8d67280efbeb5d')

    CentroidDType = np.dtype([('longitude', np.float64), ('latitude', np.float64)])

    census_block_groups = block_groups(state=STATE, county=COUNTY, year=YEAR, cache=False)
    census_block_groups = census_block_groups.to_crs(crs_to_use)

    state_fips_list = census_block_groups['STATEFP'].unique()
    county_fips_list = census_block_groups['COUNTYFP'].unique()

    state_fips = ','.join(map(str, state_fips_list))
    county_fips = ','.join(map(str, county_fips_list))
    census_block_groups = census_block_groups[['GEOID', 'geometry', 'ALAND']]
    CNTY = state_fips + county_fips

    AGE_VARS = [
        "B01001_003E", "B01001_004E", "B01001_005E", "B01001_006E", "B01001_007E",
        "B01001_008E", "B01001_009E", "B01001_010E", "B01001_011E", "B01001_012E",
        "B01001_013E", "B01001_014E", "B01001_015E", "B01001_016E", "B01001_017E",
        "B01001_018E", "B01001_019E", "B01001_020E", "B01001_021E", "B01001_022E",
        "B01001_023E", "B01001_024E", "B01001_025E"
    ]

    SCHOOL_VARS = [
        "B14007_002E", "B14007_003E", "B14007_004E", "B14007_005E", "B14007_006E",
        "B14007_007E", "B14007_008E", "B14007_009E", "B14007_010E", "B14007_011E",
        "B14007_012E", "B14007_013E", "B14007_014E", "B14007_015E"
    ]

    cbgs = census_block_groups.copy()
    cbgs['centroid'] = cbgs.centroid
    cbgs['area'] = cbgs.area / 1e6
    cbgs_age = pd.DataFrame(columns=AGE_VARS)
    cbgs_erollemt = pd.DataFrame(columns=SCHOOL_VARS)

    cbgs_age_2 = cbgs_age.copy()

    def enrollment_bracketize(cbgs_enrollment: pd.DataFrame) -> pd.DataFrame:
        brackets = {
            "Preschool": ("B14007_002E", "B14007_002E"),
            "Elementary": ("B14007_003E", "B14007_008E"),
            "Middle": ("B14007_009E", "B14007_011E"),
            "High": ("B14007_012E", "B14007_015E")
        }

        school_ranges = {}
        for level, (start_var, end_var) in brackets.items():
            start_index = SCHOOL_VARS.index(start_var)
            end_index = SCHOOL_VARS.index(end_var) + 1
            school_ranges[level] = (start_index, end_index)

        return pd.DataFrame({
            level: cbgs_enrollment.iloc[:, start:end].sum(axis=1)
            for level, (start, end) in school_ranges.items()
        }, dtype=np.int64)

    enrollment_summary = enrollment_bracketize(cbgs_erollemt)
    data = {
        'label': cbgs['GEOID'].to_numpy(dtype=np.str_),
        'centroid': cbgs['centroid'].to_numpy(dtype=CentroidDType),
        'area km^2': cbgs['area'].to_numpy(dtype=np.int64),
        'population': cbgs['population'].to_numpy(dtype=np.int64),
        'population_by_school': enrollment_summary.to_numpy(dtype=np.int64),
        'population_by_age': cbgs_age_2.to_numpy(dtype=np.int64),
        'median_age': cbgs['median_age'].to_numpy(dtype=np.float64),
        'median_income': cbgs['median_income'].to_numpy(dtype=np.int64),
        'average_household_size': cbgs['average_household_size'].to_numpy(dtype=np.float64),
        'pop_density_km2': cbgs['pop_density_km2'].to_numpy(dtype=np.float64),
    }

    census_df = pd.DataFrame({k: list(v) for k, v in data.items()})
    census_block_groups = pd.merge(census_df, census_block_groups, left_on='label', right_on='GEOID')
    census_block_groups = gpd.GeoDataFrame(census_block_groups, geometry=census_block_groups['geometry']).set_crs(epsg=4269, inplace=True)

    census_df = census_block_groups

# Call the function to initialize the DataFrame
build_census_df()
