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

from epymorph import geo_library
from epymorph.data_shape import Shapes
from epymorph.error import GeoValidationException
from epymorph.geo.cache import load_from_cache, save_to_cache
from epymorph.geo.spec import AttribDef, CentroidDType, StaticGeoSpec, Year
from epymorph.geo.static import StaticGeo

########################################################## Enduser Input #################################################################
STATE = input('Enter a State: ')
COUNTY = input('Enter a County: ')
YEAR = int(input('Enter a Year: '))
crs_to_use = 'EPSG:4269' 


dir = os.path.expanduser('~/Desktop/Github/EpiMoRPH-Modeling/social-determinate-of-health/scratch/Movement_model')
powerpoint = os.path.expanduser('~/Desktop/Github/EpiMoRPH-Modeling/social-determinate-of-health/scratch/powerpoints/School_movement_01-22-24')

census = Census('98a3a86a5708b65b92a50cc76d8d67280efbeb5d')

CentroidDType = np.dtype([('longitude', np.float64), ('latitude', np.float64)])

census_block_groups = block_groups(state=STATE, county=COUNTY, year=YEAR, cache=False)
census_block_groups = census_block_groups.to_crs(crs_to_use)

districts = school_districts(state = STATE, year = YEAR)

state_fips_list = census_block_groups['STATEFP'].unique()
county_fips_list = census_block_groups['COUNTYFP'].unique()

state_fips = ','.join(map(str, state_fips_list))
county_fips = ','.join(map(str, county_fips_list))

CNTY = state_fips + county_fips

################################################ Functions ################################################
def haversine(centroid_1_lon: float, centroid_1_lat: float,
              centroid_2_lon: float, centroid_2_lat: float) -> np.double:
    R = 3959.87433
    dLat = np.radians(centroid_1_lat - centroid_2_lat)
    dLon = np.radians(centroid_1_lon - centroid_2_lon)
    lat1 = np.radians(centroid_1_lat)
    lat2 = np.radians(centroid_2_lat)

    a = np.sin(dLat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def classify_school_type(low_grade, high_grade):
    if low_grade in ['PK'] and high_grade in ['PK']:
        return 'Preschool'
    if low_grade in ['KG'] and high_grade in ['KG']:
        return 'kindergarten School'
    if low_grade in ['PK', 'KG', '01', '02', '03', '04', '05'] and high_grade in ['PK', 'KG', '01', '02', '03', '04', '05']:
        return 'Elementary School'
    elif low_grade in ['06', '07', '08'] and high_grade in ['06', '07', '08']:
        return 'Middle School'
    elif low_grade in ['09', '10', '11', '12'] and high_grade in ['09', '10', '11', '12']:
        return 'High School'
    elif low_grade in ['PK', 'KG', '01', '02', '03', '04', '05'] and high_grade in ['06', '07', '08']:
        return 'Elementary/Middle School'
    elif low_grade in ['06', '07', '08'] and high_grade in ['09', '10', '11', '12']:
        return 'Middle/High School'
    else:
        return 'Elementary/Middle/High School'
    
def random_color():
    r, g, b = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return (r / 255.0, g / 255.0, b / 255.0, 1.0)

def read_excel_file(directory, file_path, skip_rows, dtype, usecols):
    """Reads an Excel file and returns a DataFrame based on the given parameters."""
    full_path = os.path.join(directory, file_path)
    return pd.read_excel(full_path, dtype=dtype, usecols=usecols, skiprows=skip_rows)
    

################################# National Center for Education Statistics ################################################
random_colors = [random_color() for _ in range(len(districts))]

districts['color'] = random_colors

districts['Centroid'] = districts['geometry'].representative_point()
districts['coords'] = districts['Centroid'].apply(lambda point: (point.x, point.y))

districts = districts.to_crs(epsg=4326)  

directory = os.path.expanduser('~/Desktop/Github/EpiMoRPH-Modeling/social-determinate-of-health/scratch/Movement_model')  
file_configs = [
    {
        "file_path": "shapefiles/EDGE_GEOCODE_PUBLICSCH_1920/EDGE_GEOCODE_PUBLICSCH_1920.xlsx",  
        "dtype": {"CNTY": object},
        "usecols": ['NCESSCH', 'CNTY', 'STREET', 'CITY', 'ZIP', 'LAT', 'LON', 'SCHOOLYEAR'],
        "skip_rows": None
    },
    {
        "file_path": f"School_movement/NCES_public_schools_{STATE}.xltx",  
        "dtype": None,
        "usecols": ['NCES School ID', 'Low Grade*', 'High Grade*', 'School Name', 'District', 'Students*',
                    'Teachers*', 'Student Teacher Ratio*'],
        "skip_rows": 14
    },
    {
        "file_path": f"School_movement/NCES_public_schools_{STATE}.xltx",  
        "dtype": None,
        "usecols": ['NCES School ID', 'Low Grade', 'High Grade', 'School Name', 'District', 'Students',
                    'Teachers', 'Student Teacher Ratio'],
        "skip_rows": 11
    }
]

schools_location_df = read_excel_file(directory, file_configs[0]['file_path'], skip_rows=file_configs[0]['skip_rows'], dtype=file_configs[0]['dtype'], usecols=file_configs[0]['usecols'])

if STATE != 'AZ':
    schools_info_df = read_excel_file(directory, file_configs[1]['file_path'], 
                                      skip_rows=file_configs[1]['skip_rows'], 
                                      dtype=file_configs[1]['dtype'], 
                                      usecols=file_configs[1]['usecols'])

    schools_info_df['school type'] = schools_info_df.apply(
        lambda row: classify_school_type(row['Low Grade*'], row['High Grade*']), axis=1)
    schools_info_df['Students*'] = pd.to_numeric(schools_info_df['Students*'], errors='coerce')
else:
    schools_info_df = read_excel_file(directory, file_configs[2]['file_path'], 
                                      skip_rows=file_configs[2]['skip_rows'], 
                                      dtype=file_configs[2]['dtype'], 
                                      usecols=file_configs[2]['usecols'])
    schools_info_df['school type'] = schools_info_df.apply(
        lambda row: classify_school_type(row['Low Grade'], row['High Grade']), axis=1)
    schools_info_df['Students'] = pd.to_numeric(schools_info_df['Students'], errors='coerce')

NCES_public_schools_df = pd.merge(schools_info_df, schools_location_df, left_on='NCES School ID', right_on='NCESSCH')

NCES_public_schools_df = NCES_public_schools_df[NCES_public_schools_df['CNTY'] == CNTY]
NCES_public_schools_df.drop(columns=['NCESSCH'], inplace=True)

########################################################################################################################
dir_path = os.path.expanduser('~/Desktop/Github/EpiMoRPH-Modeling/social-determinate-of-health/scratch/Movement_model')
filename = 'matrix_array.npy'
file_path = os.path.join(dir_path, filename)

loaded_array = np.load(file_path)

####################################################### Census ##########################################################

SCHOOL_VARS = [
    "B14007_002E",  # Enrolled in nursery school, preschool
    "B14007_003E",  # Enrolled in kindergarten
    "B14007_004E",  # Enrolled in grade 1
    "B14007_005E",  # Enrolled in grade 2
    "B14007_006E",  # Enrolled in grade 3
    "B14007_007E",  # Enrolled in grade 4
    "B14007_008E",  # Enrolled in grade 5
    "B14007_009E",  # Enrolled in grade 6
    "B14007_010E",  # Enrolled in grade 7
    "B14007_011E",  # Enrolled in grade 8
    "B14007_012E",  # Enrolled in grade 9
    "B14007_013E",  # Enrolled in grade 10
    "B14007_014E",  # Enrolled in grade 11
    "B14007_015E",  # Enrolled in grade 12
]

# CBG data
query1 = {'for': 'block group: *',
          'in': f'state: {state_fips} county: {county_fips}'}
cbgs_raw = census.acs5.get([
    "B01003_001E",  # Total Population
    "B01003_001M",  # Total Population MOE

], query1, year=YEAR)

cbgs_errollemt_raw = census.acs5.get(SCHOOL_VARS, query1, year=YEAR)

cbgs_geog = pygris.block_groups(
    state=state_fips,
    county=county_fips,
    year=YEAR,
    cache=True
)

cbgs = pd.DataFrame.from_records(cbgs_raw)
cbgs.fillna(0, inplace=True)
cbgs.replace(-666666666, 0, inplace=True)

tract_geoids = cbgs['state'] + cbgs['county'] + cbgs['tract']
geoids = tract_geoids + cbgs['block group']

cbgs = pd.DataFrame({
    'geoid': geoids,
    'population': cbgs['B01003_001E'].astype(np.int64),
    'population_MOE': cbgs['B01003_001M'].astype(np.float64),

})

# Merge geo info
cbgs = cbgs.merge(pd.DataFrame({
    'geoid': cbgs_geog['GEOID'],
    'centroid': cbgs_geog['geometry'].apply(lambda row: row.centroid.coords[0]),
    # TIGER areas are in m^2; divide by 1e6 to get km^2
    'area': cbgs_geog.ALAND / 1e6
}), on='geoid')


cbgs.sort_values(by='geoid', inplace=True)
cbgs.reset_index(drop=True, inplace=True)

cbgs_errollemt = pd.DataFrame.from_records(cbgs_errollemt_raw)
cbgs_errollemt.astype(np.int64)


geoids = cbgs_errollemt['state'] + cbgs_errollemt['county'] + \
    cbgs_errollemt['tract'] + cbgs_errollemt['block group']
cbgs_errollemt.insert(len(cbgs_errollemt.columns), 'geoid', geoids)

cbgs_errollemt = cbgs[['geoid']].merge(cbgs_errollemt, how='inner', on='geoid')
cbgs_errollemt.reset_index(drop=True, inplace=True)
cbgs_errollemt.drop(columns=['geoid'], inplace=True)


def enrollment_bracketize(cbgs_enrollment: pd.DataFrame) -> pd.DataFrame:
    """Using the enrollment info sum the age groups for elementary, middle, and high school."""

    brackets = {
        "Preschool": ("B14007_002E", "B14007_002E"),  # Preschool
        # Kindergarten to Grade 5
        "Elementary": ("B14007_003E", "B14007_008E"),
        "Middle": ("B14007_009E", "B14007_011E"),      # Grade 6 to Grade 8
        "High": ("B14007_012E", "B14007_015E")         # Grade 9 to Grade 12
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


enrollment_summary = enrollment_bracketize(cbgs_errollemt)

data = {
    'label': cbgs['geoid'].to_numpy(dtype=np.str_),
    'centroid': cbgs['centroid'].to_numpy(dtype=CentroidDType),
    'area km^2': cbgs['area'].to_numpy(dtype=np.int64),
    'population': cbgs['population'].to_numpy(dtype=np.int64),
    'population_MOE': cbgs['population_MOE'].to_numpy(dtype=np.float64),
    'population_by_school': enrollment_summary.to_numpy(dtype=np.int64),
}

census_df = pd.DataFrame({k: list(v) for k, v in data.items()})


############################################################################################
spec = StaticGeoSpec(
    attributes=[
        AttribDef('label', np.str_, Shapes.N),
        AttribDef('centroid', CentroidDType, Shapes.N),
        AttribDef('population', np.int64, Shapes.N),  
    ],
    time_period=Year(2019))

label = np.array(census_df['label'], dtype=np.str_)
population = np.array(census_df['population'], dtype=np.int64)
centroid = np.array(census_df['centroid'], dtype=CentroidDType)

geo = StaticGeo(spec, {
    'label': label,
    'centroid': centroid,
    'population': population,
})

try:
    geo.validate()
except GeoValidationException as e:
    print(e.pretty())


from pathlib import Path

from epymorph.geo.static import StaticGeoFileOps

filename = StaticGeoFileOps.to_archive_filename(f'{STATE}_{COUNTY}_{YEAR}')
filepath = Path('epymorph/data/geo') / filename

geo.save(filepath)
