from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygris
from numpy.typing import NDArray

from epymorph.engine.standard_sim import Output
from epymorph.geo.geo import Geo

state_info = {
    'AL': {'name': 'Alabama', 'fips': '01'},
    'AK': {'name': 'Alaska', 'fips': '02'},
    'AZ': {'name': 'Arizona', 'fips': '04'},
    'AR': {'name': 'Arkansas', 'fips': '05'},
    'CA': {'name': 'California', 'fips': '06'},
    'CO': {'name': 'Colorado', 'fips': '08'},
    'CT': {'name': 'Connecticut', 'fips': '09'},
    'DE': {'name': 'Delaware', 'fips': '10'},
    'FL': {'name': 'Florida', 'fips': '12'},
    'GA': {'name': 'Georgia', 'fips': '13'},
    'HI': {'name': 'Hawaii', 'fips': '15'},
    'ID': {'name': 'Idaho', 'fips': '16'},
    'IL': {'name': 'Illinois', 'fips': '17'},
    'IN': {'name': 'Indiana', 'fips': '18'},
    'IA': {'name': 'Iowa', 'fips': '19'},
    'KS': {'name': 'Kansas', 'fips': '20'},
    'KY': {'name': 'Kentucky', 'fips': '21'},
    'LA': {'name': 'Louisiana', 'fips': '22'},
    'ME': {'name': 'Maine', 'fips': '23'},
    'MD': {'name': 'Maryland', 'fips': '24'},
    'MA': {'name': 'Massachusetts', 'fips': '25'},
    'MI': {'name': 'Michigan', 'fips': '26'},
    'MN': {'name': 'Minnesota', 'fips': '27'},
    'MS': {'name': 'Mississippi', 'fips': '28'},
    'MO': {'name': 'Missouri', 'fips': '29'},
    'MT': {'name': 'Montana', 'fips': '30'},
    'NE': {'name': 'Nebraska', 'fips': '31'},
    'NV': {'name': 'Nevada', 'fips': '32'},
    'NH': {'name': 'New Hampshire', 'fips': '33'},
    'NJ': {'name': 'New Jersey', 'fips': '34'},
    'NM': {'name': 'New Mexico', 'fips': '35'},
    'NY': {'name': 'New York', 'fips': '36'},
    'NC': {'name': 'North Carolina', 'fips': '37'},
    'ND': {'name': 'North Dakota', 'fips': '38'},
    'OH': {'name': 'Ohio', 'fips': '39'},
    'OK': {'name': 'Oklahoma', 'fips': '40'},
    'OR': {'name': 'Oregon', 'fips': '41'},
    'PA': {'name': 'Pennsylvania', 'fips': '42'},
    'RI': {'name': 'Rhode Island', 'fips': '44'},
    'SC': {'name': 'South Carolina', 'fips': '45'},
    'SD': {'name': 'South Dakota', 'fips': '46'},
    'TN': {'name': 'Tennessee', 'fips': '47'},
    'TX': {'name': 'Texas', 'fips': '48'},
    'UT': {'name': 'Utah', 'fips': '49'},
    'VT': {'name': 'Vermont', 'fips': '50'},
    'VA': {'name': 'Virginia', 'fips': '51'},
    'WA': {'name': 'Washington', 'fips': '53'},
    'WV': {'name': 'West Virginia', 'fips': '54'},
    'WI': {'name': 'Wisconsin', 'fips': '55'},
    'WY': {'name': 'Wyoming', 'fips': '56'}
}

vaild_county_fips_codes = {
    'Apache': {'county_fips':'04001', 'fips': '001'}, 
    'Cochise': {'county_fips':'04003', 'fips': '003'}, 
    'Coconino': {'county_fips':'04005', 'fips': '005'}, 
    'Gila': {'county_fips':'04007', 'fips': '007'}, 
    'Graham': {'county_fips':'04009', 'fips': '009'}, 
    'Greenlee': {'county_fips':'04011', 'fips': '011'}, 
    'La Paz': {'county_fips':'04012', 'fips': '012'}, 
    'Maricopa': {'county_fips':'04013', 'fips': '013'}, 
    'Mohave': {'county_fips':'04015', 'fips': '015'}, 
    'Navajo': {'county_fips':'04017', 'fips': '017'}, 
    'Pima': {'county_fips':'04019', 'fips': '019'}, 
    'Pinal': {'county_fips':'04021', 'fips': '021'}, 
    'Santa Cruz': {'county_fips':'04023', 'fips': '023'}, 
    'Yavapai': {'county_fips':'04025', 'fips': '025'}, 
    'Yuma': {'county_fips':'04027', 'fips': '027'}, 
}

def get_state_details(state_info):
    state_mapping = {}
    for abbrev, state_data in state_info.items():
        state_mapping[abbrev.lower()] = abbrev
        state_mapping[state_data['name'].lower()] = abbrev
        state_mapping[state_data['fips']] = abbrev

    state_details = []

    while True:
        user_input = input("Enter a state abbreviation, full name, or FIPS code (type 'stop' to finish): ")
        if user_input.lower() == 'stop':
            break
        elif user_input.lower() not in state_mapping:
            print("Invalid input. Please enter a valid state abbreviation, full name, or FIPS code.")
            continue
        state_abbrev = state_mapping[user_input.lower()]
        state = state_info[state_abbrev]
        state_details.append(state['fips'])

    return state_details


def get_county_fips_codes(valid_county_fips_codes):
    county_fips_codes = []

    while True:
        user_input = input("Enter a county name, full FIPS code, or last three digits of FIPS code (type 'stop' to finish): ")
        if user_input.lower() == 'stop':
            break
        elif user_input.lower() == 'all':
            county_fips_codes = [codes['county_fips'][2:] for codes in valid_county_fips_codes.values()]
            break
        elif user_input in valid_county_fips_codes:
            county_fips_codes.append(valid_county_fips_codes[user_input]['county_fips'][2:])
        elif user_input.isdigit() and len(user_input) == 3:
            for county, codes in valid_county_fips_codes.items():
                if codes['county_fips'][2:] == user_input:
                    county_fips_codes.append(codes['county_fips'][2:])
                    break
        elif user_input.isdigit() and len(user_input) == 5 and user_input[:2] == '04':
            for county, codes in valid_county_fips_codes.items():
                if codes['county_fips'] == user_input:
                    county_fips_codes.append(codes['county_fips'][2:])
                    break
        else:
            print("Invalid input. Please enter a valid county name, full FIPS code, or last three digits of FIPS code.")

    return county_fips_codes

###################################### Alex's Code ######################################

# State Data
def map_user_data_by_state(
    geo: Geo,
    *,
    title: str,
    cmap: Any | None = None,
) -> None:
    """
    Draw a state-level choropleth map using the given `data`. This must be a numpy array whose
    ordering is the same as the nodes in the geo.
    Assumes that the geo contains an attribute (`geoid`) containing the geoids of its nodes.
    (This information is needed to fetch the map shapes.)
    """
    state_fips = {s[:2] for s in geo['geoid']}
    df_states = pygris.states(cb=True, resolution='5m', cache=True, year=2020)
    df_states = df_states.loc[df_states['GEOID'].isin(state_fips)]
   


    return _map_user_data_by_geo(geo, df_states, title=title, cmap=cmap)

def _map_user_data_by_geo(
    geo: Geo,
    df_nodes: pd.DataFrame,
    *,
    df_borders: pd.DataFrame | None = None,
    title: str,
    cmap: Any | None = None,

) -> None:
    """
    Draw a choropleth map for the geo information given.
    This is an internal function to abstract commonalities -- see `map_data_by_county()` and `map_data_by_state()`.
    """
    df_merged = pd.merge(
        on="GEOID",
        left=df_nodes,
        right=pd.DataFrame({'GEOID': geo['geoid']}),
    )
    USA = pygris.states(cb=True, resolution='5m', cache=True, year=2020)
    USA = USA[~USA['STUSPS'].isin(['AK', 'HI', 'GU', 'MP', 'VI', 'PR', 'AS'])]

    fig, ax = plt.subplots(figsize=(6, 4), dpi = 300)
    ax.axis('off')
    ax.set_title(title)
    USA.plot(ax=ax, color='None', linewidth=1, edgecolor='black')
    df_merged.plot(ax=ax, legend=True, color = cmap)
    if df_borders is None:
        df_borders = df_nodes
    df_borders.plot(ax=ax, linewidth=1, edgecolor='black', color='none', alpha=0.8)
    fig.tight_layout()
    plt.show()

# County Data 
def map_user_data_by_county(
    geo: Geo,
    *,
    df_borders: pd.DataFrame | None = None,
    title: str,
    cmap: Any | None = None,
    outline: Literal['states', 'counties'] = 'counties',
) -> None:
    """
    Draw a county-level choropleth map using the given `data`. This must be a numpy array whose
    ordering is the same as the nodes in the geo.
    Assumes that the geo contains an attribute (`geoid`) containing the geoids of its nodes.
    (This information is needed to fetch the map shapes.)
    """
    state_fips = {s[:2] for s in geo['geoid']}
    df_states = pygris.states(cb=True, resolution='5m', cache=True, year=2020)
    df_states = df_states.loc[df_states['GEOID'].isin(state_fips)]
    df_counties = pd.concat([
        pygris.counties(state=s, cb=True, resolution='5m', cache=True, year=2020)
        for s in state_fips
    ])
    if outline == 'counties':
        df_borders = df_counties
    else:
        df_borders = df_states
    return _map_user_county_data_by_geo(geo, df_counties, df_borders=df_borders, title=title, cmap=cmap)
    
def _map_user_county_data_by_geo(
    geo: Geo,
    df_nodes: pd.DataFrame,
    *,
    df_borders: pd.DataFrame | None = None,
    title: str,
    cmap: Any | None = None,

) -> None:
    """
    Draw a choropleth map for the geo information given.
    This is an internal function to abstract commonalities -- see `map_data_by_county()` and `map_data_by_state()`.
    """
    df_merged = pd.merge(
        on="GEOID",
        left=df_nodes,
        right=pd.DataFrame({'GEOID': geo['geoid']}),
    )
    state_fips = {s[:2] for s in geo['geoid']}
    df_states = pygris.states(cb=True, resolution='5m', cache=True, year=2020)
    df_states = df_states.loc[df_states['GEOID'].isin(state_fips)]
    df_counties = pd.concat([
        pygris.counties(state=s, cb=True, resolution='5m', cache=True, year=2020)
        for s in state_fips
    ])

    fig, ax = plt.subplots(figsize=(6, 4), dpi = 300)
    ax.axis('off')
    ax.set_title(title)
    df_counties.plot(ax=ax, color='None', linewidth=1, edgecolor='black')
    df_merged.plot(ax=ax, legend=True, color = cmap)
    if df_borders is None:
        df_borders = df_nodes
    df_borders.plot(ax=ax, linewidth=1, edgecolor='black', color='none', alpha=0.8)
    fig.tight_layout()
    plt.show()

# Census Tract 

def map_user_data_by_tract(
    geo: Geo,
    *,
    df_borders: pd.DataFrame | None = None,
    title: str,
    cmap: Any | None = None,
    outline: Literal['states', 'counties'] = 'counties',
) -> None:
    """
    Draw a county-level choropleth map using the given `data`. This must be a numpy array whose
    ordering is the same as the nodes in the geo.
    Assumes that the geo contains an attribute (`geoid`) containing the geoids of its nodes.
    (This information is needed to fetch the map shapes.)
    """
    state_fips = {s[:2] for s in geo['geoid']}
    df_states = pygris.states(cb=True, resolution='5m', cache=True, year=2020)
    df_states = df_states.loc[df_states['GEOID'].isin(state_fips)]
    df_counties = pygris.tracts(state='AZ',county='Maricopa', cb=True, cache=True, year=2020)
        
    if outline == 'counties':
        df_borders = df_counties
    else:
        df_borders = df_states
    return _map_user_tract_data_by_geo(geo, df_counties, df_borders=df_borders, title=title, cmap=cmap)

def _map_user_tract_data_by_geo(
    geo: Geo,
    df_nodes: pd.DataFrame,
    *,
    df_borders: pd.DataFrame | None = None,
    title: str,
    cmap: Any | None = None,

) -> None:
    """
    Draw a choropleth map for the geo information given.
    This is an internal function to abstract commonalities -- see `map_data_by_county()` and `map_data_by_state()`.
    """
    df_merged = pd.merge(
        on="GEOID",
        left=df_nodes,
        right=pd.DataFrame({'GEOID': geo['geoid']}),
    )       

    fig, ax = plt.subplots(figsize=(6, 4), dpi = 300)
    ax.axis('off')
    ax.set_title(title)
    df_merged.plot(ax=ax, legend=True, color = cmap)
    if df_borders is None:
        df_borders = df_nodes
    df_borders.plot(ax=ax, linewidth=0.2, edgecolor='black', color='none', alpha=0.8)
    fig.tight_layout()
    plt.show()

# Census Block Group
def map_user_data_by_CBG(
    geo: Geo,
    *,
    df_borders: pd.DataFrame | None = None,
    title: str,
    cmap: Any | None = None,
    outline: Literal['states', 'counties'] = 'counties',
) -> None:
    """
    Draw a county-level choropleth map using the given `data`. This must be a numpy array whose
    ordering is the same as the nodes in the geo.
    Assumes that the geo contains an attribute (`geoid`) containing the geoids of its nodes.
    (This information is needed to fetch the map shapes.)
    """
    state_fips = {s[:2] for s in geo['geoid']}
    df_states = pygris.states(cb=True, resolution='5m', cache=True, year=2020)
    df_states = df_states.loc[df_states['GEOID'].isin(state_fips)]
    df_counties = pygris.block_groups(state='AZ',county='Maricopa', cb=True, cache=True, year=2020)
     
    if outline == 'counties':
        df_borders = df_counties
    else:
        df_borders = df_states
    return _map_user_CBG_data_by_geo(geo, df_counties, df_borders=df_borders, title=title, cmap=cmap)

def _map_user_CBG_data_by_geo(
    geo: Geo,
    df_nodes: pd.DataFrame,
    *,
    df_borders: pd.DataFrame | None = None,
    title: str,
    cmap: Any | None = None,

) -> None:
    """
    Draw a choropleth map for the geo information given.
    This is an internal function to abstract commonalities -- see `map_data_by_county()` and `map_data_by_state()`.
    """
    df_merged = pd.merge(
        on="GEOID",
        left=df_nodes,
        right=pd.DataFrame({'GEOID': geo['geoid']}),
    )
    state_fips = {s[:2] for s in geo['geoid']}
    df_states = pygris.states(cb=True, resolution='5m', cache=True, year=2020)
    df_states = df_states.loc[df_states['GEOID'].isin(state_fips)]

    fig, ax = plt.subplots(figsize=(6, 4), dpi = 300)
    ax.axis('off')
    ax.set_title(title)
    df_merged.plot(ax=ax, legend=True, color = cmap)
    if df_borders is None:
        df_borders = df_nodes
    df_borders.plot(ax=ax, linewidth=0.2, edgecolor='black', color='none', alpha=0.8)
    fig.tight_layout()
    plt.show()
 
###################################### Alex's Code ######################################
