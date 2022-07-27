from shapely.geometry import Point
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from haversine import haversine
from datetime import datetime
import seaborn as sns
import matplotlib
from matplotlib.pyplot import figure
from requests import get
from pandas import json_normalize
#import geopandas as gpd
#from geopandas import GeoDataFrame

df = pd.read_csv('../../data/tidy/stops-lon-lat.csv', delimiter=',', low_memory=False)

def get_elevation(lat = None, lon = None):
    '''
        script for returning elevation in m from lat, long
    '''
    if lat is None or lon is None: return None
    
    query = ('https://api.open-elevation.com/api/v1/lookup'
             f'?locations={lat},{lon}')
    
    # Request with a timeout for slow responses
    r = get(query, timeout = 20)

    # Only get the json response in case of 200 or 201
    if r.status_code == 200 or r.status_code == 201:
        elevation = json_normalize(r.json(), 'results')['elevation'].values[0]
    else: 
        elevation = None
    return elevation

def main():
	for i in df.index[1:5]:
    elevation = get_elevation(lat = df.loc[i , 'lat'], lon = df.loc[i , 'lon'])
    df.at[i, 'elevation'] = elevation
    df.to_csv('../../results/stop-elevations.csv')

if __name__ == "__main__":
    main()    
