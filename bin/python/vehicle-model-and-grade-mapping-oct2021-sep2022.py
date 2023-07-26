# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 22:42:30 2023

@author: Mahsa
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np

df = pd.read_csv(r'../../results/computed-trajectories-Oct2021toSep2022.csv', delimiter=',', skiprows=0, low_memory=False)
df2 = pd.read_csv(r'../../data/tidy/vehicles-summary.csv', delimiter=',', skiprows=0, low_memory=False)


### Map powertrain
mydict = df2.groupby('Type')['Equipment ID'].agg(list).to_dict()
d = {val:key for key, lst in mydict.items() for val in lst}
df['Powertrain'] = df['Vehicle'].map(d)


### Map vehicle models
mydict = df2.groupby('Model')['Equipment ID'].agg(list).to_dict()
d = {val:key for key, lst in mydict.items() for val in lst}
df['VehicleModel'] = df['Vehicle'].map(d)


# Assign weights based on vehicle models
thisdict = {
  "Gillig 30 ":25700,
  "Gillig 35":26440,
  "Gillig 40":27180,
  "NEW FLYER XD35": 26900 ,
  "NEW FLYER XD40": 27180 ,
  "NEW FLYER XDE40": 28250 ,
  "NEW FLYER XDE60 (ARTIC)": 42250 ,
  "NEW FLYER XE35": 26900 ,
  "NEW FLYER XE40": 32770 ,
  "PROTERRA CATALYST BE-40": 27370 ,
}
df['VehiclWeight(lb)'] = df['VehicleModel'].map(thisdict)


# map grades
df_trajectories = df.copy()
df_elevation = pd.read_csv('../../results/stop-elevations.csv', delimiter=',', low_memory=False)
mydict = df_elevation.set_index('stopid').to_dict()['elevation']
df_trajectories['elevation'] = df_trajectories['Stop'].map(mydict)
elevation_dict = df_elevation.set_index('stopid')['elevation'].to_dict()
df_trajectories['elevation'] = df_trajectories['Stop'].map(elevation_dict)
df_trajectories.sort_values(by=['Vehicle', 'ServiceDateTime'], ascending=True, inplace=True)

# Convert required columns to numpy arrays
elevation = df_trajectories['elevation'].values
distance = df_trajectories['dist'].values

# Compute elevation differences and distances using numpy arrays
elevation_diff = (elevation[1:] - elevation[:-1]) / 1000
distance = distance[1:] * 1.61

# Compute the grade using numpy operations
elev_squared = elevation_diff**2
dist_squared = distance**2
cond = elev_squared < dist_squared
grade = np.zeros(len(df_trajectories))
grade[1:][cond] = elevation_diff[cond] / np.sqrt(dist_squared[cond] - elev_squared[cond])

# Assign the computed grades to the dataframe
df_trajectories['grade'] = grade

# Save tidy csv file
df_trajectories.to_csv(r'../../data/tidy/large/trajectories-mapped-powertrain-weight-grade-oct2021-sep2022.csv')
