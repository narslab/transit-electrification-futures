# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 12:36:05 2022

@author: Mahsa
"""
#import required packages
import pandas as pd
import numpy as np

# Read data
df = pd.read_csv(r'../../data/raw/Apr2022LatLon.csv', delimiter=',', skiprows=0, low_memory=False)

# Join H,M, and S
df['time'] = (pd.to_datetime(df['Date'].astype(str) + '-' +df['Hour'].astype(str) + ':' + df['Minute'].astype(str)+ ':' + df['Second'].astype(str)).dt.time)

# Format date columns
df["ServiceDateTime"] = pd.to_datetime(df["ServiceDateTime"], errors="coerce")

# Set index 
df = df.set_index('ServiceDateTime')

# Sort data
df.sort_values(by=['Vehicle','Date','Hour','Minute','Second'], ascending=True, inplace=True)

# Set index 
df = df.reset_index()

# Assign previous Lon and Lat to nan values if the vehicle and dates are same
sel = df['Lon'].isnull() & df['Lon'].isnull()
for i in df.loc[sel]['Unnamed: 0']:
    idx = df[df['Unnamed: 0']== i].index.tolist()
    if df['Vehicle'].loc[idx[0]]==df['Vehicle'].loc[idx[0]-1]:
        if df['Date'].loc[idx[0]]==df['Date'].loc[idx[0]-1]:             
            df['Lon'][idx[0]] = df['Lon'][idx[0]-1]    
            df['Lat'][idx[0]] = df['Lat'][idx[0]-1]


# Set index 
df = df.set_index('ServiceDateTime')

# Sort data
df.sort_values(by=['Vehicle','Date','Hour','Minute','Second'], ascending=True, inplace=True)

# Save tidy csv file
df.to_csv(r'../../data/tidy/large/vehicle-locations-Apr2022.csv')