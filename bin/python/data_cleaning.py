# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 12:36:05 2022

@author: Mahsa
"""
#import required packages
import pandas as pd

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

# Save tidy csv file
df.to_csv(r'../../data/tidy/vehicle-locations.csv')