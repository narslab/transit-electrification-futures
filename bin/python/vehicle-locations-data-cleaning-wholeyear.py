# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 17:26:44 2023

@author: Mahsa
"""

import pandas as pd
import numpy as np

# Read data
df7 = pd.read_csv(r'../../data/raw/Apr2022LatLon.csv', delimiter=',', skiprows=0, low_memory=False)
df7 = df7.drop(df7.columns[[0, 27, 28, 29, 30, 31, 32]], axis=1)
#df7 = df7.drop(df7.columns[[0, 32]], axis=1)
df1 = pd.read_csv(r'../../data/raw/DateTripStop_Oct2021LatLon.csv', delimiter=',', skiprows=0, low_memory=False)
df2 = pd.read_csv(r'../../data/raw/DateTripStop_Nov2021LatLon.csv', delimiter=',', skiprows=0, low_memory=False)
df3 = pd.read_csv(r'../../data/raw/DateTripStop_Dec2021LatLon.csv', delimiter=',', skiprows=0, low_memory=False)
df4 = pd.read_csv(r'../../data/raw/DateTripStop_Jan2022LatLon.csv', delimiter=',', skiprows=0, low_memory=False)
df5 = pd.read_csv(r'../../data/raw/DateTripStop_Feb2022LatLon.csv', delimiter=',', skiprows=0, low_memory=False)
df6 = pd.read_csv(r'../../data/raw/DateTripStop_Mar2022LatLon.csv', delimiter=',', skiprows=0, low_memory=False)
df8 = pd.read_csv(r'../../data/raw/DateTripStop_May2022LatLon.csv', delimiter=',', skiprows=0, low_memory=False)
df9 = pd.read_csv(r'../../data/raw/DateTripStop_Jun2022LatLon.csv', delimiter=',', skiprows=0, low_memory=False)
df10 = pd.read_csv(r'../../data/raw/DateTripStop_Jul2022LatLon.csv', delimiter=',', skiprows=0, low_memory=False)
df11 = pd.read_csv(r'../../data/raw/DateTripStop_Aug2022LatLon.csv', delimiter=',', skiprows=0, low_memory=False)
df12 = pd.read_csv(r'../../data/raw/DateTripStop_Sep2022LatLon.csv', delimiter=',', skiprows=0, low_memory=False)

df_list = [df1, df2, df3, df4, df5, df6, df8, df9, df10, df11, df12]
df_all = pd.concat(df_list, axis=0, ignore_index=True)
df = pd.concat([df7, df_all], axis=0, ignore_index=True)

# Format timestamp columns
df["ServiceDateTime"] = pd.to_datetime(df['ServiceDateTime'], errors="coerce")
#df1["ServiceDateTime"] = pd.to_datetime(df1['ServiceDateTime'], errors="coerce")
#df2["ServiceDateTime"] = pd.to_datetime(df2['ServiceDateTime'], errors="coerce")
#df3["ServiceDateTime"] = pd.to_datetime(df3['ServiceDateTime'], errors="coerce")
#df4["ServiceDateTime"] = pd.to_datetime(df4['ServiceDateTime'], errors="coerce")
#df5["ServiceDateTime"] = pd.to_datetime(df5['ServiceDateTime'], errors="coerce")
#df6["ServiceDateTime"] = pd.to_datetime(df6['ServiceDateTime'], errors="coerce")
#df7["ServiceDateTime"] = pd.to_datetime(df7['ServiceDateTime'], errors="coerce")
#df8["ServiceDateTime"] = pd.to_datetime(df8['ServiceDateTime'], errors="coerce")
#df9["ServiceDateTime"] = pd.to_datetime(df9['ServiceDateTime'], errors="coerce")
#df10["ServiceDateTime"] = pd.to_datetime(df10['ServiceDateTime'], errors="coerce")
#df11["ServiceDateTime"] = pd.to_datetime(df11['ServiceDateTime'], errors="coerce")
#df12["ServiceDateTime"] = pd.to_datetime(df12['ServiceDateTime'], errors="coerce")


# Add columns related to timestamped
#df['Weekday'] = df['ServiceDateTime'].dt.weekday_name
df['Hour'] = df['ServiceDateTime'].dt.hour
df['Minute'] = df['ServiceDateTime'].dt.minute
df['Second'] = df['ServiceDateTime'].dt.second
df['Date'] = df['ServiceDateTime'].dt.date

# =============================================================================
# df1['Weekday'] = df1['ServiceDateTime'].dt.weekday_name
# df1['Hour'] = df1['ServiceDateTime'].dt.hour
# df1['Minute'] = df1['ServiceDateTime'].dt.minute
# df1['Second'] = df1['ServiceDateTime'].dt.second
# df1['Date'] = df1['ServiceDateTime'].dt.date
# df2['Weekday'] = df2['ServiceDateTime'].dt.weekday_name
# df2['Hour'] = df2['ServiceDateTime'].dt.hour
# df2['Minute'] = df2['ServiceDateTime'].dt.minute
# df2['Second'] = df2['ServiceDateTime'].dt.second
# df2['Date'] = df2['ServiceDateTime'].dt.date
# df3['Weekday'] = df3['ServiceDateTime'].dt.weekday_name
# df3['Hour'] = df3['ServiceDateTime'].dt.hour
# df3['Minute'] = df3['ServiceDateTime'].dt.minute
# df3['Second'] = df3['ServiceDateTime'].dt.second
# df3['Date'] = df3['ServiceDateTime'].dt.date
# df4['Weekday'] = df4['ServiceDateTime'].dt.weekday_name
# df4['Hour'] = df4['ServiceDateTime'].dt.hour
# df4['Minute'] = df4['ServiceDateTime'].dt.minute
# df4['Second'] = df4['ServiceDateTime'].dt.second
# df4['Date'] = df4['ServiceDateTime'].dt.date
# df5['Weekday'] = df5['ServiceDateTime'].dt.weekday_name
# df5['Hour'] = df5['ServiceDateTime'].dt.hour
# df5['Minute'] = df5['ServiceDateTime'].dt.minute
# df5['Second'] = df5['ServiceDateTime'].dt.second
# df5['Date'] = df5['ServiceDateTime'].dt.date
# df6['Weekday'] = df6['ServiceDateTime'].dt.weekday_name
# df6['Hour'] = df6['ServiceDateTime'].dt.hour
# df6['Minute'] = df6['ServiceDateTime'].dt.minute
# df6['Second'] = df6['ServiceDateTime'].dt.second
# df6['Date'] = df6['ServiceDateTime'].dt.date
# df8['Weekday'] = df8['ServiceDateTime'].dt.weekday_name
# df8['Hour'] = df8['ServiceDateTime'].dt.hour
# df8['Minute'] = df8['ServiceDateTime'].dt.minute
# df8['Second'] = df8['ServiceDateTime'].dt.second
# df8['Date'] = df8['ServiceDateTime'].dt.date
# df9['Weekday'] = df9['ServiceDateTime'].dt.weekday_name
# df9['Hour'] = df9['ServiceDateTime'].dt.hour
# df9['Minute'] = df9['ServiceDateTime'].dt.minute
# df9['Second'] = df9['ServiceDateTime'].dt.second
# df9['Date'] = df9['ServiceDateTime'].dt.date
# df10['Weekday'] = df10['ServiceDateTime'].dt.weekday_name
# df10['Hour'] = df10['ServiceDateTime'].dt.hour
# df10['Minute'] = df10['ServiceDateTime'].dt.minute
# df10['Second'] = df10['ServiceDateTime'].dt.second
# df10['Date'] = df10['ServiceDateTime'].dt.date
# df11['Weekday'] = df11['ServiceDateTime'].dt.weekday_name
# df11['Hour'] = df11['ServiceDateTime'].dt.hour
# df11['Minute'] = df11['ServiceDateTime'].dt.minute
# df11['Second'] = df11['ServiceDateTime'].dt.second
# df11['Date'] = df11['ServiceDateTime'].dt.date
# df12['Weekday'] = df12['ServiceDateTime'].dt.weekday_name
# df12['Hour'] = df12['ServiceDateTime'].dt.hour
# df12['Minute'] = df12['ServiceDateTime'].dt.minute
# df12['Second'] = df12['ServiceDateTime'].dt.second
# df12['Date'] = df12['ServiceDateTime'].dt.date
# =============================================================================


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
df.to_csv(r'../../data/tidy/large/vehicle-locations-Oct2021toSep2022.csv')