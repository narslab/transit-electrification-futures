# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 08:54:43 2023

@author: Mahsa
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from haversine import haversine
from datetime import datetime
import seaborn as sns
import matplotlib
from matplotlib.pyplot import figure
from matplotlib.dates import DateFormatter
from matplotlib.lines import Line2D
import numpy as np

df_initial = pd.read_csv(r'../../data/tidy/large/vehicle-locations-Oct2021toSep2022.csv', delimiter=',', skiprows=0, low_memory=False)
df=df_initial.copy()
df["ServiceDateTime"] = pd.to_datetime(df['ServiceDateTime'], errors="coerce")
df.sort_values(by=['Vehicle','ServiceDateTime'], ascending=True, inplace=True)
df['time_delta'] = df.groupby(['Vehicle', 'Date'])['ServiceDateTime'].diff()
df.time_delta=df.time_delta.astype(str).str.replace('0 days ', '')
df['time_delta']= pd.to_datetime(df['time_delta'])
df['time_delta_in_seconds'] = df['time_delta'].dt.hour * 3600 + df['time_delta'].dt.minute * 60 + df['time_delta'].dt.second

#calculating distances 
dist=[]
for i in df.index:
    if i==0:
        dist.append(0) 
    else:
        if df['Vehicle'].loc[i]==df['Vehicle'].loc[i-1]:
            if df['Date'].loc[i]==df['Date'].loc[i-1]:           
                coordinate_x = (df['Lon'].loc[i] ,df['Lat'].loc[i]) # (lat, lon) for x
                coordinate_x_minus_1 = (df['Lon'].loc[i-1] ,df['Lat'].loc[i-1]) #(lat, lon) for x-1
                distance=haversine(coordinate_x_minus_1, coordinate_x, unit='mi')
                dist.append(distance) 
            else:
                #dist.append('nan')
                dist.append(0) 
        else:   
            #dist.append('nan')
            dist.append(0) 
df['dist'] = dist

#calculating speed
speed=[]
for i in df.index:
    if i==0 or i==1:
        speed.append(0) 
    else:
        if df['Vehicle'].loc[i]==df['Vehicle'].loc[i-1]:
            if df['Date'].loc[i]==df['Date'].loc[i-1]:
                FMT = '%H:%M:%S'
                time_diff = abs(datetime.strptime(df['time'].loc[i-1], FMT) - datetime.strptime(df['time'].loc[i], FMT)).total_seconds() / 3600.0
                distance = df['dist'].loc[i]
                if time_diff==0:
                    speed.append(0)
                else:
                    speed_current=distance/time_diff
                    speed.append(speed_current) 
                    #if speed_current>=90:
                        #speed.append(90) 
                    #else:    
                        #speed.append(speed_current)
            else:
                #speed.append('nan')
                speed.append(0) 
        else:   
            #speed.append('nan')
            speed.append(0) 
df['speed'] = speed

#calculating acceleration by unique vehicle Id and unique day
acc=[]
for i in df.index:
    if i==0 or i==1 or i==2:
        acc.append(0) 
    else:
        if df['Vehicle'].loc[i]==df['Vehicle'].loc[i-1]:
            if df['Date'].loc[i]==df['Date'].loc[i-1]:
                FMT = '%H:%M:%S'
                time_diff = abs(datetime.strptime(df['time'].loc[i-1], FMT) - datetime.strptime(df['time'].loc[i], FMT)).total_seconds() / 3600.0
                #if df['speed'].loc[i-1]=='nan':
                #    df['speed'].loc[i-1]==0
                #else:
                #    pass
                speed_diff = df['speed'].loc[i] - df['speed'].loc[i-1]
                if time_diff==0:
                    acc.append(0) 
                else:
                    acc_current=(speed_diff/time_diff)*0.00012417777777778 # 1 miles/h2 = 0.00012417777777778 m/s2
                    #if -5<acc_current<5:
                    #    acc.append(acc_current)
                    #else:
                    #    acc.append(0)
                    acc.append(acc_current)
            else:
                #acc.append('nan')
                acc.append(0) 
        else:   
            #acc.append('nan')
            acc.append(0) 
df['acc'] = acc
df.to_csv(r'../../results/computed-trajectories-oct2021tosep2022.csv')