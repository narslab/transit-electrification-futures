# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:39:25 2022

@author: Mahsa
"""

import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np


# Read fuel rates
df = pd.read_csv(r'../../results/computed-fuel-rates-oct2021-sep2022-test-10222023.csv', delimiter=',', skiprows=0, low_memory=False)


# Subsetting data frame for "Conventional", "hybrid", and "electric" buses
df_conventional=df.loc[df['Powertrain'] == 'conventional'].copy()
df_hybrid=df.loc[df['Powertrain'] == 'hybrid'].copy()


# Read validation 
df_validation = pd.read_csv(r'../../data/tidy/fuel-tickets-clean-oct2021-sep2022.csv', delimiter=',', skiprows=0, low_memory=False)
df_validation=df_validation.loc[df_validation['Qty']>0]
df_validation.sort_values(by=['Equipment ID','Transaction Date'], inplace=True)
df_validation.drop(['Unnamed: 0'], axis=1, inplace=True)
df_validation.rename(columns={"Transaction Date": "ServiceDateTime","Equipment ID":"Vehicle"}, inplace=True)
df_validation.sort_values(by=['Vehicle','ServiceDateTime'], inplace=True)
df_validation['ServiceDateTime'] = pd.to_datetime(df_validation['ServiceDateTime'])
df_validation=df_validation.reset_index()
print(df_validation.columns)

### Map powertrain in the validation dataset
df2 = pd.read_csv(r'../../data/tidy/vehicles-summary.csv', delimiter=',', skiprows=0, low_memory=False)
mydict = df2.groupby('Type')['Equipment ID'].agg(list).to_dict()
d = {val:key for key, lst in mydict.items() for val in lst}
df_validation['Powertrain'] = df_validation['Vehicle'].map(d)

def compute_metrics(df):
    energy = df['Energy']
    qty = df['Qty']
    mape = mean_absolute_percentage_error(energy, qty)
    rmse = np.sqrt(mean_squared_error(energy, qty))
    return mape, rmse

def validation(hybrid=False):
    df = df_hybrid.copy() if hybrid else df_conventional.copy()
    validation = df_validation[df_validation.Powertrain == ('hybrid' if hybrid else 'conventional')].copy()
    
    for data in [df, validation]:
        data['ServiceDateTime'] = pd.to_datetime(data['ServiceDateTime'])
        data.sort_values(by=['Vehicle', 'ServiceDateTime'], inplace=True)
    
    df_integrated = validation.copy().reset_index(drop=True)
    vehicle_groups = df.groupby('Vehicle')
    
    for i, row in df_integrated.iterrows():
        if i == 0 or row['Vehicle'] != df_integrated.loc[i - 1, 'Vehicle']:
            continue
        vehicle_data = vehicle_groups.get_group(row['Vehicle'])
        df_filtered = vehicle_data[(df_integrated.at[i - 1, 'ServiceDateTime'] < vehicle_data['ServiceDateTime']) & (vehicle_data['ServiceDateTime'] < row['ServiceDateTime'])]
        df_integrated.at[i, 'dist'] = df_filtered['dist'].sum()
        df_integrated.at[i, 'Energy'] = df_filtered['Energy'].sum()
    
    df_integrated_clean = df_integrated.dropna(subset=['Energy', 'Qty'])
    df_integrated_clean = df_integrated_clean[(~df_integrated_clean['Energy'].isin([np.nan, np.inf, -np.inf])) & (~df_integrated_clean['Qty'].isin([np.nan, np.inf, -np.inf]))]
    df_integrated_clean = df_integrated_clean[df_integrated_clean['Qty'] != 0]
    df_integrated_clean = df_integrated_clean[df_integrated_clean['Energy'] != 0]

    
    file_name = f"../../results/validation-vs-computed-fuel-rates-clean-{'heb' if hybrid else 'cdb'}-oct2021-sep2022-test-10222023.csv"
    df_integrated_clean.to_csv(file_name)

    return compute_metrics(df_integrated_clean)


#hybrid
mape_heb, rmse_heb = validation(hybrid=True)
print("mape_heb:", mape_heb, "rmse_heb:", rmse_heb)


#conventional
mape_cdb, rmse_cdb = validation(hybrid=False)
print("mape_cdb:", mape_cdb, "rmse_cdb:", rmse_cdb)



