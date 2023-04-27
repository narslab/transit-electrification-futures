# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:39:25 2022

@author: Mahsa
"""

import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np


# Read fuel rates
df = pd.read_csv(r'../../results/computed-fuel-rates-oct2021-sep2022.csv', delimiter=',', skiprows=0, low_memory=False)


# Subsetting data frame for "Conventional", "hybrid", and "electric" buses
df_conventional=df.loc[df['Powertrain'] == 'conventional'].copy()
df_hybrid=df.loc[df['Powertrain'] == 'hybrid'].copy()


# Read validation 
df_validation = pd.read_csv(r'../../data/tidy/fuel-tickets-oct2021-sep2022.csv', delimiter=',', skiprows=0, low_memory=False)
df_validation=df_validation.loc[df_validation['Qty']>0]
df_validation.sort_values(by=['Equipment ID','Transaction Date'], inplace=True)
df_validation.drop(['Unnamed: 0'], axis=1, inplace=True)
df_validation.rename(columns={"Transaction Date": "ServiceDateTime","Equipment ID":"Vehicle"}, inplace=True)
df_validation.sort_values(by=['Vehicle','ServiceDateTime'], inplace=True)
df_validation['ServiceDateTime'] = pd.to_datetime(df_validation['ServiceDateTime'])
df_validation=df_validation.reset_index()

### Map powertrain in the validation dataset
df2 = pd.read_csv(r'../../data/tidy/vehicles-summary.csv', delimiter=',', skiprows=0, low_memory=False)
mydict = df2.groupby('Type')['Equipment ID'].agg(list).to_dict()
d = {val:key for key, lst in mydict.items() for val in lst}
df_validation['Powertrain'] = df_validation['Vehicle'].map(d)

def validation(hybrid=False):
    # Create separate dataframes for hybrid and conventional vehicles
    if hybrid:
        df = df_hybrid.copy()
        validation = df_validation[df_validation.Powertrain == 'hybrid'].copy()
    else:
        df = df_conventional.copy()
        validation = df_validation[df_validation.Powertrain == 'conventional'].copy()
      
    # Convert ServiceDateTime column to datetime data type
    df['ServiceDateTime'] = pd.to_datetime(df['ServiceDateTime'])
    validation['ServiceDateTime'] = pd.to_datetime(validation['ServiceDateTime'])
    
    # Sort dataframes by vehicle and service datetime
    df.sort_values(by=['Vehicle', 'ServiceDateTime'], inplace=True)
    validation.sort_values(by=['Vehicle', 'ServiceDateTime'], inplace=True)    
   
    # Creat integerated df by merging df and validation
    df_integrated = validation.copy()
    df_integrated.sort_values(by=['Vehicle','ServiceDateTime'], inplace=True)
    df_integrated=df_integrated.reset_index()
    for i in df_integrated.index:
        if i==0:
            pass 
        else:
            if df_integrated['Vehicle'][i]==df_integrated['Vehicle'][i-1]:
                df_filtered=df.loc[(df['Vehicle']==df_integrated['Vehicle'][i])&((df_integrated['ServiceDateTime'][i-1]<df['ServiceDateTime']))&((df['ServiceDateTime']<df_integrated['ServiceDateTime'][i]))]
                df_integrated.loc[i,'dist']=df_filtered['dist'].sum()
                df_integrated.loc[i,'Energy']=df_filtered['Energy'].sum()
                df_filtered=df_filtered.reset_index()
            else:
                pass               
    # Drop rows with missing values in 'Energy' and 'Qty' columns
    df_integrated_clean = df_integrated.dropna(subset=['Energy', 'Qty'])
    df_integrated_clean = df_integrated_clean[~(df_integrated_clean['Energy'].isin([np.nan, np.inf, -np.inf]))]
    df_integrated_clean = df_integrated_clean[~(df_integrated_clean['Qty'].isin([np.nan, np.inf, -np.inf]))]
    
    # save the integerated dataframe
    if hybrid:
        df_integrated_clean.to_csv(r'../../results/validation-vs-computed-fuel-rates-heb-oct2021-sep2022.csv')
    else:
        df_integrated_clean.to_csv(r'../../results/validation-vs-computed-fuel-rates-cdb-oct2021-sep2022.csv')

    # Extract the two columns from your dataframe
    energy = df_integrated_clean['Energy']
    qty = df_integrated_clean['Qty']

    # Compute the MAPE and RMSE
    mape = mean_absolute_percentage_error(energy, qty)
    rmse = np.sqrt(mean_squared_error(energy, qty))
    return (mape, rmse)

#hybrid
mape_heb, mape2_heb, rmse_heb = validation(hybrid=True)
print("mape_heb:", mape_heb, "rmse_heb:", rmse_heb)


#conventional
mape_cdb, rmse_cdb = validation(hybrid=False)
print("mape_cdb:", mape_cdb, "rmse_cdb:", rmse_cdb)



