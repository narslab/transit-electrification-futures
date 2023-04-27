# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:39:25 2022

@author: Mahsa
"""

import yaml
import pandas as pd
import numpy as np



#import model


f = open('params.yaml')
parameters = yaml.safe_load(f)
f.close()

class vehicleParams():
	def __init__(self, **entries):
		self.__dict__.update(entries)

# Define model parameters
p = vehicleParams(**parameters)
rho = p.air_density
C_D = p.drag_coefficient
C_h= p.altitude_correction_factor
A_f_cdb = p.frontal_area_cdb
A_f_heb = p.frontal_area_heb
A_f_beb = p.frontal_area_beb
g = p.gravitational_acceleration
C_r = p.rolling_coefficient
c1 = p.rolling_resistance_coef1
c2 = p.rolling_resistance_coef2
eta_d_dis = p.driveline_efficiency_d_dis
eta_d_beb = p.driveline_efficiency_d_beb
P_mfo = p.idling_mean_fuel_pressure
omega = p.idling_speed
d = p.engine_displacement
Q = p.fuel_lower_heating_value
N = p.number_of_engine_cylinders
FE_city_p = p.fuel_economy_city
FE_hwy_p = p.fuel_economy_hwy
eps = p.epsilon
eta_batt = p.battery_efficiency
eta_m = p.motor_efficiency
a0_cdb = p.alpha_0_cdb
a1_cdb = p.alpha_1_cdb
a2 = p.alpha_2
a0_heb = p.alpha_0_heb
a1_heb = p.alpha_1_heb
b=p.beta

# Define power function for diesel vehicle
def power_d(df_input, hybrid=False):
    if hybrid == True:
       A_f_d=A_f_heb
    else:
       A_f_d=A_f_cdb        
    df = df_input
    v = df.speed
    a = df.acc
    gr = df.grade
    m = (df.Vehicle_mass+df.Onboard*179)*0.453592 # converts lb to kg
    P_t = (1/float(3600*eta_d_dis))*((1./25.92)*rho*C_D*C_h*A_f_d*v*v + m*g*C_r*(c1*v + c2)/1000 + 1.2*m*a+m*g*gr)*v
    return P_t


# Define fuel rate function for diesel vehicle
def regenerative_braking(df_input, hybrid=True):
	# Apply regenerative braking for HEBs 
    df = df_input
    if hybrid == True:
        factor = df.acc.apply(lambda a: 1 if a >= 0 else np.exp(-(0.0411/abs(a))))
        P_t = factor * power_d(df_input, hybrid=True)
    else:
        P_t = power_d(df_input, hybrid=False)
    return P_t


# Define Energy consumption function for diesel vehicle
def energyConsumption_d(df_input, hybrid=False):
	# Estimates energy consumed (gallons)     
    df = df_input
    t = df.time_delta_in_seconds
    if hybrid == True:
        P_t = regenerative_braking(df_input, hybrid=True)
    else:
        P_t = regenerative_braking(df_input, hybrid=False)
    E_t = (P_t * t)/3.78541
    return E_t


# Read trajectories df
df_trajectories = pd.read_csv(r'../../data/tidy/large/trajectories-mapped-powertrain-weight-grade-oct2021-sep2022.csv', delimiter=',', skiprows=0, low_memory=False)
df_trajectories.rename(columns={"VehiclWeight(lb)": "Vehicle_mass"}, inplace=True)
df_trajectories['Date']=pd.to_datetime(df_trajectories['Date'])
df_trajectories.speed = df_trajectories.speed *1.60934 # takes speed in km/h (Convert from mph to km/h)
df_trajectories = df_trajectories.fillna(0)

# Subsetting data frame for "Conventional", "hybrid", and "electric" buses
df_conventional=df_trajectories.loc[df_trajectories['Powertrain'] == 'conventional'].copy()
df_hybrid=df_trajectories.loc[df_trajectories['Powertrain'] == 'hybrid'].copy()



# read validation df
df_validation = pd.read_csv(r'../../data/tidy/fuel-tickets-oct2021-sep2022.csv', delimiter=',', skiprows=0, low_memory=False)
df_validation=df_validation.loc[df_validation['Qty']>0]
df_validation.sort_values(by=['Equipment ID','Transaction Date'], inplace=True)
df_validation.drop(['Unnamed: 0'], axis=1, inplace=True)
df_validation.rename(columns={"Transaction Date": "ServiceDateTime","Equipment ID":"Vehicle"}, inplace=True)
df_validation["dist"] = np.nan
df_validation["Energy"] = np.nan
df_validation.sort_values(by=['Vehicle','ServiceDateTime'], inplace=True)
df_validation['ServiceDateTime'] = pd.to_datetime(df_validation['ServiceDateTime'])
df_validation=df_validation.reset_index()

### Map powertrain in the validation dataset
df2 = pd.read_csv(r'../../data/tidy/vehicles-summary.csv', delimiter=',', skiprows=0, low_memory=False)
mydict = df2.groupby('Type')['Equipment ID'].agg(list).to_dict()
d = {val:key for key, lst in mydict.items() for val in lst}
df_validation['Powertrain'] = df_validation['Vehicle'].map(d)

# =============================================================================
# def calibrate_parameter(hybrid=False):
#      # Create separate dataframes for hybrid and conventional vehicles
#      if hybrid:
#          df = df_hybrid.copy()
#          validation = df_validation[df_validation.Powertrain == 'hybrid'].copy()
#      else:
#          df = df_conventional.copy()
#          validation = df_validation[df_validation.Powertrain == 'conventional'].copy()
#      
#      # Calculate energy consumption for each trip
#      df['Energy'] = energyConsumption_d(df, hybrid=hybrid)
#      
#      
#      # Convert ServiceDateTime column to datetime data type
#      df['ServiceDateTime'] = pd.to_datetime(df['ServiceDateTime'])
#      validation['ServiceDateTime'] = pd.to_datetime(validation['ServiceDateTime'])
#      
#      # Sort dataframes by vehicle and service datetime
#      df.sort_values(by=['Vehicle', 'ServiceDateTime'], inplace=True)
#      validation.sort_values(by=['Vehicle', 'ServiceDateTime'], inplace=True)    
#     
#      # creat integerated df by merging df and validation
#      df_integrated = validation.copy()
#      df_integrated.sort_values(by=['Vehicle','ServiceDateTime'], inplace=True)
#      df_integrated=df_integrated.reset_index()
#      for i in df_integrated.index:
#          if i==0:
#              pass 
#          else:
#              if df_integrated['Vehicle'][i]==df_integrated['Vehicle'][i-1]:
#                  df_filtered=df.loc[(df['Vehicle']==df_integrated['Vehicle'][i])&(df_integrated['ServiceDateTime'][i-1]<df['ServiceDateTime'])&(df['ServiceDateTime']<df_integrated['ServiceDateTime'][i])]
#                  df_integrated.loc[i,'dist']=df_filtered['dist'].sum()
#                  df_integrated.loc[i,'Energy']=df_filtered['Energy'].sum()
#              else:
#                  pass               
#      # Drop rows with missing values in 'Energy' and 'Qty' columns
#      df_integrated_clean = df_integrated.dropna(subset=['Energy', 'Qty'])
#  
#      # Perform the polyfit using cleaned data
#      coefficients, residuals, _, _, _ = np.polyfit(df_integrated_clean['Energy'], df_integrated_clean['Qty'], deg=2, full=True)
#      a0_cdb, a1_cdb, a2_cdb = coefficients
#  
#      # Return coefficients
#      return coefficients
# =============================================================================


def calibrate_parameter(hybrid=False):
    # Create separate dataframes for hybrid and conventional vehicles
    if hybrid:
        df = df_hybrid.copy()
        validation = df_validation[df_validation.Powertrain == 'hybrid'].copy()
    else:
        df = df_conventional.copy()
        validation = df_validation[df_validation.Powertrain == 'conventional'].copy()

    # Calculate energy consumption for each trip
    df['Energy'] = energyConsumption_d(df, hybrid=hybrid)

    # Convert ServiceDateTime column to datetime data type
    df['ServiceDateTime'] = pd.to_datetime(df['ServiceDateTime'])
    validation['ServiceDateTime'] = pd.to_datetime(validation['ServiceDateTime'])

    # Sort dataframes by vehicle and service datetime
    df.sort_values(by=['Vehicle', 'ServiceDateTime'], inplace=True)
    validation.sort_values(by=['Vehicle', 'ServiceDateTime'], inplace=True)
    validation.reset_index(drop=True, inplace=True)  


    # Initialize dist and Energy columns in validation with 0
    validation['dist'] = 0
    validation['Energy'] = 0

    # Calculate the sum of dist and Energy values between two consecutive ServiceDateTime in validation
    for index, row in validation.iterrows():
        if index > 0 and row['Vehicle'] == validation.iloc[index - 1]['Vehicle']:
            lower_bound = validation.iloc[index - 1]['ServiceDateTime']
            upper_bound = row['ServiceDateTime']
            dist_sum = df.loc[(df['Vehicle'] == row['Vehicle']) & (df['ServiceDateTime'] > lower_bound) & (df['ServiceDateTime'] <= upper_bound), 'dist'].sum()
            energy_sum = df.loc[(df['Vehicle'] == row['Vehicle']) & (df['ServiceDateTime'] > lower_bound) & (df['ServiceDateTime'] <= upper_bound), 'Energy'].sum()
            validation.loc[index, 'dist'] = dist_sum
            validation.loc[index, 'Energy'] = energy_sum

    # Drop rows with missing or invalid values in 'Energy' and 'Qty' columns
    df_integrated_clean = validation.dropna(subset=['Energy', 'Qty'])
    df_integrated_clean = df_integrated_clean[~(df_integrated_clean['Energy'].isin([np.nan, np.inf, -np.inf]))]
    df_integrated_clean = df_integrated_clean[~(df_integrated_clean['Qty'].isin([np.nan, np.inf, -np.inf]))]

    # Add a small constant to the input data to improve conditioning
    epsilon = 1e-8
    energy = df_integrated_clean['Energy'] + epsilon
    qty = df_integrated_clean['Qty'] + epsilon

    # Perform the polyfit using cleaned data
    coefficients, residuals, _, _, _ = np.polyfit(energy, qty, deg=2, full=True)
    a0, a1, a2 = coefficients

    # Return coefficients
    return coefficients


#hybrid
a0_heb, a1_heb, a2_heb = calibrate_parameter(hybrid=True)
print("a0_heb:", a0_heb, "a1_heb:", a1_heb, "a2_heb:", a2_heb)


#conventional
a0_cdb, a1_cdb, a2_cdb = calibrate_parameter(hybrid=False)
print("a0_cdb:", a0_cdb, "a1_cdb:", a1_cdb, "a2_cdb:", a2_heb)




