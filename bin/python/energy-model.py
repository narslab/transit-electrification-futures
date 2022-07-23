# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:39:25 2022

@author: Mahsa
"""

import yaml
import pandas as pd
import numpy as np

f = open('params.yaml')
parameters = yaml.safe_load(f)
f.close()

class vehicleParams():
	def __init__(self, **entries):
		self.__dict__.update(entries)

# Read trajectories df
df = pd.read_csv(r'../../results/trajectories-mapped-powertrain-weight.csv', delimiter=',', skiprows=0, low_memory=False)
df.speed = df.speed *1.60934 # takes speed in km/h (Convert from mph to km/h)
df.rename(columns={"speed": "Speed", "acc": "Acceleration", "VehiclWeight(lb)": "Vehicle_mass"}, inplace=True)
df = df.fillna(0)


# Subsetting data frame for "Conventional", "hybrid", and "electric" buses
df_conventional=df.loc[df['Powertrain'] == 'conventional'].copy()
df_hybrid=df.loc[df['Powertrain'] == 'hybrid'].copy()
df_electric=df.loc[df['Powertrain'] == 'electric'].copy()

# Define model parameters
p = vehicleParams(**parameters)
rho = p.air_density
C_D = p.drag_coefficient
A_f = p.frontal_area
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


# Define power function for diesel vehicle
def power_d(df_input):
    df = df_input
    v = df.Speed
    a = df.Acceleration
    m = df.Vehicle_mass+(df.Onboard*179)
    P_t = (1/float(3600*eta_d_dis))*((1./25.92)*rho*C_D*A_f*v*v + m*g*C_r*(c1*v + c2)/1000 + 1.1*m*a)*v
    return P_t


# Define fuel rate function for diesel vehicle
def fuelRate_d(df_input, hybrid=False):
	# Estimates fuel consumed (liters per second) 
    if hybrid == True:
        a0 = a0_heb 
        a1 = a1_heb        
        P_t = power_d(df_input)
        FC_t = P_t.apply(lambda x: a0 + a1*x +a2*x*x if x >= 0 else a0)
        FC_t=FC_t*0.85
    else:
        a0 = a0_cdb
        a1 = a1_cdb
        P_t = power_d(df_input)
        FC_t = P_t.apply(lambda x: a0 + a1*x +a2*x*x if x >= 0 else a0)  
    return FC_t

# Define power function for electric vehicle
def power_e(df_input):
    df = df_input
    v = df.Speed
    a = df.Acceleration
    m = df.Vehicle_mass+(df.Onboard*179)
    P_t = (eta_batt/eta_m*eta_d_beb)*(1/float(3600*eta_d_beb))*((1./25.92)*rho*C_D*A_f*v*v + m*g*C_r*(c1*v + c2)/1000 + 1.1*m*a)*v
    factor=[]
    for index, value in a.items():
        if value<0:
            factor.append(np.exp(-(0.0411/abs(a))))
        else:
            factor.append(1)
    P_t_e=factor*P_t
    return P_t_e

# Define fuel rate function for diesel vehicle
def EnergyConsumption_e(df_input):
	# Estimates energy consumed (KWh)     
    df = df_input
    t = df.time_delta_in_seconds/3600
    P_t = power_e(df_input)
    FC_t = P_t * t
    return FC_t

# Compute fuel rate for "Conventional", and "hybrid" buses
df_conventional['FuelRate/Energy']=fuelRate_d(df_conventional)
df_hybrid['FuelRate/Energy']=fuelRate_d(df_hybrid, hybrid=True)
df_electric['FuelRate/Energy']=EnergyConsumption_e(df_electric)



#megre subset dataframes 
df_final=pd.concat([df_conventional, df_hybrid, df_electric])

# Sort dataframe
df_final.sort_values(by=['Vehicle','ServiceDateTime'], ascending=True, inplace=True)

# Change mistakenly computed fuel rates to zero
# =============================================================================
# for i in df_final.index:
#     if i==0:
#         df_final.at[i , 'FuelRate(L/s)'] = 0
#     else:
#         if df_final['Vehicle'].loc[i]==df_final['Vehicle'].loc[i-1]:
#             if df_final['Date'].loc[i]!=df_final['Date'].loc[i-1]:
#                 df_final.at[i , 'FuelRate(L/s)'] = 0
#         else:
#             df_final.at[i , 'FuelRate(L/s)'] = 0  
# =============================================================================
### This resulted an error in the script so,
### I had to run this part in Jupyter "modify-fuel-rates"
### You need to run "modify-fuel-rates.ipynb" after this

        
# Save df_final
df_final.to_csv(r'../../results/computed-fuel-rates.csv')


