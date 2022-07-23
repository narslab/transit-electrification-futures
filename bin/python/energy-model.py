# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:39:25 2022

@author: Mahsa
"""

import yaml
import pandas as pd

f = open('params.yaml')
parameters = yaml.safe_load(f)
f.close()

class vehicleParams():
	def __init__(self, **entries):
		self.__dict__.update(entries)


df = pd.read_csv(r'../../results/trajectories-mapped-powertrain-weight.csv', delimiter=',', skiprows=0, low_memory=False)
df.speed = df.speed *1.60934 # takes speed in km/h (Convert from mph to km/h)
df.rename(columns={"speed": "Speed", "acc": "Acceleration", "VehiclWeight(lb)": "Vehicle_mass"}, inplace=True)
df = df.fillna(0)
#df=df.set_index("ServiceDateTime")


p = vehicleParams(**parameters)
rho = p.air_density
C_D = p.drag_coefficient
A_f = p.frontal_area
g = p.gravitational_acceleration
C_r = p.rolling_coefficient
c1 = p.rolling_resistance_coef1
c2 = p.rolling_resistance_coef2
eta_d = p.driveline_efficiency
P_mfo = p.idling_mean_fuel_pressure
omega = p.idling_speed
d = p.engine_displacement
Q = p.fuel_lower_heating_value
N = p.number_of_engine_cylinders
FE_city_p = p.fuel_economy_city
FE_hwy_p = p.fuel_economy_hwy
eps = p.epsilon
a0 = p.alpha_0
a1 = p.alpha_1
a2 = p.alpha_2



def power():
	v = df.Speed
	a = df.Acceleration
	m = df.Vehicle_mass+(df.Onboard*179)
	P_t = (1/float(3600*eta_d))*((1./25.92)*rho*C_D*A_f*v*v + m*g*C_r*(c1*v + c2)/1000 + 1.04*m*a)*v
	return P_t



def fuelRate(factor=1):
	# Estimates fuel consumed (liters per second) based on input speed profile
	# Input must be dataframe with columns "Time", "Speed", and "Acceleration"
	P_t = power()
	FC_t = P_t.apply(lambda x: a0 + a1*x +a2*x*x if x >= 0 else a0)
	FC_t=FC_t*factor
    # FC_t = FC_t.sum()
	return FC_t


# Subsetting data frame for "Conventional", "hybrid", and "electric" buses
df_conventional=df.loc[df['Powertrain'] == 'conventional'].copy()
df_hybrid=df.loc[df['Powertrain'] == 'hybrid'].copy()
df_electric=df.loc[df['Powertrain'] == 'electric'].copy()


# Compute fuel rate for "Conventional", and "hybrid" buses
df_conventional['FuelRate(L/s)']=fuelRate(factor=1)
df_hybrid['FuelRate(L/s)']=fuelRate(factor=0.85)


#megre subset dataframes 
df_final=pd.concat([df_conventional, df_hybrid])

# Sort dataframe, remove mistakenly computed fuel rates
df_final.sort_values(by=['Vehicle','ServiceDateTime'], ascending=True, inplace=True)

# remove mistakenly computed fuel rates
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
### This resulted an error in the script so I had to run this part in Jupyter "modify-fuel-rates"


        
# Save df_final
df_final.to_csv(r'../../results/fuel-rates-CDB-CHB.csv')


