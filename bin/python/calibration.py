# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:39:25 2022

@author: Mahsa
"""

import yaml
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
import energy-model



f = open('params.yaml')
parameters = yaml.safe_load(f)
f.close()

class vehicleParams():
    def __init__(self, **entries):
        self.__dict__.update(entries)
    
    def modify_parameter(self, parameter, new_value):
        self.parameter=new_value
        return (self.parameter)
    

# Read trajectories df
df = pd.read_csv(r'../../results/trajectories-mapped-powertrain-weight.csv', delimiter=',', skiprows=0, low_memory=False)
df.speed = df.speed *1.60934 # takes speed in km/h (Convert from mph to km/h)
df.rename(columns={"speed": "Speed", "acc": "Acceleration", "VehiclWeight(lb)": "Vehicle_mass"}, inplace=True)
df = df.fillna(0)


# Subsetting data frame for "Conventional", "hybrid", and "electric" buses
#df_conventional=df.loc[df['Powertrain'] == 'conventional'].copy()
#df_hybrid=df.loc[df['Powertrain'] == 'hybrid'].copy()
#df_electric=df.loc[df['Powertrain'] == 'electric'].copy()
#df_conventional['Date']=pd.to_datetime(df_conventional['Date'])
#df_hybrid['Date']=pd.to_datetime(df_hybrid['Date'])
#df_electric['Date']=pd.to_datetime(df_electric['Date'])


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
b=p.beta

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
        FC_t=FC_t*b
    else:
        a0 = a0_cdb
        a1 = a1_cdb
        P_t = power_d(df_input)
        FC_t = P_t.apply(lambda x: a0 + a1*x +a2*x*x if x >= 0 else a0)  
    return FC_t


# Define Energy consumption function for electric vehicle
def energyConsumption_d(df_input, hybrid=False):
	# Estimates energy consumed (gallons)     
    df = df_input
    t = df.time_delta_in_seconds
    FC_t = fuelRate_d(df_input, hybrid)
    E_t = FC_t * t/3.78541
    return E_t


# Define power function for electric vehicle
def power_e(df_input):
    df = df_input
    v = df.Speed
    a = df.Acceleration
    m = df.Vehicle_mass+(df.Onboard*179)
    factor = df.Acceleration.apply(lambda a: 1 if a >= 0 else np.exp(-(0.0411/abs(a))))
    P_t = factor*(eta_batt/eta_m*eta_d_beb)*(1/float(3600*eta_d_beb))*((1./25.92)*rho*C_D*A_f*v*v + m*g*C_r*(c1*v + c2)/1000 + 1.1*m*a)*v
    return P_t


# Define Energy consumption function for electric vehicle
def energyConsumption_e(df_input):
	# Estimates energy consumed (KWh)     
    df = df_input
    t = df.time_delta_in_seconds/3600
    P_t = power_e(df_input)
    E_t = P_t * t
    return E_t


# read validation df
df_validation = pd.read_csv(r'../../data/tidy/energy_validation_april2022.csv', delimiter=',', skiprows=0, low_memory=False)
df_validation['date']=pd.to_datetime(df_validation['date'])
df_validation.rename(
    columns={"equipment_id":"Vehicle",
                "date":"Date",
                "gallons":"Real_Energy"}
          ,inplace=True)
df_validation=df_validation[['Vehicle','Date','Real_Energy']]


### calibrating hybrid model for parameter b
b_values=[0.5, 0.6, 0.7, 0.8, 0.9]
RMSE=[]
for i in b_values:
    b=i
    df_hybrid=df.loc[df['Powertrain'] == 'hybrid'].copy()
    df_hybrid['Date']=pd.to_datetime(df_hybrid['Date'])
    df_hybrid['Energy']=energyConsumption_d(df_hybrid, hybrid=True)
    train_hybrid, test_hybrid = train_test_split(df_hybrid, test_size=0.2, random_state=(42))    
    train_hybrid = train_hybrid.groupby(['Vehicle', 'Date']).agg({'Energy': ['sum'] ,'Powertrain': ['max'], 'dist': ['sum']}).reset_index()
    train_hybrid.columns = train_hybrid.columns.droplevel()
    train_hybrid.columns =['Vehicle', 'Date', 'Energy', 'Powertrain', 'Distance']
    cols = ['Vehicle', 'Date']
    df_integrated_hybrid=train_hybrid.join(df_validation.set_index(cols), on=cols)
    df_integrated_hybrid['Fuel/energy_economy']=df_integrated_hybrid['Distance']/df_integrated_hybrid['Energy']
    df_integrated_hybrid['Real_Fuel/energy_economy']=df_integrated_hybrid['Distance']/df_integrated_hybrid['Real_Energy']
    df_integrated_hybrid=df_integrated_hybrid.dropna()
    df_integrated_hybrid = df_integrated_hybrid.reset_index()
    MSE = np.square(np.subtract(df_integrated_hybrid['Real_Fuel/energy_economy'],df_integrated_hybrid['Fuel/energy_economy'])).mean() 
    rmse_hybrid = math.sqrt(MSE)
    #rmse_hybrid = mean_squared_error(df_integrated_hybrid['Real_Fuel/energy_economy'], df_integrated_hybrid['Fuel/energy_economy'], squared=False)
    RMSE.append(rmse_hybrid)
    print(rmse_hybrid)
print(RMSE)

# Get RMSE on test set
b = 0.9
b=i
df_hybrid=df.loc[df['Powertrain'] == 'hybrid'].copy()
df_hybrid['Date']=pd.to_datetime(df_hybrid['Date'])
df_hybrid['Energy']=energyConsumption_d(df_hybrid, hybrid=True)
train_hybrid, test_hybrid = train_test_split(df_hybrid, test_size=0.2, random_state=(42))    
test_hybrid = test_hybrid.groupby(['Vehicle', 'Date']).agg({'Energy': ['sum'] ,'Powertrain': ['max'], 'dist': ['sum']}).reset_index()
test_hybrid.columns = test_hybrid.columns.droplevel()
test_hybrid.columns =['Vehicle', 'Date', 'Energy', 'Powertrain', 'Distance']
cols = ['Vehicle', 'Date']
df_integrated_hybrid=test_hybrid.join(df_validation.set_index(cols), on=cols)
df_integrated_hybrid['Fuel/energy_economy']=df_integrated_hybrid['Distance']/df_integrated_hybrid['Energy']
df_integrated_hybrid['Real_Fuel/energy_economy']=df_integrated_hybrid['Distance']/df_integrated_hybrid['Real_Energy']
df_integrated_hybrid=df_integrated_hybrid.dropna()
df_integrated_hybrid = df_integrated_hybrid.reset_index()
MSE = np.square(np.subtract(df_integrated_hybrid['Real_Fuel/energy_economy'],df_integrated_hybrid['Fuel/energy_economy'])).mean() 
rmse_hybrid = math.sqrt(MSE)
print('RMSE on the test set for b=.9:', rmse_hybrid)

# Save df_final
#df_final.to_csv(r'../../results/computed-fuel-rates.csv')





