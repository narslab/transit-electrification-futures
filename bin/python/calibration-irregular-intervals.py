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
from sklearn.metrics import mean_squared_error
import time


#import model


f = open('params.yaml')
parameters = yaml.safe_load(f)
f.close()

class vehicleParams():
    def __init__(self, **entries):
        self.__dict__.update(entries)
    
    def modify_parameter(self, parameter, new_value):
        self.parameter=new_value
        return (self.parameter)
    

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
gamma=p.gamma

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
def fuelRate_d(df_input, hybrid=False):
	# Estimates fuel consumed (liters per second)
    if hybrid == True:
        a0 = a0_heb
        #print(a0)
        a1 = a1_heb        
        P_t = power_d(df_input, hybrid=True)
        FC_t = P_t.apply(lambda x: a0 + a1*x +a2*x*x if x >= 0 else a0)
        FC_t=FC_t*b
    else:
        a0 = a0_cdb
        #print(a0)
        a1 = a1_cdb
        P_t = power_d(df_input, hybrid=False)
        FC_t = P_t.apply(lambda x: a0 + a1*x +a2*x*x if x >= 0 else a0)  
    return FC_t


# Define Energy consumption function for electric vehicle
def energyConsumption_d(df_input,  hybrid=False):
	# Estimates energy consumed (gallons)
    df = df_input
    t = df.time_delta_in_seconds
    FC_t = fuelRate_d(df_input , hybrid)
    E_t = (FC_t * t)/3.78541
    return E_t


# Define power function for electric vehicle
def power_e(df_input):
    df = df_input
    v = df.speed
    a = df.acc
    gr = df.grade
    m = df.Vehicle_mass+(df.Onboard*179)*0.453592 # converts lb to kg
    factor = df.acc.apply(lambda a: 1 if a >= 0 else np.exp(-(gamma/abs(a))))
    P_t = factor*(eta_batt/eta_m*eta_d_beb)*(1/float(3600*eta_d_beb))*((1./25.92)*rho*C_D*C_h*A_f_beb*v*v + m*g*C_r*(c1*v + c2)/1000 + 1.2*m*a+m*g*gr)*v
    return P_t


# Define Energy consumption function for electric vehicle
def energyConsumption_e(df_input):
	# Estimates energy consumed (KWh)     
    df = df_input
    t = df.time_delta_in_seconds/3600
    P_t = power_e(df_input)
    E_t = P_t * t
    return E_t



# Read trajectories df
df_trajectories = pd.read_csv(r'../../results/trajectories-mapped-powertrain-weight-grade.csv', delimiter=',', skiprows=0, low_memory=False)
#df_trajectories.rename(columns={"acc": "Acceleration", "VehiclWeight(lb)": "Vehicle_mass"}, inplace=True)
df_trajectories.rename(columns={"VehiclWeight(lb)": "Vehicle_mass"}, inplace=True)
df_trajectories['Date']=pd.to_datetime(df_trajectories['Date'])
df_trajectories.speed = df_trajectories.speed *1.60934 # takes speed in km/h (Convert from mph to km/h)
df_trajectories = df_trajectories.fillna(0)

# Subsetting data frame for "Conventional", "hybrid", and "electric" buses
df_conventional=df_trajectories.loc[df_trajectories['Powertrain'] == 'conventional'].copy()
df_hybrid=df_trajectories.loc[df_trajectories['Powertrain'] == 'hybrid'].copy()



# read validation df
df_validation = pd.read_csv(r'../../data/tidy/energy-validation-april2022-31march.csv', delimiter=',', skiprows=0, low_memory=False)
df_validation=df_validation.loc[df_validation['gallons']>0]
df_validation.sort_values(by=['equipment_id','timestamp'], inplace=True)
df_validation.drop(['Unnamed: 0'], axis=1, inplace=True)
df_validation.rename(columns={"timestamp": "ServiceDateTime","equipment_id":"Vehicle"}, inplace=True)
df_validation["dist"] = np.nan
df_validation["Energy"] = np.nan
#df_validation["VehicleModel"] = np.nan
df_validation.sort_values(by=['Vehicle','ServiceDateTime'], inplace=True)
df_validation['ServiceDateTime'] = pd.to_datetime(df_validation['ServiceDateTime'])
df_validation=df_validation.reset_index()


### calibrating model
def calibrate_parameter(start1, stop1, start2, stop2, hybrid=False):
    start_time = time.time()
    parameter1_values=[]
    parameter2_values=[]
    RMSE_Energy=[]
    MAPE_Energy=[]
    RMSE_Economy=[]
    MAPE_Economy=[]
    if hybrid==True:
        df=df_hybrid
        validation=df_validation
        #parameter1=a0_heb
        #parameter2=a1_heb
        count=0
        for a0 in np.linspace(start1, stop1, 100):
            for a1 in np.linspace(start2, stop2, 100):
                count+=1
                print(count)
                global a0_heb
                a0_heb=a0
                global a1_heb
                a1_heb=a1
                #global b
                #b=j
                df_new=df.copy()
                validation_new=validation.copy()
                df_new['Energy']=energyConsumption_d(df, hybrid=True)
                df_new.sort_values(by=['Vehicle','ServiceDateTime'], inplace=True)
                df_new['ServiceDateTime']=pd.to_datetime(df_new['ServiceDateTime'])
                df_integrated_hybrid = validation_new[(validation_new.Powertrain == 'hybrid')].copy()
                df_integrated_hybrid.sort_values(by=['Vehicle','ServiceDateTime'], inplace=True)
                df_integrated_hybrid=df_integrated_hybrid.reset_index()
                for i in df_integrated_hybrid.index:
                    if i==0:
                        pass 
                    else:
                        if df_integrated_hybrid['Vehicle'][i]==df_integrated_hybrid['Vehicle'][i-1]:
                            df_filtered=df_new.loc[(df_new['Vehicle']==df_integrated_hybrid['Vehicle'][i])&(df_integrated_hybrid['ServiceDateTime'][i-1]<df_new['ServiceDateTime'])&(df_new['ServiceDateTime']<df_integrated_hybrid['ServiceDateTime'][i])]
                            df_integrated_hybrid.loc[i,'dist']=df_filtered['dist'].sum()
                            df_integrated_hybrid.loc[i,'Energy']=df_filtered['Energy'].sum()
                            #df_integrated_hybrid['VehicleModel'][i]=df_filtered['VehicleModel'].mode()
                        else:
                            pass
                df_integrated_hybrid['Energy'].fillna(0, inplace=True)
                df_integrated_hybrid=df_integrated_hybrid[df_integrated_hybrid['Energy']!=0]
                df_integrated_hybrid['Fuel_economy']=df_integrated_hybrid['dist']/df_integrated_hybrid['Energy']
                df_integrated_hybrid['Real_Fuel_economy']=df_integrated_hybrid['dist']/df_integrated_hybrid['gallons']
                df_integrated_hybrid['Energy'] = df_integrated_hybrid['Energy'].astype(float)
                df_integrated_hybrid['dist'] = df_integrated_hybrid['dist'].astype(float)
                train_dates=['2022-04-01','2022-04-02','2022-04-03','2022-04-04','2022-04-05','2022-04-06','2022-04-07','2022-04-08','2022-04-09','2022-04-10','2022-04-11','2022-04-12','2022-04-13', '2022-04-14','2022-04-15','2022-04-16','2022-04-17','2022-04-18','2022-04-19','2022-04-20','2022-04-21','2022-04-22','2022-04-23']
                train = df_integrated_hybrid[df_integrated_hybrid.date.isin(train_dates)]
                MSE_Energy_current = mean_squared_error(train['gallons'], train['Energy'])
                RMSE_Energy_current = math.sqrt(MSE_Energy_current)
                MAPE_Energy_current = np.mean(np.abs((train['gallons'] - train['Energy']) / train['gallons'])) * 100
                RMSE_Economy_current = mean_squared_error(train['Real_Fuel_economy'], train['Fuel_economy'], squared=False)
                MAPE_Economy_current = np.mean(np.abs((train['Real_Fuel_economy'] - train['Fuel_economy']) / train['Real_Fuel_economy'])) * 100
                parameter1_values.append(a0)
                parameter2_values.append(a1)
                RMSE_Energy.append(RMSE_Energy_current)
                MAPE_Energy.append(MAPE_Energy_current)
                RMSE_Economy.append(RMSE_Economy_current)
                MAPE_Economy.append(MAPE_Economy_current)
        else:
            pass
    else:
        df=df_conventional
        validation=df_validation
        #parameter1=a0_cdb
        #parameter2=a1_cdb
        count=0
        for a0 in np.linspace(start1, stop1, 100):
            for a1 in np.linspace(start2, stop2, 100):
                count+=1
                print(count)
                global a0_cdb
                a0_cdb=a0
                global a1_cdb
                a1_cdb=a1
                df_new=df.copy()
                validation_new=validation.copy()
                df_new['Energy']=energyConsumption_d(df, hybrid=False)
                df_new.sort_values(by=['Vehicle','ServiceDateTime'], inplace=True)
                df_new['ServiceDateTime']=pd.to_datetime(df_new['ServiceDateTime'])
                df_integrated_conventional = validation_new[(validation_new.Powertrain == 'conventional')].copy()
                df_integrated_conventional.sort_values(by=['Vehicle','ServiceDateTime'], inplace=True)
                df_integrated_conventional=df_integrated_conventional.reset_index()
                for i in df_integrated_conventional.index:
                    if i==0:
                        pass 
                    else:
                        if df_integrated_conventional['Vehicle'][i]==df_integrated_conventional['Vehicle'][i-1]:
                            df_filtered=df_new.loc[(df_new['Vehicle']==df_integrated_conventional['Vehicle'][i])&(df_integrated_conventional['ServiceDateTime'][i-1]<df_new['ServiceDateTime'])&(df_new['ServiceDateTime']<df_integrated_conventional['ServiceDateTime'][i])]
                            df_integrated_conventional.loc[i,'dist']=df_filtered['dist'].sum()
                            df_integrated_conventional.loc[i,'Energy']=df_filtered['Energy'].sum()
                            #df_integrated_conventional['VehicleModel'][i]=df_filtered['VehicleModel'].mode()
                        else:
                            pass
                df_integrated_conventional['Energy'].fillna(0, inplace=True)
                df_integrated_conventional=df_integrated_conventional[df_integrated_conventional['Energy']!=0]
                df_integrated_conventional['Fuel_economy']=df_integrated_conventional['dist']/df_integrated_conventional['Energy']
                df_integrated_conventional['Real_Fuel_economy']=df_integrated_conventional['dist']/df_integrated_conventional['gallons']
                df_integrated_conventional['Energy'] = df_integrated_conventional['Energy'].astype(float)
                df_integrated_conventional['dist'] = df_integrated_conventional['dist'].astype(float)
                train_dates=['2022-04-01','2022-04-02','2022-04-03','2022-04-04','2022-04-05','2022-04-06','2022-04-07','2022-04-08','2022-04-09','2022-04-10','2022-04-11','2022-04-12','2022-04-13', '2022-04-14','2022-04-15','2022-04-16','2022-04-17','2022-04-18','2022-04-19','2022-04-20','2022-04-21','2022-04-22','2022-04-23']
                train = df_integrated_conventional[df_integrated_conventional.date.isin(train_dates)]
                MSE_Energy_current = mean_squared_error(train['gallons'], train['Energy'])
                RMSE_Energy_current = math.sqrt(MSE_Energy_current)
                MAPE_Energy_current = np.mean(np.abs((train['gallons'] - train['Energy']) / train['gallons'])) * 100
                RMSE_Economy_current = mean_squared_error(train['Real_Fuel_economy'], train['Fuel_economy'], squared=False)
                MAPE_Economy_current = np.mean(np.abs((train['Real_Fuel_economy'] - train['Fuel_economy']) / train['Real_Fuel_economy'])) * 100
                parameter1_values.append(a0)
                parameter2_values.append(a1)
                RMSE_Energy.append(RMSE_Energy_current)
                MAPE_Energy.append(MAPE_Energy_current)
                RMSE_Economy.append(RMSE_Economy_current)
                MAPE_Economy.append(MAPE_Economy_current)
    results = pd.DataFrame(list(zip(parameter1_values, parameter2_values, RMSE_Energy, MAPE_Energy, RMSE_Economy, MAPE_Economy)),
               columns =['parameter1_values', 'parameter2_values','RMSE_Energy','MAPE_Energy', 'RMSE_Economy', 'MAPE_Economy'])
    results.to_csv(r'../../results/calibration-irregular-intervals-results.csv')
    print("--- %s seconds ---" % (time.time() - start_time))




#hybrid
#calibrate_parameter(0.000008, 0.0168, 0.0000011, 0.000411, hybrid=True)
#calibrate_parameter(0.001195, 0.001195, 0.5 , 0.95 , hybrid=True)
#calibrate_parameter(0.0009, 0.002, 0.00003, 0.0001, hybrid=True)
##test
#calibrate_parameter(0.0001, 0.0001, 0.00001, 0.00001, hybrid=True)


#conventional
#calibrate_parameter(0.0001, 0.01, 0.000001, 0.0001, hybrid=False)
calibrate_parameter(0.0005, 0.0030, 0.00001, 0.0002, hybrid=False)
##test
#calibrate_parameter(0.0001, 0.0001, 0.000001, 0.000001, hybrid=False)



