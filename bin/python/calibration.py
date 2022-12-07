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
        print(a0)
        a1 = a1_heb        
        P_t = power_d(df_input, hybrid=True)
        FC_t = P_t.apply(lambda x: a0 + a1*x +a2*x*x if x >= 0 else a0)
        FC_t=FC_t*b
    else:
        a0 = a0_cdb
        print(a0)
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
df_electric=df_trajectories.loc[df_trajectories['Powertrain'] == 'electric'].copy()



# read validation df
df_validation = pd.read_csv(r'../../data/tidy/energy_validation_april2022.csv', delimiter=',', skiprows=0, low_memory=False)
df_validation['date']=pd.to_datetime(df_validation['date'])
df_validation = df_validation.groupby(['equipment_id', 'date']).agg({'gallons': ['sum']}).reset_index()
df_validation.columns = df_validation.columns.droplevel()
df_validation.columns =['Vehicle', 'Date', 'Real_Energy']
#df_validation.rename(
#    columns={"equipment_id":"Vehicle",
#                "date":"Date",
#                "gallons":"Real_Energy"}
#          ,inplace=True)
#df_validation=df_validation[['Vehicle','Date','Real_Energy']]
df_validation['Date']=pd.to_datetime(df_validation['Date'])


### calibrating hybrid model for parameter b
def calibrate_parameter(start1, stop1, start2, stop2, hybrid=False, electric=False):
    start_time = time.time()
    parameter1_values=[]
    parameter2_values=[]
    RMSE_Economy=[]
    RMSE_Energy=[]
    MAPE_Energy=[]
    if hybrid==True:
        df=df_hybrid
        #parameter1=a0_heb
        #parameter2=a1_heb
        for i in np.linspace(start1, stop1, 100):
            for j in np.linspace(start2, stop2, 100):
                global a0_heb
                a0_heb=i
                global a1_heb
                a1_heb=j
                #global b
                #b=j
                df_new=df.copy()
                df_new['Energy']=energyConsumption_d(df, hybrid=True)
                df_grouped = df_new.groupby(['Vehicle', 'Date']).agg({'Energy': ['sum'] ,'Powertrain': ['max'], 'dist': ['sum']}).reset_index()
                df_grouped.columns = df_grouped.columns.droplevel()
                df_grouped.columns =['Vehicle', 'Date', 'Energy', 'Powertrain', 'Distance']
                cols = ['Vehicle', 'Date']
                df_integrated=df_grouped.join(df_validation.set_index(cols), on=cols)
                df_integrated['Fuel/energy_economy']=df_integrated['Distance']/df_integrated['Energy']
                df_integrated['Real_Fuel/energy_economy']=df_integrated['Distance']/df_integrated['Real_Energy']
                df_integrated=df_integrated.dropna()
                df_integrated = df_integrated.reset_index()
                #train, test = train_test_split(df_integrated, test_size=0.2, random_state=(42))    
                train_dates=['2022-04-01','2022-04-02','2022-04-03','2022-04-04','2022-04-05','2022-04-06','2022-04-07','2022-04-08','2022-04-09','2022-04-10','2022-04-11','2022-04-12','2022-04-13', '2022-04-14','2022-04-15','2022-04-16','2022-04-17','2022-04-18','2022-04-19','2022-04-20','2022-04-21','2022-04-22','2022-04-23']
                train = df_integrated[df_integrated.Date.isin(train_dates)]
                #print('train',train['Date'].unique())
                test_dates=['2022-04-24','2022-04-25','2022-04-26','2022-04-27','2022-04-28','2022-04-29','2022-04-30']
                test = df_integrated[df_integrated.Date.isin(test_dates)]
                #print('test',test['Date'].unique())
                MSE_Economy = np.square(np.subtract(train['Real_Fuel/energy_economy'],train['Fuel/energy_economy'])).mean() 
                RMSE_Economy_current = math.sqrt(MSE_Economy)
                MSE_Energy = np.square(np.subtract(train['Real_Energy'],train['Energy'])).mean() 
                RMSE_Energy_current = math.sqrt(MSE_Energy)
                MAPE_Energy_current = np.mean(np.abs((train['Real_Energy'] - train['Energy']) / train['Real_Energy'])) * 100
                parameter1_values.append(i)
                parameter2_values.append(j)
                RMSE_Economy.append(RMSE_Economy_current)
                RMSE_Energy.append(RMSE_Energy_current)
                MAPE_Energy.append(MAPE_Energy_current)
    else:
        if electric==False:
            df=df_conventional
            #parameter1=a0_cdb
            #parameter2=a1_cdb
            for i in np.linspace(start1, stop1, 50):
                for j in np.linspace(start2, stop2, 50):
                    global a0_cdb
                    a0_cdb=i
                    global a1_cdb
                    a1_cdb=j
                    df_new=df.copy()
                    df_new['Energy']=energyConsumption_d(df, hybrid=False)
                    df_grouped = df_new.groupby(['Vehicle', 'Date']).agg({'Energy': ['sum'] ,'Powertrain': ['max'], 'dist': ['sum']}).reset_index()
                    df_grouped.columns = df_grouped.columns.droplevel()
                    df_grouped.columns =['Vehicle', 'Date', 'Energy', 'Powertrain', 'Distance']
                    cols = ['Vehicle', 'Date']
                    df_integrated=df_grouped.join(df_validation.set_index(cols), on=cols)
                    df_integrated['Fuel/energy_economy']=df_integrated['Distance']/df_integrated['Energy']
                    df_integrated['Real_Fuel/energy_economy']=df_integrated['Distance']/df_integrated['Real_Energy']
                    df_integrated=df_integrated.dropna()
                    df_integrated = df_integrated.reset_index()
                    #train, test = train_test_split(df_integrated, test_size=0.2, random_state=(42))    
                    train_dates=['2022-04-01','2022-04-02','2022-04-03','2022-04-04','2022-04-05','2022-04-06','2022-04-07','2022-04-08','2022-04-09','2022-04-10','2022-04-11','2022-04-12','2022-04-13', '2022-04-14','2022-04-15','2022-04-16','2022-04-17','2022-04-18','2022-04-19','2022-04-20','2022-04-21','2022-04-22','2022-04-23']
                    train = df_integrated[df_integrated.Date.isin(train_dates)]
                    #print('train',train['Date'].unique())
                    test_dates=['2022-04-24','2022-04-25','2022-04-26','2022-04-27','2022-04-28','2022-04-29','2022-04-30']
                    test = df_integrated[df_integrated.Date.isin(test_dates)]
                    #print('test',test['Date'].unique())
                    MSE_Economy = np.square(np.subtract(train['Real_Fuel/energy_economy'],train['Fuel/energy_economy'])).mean() 
                    RMSE_Economy_current = math.sqrt(MSE_Economy)
                    #RMSE_Economy_current = mean_squared_error(train['Real_Fuel/energy_economy'], train['Fuel/energy_economy'], squared=False)
                    MSE_Energy = np.square(np.subtract(train['Real_Energy'],train['Energy'])).mean() 
                    RMSE_Energy_current = math.sqrt(MSE_Energy)
                    MAPE_Energy_current = np.mean(np.abs((train['Real_Energy'] - train['Energy']) / train['Real_Energy'])) * 100
                    parameter1_values.append(i)
                    parameter2_values.append(j)
                    RMSE_Economy.append(RMSE_Economy_current)
                    RMSE_Energy.append(RMSE_Energy_current)
                    MAPE_Energy.append(MAPE_Energy_current)
        else:
            df=df_electric
            #parameter1=gamma
            #parameter2=eta_batt
            for i in np.linspace(start1, stop1, 200):
                for j in np.linspace(start2, stop2, 20):
                    global gamma
                    gamma=i
                    global eta_batt
                    eta_batt=j
                    df_new=df.copy()
                    df_new['Energy']=energyConsumption_e(df)
                    df_grouped = df_new.groupby(['Vehicle', 'Date']).agg({'Energy': ['sum'] ,'Powertrain': ['max'], 'dist': ['sum']}).reset_index()
                    df_grouped.columns = df_grouped.columns.droplevel()
                    df_grouped.columns =['Vehicle', 'Date', 'Energy', 'Powertrain', 'Distance']
                    cols = ['Vehicle', 'Date']
                    df_integrated=df_grouped.join(df_validation.set_index(cols), on=cols)
                    df_integrated['Fuel/energy_economy']=df_integrated['Distance']/df_integrated['Energy']
                    df_integrated['Real_Fuel/energy_economy']=df_integrated['Distance']/df_integrated['Real_Energy']
                    df_integrated=df_integrated.dropna()
                    df_integrated = df_integrated.reset_index()
                    train, test = train_test_split(df_integrated, test_size=0.2, random_state=(42))    
                    MSE_Economy = np.square(np.subtract(0.438,train['Fuel/energy_economy'])).mean() 
                    RMSE_Economy_current = math.sqrt(MSE_Economy)
                    MSE_Energy = 0 
                    RMSE_Energy_current = 0
                    parameter1_values.append(i)
                    parameter2_values.append(j)
                    RMSE_Economy.append(RMSE_Economy_current)
                    RMSE_Energy.append(RMSE_Energy_current)
    results = pd.DataFrame(list(zip(parameter1_values, parameter2_values, RMSE_Economy, RMSE_Energy, MAPE_Energy)),
               columns =['parameter1_values', 'parameter2_values', 'RMSE_Economy','RMSE_Energy','MAPE_Energy'])
    results.to_csv(r'../../results/calibration-results.csv')
    print("--- %s seconds ---" % (time.time() - start_time))
#hybrid
#calibrate_parameter(0.000008, 0.0168, 0.0000011, 0.000411, hybrid=True, electric=False)
#calibrate_parameter(0.001195, 0.001195, 0.5 , 0.95 , hybrid=True, electric=False)
#calibrate_parameter(0.0001, 0.002, 0.00001, 0.0002, hybrid=True, electric=False)


#conventional
calibrate_parameter(0.0001, 0.01, 0.000001, 0.0001, hybrid=False, electric=False)


#electric
#calibrate_parameter(0.0299, 5 , 0.75, 0.95, hybrid=False, electric=True)

