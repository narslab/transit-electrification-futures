import yaml
import pandas as pd
import numpy as np
import gc   # For manual garbage collection

f = open('params-oct2021-sep2022-test10222023.yaml')
parameters = yaml.safe_load(f)
f.close()

class vehicleParams():
	def __init__(self, **entries):
		self.__dict__.update(entries)

# Read only required columns for computation from trajectories df. Change this to the actual column names you use.
#required_columns = ["speed", "acc", "VehiclWeight(lb)", "Powertrain", "time_delta_in_seconds", "Vehicle", "ServiceDateTime", "Onboard", "grade"]
df = pd.read_csv(r'../../data/tidy/large/trajectories-mapped-powertrain-weight-grade-oct2021-sep2022.csv', delimiter=',', skiprows=0, low_memory=False)


df.speed = df.speed *1.60934 # takes speed in km/h (Convert from mph to km/h)
df.rename(columns={"speed": "Speed", "acc": "Acceleration", "VehiclWeight(lb)": "Vehicle_mass"}, inplace=True)
df = df.fillna(0)


# Subsetting data frame for "Conventional", "hybrid", and "electric" buses
df_conventional=df.loc[df['Powertrain'] == 'conventional'].copy()
df_hybrid=df.loc[df['Powertrain'] == 'hybrid'].copy()
df_electric=df.loc[df['Powertrain'] == 'electric'].copy()
print('Done with reading dataframes for CDB, HEB, and BEB')

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
eta_d_heb = p.driveline_efficiency_d_beb
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
a2_cdb = p.alpha_2_cdb
a0_heb = p.alpha_0_heb
a1_heb = p.alpha_1_heb
a2_heb = p.alpha_2_heb
b=p.beta
gamma_heb=p.gamma
gamma_beb=p.gamma
a2_heb = p.alpha_2_heb

# Define power function for diesel vehicle
def power_d(df_input, hybrid=False):
    if hybrid == True:
       A_f_d=A_f_heb
    else:
       A_f_d=A_f_cdb        
    df = df_input
    v = df.Speed
    a = df.Acceleration
    gr = df.grade
    m = (df.Vehicle_mass+df.Onboard*179)*0.453592 # converts lb to kg
    H = df.elevation/1000 # df.elevation is in meters and we need to convert it to km 
    #C_h = 1 - (0.085*H)
    P_t = (1/float(3600*eta_d_dis))*((1./25.92)*rho*C_D*C_h*A_f_d*v*v + m*g*C_r*(c1*v + c2)/1000 + 1.1*m*a+m*g*gr)*v
    P_t = factor*(eta_batt/(eta_m*eta_d_beb))*(1/float(3600*eta_d_beb))*((1./25.92)*rho*C_D*C_h*A_f_beb*v*v + m*g*C_r*(c1*v + c2)/1000 + 1.2*m*a+m*g*gr)*v

    return P_t

# Define fuel rate function for diesel vehicle
def fuelRate_d(df_input, hybrid=False):
	# Estimates fuel consumed (liters per second) 
    if hybrid == True:
        a0 = a0_heb 
        a1 = a1_heb 
        a2 = a2_heb 
        P_t = power_d(df_input, hybrid=True)
        FC_t = P_t.apply(lambda x: a0 + a1*x +a2*x*x if x >= 0 else a0)
        FC_t=FC_t*b
    else:
        a0 = a0_cdb
        a1 = a1_cdb
        a2 = a2_cdb
        P_t = power_d(df_input, hybrid=False)
        FC_t = P_t.apply(lambda x: a0 + a1*x +a2*x*x if x >= 0 else a0)  
    return FC_t


# Define Energy consumption function for diesel vehicle
def energyConsumption_d(df_input, hybrid=False):
	# Estimates energy consumed (gallons)     
    df = df_input
    t = df.time_delta_in_seconds
    if hybrid == True:
        FC_t = fuelRate_d(df_input, hybrid=True)
    else:
        FC_t = fuelRate_d(df_input, hybrid=False)
    E_t = (FC_t * t)/3.78541
    return E_t


# Define power function for electric vehicle
def power_e(df_input):
    df = df_input
    v = df.Speed
    a = df.Acceleration
    gr = df.grade
    m = df.Vehicle_mass+(df.Onboard*179)*0.453592 # converts lb to kg
    H = df.elevation/1000 # df.elevation is in meters and we need to convert it to km 
    #C_h = 1 - (0.085*H)
    factor = df.Acceleration.apply(lambda a: 1 if a >= 0 else np.exp(-(gamma_beb/abs(a))))
    P_t = factor*(eta_batt/(eta_m*eta_d_beb))*(1/float(3600*eta_d_beb))*((1./25.92)*rho*C_D*C_h*A_f_beb*v*v + m*g*C_r*(c1*v + c2)/1000 + 1.2*m*a+m*g*gr)*v
    return P_t


# Define Energy consumption function for electric vehicle
def energyConsumption_e(df_input):
	# Estimates energy consumed (KWh)     
    df = df_input
    t = df.time_delta_in_seconds/3600
    P_t = power_e(df_input)
    E_t = P_t * t
    return E_t

# Compute energy consumption for "Conventional", "hybrid" and "electric" buses
df_conventional['Energy']=energyConsumption_d(df_conventional)
print('Done calculating energy for CDB')
df_hybrid['Energy']=energyConsumption_d(df_hybrid, hybrid=True)
print('Done calculating energy for HEB')
df_electric['Energy']=energyConsumption_e(df_electric)
print('Done calculating energy for BEB')


#megre subset dataframes 
df_final=pd.concat([df_conventional, df_hybrid, df_electric])
print('Done concatinating')

# Delete the no longer needed dataframes and perform garbage collection
del df_conventional, df_hybrid, df_electric
gc.collect()

# Sort dataframe
df_final.sort_values(by=['Vehicle','ServiceDateTime'], ascending=True, inplace=True)
print('Done sorting')

#df_final.to_csv(r'../../results/computed-fuel-rates-oct2021-sep2022-test-10222023.csv')
df_final.to_csv(r'../../results/computed-fuel-rates-oct2021-sep2022-test-10222023-v02.csv')


