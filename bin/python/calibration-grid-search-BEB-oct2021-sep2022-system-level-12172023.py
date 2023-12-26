import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split

f = open('params-oct2021-sep2022-test10222023.yaml')
parameters = yaml.safe_load(f)
f.close()

class vehicleParams():
	def __init__(self, **entries):
		self.__dict__.update(entries)

# Define model parameters
p = vehicleParams(**parameters)
rho = p.air_density
C_D = p.drag_coefficient
#C_h= p.altitude_correction_factor
A_f_cdb = p.frontal_area_cdb
A_f_heb = p.frontal_area_heb
A_f_beb = p.frontal_area_beb
g = p.gravitational_acceleration
C_r = p.rolling_coefficient
c1 = p.rolling_resistance_coef1
c2 = p.rolling_resistance_coef2
eta_d_dis = p.driveline_efficiency_d_dis
#eta_d_beb = p.driveline_efficiency_d_beb
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
#gamma=0.0411

# Define power function for electric vehicle
def power_e(df_input, gamma, driveline_efficiency):
    df = df_input
    v = df.speed
    a = df.acc
    gr = df.grade
    m = df.Vehicle_mass+(df.Onboard*179)*0.453592 # converts lb to kg
    #eta_d_beb = driveline_efficiency
    H = df.elevation/1000 # df.elevation is in meters and we need to convert it to km 
    C_h = 1 - (0.085*H)
    factor = df.acc.apply(lambda a: 1 if a >= 0 else np.exp(-(gamma/abs(a))))
    #P_t = factor*(eta_batt/eta_m*eta_d_beb)*(1/float(3600*eta_d_beb))*((1./25.92)*rho*C_D*C_h*A_f_beb*v*v + m*g*C_r*(c1*v + c2)/1000 + 1.1*m*a+m*g*gr)*v
    P_t = factor*(eta_batt/eta_m*driveline_efficiency)*(1/float(3600*driveline_efficiency))*((1./25.92)*rho*C_D*C_h*A_f_beb*v*v + m*g*C_r*(c1*v + c2)/1000 + 1.1*m*a+m*g*gr)*v
    return P_t


# Define Energy consumption function for electric vehicle
def energyConsumption_e(df_input, gamma, driveline_efficiency):
	# Estimates energy consumed (KWh)     
    df = df_input
    t = df.time_delta_in_seconds/3600
    P_t = power_e(df_input, gamma, driveline_efficiency)
    E_t = P_t * t
    return E_t

# Read computed fuel rates
df_trajectories = pd.read_csv(r'../../data/tidy/large/trajectories-mapped-powertrain-weight-grade-oct2021-sep2022.csv', delimiter=',', skiprows=0, low_memory=False)
df_trajectories.rename(columns={"VehiclWeight(lb)": "Vehicle_mass"}, inplace=True)
df_trajectories['Date']=pd.to_datetime(df_trajectories['Date'])
df_trajectories.speed = df_trajectories.speed *1.60934 # takes speed in km/h (Convert from mph to km/h)
df_trajectories = df_trajectories.fillna(0)

# Subsetting data frame for "Conventional", "hybrid", and "electric" buses
df_beb=df_trajectories.loc[df_trajectories['Powertrain'] == 'electric'].copy()
del df_trajectories

# read validation df
df_validation = pd.read_excel(r'../../data/tidy/BEB-validation.xlsx')
df_validation.rename(columns={"Transaction Date": "ServiceDateTime","Equipment ID":"Vehicle"}, inplace=True)
df_validation["dist"] = np.nan
df_validation["Energy"] = np.nan
df_validation['ServiceDateTime'] = pd.to_datetime(df_validation['ServiceDateTime'])
df_validation.sort_values(by=['Vehicle','ServiceDateTime'], inplace=True)


### Map powertrain in the validation dataset
df2 = pd.read_csv(r'../../data/tidy/vehicles-summary.csv', delimiter=',', skiprows=0, low_memory=False)
mydict = df2.groupby('Type')['Equipment ID'].agg(list).to_dict()
d = {val:key for key, lst in mydict.items() for val in lst}
df_validation['Powertrain'] = df_validation['Vehicle'].map(d)


def process_dataframe(df, validation, gamma, driveline_efficiency):
    df_new = df.copy()
    validation_new = validation.copy()

    df_new['Energy'] = energyConsumption_e(df, gamma, driveline_efficiency)
    df_new['ServiceDateTime'] = pd.to_datetime(df_new['ServiceDateTime'])
    df_new = df_new.groupby(['Date', 'Vehicle'])[['Energy', 'dist']].sum().reset_index()

    # Select only 'ServiceDateTime', 'Vehicle', and 'trip' columns from validation_new
    validation_subset = validation_new[['ServiceDateTime', 'Vehicle', 'trip']]

    # Merge df_new with the subset of validation_new
    df_integrated = pd.merge(validation_subset, df_new, 
                             left_on=['ServiceDateTime', 'Vehicle'], 
                             right_on=['Date', 'Vehicle'], 
                             how='left').copy()
    #df_integrated['residual']=df_integrated['trip']-df_integrated['Energy']
    df_integrated = df_integrated.dropna(subset=['trip', 'Energy'])

    # Drop rows where 'trip' or 'Predicted Energy' is 0
    df_integrated = df_integrated.query("trip != 0 and `Energy` != 0")
    
    
    df_integrated['Fuel_economy'] = np.divide(df_integrated['dist'], df_integrated['Energy'], where=df_integrated['Energy'] != 0)
    df_integrated['Real_Fuel_economy'] = np.divide(df_integrated['dist'], df_integrated['trip'], where=df_integrated['trip'] != 0)
    
    return df_integrated

def calibrate_parameter(args):
    start_1, stop_1, n_points_1, start_2, stop_2, n_points_2 = args
    start_time = time.time()
    parameter1_values = []
    parameter2_values = []
    RMSE_Energy_train = []
    MAPE_Energy_train = []
    RMSE_Energy_test = []
    MAPE_Energy_test = []

    df = df_beb.copy()
    validation = df_validation.copy()
    validation.reset_index(inplace=True)        
    decimal_places = 8  # Set the desired number of decimal places
    gamma_values = np.around(np.linspace(start_1, stop_1, n_points_1), decimals=decimal_places)
    driveline_efficiencies = np.linspace(start_2, stop_2, n_points_2)

    for gamma in tqdm(gamma_values, desc="Processing gamma values"):
        for driveline_efficiency in tqdm(driveline_efficiencies, desc="Processing driveline_efficiencies values"):
            df_integrated = process_dataframe(df, validation, gamma, driveline_efficiency)
            df_integrated.dropna(subset=['trip', 'Energy'], inplace=True)
            df_integrated = df_integrated.loc[df_integrated['Energy']!=0]
            df_integrated['Transaction Date'] = df_integrated['ServiceDateTime'].dt.date
            df_integrated = df_integrated.groupby('Transaction Date')[['dist','trip', 'Energy']].sum().reset_index()       
            df_train, df_test = train_test_split(df_integrated, test_size=0.2, random_state=42)
            df_integrated=df_integrated.reset_index()
            RMSE_Energy_train_current = np.sqrt(mean_squared_error(df_train['trip'], df_train['Energy']))
            MAPE_Energy_train_current = mean_absolute_percentage_error(df_train['trip'] , df_train['Energy'])
            RMSE_Energy_test_current = np.sqrt(mean_squared_error(df_test['trip'], df_test['Energy']))
            MAPE_Energy_test_current = mean_absolute_percentage_error(df_test['trip'] , df_test['Energy'])
            RMSE_Energy_train.append(RMSE_Energy_train_current)
            MAPE_Energy_train.append(MAPE_Energy_train_current)
            RMSE_Energy_test.append(RMSE_Energy_test_current)
            MAPE_Energy_test.append(MAPE_Energy_test_current)
            # Append parameter values
            parameter1_values.append(gamma)
            parameter2_values.append(driveline_efficiency)
            print("driveline_efficiency:",driveline_efficiency)
            print("MAPE_Energy_test:", MAPE_Energy_test_current)



    results = pd.DataFrame(list(zip(parameter1_values, parameter2_values, RMSE_Energy_train, MAPE_Energy_train, RMSE_Energy_test, MAPE_Energy_test)),
                           columns=['parameter1_values', 'parameter2_values', 'RMSE_Energy_train', 'MAPE_Energy_train', 'RMSE_Energy_test', 'MAPE_Energy_test'])
    results.to_csv((r'../../results/calibration-grid-search-BEB-oct2021-sep2022_12172023.csv'))
    print("--- %s seconds ---" % (time.time() - start_time))

    
calibrate_parameter((0.00000001, 0.6, 10, 0.8, 0.99,5))