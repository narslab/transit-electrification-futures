import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import time
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

f = open('params-oct2021-sep2022.yaml')
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
gamma_heb=p.gamma

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
    P_t = (1/float(3600*eta_d_dis)) * ((1./25.92)*rho*C_D*C_h*A_f_d*v*v + m*g*C_r*(c1*v + c2)/1000 + 1.2*m*a+m*g*gr) * v
    return P_t


# Define fuel rate function for diesel vehicle
def fuelRate_d(df_input, hybrid=False):
    # Set the coefficients based on the hybrid flag
    if hybrid:
        a0_local = a0_heb
        a1_local = a1_heb
    else:
        a0_local = a0_cdb
        a1_local = a1_cdb

    # Compute power
    P_t = power_d(df_input, hybrid=hybrid)

    # Vectorized computation of fuel rate based on the condition
    condition = P_t >= 0
    FC_t = np.where(condition, a0_local + a1_local*P_t + a2*P_t*P_t, a0_local)

    # Adjust the value if it's a hybrid vehicle
    if hybrid:
        FC_t *= b

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
    E_t = (FC_t * t)/3.78541 #The value 3.78541 represents the number of liters in one US gallon
    return E_t


# Read trajectories df
df_trajectories = pd.read_csv(r'../../data/tidy/large/trajectories-mapped-powertrain-weight-grade-oct2021-sep2022.csv', delimiter=',', skiprows=0, low_memory=False)
df_trajectories.rename(columns={"VehiclWeight(lb)": "Vehicle_mass"}, inplace=True)
df_trajectories['Date'] = pd.to_datetime(df_trajectories['Date'])
df_trajectories.speed = df_trajectories.speed *1.60934  # Convert from mph to km/h
df_trajectories = df_trajectories.fillna(0)

# Subsetting data frame
df_conventional = df_trajectories.loc[df_trajectories['Powertrain'] == 'conventional']
df_hybrid = df_trajectories.loc[df_trajectories['Powertrain'] == 'hybrid']
del df_trajectories

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

# Delete unnecessary dataframes
del df2, mydict

def process_dataframe(df, validation, a0, a1, hybrid):

    df['Energy'] = energyConsumption_d(df, hybrid=hybrid)
    df.sort_values(by=['Vehicle', 'ServiceDateTime'], inplace=True)
    df['ServiceDateTime'] = pd.to_datetime(df['ServiceDateTime'])

    df_integrated = validation.copy()
    df_integrated.sort_values(by=['Vehicle', 'ServiceDateTime'], inplace=True)
    df_integrated['ServiceDateTime_prev'] = df_integrated.groupby('Vehicle')['ServiceDateTime'].shift(1)
    df_integrated = df_integrated.dropna(subset=['ServiceDateTime_prev'])

    # Vectorized operation: 
    mask = (df['Vehicle'].isin(df_integrated['Vehicle'])) & \
           (df['ServiceDateTime'].gt(df_integrated['ServiceDateTime_prev'])) & \
           (df['ServiceDateTime'].lt(df_integrated['ServiceDateTime']))
    
    filtered_data = df.loc[mask].groupby(['Vehicle', 'ServiceDateTime']).agg({
    'dist': 'sum',
    'Energy': 'sum'
    }).reset_index().rename(columns={'dist': 'dist_sum', 'Energy': 'Energy_sum'})


    df_integrated = df_integrated.merge(filtered_data, on=['Vehicle', 'ServiceDateTime'])

    print("df_integrated columns",df_integrated.columns)
    # Drop rows with NaN values in 'Energy' or 'Qty' columns
    df_integrated.dropna(subset=['Energy_sum', 'Qty'], inplace=True)
    print("1",df_integrated)
    df_integrated['Fuel_economy'] = np.divide(df_integrated['dist_sum'], df_integrated['Energy_sum'], where=df_integrated['Energy_sum'] != 0)
    print("2",df_integrated)
    df_integrated['Real_Fuel_economy'] = np.divide(df_integrated['dist_sum'], df_integrated['Qty'], where=df_integrated['Energy_sum'] != 0)
    print("3",df_integrated)
    df_integrated.dropna(subset=['Fuel_economy'], inplace=True)
    print("4",df_integrated)
    df_integrated.dropna(subset=['Real_Fuel_economy'], inplace=True)
    print("5",df_integrated)
    return df_integrated


# Configuration Section
START1_VAL = 0.005
STOP1_VAL = 0.0025
START2_VAL = 0.005
STOP2_VAL = 0.0025
N_POINTS = 50

# Calibrate parameters with Dask + Joblib for parallel processing
def calibrate_parameter(a0, a1, hybrid):
    
    start_time = time.time()

    if hybrid:
        df = df_hybrid.copy()
        validation = df_validation[df_validation.Powertrain == 'hybrid'].copy()
    else:
        df = df_conventional.copy()
        validation = df_validation[df_validation.Powertrain == 'conventional'].copy()

    validation.reset_index(drop=True, inplace=True)    
    
    #print("hybrid:",hybrid)
    #print("df columns:",df.columns)
    df_integrated = process_dataframe(df, validation, a0, a1, hybrid)
    train, test = train_test_split(df_integrated, test_size=0.2, random_state=42)

    # Training Data
    MSE_Energy_train = mean_squared_error(train['Qty'], train['Energy_sum'])
    RMSE_Energy_train = math.sqrt(MSE_Energy_train)
    MAPE_Energy_train = np.mean(np.abs((train['Qty'] - train['Energy_sum']) / train['Qty'])) * 100
    RMSE_Economy_train = mean_squared_error(train['Real_Fuel_economy'], train['Fuel_economy'], squared=False)
    MAPE_Economy_train = np.mean(np.abs((train['Real_Fuel_economy'] - train['Fuel_economy']) / train['Real_Fuel_economy'])) * 100
    
    # Testing Data
    MSE_Energy_test = mean_squared_error(test['Qty'], test['Energy_sum'])
    RMSE_Energy_test = math.sqrt(MSE_Energy_test)
    MAPE_Energy_test = np.mean(np.abs((test['Qty'] - test['Energy_sum']) / test['Qty'])) * 100
    RMSE_Economy_test = mean_squared_error(test['Real_Fuel_economy'], test['Fuel_economy'], squared=False)
    MAPE_Economy_test = np.mean(np.abs((test['Real_Fuel_economy'] - test['Fuel_economy']) / test['Real_Fuel_economy'])) * 100
            
    results_df = pd.DataFrame({
        'parameter1_values': [a0], 
        'parameter2_values': [a1],
        'RMSE_Energy_train': [RMSE_Energy_train], 
        'MAPE_Energy_train': [MAPE_Energy_train], 
        'RMSE_Economy_train': [RMSE_Economy_train], 
        'MAPE_Economy_train': [MAPE_Economy_train],
        'RMSE_Energy_test': [RMSE_Energy_test], 
        'MAPE_Energy_test': [MAPE_Energy_test], 
        'RMSE_Economy_test': [RMSE_Economy_test], 
        'MAPE_Economy_test': [MAPE_Economy_test]
    })

    file_name = f"../../results/calibration-grid-search-oct2021-sep2022-{'heb' if hybrid else 'cdb'}-oct2021-sep2022-10112023.csv"
    results_df.to_csv(file_name, mode='a', header=False)  # Append mode so you don't overwrite for each a0, a1, hybrid combination
    
    print("--- %s seconds ---" % (time.time() - start_time))


def process_with_error_handling(args):
    try:
        return calibrate_parameter(*args)
    except Exception as e:
        return f"Error: {e}"

def main():
    # Use all available CPUs
    n_processes = 32
    START1_VAL = 0.0001
    STOP1_VAL = 0.005
    START2_VAL = 0.00001
    STOP2_VAL = 0.00001
    N_POINTS = 10

    # Split the parameter grid equally among the available CPUs
    param_grid = [
        (s1, s2, hybrid) 
        for s1 in np.linspace(START1_VAL, STOP1_VAL, N_POINTS) 
        for s2 in np.linspace(START2_VAL, STOP2_VAL, N_POINTS)
        for hybrid in [True, False]
    ]

    with Pool(processes=n_processes) as pool:
        with tqdm(total=len(param_grid), desc="Processing", unit="task") as pbar:
            for _ in pool.starmap(calibrate_parameter, param_grid):
                pbar.update()
                
    # Close the pool
    pool.close()

    # Use get to extract the result from each async result
    # This will block until the result is available
    results = [result.get() for result in results]

    # Wait for all processes to finish
    pool.join()
    
    # Handle errors (if any)
    for res in results:
        if "Error" in res:
            print(res)

if __name__ == '__main__':
    main()
