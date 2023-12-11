import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import time
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval



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
a2_cdb = p.alpha_2_cdb
a0_heb = p.alpha_0_heb
a1_heb = p.alpha_1_heb
a2_heb = p.alpha_2_heb
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
def fuelRate_d(df_input, a0, a1, a2, hybrid=False):
	# Estimates fuel consumed (liters per second) 
    df = df_input
    if hybrid == True:
        #a0 = a0_heb 
        #a1 = a1_heb
        #factor = df.acc.apply(lambda a: 1 if a >= 0 else np.exp(-(0.0411/abs(a))))
        #factor=1
        P_t = power_d(df_input, hybrid=True)
        FC_t = P_t.apply(lambda x: a0 + a1*x +a2_cdb*x*x if x >= 0 else a0 + eta_batt*np.exp(-(0.0411/abs(df.acc)))/eta_m*eta_d_beb*x)  

    else:
        #a0 = a0_cdb
        #a1 = a1_cdb
        P_t = power_d(df_input, hybrid=False)
        FC_t = P_t.apply(lambda x: a0 + a1*x +a2_heb*x*x if x >= 0 else a0)  
    return FC_t


# Define Energy consumption function for diesel vehicle
def energyConsumption_d(df_input, a0, a1, a2, hybrid=False):
	# Estimates energy consumed (gallons)     
    df = df_input
    t = df.time_delta_in_seconds
    if hybrid == True:
        FC_t = fuelRate_d(df_input, a0, a1, a2, hybrid=True)
    else:
        FC_t = fuelRate_d(df_input, a0, a1, a2, hybrid=False)
    E_t = (FC_t * t)/3.78541
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


def process_dataframe(df, validation, a0, a1, a2, hybrid):
    df_new = df.copy()
    validation_new = validation.copy()

    df_new['Energy'] = energyConsumption_d(df, a0, a1, a2, hybrid=hybrid)
    df_new.sort_values(by=['Vehicle', 'ServiceDateTime'], inplace=True)
    df_new['ServiceDateTime'] = pd.to_datetime(df_new['ServiceDateTime'])

    df_integrated = validation_new.copy()
    df_integrated.sort_values(by=['Vehicle', 'ServiceDateTime'], inplace=True)
    df_integrated['ServiceDateTime_prev'] = df_integrated.groupby('Vehicle')['ServiceDateTime'].shift(1)
    df_integrated = df_integrated.dropna(subset=['ServiceDateTime_prev'])

    def process_group(group):
        return pd.Series({'dist_sum': group['dist'].sum(), 'Energy_sum': group['Energy'].sum()})

    df_filtered = pd.DataFrame()
    for _, row in tqdm(df_integrated.iterrows(), total=len(df_integrated)):
        vehicle, cur_time, prev_time = row['Vehicle'], row['ServiceDateTime'], row['ServiceDateTime_prev']
        group = df_new[(df_new['Vehicle'] == vehicle) & (df_new['ServiceDateTime'] > prev_time) & (df_new['ServiceDateTime'] < cur_time)]
        filtered_group = process_group(group)
        filtered_group['Vehicle'] = vehicle
        filtered_group['ServiceDateTime_cur'] = cur_time
        filtered_group['ServiceDateTime_prev'] = prev_time
        df_filtered = pd.concat([df_filtered, filtered_group.to_frame().T], ignore_index=True)
    df_integrated = df_integrated.merge(df_filtered, left_on=['Vehicle', 'ServiceDateTime', 'ServiceDateTime_prev'],
                                        right_on=['Vehicle', 'ServiceDateTime_cur', 'ServiceDateTime_prev'])

    # Drop rows with NaN values in 'Energy' or 'Qty' columns
    df_integrated.dropna(subset=['Energy_sum', 'Qty'], inplace=True)
    
    # Drop rows with 0 values in 'Energy' or 'Qty' columns
    df_integrated = df_integrated.query("Qty != 0 and `Energy_sum` != 0")
    
    df_integrated['Fuel_economy'] = np.divide(df_integrated['dist_sum'], df_integrated['Energy_sum']) #, where=df_integrated['Energy_sum'] != 0)
    df_integrated['Real_Fuel_economy'] = np.divide(df_integrated['dist_sum'], df_integrated['Qty']) #, where=df_integrated['Energy_sum'] != 0)

    return df_integrated

# Define the worker function
def worker_function(params, hybrid):
    a0, a1, a2 = params
    #gamma = params['gamma']
    #driveline_efficiency_d_beb = params['driveline_efficiency_d_beb']
    #battery_efficiency = params['battery_efficiency']
    #motor_efficiency = params['motor_efficiency']
    if hybrid == True:
        df = df_hybrid.copy()
    else:
        df=df_conventional.copy()

    df_integrated = process_dataframe(df, df_validation.copy(), a0, a1, a2)
    df_train, df_test = train_test_split(df_integrated, test_size=0.2, random_state=42)

    # Calculate metrics
    RMSE_Energy_train_current = np.sqrt(mean_squared_error(df_train['trip'], df_train['Energy']))
    MAPE_Energy_train_current = mean_absolute_percentage_error(df_train['trip'] , df_train['Energy'])
    RMSE_Energy_test_current = np.sqrt(mean_squared_error(df_test['trip'], df_test['Energy']))
    MAPE_Energy_test_current = mean_absolute_percentage_error(df_test['trip'] , df_test['Energy'])

    RMSE_economy_train_current = np.sqrt(mean_squared_error(df_train['Real_Fuel_economy'], df_train['Fuel_economy']))
    MAPE_economy_train_current = mean_absolute_percentage_error(df_train['Real_Fuel_economy'] , df_train['Fuel_economy'])
    RMSE_economy_test_current = np.sqrt(mean_squared_error(df_test['Real_Fuel_economy'], df_test['Fuel_economy']))
    MAPE_economy_test_current = mean_absolute_percentage_error(df_test['Real_Fuel_economy'] , df_test['Fuel_economy'])

    return (a0, a1, a2, RMSE_Energy_train_current, MAPE_Energy_train_current, RMSE_Energy_test_current, MAPE_Energy_test_current, RMSE_economy_train_current, MAPE_economy_train_current, RMSE_economy_test_current, MAPE_economy_test_current)

    
def calibrate_parameter(args):
    start_time = time.time()
    param1_range, param2_range, param3_range = args
    parameter1_values = []
    parameter2_values = []
    RMSE_Energy = []
    MAPE_Energy = []
    RMSE_Economy = []
    MAPE_Economy = []

    if hybrid:
        df = df_hybrid
        validation = df_validation[df_validation.Powertrain == 'hybrid'].copy()
        a0_global_name, a1_global_name = 'a0_heb', 'a1_heb'
        n_points = 30
    else:
        df = df_conventional
        validation = df_validation[df_validation.Powertrain == 'conventional'].copy()
        a0_global_name, a1_global_name = 'a0_cdb', 'a1_cdb'
        n_points = 10

    validation.reset_index(inplace=True)    
    
    decimal_places = 6  # Set the desired number of decimal places
    a0_space = np.around(np.linspace(start1, stop1, n_points), decimals=decimal_places)
    a1_space = np.around(np.linspace(start2, stop2, n_points), decimals=decimal_places)

    for a0 in tqdm(a0_space, desc="a0"):
        for a1 in tqdm(a1_space, desc="a1", leave=False):
            globals()[a0_global_name] = a0
            globals()[a1_global_name] = a1

            df_integrated = process_dataframe(df, validation, a0, a1, hybrid)
            df_integrated.dropna(subset=['Qty', 'Energy_sum'], inplace=True)
            train, test = train_test_split(df_integrated, test_size=0.2, random_state=42)

            MSE_Energy_current = mean_squared_error(train['Qty'], train['Energy_sum'])
            RMSE_Energy_current = math.sqrt(MSE_Energy_current)
            MAPE_Energy_current = np.mean(np.abs((train['Qty'] - train['Energy_sum']) / train['Qty'])) * 100
            RMSE_Economy_current = mean_squared_error(train['Real_Fuel_economy'], train['Fuel_economy'], squared=False)
            MAPE_Economy_current = np.mean(np.abs((train['Real_Fuel_economy'] - train['Fuel_economy']) / train['Real_Fuel_economy'])) * 100

            parameter1_values.append(a0)
            parameter2_values.append(a1)
            RMSE_Energy.append(RMSE_Energy_current)
            MAPE_Energy.append(MAPE_Energy_current)
            RMSE_Economy.append(RMSE_Economy_current)
            MAPE_Economy.append(MAPE_Economy_current)

    results = pd.DataFrame(list(zip(parameter1_values, parameter2_values, RMSE_Energy, MAPE_Energy, RMSE_Economy, MAPE_Economy)),
                           columns=['parameter1_values', 'parameter2_values', 'RMSE_Energy', 'MAPE_Energy', 'RMSE_Economy', 'MAPE_Economy'])
    file_name = f"../../results/calibration-grid-search-oct2021-sep2022-{'heb' if hybrid else 'cdb'}-oct2021-sep2022.csv"
    results.to_csv(file_name)
    print("--- %s seconds ---" % (time.time() - start_time))


# Worker function 
def hyperband_worker(params, hybrid):
    #gamma, driveline_efficiency_d_beb, battery_efficiency, motor_efficiency = params
    a0 = params['a0']
    a1 = params['a1']
    a2 = params['a2']
    
    if hybrid == True:
        df = df_hybrid.copy()
    else:
        df=df_conventional.copy()
    
    df_integrated = process_dataframe(df, df_validation.copy(), a0, a1, a2)
    df_train, df_test = train_test_split(df_integrated, test_size=0.2, random_state=42)

    # Calculate metrics
    RMSE_Energy_train_current = np.sqrt(mean_squared_error(df_train['trip'], df_train['Energy']))
    MAPE_Energy_train_current = mean_absolute_percentage_error(df_train['trip'] , df_train['Energy'])
    RMSE_Energy_test_current = np.sqrt(mean_squared_error(df_test['trip'], df_test['Energy']))
    MAPE_Energy_test_current = mean_absolute_percentage_error(df_test['trip'] , df_test['Energy'])

    RMSE_economy_train_current = np.sqrt(mean_squared_error(df_train['Real_Fuel_economy'], df_train['Fuel_economy']))
    MAPE_economy_train_current = mean_absolute_percentage_error(df_train['Real_Fuel_economy'] , df_train['Fuel_economy'])
    RMSE_economy_test_current = np.sqrt(mean_squared_error(df_test['Real_Fuel_economy'], df_test['Fuel_economy']))
    MAPE_economy_test_current = mean_absolute_percentage_error(df_test['Real_Fuel_economy'] , df_test['Fuel_economy'])
    
    # Append the results to the global DataFrame
    global results_df
    results_row = {
        'a0': a0, 
        'a1': a1, 
        'a2': a2, 
        'RMSE_Energy_train': RMSE_Energy_train_current,
        'MAPE_Energy_train': MAPE_Energy_train_current,
        'RMSE_Energy_test': RMSE_Energy_test_current,
        'MAPE_Energy_test': MAPE_Energy_test_current,
        'RMSE_economy_train': RMSE_economy_train_current,
        'MAPE_economy_train': MAPE_economy_train_current,
        'RMSE_economy_test': RMSE_economy_test_current,
        'MAPE_economy_test': MAPE_economy_test_current,
    }
    results_df = results_df.append(results_row, ignore_index=True)
    
    return {
        'loss': RMSE_Energy_test_current,  
        'status': STATUS_OK,
        'params': params,
        'RMSE_Energy_train': RMSE_Energy_train_current,
        'RMSE_Energy_test': RMSE_Energy_test_current,
        'MAPE_Energy_train': MAPE_Energy_train_current,
        'MAPE_Energy_test': MAPE_Energy_test_current,
        'RMSE_Economy_train': RMSE_economy_train_current,
        'RMSE_Economy_test': RMSE_economy_test_current,
        'MAPE_Economy_train': MAPE_economy_train_current,
        'MAPE_Economy_test': MAPE_economy_test_current,
    }



# Define the search space
space = {
    'a0': hp.uniform('a0', 0.0009, 0.003),
    'a1': hp.uniform('a1', 0.00005, 0.0002),
    'a2': hp.uniform('a2', 0.0000009, 0.00000003),
}


# Run the hyperband optimizer
trials = Trials()
best = fmin(
    fn=hyperband_worker,
    space=space,
    algo=tpe.suggest,
    max_evals=20,  
    trials=trials
)

# Save the results to a CSV file after optimization
results_df.to_csv(r'../../results/calibration-grid-search-HEB-oct2021-sep2022_12112023.csv', index=False)

print("Best parameters found: ", space_eval(space, best))


