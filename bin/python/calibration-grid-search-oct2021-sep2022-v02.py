import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval


start_time = time.time()
with open('params-oct2021-sep2022-test10222023.yaml') as f:
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
    A_f_d = A_f_heb if hybrid else A_f_cdb         
    v = df_input.speed
    a = df_input.acc
    gr = df_input.grade
    m = (df_input.Vehicle_mass + df_input.Onboard * 179) * 0.453592  # converts lb to kg
    P_t = (1 / (3600 * eta_d_dis)) * ((1. / 25.92) * rho * C_D * C_h * A_f_d * v ** 2 + m * g * C_r * (c1 * v + c2) / 1000 + 1.2 * m * a + m * g * gr) * v
    return P_t


# Define fuel rate function for diesel vehicle
def fuelRate_d(df_input, a0, a1, a2, hybrid=False):
    P_t = power_d(df_input, hybrid)
    if hybrid:
        FC_t = np.where(P_t >= 0, a0 + a1 * P_t + a2_cdb * P_t ** 2, a0 + eta_batt * np.exp(-(0.0411 / np.abs(df_input.acc))) / eta_m * eta_d_beb * P_t)
    else:
        FC_t = np.where(P_t >= 0, a0 + a1 * P_t + a2_heb * P_t ** 2, a0)
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
df_conventional=df_trajectories.loc[df_trajectories['Powertrain'] == 'conventional']
df_hybrid=df_trajectories.loc[df_trajectories['Powertrain'] == 'hybrid']

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

    # Calculate energy consumption
    df_new['Energy'] = energyConsumption_d(df_new, a0, a1, a2, hybrid=hybrid)
    df_new.sort_values(by=['Vehicle', 'ServiceDateTime'], inplace=True)
    df_new['ServiceDateTime'] = pd.to_datetime(df_new['ServiceDateTime'])

    # Prepare integrated dataframe
    df_integrated = validation_new
    df_integrated.sort_values(by=['Vehicle', 'ServiceDateTime'], inplace=True)
    df_integrated['ServiceDateTime_prev'] = df_integrated.groupby('Vehicle')['ServiceDateTime'].shift(1)
    df_integrated = df_integrated.dropna(subset=['ServiceDateTime_prev'])

    # Function to process each group
    def process_group(group):
        return pd.Series({'dist_sum': group['dist'].sum(), 'Energy_sum': group['Energy'].sum()})

    # Process each group and store in a list
    filtered_groups = []
    with tqdm(total=len(df_integrated), desc="Processing Integrated Dataframe") as pbar_outer:
        for _, row in df_integrated.iterrows():
            pbar_outer.update(1)
            vehicle, cur_time, prev_time = row['Vehicle'], row['ServiceDateTime'], row['ServiceDateTime_prev']
            group = df_new[(df_new['Vehicle'] == vehicle) & (df_new['ServiceDateTime'] > prev_time) & (df_new['ServiceDateTime'] <= cur_time)]
        
            # Adding an inner tqdm progress bar
            with tqdm(total=len(group), desc=f"Processing Group for Vehicle {vehicle}", leave=False) as pbar_inner:
                filtered_group = process_group(group)
                pbar_inner.update(len(group))  # Update inner progress bar after processing the group

            filtered_group['Vehicle'] = vehicle
            filtered_group['ServiceDateTime_cur'] = cur_time
            filtered_group['ServiceDateTime_prev'] = prev_time
            filtered_groups.append(filtered_group.to_frame().T)

    # Concatenate all groups at once
    df_filtered = pd.concat(filtered_groups, ignore_index=True)

    # Merge and process the integrated dataframe
    print("df_integrated",df_integrated.columns)
    print("df_filtered",df_filtered.columns)

    #df_integrated = df_integrated.merge(df_filtered, on=['Vehicle', 'ServiceDateTime', 'ServiceDateTime_prev'])
    df_integrated = df_integrated.merge(df_filtered, left_on=['Vehicle', 'ServiceDateTime', 'ServiceDateTime_prev'], right_on=['Vehicle', 'ServiceDateTime_cur', 'ServiceDateTime_prev'])

    df_integrated.dropna(subset=['Energy_sum', 'Qty'], inplace=True)
    df_integrated = df_integrated.query("Qty != 0 and Energy_sum != 0")

    df_integrated['Fuel_economy'] = df_integrated['dist_sum'] / df_integrated['Energy_sum']
    df_integrated['Real_Fuel_economy'] = df_integrated['dist_sum'] / df_integrated['Qty']

    return df_integrated

# Worker function 
def hyperband_worker(params, hybrid):
    #gamma, driveline_efficiency_d_beb, battery_efficiency, motor_efficiency = params
    a0 = params['a0']
    a1 = params['a1']
    a2 = params['a2']
    
    if hybrid == True:
        df = df_hybrid
    else:
        df=df_conventional
    
    df_integrated = process_dataframe(df, df_validation.copy(), a0, a1, a2, hybrid)
    df_train, df_test = train_test_split(df_integrated, test_size=0.2, random_state=42)

    # Calculate metrics
    RMSE_Energy_train_current = np.sqrt(mean_squared_error(df_train['Qty'], df_train['Energy_sum']))
    MAPE_Energy_train_current = mean_absolute_percentage_error(df_train['Qty'] , df_train['Energy_sum'])
    RMSE_Energy_test_current = np.sqrt(mean_squared_error(df_test['Qty'], df_test['Energy_sum']))
    MAPE_Energy_test_current = mean_absolute_percentage_error(df_test['Qty'] , df_test['Energy_sum'])

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
    'a0': hp.uniform('a0', 0.0003, 0.005),
    'a1': hp.uniform('a1', 0.00005, 0.0002),
    'a2': hp.uniform('a2', 0.000000008, 0.00000005),
}

hybrid = True  # Set this to False if working with conventional buses


# Run the hyperband optimizer
trials = Trials()
best = fmin(
    fn=lambda params: hyperband_worker(params, hybrid=hybrid),
    space=space,
    algo=tpe.suggest,
    max_evals=20,  
    trials=trials
)

# Save the results to a CSV file after optimization
if hybrid:
    results_df.to_csv(r'../../results/calibration-grid-search-HEB-oct2021-sep2022_12112023.csv', index=False)
else:
    results_df.to_csv(r'../../results/calibration-grid-search-CDB-oct2021-sep2022_12112023.csv', index=False)


print("Best parameters found: ", space_eval(space, best))
print("--- %s seconds ---" % (time.time() - start_time))

