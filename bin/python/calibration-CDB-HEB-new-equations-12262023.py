import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from dask import compute, delayed
from dask.distributed import Client
import itertools


f = open('params-oct2021-sep2022.yaml')
parameters = yaml.safe_load(f)

class vehicleParams():
	def __init__(self, **entries):
		self.__dict__.update(entries)

# Define model parameters
p = vehicleParams(**parameters)
rho = p.air_density
C_D = p.drag_coefficient
A_f_cdb = p.frontal_area_cdb
A_f_heb = p.frontal_area_heb
A_f_beb = p.frontal_area_beb
g = p.gravitational_acceleration
C_r1 = p.rolling_resistance_coef1
C_r2 = p.rolling_resistance_coef2
eta_d_cdb = p.driveline_efficiency_d_dis
eta_d_heb = p.driveline_efficiency_d_dis
eta_d_beb = p.driveline_efficiency_d_beb
eta_batt = p.battery_efficiency
eta_m = p.motor_efficiency
a0_cdb = p.alpha_0_cdb
a1_cdb = p.alpha_1_cdb
a2_cdb = p.alpha_2_cdb
a0_heb = p.alpha_0_heb
a1_heb = p.alpha_1_heb
a2_heb = p.alpha_2_heb
gamma_beb=p.gamma

# Define power function for diesel vehicle
def power(df_input, hybrid=False, electric=False):
    if hybrid == True:
       A=A_f_heb
       eta_d = eta_d_heb
    elif electric == True:
        A=A_f_beb
        eta_d = eta_d_beb
    else:
       A=A_f_cdb 
       eta_d = eta_d_cdb
       
    df = df_input
    v = df.speed
    a = df.acc
    G = df.grade
    m = (df.Vehicle_mass+df.Onboard*179)*0.453592 # converts lb to kg
    H = df.elevation/1000 # df.elevation is in meters and we need to convert it to km 
    C_h = 1 - (0.085*H)
    #P_t = (v/float(1000*eta_d))*(rho*C_D*C_h*A*v*v +m(g(C_r1*v+C_r2+G)+1.2*a))
    P_t = (v / (1000 * eta_d)) * (rho * C_D * C_h * A * v * v + m * (g * (C_r1 * v + C_r2 + G) + 1.2 * a))

    return P_t

# Define fuel rate function for diesel vehicle
def fuelRate_d(df_input, a0, a1, a2, hybrid=False):
	# Estimates fuel consumed (liters per second) 
    a0 = a0_heb 
    a1 = a1_heb 
    a2 = a2_heb 
    P_t = power(df_input, hybrid)
    FC_t = P_t.apply(lambda x: a0 + a1*x +a2*x*x if x >= 0 else a0)  
    return FC_t


# Define Energy consumption function for diesel vehicle
def energyConsumption_d(df_input, a0, a1, a2, hybrid=False):
	# Estimates energy consumed (gallons)     
    df = df_input
    t = df.time_delta_in_seconds
    FC_t = fuelRate_d(df_input, a0, a1, a2, hybrid)
    E_t = (FC_t * t)/3.78541 # to convert liters to gals
    return E_t


# Read trajectories df
df_trajectories = pd.read_csv(r'../../data/tidy/large/trajectories-mapped-powertrain-weight-grade-oct2021-sep2022.csv', delimiter=',', skiprows=0, low_memory=False)
df_trajectories.rename(columns={"VehiclWeight(lb)": "Vehicle_mass"}, inplace=True)
df_trajectories['Date'] = pd.to_datetime(df_trajectories['Date'])
df_trajectories.speed = df_trajectories.speed *0.44704 # Convert from mph to m/s
df_trajectories = df_trajectories.fillna(0)

# Remove accelerations other than -5 to 3 m/s^2
total_rows = len(df_trajectories)
df_trajectories = df_trajectories[(df_trajectories['acc'] >= -5) & (df_trajectories['acc'] <= 3)]
remaining_rows = len(df_trajectories)
removed_rows = total_rows - remaining_rows
removed_percentage = (removed_rows / total_rows) * 100
print(f"Percentage of removed data: {removed_percentage:.2f}%")

# Subsetting data frame
df_conventional = df_trajectories.loc[df_trajectories['Powertrain'] == 'conventional'].copy()
df_hybrid = df_trajectories.loc[df_trajectories['Powertrain'] == 'hybrid'].copy()
del df_trajectories

# read validation df
df_validation = pd.read_csv(r'../../data/tidy/fuel-tickets-clean-oct2021-sep2022.csv', delimiter=',', skiprows=0, low_memory=False)
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


# def process_dataframe(df, validation, a0, a1, a2, hybrid):
#     df_new = df.copy()
#     validation_new = validation.copy()

#     df_new['Energy'] = energyConsumption_d(df_new,a0, a1, a2, hybrid=hybrid)
#     df_new.sort_values(by=['Vehicle', 'ServiceDateTime'], inplace=True)
#     df_new['ServiceDateTime'] = pd.to_datetime(df_new['ServiceDateTime'])

#     df_integrated = validation_new.copy()
#     df_integrated.sort_values(by=['Vehicle', 'ServiceDateTime'], inplace=True)
#     df_integrated['ServiceDateTime_prev'] = df_integrated.groupby('Vehicle')['ServiceDateTime'].shift(1)
#     df_integrated = df_integrated.dropna(subset=['ServiceDateTime_prev'])

#     def process_group(group):
#         return pd.Series({'dist_sum': group['dist'].sum(), 'Energy_sum': group['Energy'].sum()})

#     df_filtered = pd.DataFrame()
#     for _, row in df_integrated.iterrows():
#         vehicle, cur_time, prev_time = row['Vehicle'], row['ServiceDateTime'], row['ServiceDateTime_prev']
#         group = df_new[(df_new['Vehicle'] == vehicle) & (df_new['ServiceDateTime'] > prev_time) & (df_new['ServiceDateTime'] < cur_time)]
#         filtered_group = process_group(group)
#         filtered_group['Vehicle'] = vehicle
#         filtered_group['ServiceDateTime_cur'] = cur_time
#         filtered_group['ServiceDateTime_prev'] = prev_time
#         df_filtered = pd.concat([df_filtered, filtered_group.to_frame().T], ignore_index=True)
#     df_integrated = df_integrated.merge(df_filtered, left_on=['Vehicle', 'ServiceDateTime', 'ServiceDateTime_prev'],
#                                         right_on=['Vehicle', 'ServiceDateTime_cur', 'ServiceDateTime_prev'])
    
#     df_integrated.dropna(subset=['Energy_sum', 'Qty'], inplace=True)
#     df_integrated = df_integrated.query("Qty != 0 and Energy_sum != 0")

#     df_integrated['actual_mpg'] = df_integrated['dist_sum'] / df_integrated['Qty']
#     df_integrated['pred_mpg'] = df_integrated['dist_sum'] / df_integrated['Energy_sum']
#     return df_integrated

# vectorized version of process dataframe
def process_dataframe(df, validation, a0, a1, a2, hybrid):
    # Copying dataframes as in the original function
    df_new = df.copy()
    validation_new = validation.copy()

    # Assuming energyConsumption_d() handles the data correctly
    df_new['Energy'] = energyConsumption_d(df_new, a0, a1, a2, hybrid=hybrid)
    df_new.sort_values(by=['Vehicle', 'ServiceDateTime'], inplace=True)
    df_new['ServiceDateTime'] = pd.to_datetime(df_new['ServiceDateTime'])

    # Preparing the validation dataframe
    df_integrated = validation_new.copy()
    df_integrated.sort_values(by=['Vehicle', 'ServiceDateTime'], inplace=True)
    df_integrated['ServiceDateTime_prev'] = df_integrated.groupby('Vehicle')['ServiceDateTime'].shift(1)
    df_integrated = df_integrated.dropna(subset=['ServiceDateTime_prev'])

    # Set multi-level index for merging and filtering
    df_new.set_index(['Vehicle', 'ServiceDateTime'], inplace=True)
    df_integrated.set_index(['Vehicle', 'ServiceDateTime'], inplace=True)

    # Merging dataframes
    df_integrated['join_key'] = df_integrated.index
    merged_df = df_integrated.join(df_new, on='join_key', rsuffix='_new')

    # Apply the conditions as filters to the merged DataFrame
    filtered_df = merged_df[
        (merged_df['ServiceDateTime_new'] > merged_df['ServiceDateTime_prev']) &
        (merged_df['ServiceDateTime_new'] < merged_df.index.get_level_values('ServiceDateTime'))
    ]

    # Grouping and summarizing
    grouped = filtered_df.groupby(['Vehicle', 'ServiceDateTime', 'ServiceDateTime_prev'])
    summary_df = grouped.agg({'dist':'sum', 'Energy':'sum'}).rename(columns={'dist':'dist_sum', 'Energy':'Energy_sum'})

    # Preparing for merge
    summary_df.reset_index(inplace=True)
    df_integrated.reset_index(inplace=True)

    # Merging summary back to df_integrated
    final_df = df_integrated.merge(summary_df, left_on=['Vehicle', 'ServiceDateTime', 'ServiceDateTime_prev'], right_on=['Vehicle', 'ServiceDateTime', 'ServiceDateTime_prev'])

    # Final Calculations
    final_df['actual_mpg'] = final_df['dist_sum'] / final_df['Qty']
    final_df['pred_mpg'] = final_df['dist_sum'] / final_df['Energy_sum']

    # Drop na and filter out rows where Qty and Energy_sum are 0
    final_df.dropna(subset=['Energy_sum', 'Qty'], inplace=True)
    final_df = final_df.query("Qty != 0 and Energy_sum != 0")

    return final_df


# Calibrate parameters with Dask + Joblib for parallel processing
def calibrate_parameter(a0, a1, hybrid):
    
    if hybrid:
        df = df_hybrid.copy()
        validation = df_validation[df_validation.Powertrain == 'hybrid'].copy()
    else:
        df = df_conventional.copy()
        validation = df_validation[df_validation.Powertrain == 'conventional'].copy()

    validation.reset_index(drop=True, inplace=True)    
    
    df_integrated = process_dataframe(df, validation, a0, a1, hybrid)
    df_integrated = df_integrated.loc[df_integrated['Energy_sum']!=0]
    percentile_5 = df_integrated['Real_Fuel_economy'].quantile(0.05)
    percentile_95 = df_integrated['Real_Fuel_economy'].quantile(0.95)
    df_integrated = df_integrated[(df_integrated['pred_mpg'] >= percentile_5) & (df_integrated['pred_mpg'] <= percentile_95)]
    df_integrated = df_integrated[(df_integrated['actual_mpg'] >= percentile_5) & (df_integrated['actual_mpg'] <= percentile_95)]
    train, test = train_test_split(df_integrated, test_size=0.8, random_state=42)
    
    # Training Data
    RMSE_Energy_train = np.sqrt(mean_squared_error(train['Qty'], train['Energy_sum']))
    MAPE_Energy_train = mean_absolute_percentage_error(train['Qty'] , train['Energy_sum'])


            
    results_df = pd.DataFrame({
        'parameter1_values': [a0], 
        'parameter2_values': [a1],
        'RMSE_Energy_train': [RMSE_Energy_train], 
        'MAPE_Energy_train': [MAPE_Energy_train]
    })
    return results_df

hybrid_flag = True
# Configuration Section
START1_VAL = 0.0007
STEP_SIZE1 = 0.006

START2_VAL = 0.000059
STEP_SIZE2 = 0.00001

START3_VAL = 0.00000001
STEP_SIZE3 = 0.000000005
N_POINTS = 10

# Initialize results dataframe to store results of all iterations
all_results_df = pd.DataFrame()


def parallel_calibrate(a0, a1, a2, hybrid_flag):
    return calibrate_parameter(a0, a1, hybrid_flag)

if __name__ == '__main__':
    client = Client(n_workers=32, threads_per_worker=1)
    hybrid_flag = True

    delayed_tasks = []
    a0_values = np.linspace(START1_VAL, START1_VAL+(STEP_SIZE1 * (N_POINTS-1)), N_POINTS)
    a1_values = np.linspace(START2_VAL, START2_VAL+(STEP_SIZE2 * (N_POINTS-1)), N_POINTS)
    a2_values = np.linspace(START3_VAL, START3_VAL+(STEP_SIZE3 * (N_POINTS-1)), N_POINTS)

    start_time = time.time()  # Start timer

    # Using a single loop to iterate over cartesian product of a0 and a1 values
    for a0, a1, a2 in tqdm(itertools.product(a0_values, a1_values, a2_values), total=N_POINTS**3, desc="Processing"):
        if hybrid_flag:
            delayed_task = delayed(parallel_calibrate)(a0, a1, a2, True)
            delayed_tasks.append(delayed_task)

    results = compute(*delayed_tasks)  # compute all results in parallel

    all_results_df = pd.concat(results, ignore_index=True)

    # Save results to a CSV file
    all_results_df.to_csv('../../results/calibration_results_heb_oct2021-sep2022_12262023.csv', index=False)

    # Calculate and print the elapsed time
    elapsed_time = time.time() - start_time
    print(f"Total time taken: {elapsed_time:.2f} seconds")