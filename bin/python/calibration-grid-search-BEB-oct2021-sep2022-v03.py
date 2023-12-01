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
#gamma=0.0411

# Define power function for electric vehicle
def power_e(df_input,  gamma, driveline_efficiency_d_beb, battery_efficiency, motor_efficiency):
    df = df_input
    v = df.speed
    a = df.acc
    gr = df.grade
    m = df.Vehicle_mass+(df.Onboard*179)*0.453592 # converts lb to kg
    factor = df.acc.apply(lambda a: 1 if a >= 0 else np.exp(-(gamma/abs(a))))
    P_t = factor*(eta_batt/eta_m*eta_d_beb)*(1/float(3600*eta_d_beb))*((1./25.92)*rho*C_D*C_h*A_f_beb*v*v + m*g*C_r*(c1*v + c2)/1000 + 1.1*m*a+m*g*gr)*v
    return P_t


# Define Energy consumption function for electric vehicle
def energyConsumption_e(df_input,  gamma, driveline_efficiency_d_beb, battery_efficiency, motor_efficiency):
	# Estimates energy consumed (KWh)     
    df = df_input
    t = df.time_delta_in_seconds/3600
    P_t = power_e(df_input, gamma, driveline_efficiency_d_beb, battery_efficiency, motor_efficiency)
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
df_validation['ServiceDateTime'] = pd.to_datetime(df_validation['ServiceDateTime'])


def process_dataframe(df, validation,  gamma, driveline_efficiency_d_beb, battery_efficiency, motor_efficiency):
    df_new = df.copy()
    validation_new = validation.copy()

    df_new['Energy'] = energyConsumption_e(df,  gamma, driveline_efficiency_d_beb, battery_efficiency, motor_efficiency)
    df_new['ServiceDateTime'] = pd.to_datetime(df_new['ServiceDateTime'])
    df_new = df_new.groupby(['Date', 'Vehicle'])[['Energy', 'dist']].sum().reset_index()


    df_integrated =  pd.merge(validation_new, df_new, 
                     left_on=['ServiceDateTime', 'Vehicle'], 
                     right_on=['Date', 'Vehicle'], 
                     how='left').copy()
    #df_integrated['residual']=df_integrated['trip']-df_integrated['Energy']
    df_integrated = df_integrated.dropna(subset=['trip', 'Energy'])

    # Drop rows where 'trip' or 'Predicted Energy' is 0
    df_integrated = df_integrated.query("trip != 0 and `Energy` != 0")
    
    
    df_integrated.loc[:, 'Fuel_economy'] = np.divide(df_integrated['dist'], df_integrated['Energy'], where=df_integrated['Energy'] != 0)
    df_integrated.loc[:, 'Real_Fuel_economy'] = np.divide(df_integrated['dist'], df_integrated['trip'], where=df_integrated['trip'] != 0)



    return df_integrated

def calibrate_parameters(args):
    # Unpack the ranges for all four parameters
    param1_range, param2_range, param3_range, param4_range = args
    start_time = time.time()

    # Initialize lists to store results
    results_list = []

    # Nested loop to iterate over all combinations of the four parameters
    for gamma in tqdm(np.linspace(*param1_range), desc="Param1 Loop"):
        for driveline_efficiency_d_beb in np.linspace(*param2_range):
            for battery_efficiency in np.linspace(*param3_range):
                for motor_efficiency in np.linspace(*param4_range):
                    # Process the dataframe with the current set of parameters
                    df_integrated = process_dataframe(df_beb.copy(), df_validation.copy(), gamma, driveline_efficiency_d_beb, battery_efficiency, motor_efficiency)
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

                    # Store results
                    results_list.append(( gamma, driveline_efficiency_d_beb, battery_efficiency, motor_efficiency, RMSE_Energy_train_current, MAPE_Energy_train_current, RMSE_Energy_test_current, MAPE_Energy_test_current, RMSE_economy_train_current, MAPE_economy_train_current, RMSE_economy_test_current, MAPE_economy_test_current))

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list, columns=[ 'gamma', 'driveline_efficiency_d_beb', 'battery_efficiency', 'motor_efficiency', 'RMSE_Energy_train', 'MAPE_Energy_train', 'RMSE_Energy_test', 'MAPE_Energy_test', 'RMSE_economy_train', 'MAPE_economy_train', 'RMSE_economy_test', 'MAPE_economy_test'])
    results_df.to_csv((r'../../results/calibration-grid-search-BEB-oct2021-sep2022_12012023.csv'))
    print("--- %s seconds ---" % (time.time() - start_time))

# Example usage
calibrate_parameters(((0.000000000001,0.0001, 500), (0.7, 0.99, 100), (0.7, 0.99, 100), (0.7, 0.99, 100)))
