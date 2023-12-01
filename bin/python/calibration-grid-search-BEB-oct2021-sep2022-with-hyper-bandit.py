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
def power_e(df_input, gamma):
    df = df_input
    v = df.speed
    a = df.acc
    gr = df.grade
    m = df.Vehicle_mass+(df.Onboard*179)*0.453592 # converts lb to kg
    factor = df.acc.apply(lambda a: 1 if a >= 0 else np.exp(-(gamma/abs(a))))
    P_t = factor*(eta_batt/eta_m*eta_d_beb)*(1/float(3600*eta_d_beb))*((1./25.92)*rho*C_D*C_h*A_f_beb*v*v + m*g*C_r*(c1*v + c2)/1000 + 1.1*m*a+m*g*gr)*v
    return P_t


# Define Energy consumption function for electric vehicle
def energyConsumption_e(df_input, gamma):
	# Estimates energy consumed (KWh)     
    df = df_input
    t = df.time_delta_in_seconds/3600
    P_t = power_e(df_input, gamma)
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


def process_dataframe(df, validation, gamma):
    df_new = df.copy()
    validation_new = validation.copy()

    df_new['Energy'] = energyConsumption_e(df, gamma)
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
    
    
    df_integrated['Fuel_economy'] = np.divide(df_integrated['dist'], df_integrated['Energy'], where=df_integrated['Energy'] != 0)
    df_integrated['Real_Fuel_economy'] = np.divide(df_integrated['dist'], df_integrated['trip'], where=df_integrated['trip'] != 0)


    return df_integrated

# Hyperband helper functions
def get_hyperband_configurations(max_iter=81, eta=3):
    """ Generate the (n, r) configurations for Hyperband """
    s_max = int(np.floor(np.log(max_iter) / np.log(eta)))
    B = (s_max + 1) * max_iter

    configurations = []
    for s in range(s_max, -1, -1):
        n = int(np.ceil(B / max_iter / (s + 1) * eta ** s))
        r = max_iter * eta ** (-s)
        configurations.append((n, int(r)))
    return configurations

def calibrate_parameter(start, stop, B):
    start_time = time.time()
    configurations = get_hyperband_configurations(B)
    decimal_places = 9

    for n, r in configurations:
        # Create gamma values in log space
        gamma_values = np.logspace(np.log10(start), np.log10(stop), n)
        gamma_values = np.around(gamma_values, decimals=decimal_places)
        
        parameter1_values = []
        RMSE_Energy_train = []
        MAPE_Energy_train = []
        RMSE_Energy_test = []
        MAPE_Energy_test = []

        for gamma in tqdm(gamma_values, desc=f"Processing gamma values for n={n}, r={r}"):
            # Sample a fraction of the data based on the resource r
            sample_size = int(len(df_beb) * r / B)
            sampled_df = df_beb.sample(sample_size)

            # Process the sampled data with the current gamma
            df_integrated = process_dataframe(sampled_df, df_validation, gamma)
            df_train, df_test = train_test_split(df_integrated, test_size=0.2, random_state=42)
            
            # Calculate the performance metrics
        
            RMSE_Energy_train_current = np.sqrt(mean_squared_error(df_train['trip'], df_train['Energy']))
            MAPE_Energy_train_current = mean_absolute_percentage_error(df_train['trip'] , df_train['Energy'])
            RMSE_Energy_test_current = np.sqrt(mean_squared_error(df_test['trip'], df_test['Energy']))
            MAPE_Energy_test_current = mean_absolute_percentage_error(df_test['trip'] , df_test['Energy'])
            parameter1_values.append(gamma)
            RMSE_Energy_train.append(RMSE_Energy_train_current)
            MAPE_Energy_train.append(MAPE_Energy_train_current)
            RMSE_Energy_test.append(RMSE_Energy_test_current)
            MAPE_Energy_test.append(MAPE_Energy_test_current)


    results = pd.DataFrame(list(zip(parameter1_values, RMSE_Energy_train, MAPE_Energy_train, RMSE_Energy_test, MAPE_Energy_test)),
                           columns=['parameter1_values', 'RMSE_Energy_train', 'MAPE_Energy_train', 'RMSE_Energy_test', 'MAPE_Energy_test'])
    results.to_csv((r'../../results/calibration-grid-search-BEB-oct2021-sep2022_11282023.csv'))
    print("--- %s seconds ---" % (time.time() - start_time))

    
calibrate_parameter(0.00000001,0.09, 5000)