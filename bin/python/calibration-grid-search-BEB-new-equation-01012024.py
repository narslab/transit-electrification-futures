import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
import multiprocessing


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
#eta_m = p.motor_efficiency
a0_cdb = p.alpha_0_cdb
a1_cdb = p.alpha_1_cdb
a2_cdb = p.alpha_2_cdb
a0_heb = p.alpha_0_heb
a1_heb = p.alpha_1_heb
a2_heb = p.alpha_2_heb
#gamma_beb=p.gamma

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
    a0 = a0 
    a1 = a1 
    a2 = a2 
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

# Define Energy consumption function for electric vehicle
def energyConsumption_e(df_input, gamma_beb, eta_m, electric=True):
	# Estimates energy consumed (KWh)     
    df = df_input
    t = df.time_delta_in_seconds
    P_t = power(df_input, electric)
    eta_rb = df.Acceleration.apply(lambda a: 1 if a >= 0 else np.exp(-(gamma_beb/abs(a))))
    E_t = t * P_t * eta_rb * eta_batt /(eta_m*3600)
    return E_t

# Read computed fuel rates
df_trajectories = pd.read_csv(r'../../data/tidy/large/trajectories-mapped-powertrain-weight-grade-oct2021-sep2022.csv', delimiter=',', skiprows=0, low_memory=False)
df_trajectories.rename(columns={"VehiclWeight(lb)": "Vehicle_mass"}, inplace=True)
df_trajectories['Date']=pd.to_datetime(df_trajectories['Date'])
df_trajectories.speed = df_trajectories.speed *0.44704 # Convert from mph to m/s
df_trajectories = df_trajectories.fillna(0)
df_trajectories = df_trajectories[(df_trajectories['Acceleration'] >= -5) & (df_trajectories['Acceleration'] <= 3)]

# Subsetting data frame for "Conventional", "hybrid", and "electric" buses
df_beb=df_trajectories.loc[df_trajectories['Powertrain'] == 'electric'].copy()

del df_trajectories

# read validation df
df_validation = pd.read_excel(r'../../data/tidy/BEB-validation.xlsx')
df_validation.rename(columns={"Transaction Date": "ServiceDateTime","Equipment ID":"Vehicle"}, inplace=True)
df_validation['ServiceDateTime'] = pd.to_datetime(df_validation['ServiceDateTime'])


def process_dataframe(df, validation, gamma_beb, eta_m):
    df_new = df.copy()
    validation_new = validation.copy()

    df_new['Energy'] = energyConsumption_e(df, gamma_beb, eta_m)
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

def calibrate_parameter(args):
    start, stop, n_points = args
    start_time = time.time()
    parameter1_values = []
    RMSE_Energy_train = []
    MAPE_Energy_train = []

    df = df_beb.copy()
    validation = df_validation.copy()
    validation.reset_index(inplace=True)        
    decimal_places = 9  # Set the desired number of decimal places
    gamma_values = np.around(np.linspace(start, stop, n_points), decimals=decimal_places)

    for gamma in tqdm(gamma_values, desc="Processing gamma values"):
        df_integrated = process_dataframe(df, validation, gamma)
        df_train, df_test = train_test_split(df_integrated, test_size=0.2, random_state=42)
        
        RMSE_Energy_train_current = np.sqrt(mean_squared_error(df_train['trip'], df_train['Energy']))
        MAPE_Energy_train_current = mean_absolute_percentage_error(df_train['trip'] , df_train['Energy'])
        #RMSE_Energy_test_current = np.sqrt(mean_squared_error(df_test['trip'], df_test['Energy']))
        #MAPE_Energy_test_current = mean_absolute_percentage_error(df_test['trip'] , df_test['Energy'])
        parameter1_values.append(gamma)
        RMSE_Energy_train.append(RMSE_Energy_train_current)
        MAPE_Energy_train.append(MAPE_Energy_train_current)
        #RMSE_Energy_test.append(RMSE_Energy_test_current)
        #MAPE_Energy_test.append(MAPE_Energy_test_current)


    results = pd.DataFrame(list(zip(parameter1_values, RMSE_Energy_train, MAPE_Energy_train)),
                           columns=['parameter1_values', 'RMSE_Energy_train', 'MAPE_Energy_train'])
    results.to_csv((r'../../results/calibration-grid-search-BEB-oct2021-sep2022_01022024.csv'))
    print("--- %s seconds ---" % (time.time() - start_time))

    
calibrate_parameter((0.00000001,0.09, 100))