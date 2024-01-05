import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor

f = open('params-oct2021-sep2022-new-equation-12212023.yaml')
parameters = yaml.safe_load(f)
f.close()

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
def power(df_input, eta_d_beb, hybrid=False, electric=False):
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
    v = df.Speed
    a = df.Acceleration
    G = df.grade
    m = (df.Vehicle_mass+df.Onboard*179)*0.453592 # converts lb to kg
    H = df.elevation/1000 # df.elevation is in meters and we need to convert it to km 
    C_h = 1 - (0.085*H)
    #P_t = (v/float(1000*eta_d))*(rho*C_D*C_h*A*v*v +m(g(C_r1*v+C_r2+G)+1.2*a))
    P_t = (v / (1000 * eta_d)) * (rho * C_D * C_h * A * v * v + m * (g * (C_r1 * v + C_r2 + G) + 1.2 * a))

    return P_t


# Define Energy consumption function for electric vehicle
def energyConsumption_e(df_input, gamma_beb, eta_m, eta_d_beb, electric=True):
	# Estimates energy consumed (KWh)     
    df = df_input
    t = df.time_delta_in_seconds
    P_t = power(df_input, eta_m, electric)
    print("gamma_beb", gamma_beb)
    eta_rb = df.Acceleration.apply(lambda a: 1 if a >= 0 else np.exp(-(gamma_beb/abs(a))))
    E_t = t * P_t * eta_rb * eta_batt /(eta_m*3600)
    return eta_rb,E_t

# Read computed fuel rates
df_trajectories = pd.read_csv(r'../../data/tidy/large/trajectories-mapped-powertrain-weight-grade-oct2021-sep2022.csv', delimiter=',', skiprows=0, low_memory=False)
df_trajectories.rename(columns={"VehiclWeight(lb)": "Vehicle_mass"}, inplace=True)
df_trajectories['Date']=pd.to_datetime(df_trajectories['Date'])
df_trajectories.speed = df_trajectories.speed *0.44704 # Convert from mph to m/s
df_trajectories = df_trajectories.fillna(0)

# Subsetting data frame for "Conventional", "hybrid", and "electric" buses
df_beb=df_trajectories.loc[df_trajectories['Powertrain'] == 'electric'].copy()
del df_trajectories

# Trimming the data
df_beb.rename(columns={"speed": "Speed", "acc": "Acceleration"}, inplace=True)
quantile_1 = df_beb['Acceleration'].quantile(0.0001)
quantile_99 = df_beb['Acceleration'].quantile(0.9999)
df_beb = df_beb[(df_beb['Acceleration'] >= quantile_1) & (df_beb['Acceleration'] <= quantile_99)]

# read validation df
df_validation = pd.read_excel(r'../../data/tidy/BEB-validation.xlsx')
df_validation.rename(columns={"Transaction Date": "ServiceDateTime","Equipment ID":"Vehicle"}, inplace=True)
df_validation['ServiceDateTime'] = pd.to_datetime(df_validation['ServiceDateTime'])

def process_dataframe(df, validation, gamma_beb, eta_m, eta_d_beb):
    df_new = df.copy()
    validation_new = validation.copy()

    eta_rb,df_new['Energy'] = energyConsumption_e(df, gamma_beb, eta_m, eta_d_beb)
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
    
    df_integrated['actual_mpg']=df_integrated['dist']/df_integrated['trip']
    
    # Calculate the 5st and 95th percentiles
    percentile_5 = df_integrated['actual_mpg'].quantile(0.05)
    percentile_95 = df_integrated['actual_mpg'].quantile(0.95)

    # Filter the DataFrame
    df_integrated = df_integrated[(df_integrated['actual_mpg'] >= percentile_5) & (df_integrated['actual_mpg'] <= percentile_95)]

    return eta_rb,df_integrated

# ## Not parallel version
def calibrate_parameter(args):
      start_gamma, stop_gamma, n_points_gamma, start_eta_m, stop_eta_m, n_points_eta_m, start_eta_d_beb, stop_eta_d_beb, n_points_eta_d_beb = args
      start_time = time.time()
      #eta_rbs = []
      parameter1_values = []
      parameter2_values = []
      parameter3_values = []
      RMSE_Energy_train = []
      MAPE_Energy_train = []

      df = df_beb.copy()
      validation = df_validation.copy()
      validation.reset_index(inplace=True)        
      decimal_places = 5  # Set the desired number of decimal places
      gamma_values = np.around(np.linspace(start_gamma, stop_gamma, n_points_gamma), decimals=decimal_places)
      eta_m_values = np.around(np.linspace(start_eta_m, stop_eta_m, n_points_eta_m), decimals=decimal_places)
      eta_d_beb_values = np.around(np.linspace(start_eta_d_beb, stop_eta_d_beb, n_points_eta_d_beb), decimals=decimal_places)

      for gamma in tqdm(gamma_values, desc="Processing gamma values"):
          for eta_m in tqdm(eta_m_values, desc="Processing eta_m values"):
              for eta_d_beb in tqdm(eta_d_beb_values, desc="Processing eta_d_beb values"):
                  eta_rb, df_integrated = process_dataframe(df, validation, gamma, eta_m, eta_d_beb)
                  print(eta_rb)
                  df_train, df_test = train_test_split(df_integrated, test_size=0.2, random_state=42)
                
                  RMSE_Energy_train_current = np.sqrt(mean_squared_error(df_train['trip'], df_train['Energy']))
                  MAPE_Energy_train_current = mean_absolute_percentage_error(df_train['trip'] , df_train['Energy'])
                  #eta_rbs.append(eta_rb)
                  parameter1_values.append(gamma)
                  parameter2_values.append(eta_m)
                  parameter2_values.append(eta_d_beb)
                  RMSE_Energy_train.append(RMSE_Energy_train_current)
                  MAPE_Energy_train.append(MAPE_Energy_train_current)


      results = pd.DataFrame(list(zip(parameter1_values, parameter2_values, parameter3_values, RMSE_Energy_train, MAPE_Energy_train)),
                             columns=['parameter1_values','parameter2_values','parameter3_values', 'RMSE_Energy_train', 'MAPE_Energy_train'])
      results.to_csv((r'../../results/calibration-grid-search-BEB-oct2021-sep2022_01052024.csv'))
      print("--- %s seconds ---" % (time.time() - start_time))

    
calibrate_parameter((0.001,2, 10, 0.9,0.99, 5, 0.9,0.99, 5))

