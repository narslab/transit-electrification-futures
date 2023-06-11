import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize


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
a0_heb = p.alpha_0_heb
a1_heb = p.alpha_1_heb
a2_cdb = p.alpha_2_cdb
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
    v = df.speed
    a = df.acc
    gr = df.grade
    m = (df.Vehicle_mass+df.Onboard*179)*0.453592 # converts lb to kg
    P_t = (1/float(3600*eta_d_dis))*((1./25.92)*rho*C_D*C_h*A_f_d*v*v + m*g*C_r*(c1*v + c2)/1000 + 1.2*m*a+m*g*gr)*v
    return P_t


# Read trajectories df
df_trajectories = pd.read_csv(r'../../data/tidy/large/trajectories-mapped-powertrain-weight-grade-oct2021-sep2022.csv', delimiter=',', skiprows=0, low_memory=False)
df_trajectories.rename(columns={"VehiclWeight(lb)": "Vehicle_mass"}, inplace=True)
df_trajectories['Date']=pd.to_datetime(df_trajectories['Date'])
df_trajectories.speed = df_trajectories.speed *1.60934 # Convert from mph to km/h
#df_trajectories = df_trajectories.fillna(0)

# Subsetting data frame for "Conventional", "hybrid" buses
df_conventional=df_trajectories.loc[df_trajectories['Powertrain'] == 'conventional'].copy()
df_hybrid=df_trajectories.loc[df_trajectories['Powertrain'] == 'hybrid'].copy()

# read validation df
df_validation = pd.read_csv(r'../../data/tidy/fuel-tickets-oct2021-sep2022.csv', delimiter=',', skiprows=0, low_memory=False)
df_validation=df_validation.loc[df_validation['Qty']>0]
#df_validation.sort_values(by=['Equipment ID','Transaction Date'], inplace=True)
df_validation.drop(['Unnamed: 0'], axis=1, inplace=True)
df_validation.rename(columns={"Transaction Date": "ServiceDateTime","Equipment ID":"Vehicle"}, inplace=True)
df_validation["dist"] = np.nan
df_validation["Energy"] = np.nan
df_validation['ServiceDateTime'] = pd.to_datetime(df_validation['ServiceDateTime'])
df_validation.sort_values(by=['Vehicle','ServiceDateTime'], inplace=True)
df_validation=df_validation.reset_index()

### Map powertrain in the validation dataset
df2 = pd.read_csv(r'../../data/tidy/vehicles-summary.csv', delimiter=',', skiprows=0, low_memory=False)
mydict = df2.groupby('Type')['Equipment ID'].agg(list).to_dict()
d = {val:key for key, lst in mydict.items() for val in lst}
df_validation['Powertrain'] = df_validation['Vehicle'].map(d)

# Subsetting data frame for "Conventional", "hybrid" buses
df_conventional_val=df_validation.loc[df_validation['Powertrain'] == 'conventional'].copy()
df_hybrid_val=df_validation.loc[df_validation['Powertrain'] == 'hybrid'].copy()

# Define the two pieces of the function
def piece1(params, x):
    a0, a1, a2 = params
    return a0 + a1 * x + a2 * x**2

def piece2(params, x, df):
    a0, gamma_heb = params
    return a0 + eta_batt * np.exp(-(gamma_heb / abs(df.acc))) / (eta_m * eta_d_heb) * x

# Define cost functions for minimize
def cost_function_piece1(params, x, y):
    return ((y - piece1(params, x)) ** 2).sum()

def cost_function_combined(params, x, y, df):
    #return ((y - (piece1(params[:3], x) + piece2(params[3:], x, df))) ** 2).sum()
    #print(params)
    mask = df.acc >= 0
    cost_piece1 = ((y[mask] - piece1(params[:3], x[mask])) ** 2).sum()
    cost_piece2 = ((y[~mask] - piece2(params[3:], x[~mask], df[~mask])) ** 2).sum()
    return cost_piece1 + cost_piece2


def calibrate_parameter(hybrid=False):
    # Create separate dataframes for hybrid and conventional vehicles
    if hybrid:
        df = df_hybrid.copy()
        validation = df_hybrid_val.copy()
    else:
        df = df_conventional.copy()
        validation = df_conventional_val.copy()

    # Calculate energy consumption for each trip
    df['power'] = power_d(df, hybrid=hybrid)
    
    # convert fuel rates to energy
    df['Energy']=df['power']*df['time_delta_in_seconds']/3.78541

    # Convert ServiceDateTime column to datetime data type
    df['ServiceDateTime'] = pd.to_datetime(df['ServiceDateTime'])
    validation['ServiceDateTime'] = pd.to_datetime(validation['ServiceDateTime'])

    # Sort dataframes by vehicle and service datetime
    df.sort_values(by=['Vehicle', 'ServiceDateTime'], inplace=True)
    validation.sort_values(by=['Vehicle', 'ServiceDateTime'], inplace=True)
    validation.reset_index(drop=True, inplace=True)
   
  
    # Initialize dist and Energy columns in validation with 0
    validation['dist'] = 0
    validation['Energy'] = 0

    # Calculate the sum of dist and Energy values between two consecutive ServiceDateTime in validation
    vehicle_groups = df.groupby('Vehicle')
    for index in tqdm(validation.index, desc="Processing", unit="rows"):
        row = validation.loc[index]
        if index > 0 and row['Vehicle'] == validation.iloc[index - 1]['Vehicle']:
            lower_bound = validation.iloc[index - 1]['ServiceDateTime']
            upper_bound = row['ServiceDateTime']
            vehicle_df = vehicle_groups.get_group(row['Vehicle'])
            mask = (vehicle_df['ServiceDateTime'] > lower_bound) & (vehicle_df['ServiceDateTime'] <= upper_bound)
            validation.at[index, 'dist'] = vehicle_df.loc[mask, 'dist'].sum()
            validation.at[index, 'Energy'] = vehicle_df.loc[mask, 'Energy'].sum()
            validation.at[index,'acc'] = vehicle_df.loc[mask, 'acc'].mean()


    # Drop rows with missing or invalid values in 'Energy' and 'Qty' columns
    df_integrated_clean = validation.dropna(subset=['Energy', 'Qty'])
    df_integrated_clean = df_integrated_clean[~(df_integrated_clean['Energy'].isin([np.nan, np.inf, -np.inf]))]
    df_integrated_clean = df_integrated_clean[~(df_integrated_clean['Qty'].isin([np.nan, np.inf, -np.inf]))]

    # set the time difference to zero for the first row of each Vehicle group
    df_integrated_clean.loc[df_integrated_clean.groupby('Vehicle')['ServiceDateTime'].head(1).index, 'TimeDiff'] = 0

    # remove rows with zero values in Energy, Qty, or TimeDiff columns
    df_integrated_clean = df_integrated_clean[~(df_integrated_clean[['Energy', 'Qty', 'TimeDiff']] == 0).any(axis=1)]

    # Sort the dataframe by Vehicle and Timestamp
    df_integrated_clean = df_integrated_clean.sort_values(['Vehicle', 'ServiceDateTime'])

    x_predicted = df_integrated_clean['Energy'].copy()
    y_observed = df_integrated_clean['Qty'].copy()
    
    # Perform the least squares calibration using curve_fit for each piece
    #coefficients_piece1, _ = curve_fit(piece1, x_data_piece1, y_data_piece1)    
    if hybrid == False:
        # Perform the optimization using minimize for piece1
        initial_guess = [a0_cdb, a1_cdb, a2_cdb]
        result = minimize(cost_function_piece1, initial_guess, args=(x_predicted, y_observed), method='BFGS')
    else:
        # Perform the optimization using minimize for combined function
        initial_guess = [a0_heb, a1_heb, a2_heb, a0_heb, gamma_heb]
        #print(df_integrated_clean.columns)
        result = minimize(cost_function_combined, initial_guess, args=(x_predicted, y_observed, df_integrated_clean), method='BFGS')
    
    coefficients = result.x        
    return coefficients


#hybrid
a0_heb, a1_heb, a2_heb, a0_heb, gamma_heb = calibrate_parameter(hybrid=True)
print("a0_heb:", a0_heb, "a1_heb:", a1_heb, "a2_heb:", a2_heb, "a0_heb:", a0_heb, "gamma_heb:", gamma_heb)


#conventional
a0_cdb, a1_cdb, a2_cdb = calibrate_parameter(hybrid=False)
print("a0_cdb:", a0_cdb, "a1_cdb:", a1_cdb, "a2_cdb:", a2_cdb)



