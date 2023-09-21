import gc
import pandas as pd
from gurobipy import Model, GRB, quicksum
import gc
import pandas as pd
from gurobipy import Model, GRB, quicksum
import time
import psutil
import os
import gurobipy as grb
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


start = time.time()

# Get the current process
process = psutil.Process(os.getpid())

# Get the total memory available in bytes
total_memory = psutil.virtual_memory().total

# Convert bytes to GB
total_memory_gb = total_memory / (1024 ** 3)

print(f'Total memory: {total_memory_gb} GB')

# Define function to report memory and CPU usage
def report_usage():
    # Get the percentage of CPU usage by this process
    cpu_percent = process.cpu_percent(interval=1)

    # Get the memory usage by this process
    memory_info = process.memory_info()
    memory_usage = memory_info.rss  # in bytes


##### Reade data and organize required dataframes
# Read dataframes of all-CDB, all-HEB, and all BEB with runs included
df_CDB = pd.read_csv(r'../../results/computed-fuel-rates-all-CDB.csv', low_memory=False)
df_HEB = pd.read_csv(r'../../results/computed-fuel-rates-all-HEB.csv', low_memory=False)
df_BEB = pd.read_csv(r'../../results/computed-fuel-rates-all-BEB.csv', low_memory=False)

# Find the date with the maximum number of unique trips
date_with_max_trips = df_CDB.groupby('Date')['TripKey'].nunique().idxmax()
max_trips = df_CDB.groupby('Date')['TripKey'].nunique().max()

# Filter dataframes by the day with max trips
df_CDB = df_CDB.loc[df_CDB['Date']==date_with_max_trips]
df_HEB = df_HEB.loc[df_HEB['Date']==date_with_max_trips]
df_BEB = df_BEB.loc[df_BEB['Date']==date_with_max_trips]

## Remove Trips with zero Energy consumption with CDB vehicles
# Find the TripKey values in df_CDB where sum of Energy column is zero when grouped by TripKey
tripkeys_to_remove = df_CDB.groupby('TripKey').filter(lambda group: group['Energy'].sum() == 0)['TripKey'].unique()

# Remove the rows from all three dataframes that match the TripKey values found above
df_CDB = df_CDB[~df_CDB['TripKey'].isin(tripkeys_to_remove)]
df_HEB = df_HEB[~df_HEB['TripKey'].isin(tripkeys_to_remove)]
df_BEB = df_BEB[~df_BEB['TripKey'].isin(tripkeys_to_remove)]

# Group the dataframes to capture first and last stop, start and end time and route for each trip
df_CDB = df_CDB.groupby(['TripKey', 'Date']).agg({
'Stop': ['first', 'last'],
'ServiceDateTime': ['min', 'max'],
'Route': 'first',
'Energy':'sum',
'dist':'sum',
'Powertrain':'first'
  }).reset_index()

df_HEB = df_HEB.groupby(['TripKey', 'Date']).agg({
'Stop': ['first', 'last'],
    'ServiceDateTime': ['min', 'max'],
    'Route': 'first',
    'Energy':'sum',
    'dist':'sum',
    'Powertrain':'first'
    }).reset_index()

df_BEB = df_BEB.groupby(['TripKey', 'Date']).agg({
    'Stop': ['first', 'last'],
    'ServiceDateTime': ['min', 'max'],
    'Route': 'first',
    'Energy':'sum',
    'dist':'sum',
    'Powertrain':'first'
    }).reset_index()

# Renaming columns
df_CDB.columns = ['_'.join(col).strip() for col in df_CDB.columns.values]
df_HEB.columns = ['_'.join(col).strip() for col in df_HEB.columns.values]
df_BEB.columns = ['_'.join(col).strip() for col in df_BEB.columns.values]
df_CDB = df_CDB.rename(columns={'TripKey_': 'TripKey'})
df_CDB = df_CDB.rename(columns={'Date_': 'Date'})
df_CDB = df_CDB.rename(columns={'Route_first': 'Route'})
df_CDB = df_CDB.rename(columns={'Energy_sum': 'Energy'})
df_CDB = df_CDB.rename(columns={'dist_sum': 'dist'})
df_CDB = df_CDB.rename(columns={'Powertrain_first': 'Powertrain'})
df_HEB = df_HEB.rename(columns={'TripKey_': 'TripKey'})
df_HEB = df_HEB.rename(columns={'Date_': 'Date'})
df_HEB = df_HEB.rename(columns={'Route_first': 'Route'})
df_HEB = df_HEB.rename(columns={'Energy_sum': 'Energy'})
df_HEB = df_HEB.rename(columns={'dist_sum': 'dist'})
df_HEB = df_HEB.rename(columns={'Powertrain_first': 'Powertrain'})
df_BEB = df_BEB.rename(columns={'TripKey_': 'TripKey'})
df_BEB = df_BEB.rename(columns={'Date_': 'Date'})
df_BEB = df_BEB.rename(columns={'Route_first': 'Route'})
df_BEB = df_BEB.rename(columns={'Energy_sum': 'Energy'})
df_BEB = df_BEB.rename(columns={'dist_sum': 'dist'})
df_BEB = df_BEB.rename(columns={'Powertrain_first': 'Powertrain'})



# Groupby to compute energy consumption for each unique vehicle, date, route, and trip key
# then, create the 'Diesel' column based on the condition for 'Powertrain'

# For df_CDB
energy_CDB = df_CDB.groupby(['Date', 'Route', 'TripKey']).agg({    
    'Stop_first': 'first',
    'Stop_last': 'last',
    'ServiceDateTime_min': 'min',
    'ServiceDateTime_max': 'max',
    'Energy': 'sum', 
    'dist': 'sum', 
    'Powertrain': 'first'}).reset_index()
energy_CDB['Diesel'] = (energy_CDB['Powertrain'].isin(['conventional', 'hybrid']) * energy_CDB['Energy'])
energy_CDB_dict = energy_CDB.set_index(['TripKey']).to_dict('index')
#print("energy_CDB_dict", energy_CDB_dict)
#energy_CDB_dict = {f"CDB_{key}": value for key, value in energy_CDB.set_index(['TripKey']).to_dict('index').items()}


# For df_HEB
energy_HEB = df_HEB.groupby(['Date', 'Route', 'TripKey']).agg({    
    'Stop_first': 'first',
    'Stop_last': 'last',
    'ServiceDateTime_min': 'min',
    'ServiceDateTime_max': 'max',
    'Energy': 'sum', 
    'dist': 'sum', 
    'Powertrain': 'first'}).reset_index()
energy_HEB['Diesel'] = (energy_HEB['Powertrain'].isin(['conventional', 'hybrid']) * energy_HEB['Energy'])
energy_HEB_dict = energy_HEB.set_index(['TripKey']).to_dict('index')
#energy_HEB_dict = {f"HEB_{key}": value for key, value in energy_HEB.set_index(['TripKey']).to_dict('index').items()}


# For df_BEB
energy_BEB = df_BEB.groupby(['Date', 'Route', 'TripKey']).agg({    
    'Stop_first': 'first',
    'Stop_last': 'last',
    'ServiceDateTime_min': 'min',
    'ServiceDateTime_max': 'max',
    'Energy': 'sum', 
    'dist': 'sum', 
    'Powertrain': 'first'}).reset_index()
energy_BEB['Diesel'] = (energy_BEB['Powertrain'].isin(['conventional', 'hybrid']) * energy_BEB['Energy'])
energy_BEB_dict = energy_BEB.set_index(['TripKey']).to_dict('index')


# Define parameters
#D = len(set(df_CDB['Date'].unique()))  # Create a set of unique dates
Y = 1  # Years in simulation (including year 0)
max_number_of_buses = 1000 # 213*4 (current numnumber of fleet*4, assuming buses are going to be replaced with electric at most with ratio of 1:4)

# Batery capacity of an electric bus
battery_cap=350 #kWh 

# Maximum daily charging capacity in year y
battery_values = [23]
M_cap = [23]


# Set of scenarios
#S = {'low-cap', 'mid-cap', 'high-cap'}
S = {'high-cap'}
# Define R and Rho
R = df_CDB['Route'].nunique()
Rho = max_trips

# The cost of purchasing a new bus
cost_inv = {
  ('C'): 0,    # Assuming no cost for existing CDB buses
  ('H'): 0.9,  # in million dollars
  ('B'): 1.3   # in million dollars
    }

# Max investment per scenario per year
C_max = {
       #    'low-cap': 7,  # in million dollars
       #      'mid-cap': 14,  # in million dollars
       'high-cap': 21  # in million dollars
       }

# The maximum yearly investment
M_inv = {
       (s, y): C_max[s] for y in range(Y) for s in S
       }

# Bus ranges
range_CDB= 93-10 # mean actual values in miles 
range_HEB= 110-10 # mean actual values in miles 
range_BEB= 55-10 # mean actual values in miles 
range_CDB_list = list(range(50, 150, 5))
range_HEB_list = list(range(60, 160, 5))
range_BEB_list = list(range(30, 130, 5))


# Total number of fleet from each powertrain in year 0
N = {
  ('C', 0): 189,
  ('H', 0): 9,
  ('B', 0): 15,
   }

# Now delete the DataFrame to free up memory
del df_CDB
del df_HEB
del df_BEB
gc.collect()


def optimize():
    # Create a model
    model = Model('Minimize fleet diesel consumption')

    # Enable logging and print progress
    #model.setParam('OutputFlag', 1)
    model.setParam('OutputFlag', 0)

    # Set solver parameters
    #model.setParam('MIPFocus', 1)  # This parameter lets control the  In. The options are: 1. Finds feasible solutions quickly. This is useful if the problem is difficult to solve and you're satisfied with any feasible solution. 2: Works to improve the best bound. This is useful when the best objective bound is poor. 3: Tries to prove optimality of the best solution found. This is useful if you're sure a nearly-optimal solution exists and you want the solver to focus on proving its optimality.
    #model.setParam('Heuristics', 0.5)  # Controls the effort put into MIP heuristics (range is 0 to 1). A higher value means more effort is put into finding solutions, but at the cost of slower overall performance. 
    model.setParam('Presolve', 0)  # This parameter controls the presolve level. Presolve is a phase during which the solver tries to simplify the model before the actual optimization takes place. A higher presolve level means the solver puts more effort into simplification, which can often reduce solving time. (-1: automatic (default) - Gurobi will decide based on the problem characteristics whether to use presolve or not.0: no presolve. 1: conservative presolve. 2: aggressive presolve.)
    #model.setParam('MIPGap', 0.01) # This parameter sets the relative gap for the MIP search termination. The solver will stop as soon as the relative gap between the lower and upper objective bound is less than this value. The lower this value, the closer to optimality the solution has to be before the solver stops.  
    model.setParam('Threads', 64)  # Set number of threads to be used for parallel processing.


    # Additional keys for buses and years
    bus_keys = range(max_number_of_buses)
    year_keys = range(Y)


    # Decision variables
    keys_CDB = list(energy_CDB_dict.keys())
    keys_HEB = list(energy_HEB_dict.keys())
    keys_BEB = list(energy_BEB_dict.keys())



    # Decision variables which include two additional indices for buses (i) and years (y)
    x_CDB = model.addVars(S, year_keys, keys_CDB, vtype=GRB.BINARY, name='x_CDB')
    x_HEB = model.addVars(S, year_keys, keys_HEB, vtype=GRB.BINARY, name='x_HEB')
    x_BEB = model.addVars(S, year_keys, keys_BEB, vtype=GRB.BINARY, name='x_BEB')

    # Define y_CDB, y_HEB, and y_BEB as the number of each type of bus at each year under each scenario
    y_CDB = model.addVars(S, year_keys, vtype=GRB.INTEGER, name='y_CDB')
    y_HEB = model.addVars(S, year_keys, vtype=GRB.INTEGER, name='y_HEB')
    y_BEB = model.addVars(S, year_keys, vtype=GRB.INTEGER, name='y_BEB')

    # Define decision variables delta to show number of new buses
    model.setObjective(
        (quicksum([energy_CDB_dict[key]['Diesel'] * x_CDB[s, y, key] for s in S for key in keys_CDB for y in year_keys]) +
         quicksum([energy_HEB_dict[key]['Diesel'] * x_HEB[s, y, key] for s in S for key in keys_HEB for y in year_keys]) +
         quicksum([energy_BEB_dict[key]['Diesel'] * x_BEB[s, y, key] for s in S for key in keys_BEB for y in year_keys])),
        GRB.MINIMIZE
        )

    ## Define Constraints
    # Constraint 1: Linking the number of each type of bus at each year variable with trip assignment variables
    ### ensure the number of buses of type X (y_X[s, y]) is at least the total distance traveled by all trips assigned to buses of type X divided by the range of the bus (rounded up to ensure all trips are covered).
    # For CDB buses
    for s in S:
        for y in year_keys:
            total_distance_CDB = quicksum(energy_CDB_dict[key]['dist'] * x_CDB[s, y, key] for key in keys_CDB)
        
            model.addConstr(
                y_CDB[s, y] * range_CDB >= total_distance_CDB, 
                name=f"C1_numberofCDB_ge_{s}_{y}"
                )
            model.addConstr(
                y_CDB[s, y] * range_CDB <= total_distance_CDB + range_CDB,
                name=f"C1_numberofCDBs_le_{s}_{y}"
                )

    # For HEB buses
    for s in S:
        for y in year_keys:
            total_distance_HEB = quicksum(energy_HEB_dict[key]['dist'] * x_HEB[s, y, key] for key in keys_HEB)
        
            model.addConstr(
                y_HEB[s, y] * range_HEB >= total_distance_HEB, 
                name=f"C1_numberofHEBs_ge_{s}_{y}"
                )
            model.addConstr(
                y_HEB[s, y] * range_HEB <= total_distance_HEB + range_HEB,
                name=f"C1_numberofHEBs_le_{s}_{y}"
                )

    # For BEB buses
    for s in S:
        for y in year_keys:
            total_distance_BEB = quicksum(energy_BEB_dict[key]['dist'] * x_BEB[s, y, key] for key in keys_BEB)
        
            model.addConstr(
                y_BEB[s, y] * range_BEB >= total_distance_BEB, 
                name=f"C1_numberofBEBs_ge_{s}_{y}"
                )
            model.addConstr(
                y_BEB[s, y] * range_BEB <= total_distance_BEB + range_BEB,
                name=f"C1_numberofBEBs_le_{s}_{y}"
                )

    # Constraint 2: Each trip is assigned to exactly one bus powertrain
    #unique_keys = set(keys_CDB) | set(keys_HEB) | set(keys_BEB)  # Union of all keys
    for s in S:
        for y in year_keys:
            #for key in unique_keys:
                for key in keys_CDB:
                    model.addConstr(
                        x_CDB[s, y, key] + x_HEB[s, y, key] + x_BEB[s, y, key] == 1,
                        name=f"C2_Trip{key}_S{s}_Y{y}: Each trip is assigned to only one powertrain"
                        )
    
    # Constraint 3: Maximum daily charging capacity
    for s in S:
        for y in year_keys:
            total_BEB = y_BEB[s,y]
            model.addConstr(total_BEB <= M_cap[y], name=f"C3: daily charging capacity_{y}_{s}")
            

    # Constraint 4: Maximum yearly investment
    
    # Initial values for year 0
    initial_fleet_CDB = 189
    initial_fleet_HEB = 9
    initial_fleet_BEB = 15
    
    for s in S:
        for y in year_keys:
            z_HEB = model.addVar(vtype=GRB.BINARY, name=f"z_HEB_{s}_{y}")
            z_BEB = model.addVar(vtype=GRB.BINARY, name=f"z_BEB_{s}_{y}")

            positive_delta_fleet_HEB = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"positive_delta_HEB_{s}_{y}")
            positive_delta_fleet_BEB = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"positive_delta_BEB_{s}_{y}")

            # Adjustments for year 0 to use initial values
            if y == 0:
                delta_fleet_HEB = y_HEB[s, y] - initial_fleet_HEB
                delta_fleet_BEB = y_BEB[s, y] - initial_fleet_BEB
            else:
                delta_fleet_HEB = y_HEB[s, y] - y_HEB[s, y-1]
                delta_fleet_BEB = y_BEB[s, y] - y_BEB[s, y-1]

            big_M = 1e6
            model.addConstr(positive_delta_fleet_HEB >= delta_fleet_HEB)
            model.addConstr(positive_delta_fleet_HEB <= delta_fleet_HEB + big_M * (1 - z_HEB))
         
            model.addConstr(positive_delta_fleet_BEB >= delta_fleet_BEB)
            model.addConstr(positive_delta_fleet_BEB <= delta_fleet_BEB + big_M * (1 - z_BEB))
         
            model.addConstr(delta_fleet_HEB <= big_M * z_HEB)
            model.addConstr(delta_fleet_BEB <= big_M * z_BEB)
 
            HEB_investment = positive_delta_fleet_HEB * cost_inv[('H')]
            BEB_investment = positive_delta_fleet_BEB * cost_inv[('B')]
 
            total_investment = HEB_investment + BEB_investment
            model.addConstr(total_investment <= M_inv[s, y], name=f"C4: Max yearly investment_{y}_{s}")
 

    # Constraint 5: Total number of buses (y) (summed over all powertrain) per year cannot exceed 1000
    for s in S:
        for y in year_keys:
            model.addConstr(
                y_CDB[s, y] + y_HEB[s, y] + y_BEB[s, y] <= max_number_of_buses,
                name=f"C5: TotalFleetSize_{y}_{s}"
                )



    # Print model statistics
    model.update()
        
    # Tuning and Optimization
    model.tune()
    model.optimize()

    # Prepare dictionaries of coeesicients to save
    coeff_dict_CDB = {(s, y, key): energy_CDB_dict[key]['Diesel'] for s in S for key in keys_CDB for y in year_keys if key in energy_CDB_dict}
    coeff_dict_HEB = {(s, y, key): energy_HEB_dict[key]['Diesel'] for s in S for key in keys_HEB for y in year_keys if key in energy_HEB_dict}
    coeff_dict_BEB = {(s, y, key): energy_BEB_dict[key]['Diesel'] for s in S for key in keys_BEB for y in year_keys if key in energy_BEB_dict}

    # Combine all the dictionaries into a dataframe
    coeff_df = pd.DataFrame(list(coeff_dict_CDB.items()) + list(coeff_dict_HEB.items()) + list(coeff_dict_BEB.items()), columns=['Variable', 'Coefficient'])

    vars = model.getVars()

    # Create DataFrame directly from the variables and their values
    if model.status == grb.GRB.Status.OPTIMAL:
        df = pd.DataFrame({"Variable": [v.varName for v in vars], "Value": [v.X for v in vars]})
        #    df.to_csv(r'../../results/balanced-transition-highcap-optimized-variables.csv', index=False)
    else:
        print('No solution found')

    # Identify the smallest infeasible subset of constraints and variable bounds
    if model.status == grb.GRB.INFEASIBLE:
        model.computeIIS()
        model.write("model.ilp")

    # Generate a feasibility report
    if model.status == grb.GRB.INFEASIBLE:
        model.feasRelaxS(0, False, False, True)
        model.optimize()

    # Get optimal objective value
    optimal_value = model.ObjVal

    # Add this value to DataFrame
    df_objective = pd.DataFrame({"Year": year_keys, "Objective_Value": [optimal_value]*len(year_keys)})

    # Print objective value

    df = pd.DataFrame({"Variable": [v.varName for v in vars], "Value": [v.X for v in vars]})

    # Save the DataFrame to a CSV file
    #df.to_csv(r'../../results/highcap-SQ-optimized-variables.csv', index=False)
    end = time.time()
    return(df)


### Read busiest day info and organize
busiest_day= pd.read_csv(r'../../results/busiest-day-trips-info.csv', low_memory=False) 
busiest_day['Route'] = busiest_day['Route'].str.replace(r'\(X\)', '', regex=True) # replace (X)
busiest_day['Route'] = busiest_day['Route'].str.replace('X', '', regex=False) # replace X


# Read coeeficients
h_PP_coefficients = pd.read_csv(r'../../results/optimization-coefficients.csv', low_memory=False)


# Read energy model output
SQ = pd.read_csv(r'../../results/computed-fuel-rates-oct2021-sep2022.csv', low_memory=False)
SQ_filtered = SQ.loc[SQ['Date']=='2021-10-29'].copy()
SQ_filtered['Diesel'] = (SQ_filtered['Powertrain'].isin(['conventional', 'hybrid']) * SQ_filtered['Energy'])

def compute_energy():
    h_PP_variables = optimize()
    # Extract attributes from h_PP_variables
    h_PP_variables[['Powertrain', 'Scenario', 'Year', 'Trip']] = h_PP_variables['Variable'].str.extract(r'x_(\w+)\[(\w+-cap),(\d+),(\d+)\]')
    h_PP_variables = h_PP_variables.dropna()
    h_PP_variables['Year'] = h_PP_variables['Year'].astype(int).copy()
    # Convert 'Trip' columns to string type for both dataframes (ensure consistency)
    h_PP_variables['Trip'] = h_PP_variables['Trip'].astype(str)
    busiest_day['TripKey'] = busiest_day['TripKey'].astype(str)


    # Extract attributes from h_PP_coefficients
    h_PP_coefficients[['Scenario', 'Year', 'Trip']] = h_PP_coefficients['Variable'].str.extract(r"\('(\w+-\w+)', (\d+), (\d+)\)")
    #h_PP_coefficients = h_PP_coefficients.dropna()
    h_PP_coefficients['Year'] = h_PP_coefficients['Year'].astype(int)
    
    # Convert 'Trip' columns to string type for both dataframes (ensure consistency)
    h_PP_variables['Trip'] = h_PP_variables['Trip'].astype(str)
    h_PP_coefficients['Trip'] = h_PP_coefficients['Trip'].astype(str)
    
    # Merge the dataframes on matching values
    merged_df = h_PP_variables.merge(h_PP_coefficients[['Powertrain','Year', 'Trip', 'Coefficient']],
                                on=['Powertrain','Year', 'Trip',],
                                how='left')

    # Update h_PP_variables
    h_PP_variables = merged_df
    
    # Merge the dataframes on matching values
    merged_df = h_PP_variables.merge(busiest_day[['TripKey', 'Route', 'Stop_first', 'Stop_last', 'ServiceDateTime_min', 'ServiceDateTime_max', 'dist']], 
                                 left_on='Trip', 
                                 right_on='TripKey', 
                                 how='left')

    # Drop the TripKey column if not needed
    merged_df.drop(columns='TripKey', inplace=True)

    # Update h_PP_variables
    h_PP_variables = merged_df

    h_PP_variables['ServiceDateTime_min'] = pd.to_datetime(h_PP_variables['ServiceDateTime_min'])
    h_PP_variables['ServiceDateTime_max'] = pd.to_datetime(h_PP_variables['ServiceDateTime_max'])
    #h_PP_variables['Value'] = pd.to_numeric(h_PP_variables['Value'])
    return(h_PP_variables)


def compute_error(r_CDB, r_HEB, r_BEB):
    variables_df = compute_energy()
    variables_df['Diesel'] = variables_df['Value'] * variables_df['Coefficient']
    electrification_framework_total_diesel = variables_df['Diesel'].sum()
    energy_framework_total_diesel = SQ_filtered['Diesel'].sum()
    error_current = energy_framework_total_diesel - electrification_framework_total_diesel
    return {'error': error_current, 'CDB_range': r_CDB, 'HEB_range': r_HEB, 'BEB_range': r_BEB}

results = []

# The total number of iterations to be performed, to set the total count for tqdm
total_iterations = len(range_CDB_list) * len(range_HEB_list) * len(range_BEB_list)

# Wrap the futures in the tqdm function for the progress bar
with ThreadPoolExecutor() as executor, tqdm(total=total_iterations, desc="Processing", unit="task") as pbar:
    futures = []
    for r_CDB in range_CDB_list:
        for r_HEB in range_HEB_list:
            for r_BEB in range_BEB_list:
                futures.append(executor.submit(compute_error, r_CDB, r_HEB, r_BEB))
    
    for future in futures:
        results.append(future.result())
        pbar.update(1)  # Update the progress bar by one step for each completed task

error_df = pd.DataFrame(results)
error_df.to_csv(r'../../results/calibrate-ranges.csv', index=False)

min_error_row = error_df.iloc[error_df['error'].abs().idxmin()]
print(min_error_row)    