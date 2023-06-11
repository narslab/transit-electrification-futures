import yaml
import pandas as pd
import numpy as np
import gc   # For manual garbage collection


class vehicleParams():
    def __init__(self, **entries):
        self.__dict__.update(entries)

def compute_energy(parameters_filename, df_input, output_csv_filename):
    f = open(parameters_filename)
    parameters = yaml.safe_load(f)
    f.close()

    df = df_input
    df.speed = df.speed *1.60934 # takes speed in km/h (Convert from mph to km/h)
    df.rename(columns={"speed": "Speed", "acc": "Acceleration", "VehiclWeight(lb)": "Vehicle_mass"}, inplace=True)
    df = df.fillna(0)

    df_conventional=df.loc[df['Powertrain'] == 'conventional'].copy()
    df_hybrid=df.loc[df['Powertrain'] == 'hybrid'].copy()
    df_electric=df.loc[df['Powertrain'] == 'electric'].copy()

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
    eta_batt = p.battery_efficiency
    eta_m = p.motor_efficiency
    a0_cdb = p.alpha_0_cdb
    a1_cdb = p.alpha_1_cdb
    a0_heb = p.alpha_0_heb
    a1_heb = p.alpha_1_heb
    a2_cdb = p.alpha_2_cdb
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
        v = df.Speed
        a = df.Acceleration
        gr = df.grade
        m = (df.Vehicle_mass+df.Onboard*179)*0.453592 # converts lb to kg
        P_t = (1/float(3600*eta_d_dis))*((1./25.92)*rho*C_D*C_h*A_f_d*v*v + m*g*C_r*(c1*v + c2)/1000 + 1.2*m*a+m*g*gr)*v
        return P_t

    # Define fuel rate function for diesel vehicle
    def fuelRate_d(df_input, hybrid=False):
        # Estimates fuel consumed (liters per second) 
        if hybrid:
            a0 = float(a0_heb) 
            a1 = float(a1_heb)
            a2 = float(a2_heb)
            P_t = power_d(df_input, hybrid=True)     
            positive_pt_indices = P_t >= 0
            rg_factor = np.zeros_like(P_t)
            rg_factor[positive_pt_indices] = (a0 + eta_batt/(eta_m*eta_d_beb)*np.exp(-(gamma_heb/abs(df_input['Acceleration'][positive_pt_indices])))) * P_t[positive_pt_indices]
            FC_t = np.where(P_t >= 0, a0 + a1*P_t + a2*P_t*P_t, rg_factor)        
            
        else:
            a0 = float(a0_cdb)
            a1 = float(a1_cdb)
            a2 = float(a2_cdb)
            P_t = power_d(df_input, hybrid=False)
            FC_t = np.where(P_t >= 0, a0 + a1*P_t + a2*P_t*P_t, a0)  

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
        E_t = (FC_t * t)/3.78541
        return E_t

    # Define power function for electric vehicle
    def power_e(df_input):
        df = df_input
        v = df.Speed
        a = df.Acceleration
        gr = df.grade
        m = df.Vehicle_mass+(df.Onboard*179)*gamma_beb # converts lb to kg
        factor = df.Acceleration.apply(lambda a: 1 if a >= 0 else np.exp(-(0.0411/abs(a))))
        P_t = factor*(eta_batt/(eta_m*eta_d_beb))*(1/float(3600*eta_d_beb))*((1./25.92)*rho*C_D*C_h*A_f_beb*v*v + m*g*C_r*(c1*v + c2)/1000 + 1.1*m*a+m*g*gr)*v
        return P_t

    # Define Energy consumption function for electric vehicle
    def energyConsumption_e(df_input):
        # Estimates energy consumed (KWh)     
        df = df_input
        t = df.time_delta_in_seconds/3600
        P_t = power_e(df_input)
        E_t = P_t * t
        return E_t

    # Compute energy consumption for "Conventional", "hybrid" and "electric" buses
    df_conventional['Energy']=energyConsumption_d(df_conventional)
    df_hybrid['Energy']=energyConsumption_d(df_hybrid, hybrid=True)
    df_electric['Energy']=energyConsumption_e(df_electric)

    # Merge subset dataframes 
    df_final=pd.concat([df_conventional, df_hybrid, df_electric])

    # Delete the no longer needed dataframes and perform garbage collection
    del df_conventional, df_hybrid, df_electric
    gc.collect()

    # Sort dataframe
    df_final.sort_values(by=['Vehicle','ServiceDateTime'], ascending=True, inplace=True)

    df_final.to_csv(output_csv_filename)

    return df_final
