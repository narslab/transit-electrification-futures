# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:39:25 2022

@author: Mahsa
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


f = open('params.yaml')
parameters = yaml.safe_load(f)
f.close()

class vehicleParams:
	def __init__(self, **entries):
		self.__dict__.update(entries)


def testCycleEPA(profile): # input in mph
	df = pd.read_table('{0}-profile.txt'.format(profile),skiprows=2, names=['Time', 'Speed'])
	df.Speed = df.Speed*1.60934 # takes speed in km/h (Convert from mph to km/h)
	df['Acceleration'] = df.Speed.diff()*0.27777777  #takes accl in m/s^2
	df = df.fillna(0)
	return df


p = vehicleParams(**parameters)
rho = p.air_density
C_D = p.drag_coefficient
A_f = p.frontal_area
m = p.vehicle_mass
g = p.gravitational_acceleration
C_r = p.rolling_coefficient
c1 = p.rolling_resistance_coef1
c2 = p.rolling_resistance_coef2
eta_d = p.driveline_efficiency
P_mfo = p.idling_mean_fuel_pressure
omega = p.idling_speed
d = p.engine_displacement
Q = p.fuel_lower_heating_value
N = p.number_of_engine_cylinders
FE_city_p = p.fuel_economy_city
FE_hwy_p = p.fuel_economy_hwy
eps = p.epsilon

def power(profile):
	df = testCycleEPA(profile)
	v = df.Speed
	a = df.Acceleration
	P_t = (1/float(3600*eta_d))*((1./25.92)*rho*C_D*A_f*v*v + m*g*C_r*(c1*v + c2)/1000 + 1.04*m*a)*v
	return P_t

def powerCoefs():
	hwyDist = 16.512 # km
	cityDist = 17.767 # km
	mpg2kpl = 0.425144 #km per liter per mpg
	P_city = power('ftp').sum()
	P_hwy  = power('hwfet').sum()
	P2_city = (power('ftp')*power('ftp')).sum()
	P2_hwy = (power('hwfet')*power('hwfet')).sum()
	T_city = len(power('ftp'))
	T_hwy  = len(power('hwfet'))
	#FE_city = FE_city_p
	#FE_hwy = FE_hwy_p
	#FE_city = 1./( (1.18053/FE_city_p)-0.003259) # conversion for reporting discrepancy 
	#FE_hwy = 1./((1.3466/FE_hwy_p)-0.001376) # conversion for reporting discrepancy
	FE_city = 1.18053/( (1./FE_city_p)-0.003259) # conversion for reporting discrepancy 
	FE_hwy = 1.3466/((1./FE_hwy_p)-0.001376) # conversion for reporting discrepancy
	F_city = cityDist/float(mpg2kpl*FE_city)
	F_hwy = hwyDist/float(mpg2kpl*FE_hwy)

	alpha_0 = np.maximum((P_mfo*omega*d)/(22164*Q*N),( (F_city-F_hwy*(P_city/P_hwy)) - eps*(P2_city - P2_hwy*(P_city/P_hwy)) )/(T_city-T_hwy*(P_city/P_hwy)))
	alpha_2 = ( (F_city - F_hwy*(P_city/P_hwy)) - alpha_0*(T_city - T_hwy*(P_city/P_hwy)) )/ (P2_city - P2_hwy*(P_city/P_hwy)) 	
	alpha_1 = (F_hwy - T_hwy*alpha_0 - P2_hwy*alpha_2)/float(P_hwy)

	print ("alpha_0:", alpha_0)
	print ("alpha_1:", alpha_1)
	print ("alpha_2:", alpha_2)

	return alpha_0, alpha_1, alpha_2


nyc = testCycleEPA('nyc')

def fuelRate(profile):
	"""
	Estimates fuel consumed (liters per second) based on input speed profile
	Input must be dataframe with columns "Time", "Speed", and "Acceleration"
	"""
	a0, a1, a2 = powerCoefs()
	P_t = power(profile)
	FC_t = P_t.apply(lambda x: a0 + a1*x +a2*x*x if x >= 0 else a0)
	# FC_t = FC_t.sum()
	return FC_t

# Check output on NYC cycle
ax = fuelRate('nyc').plot()
ax.set_ylabel('Fuel consumption (l/s)')
ax.set_xlabel('Time (s)')
plt.show()
