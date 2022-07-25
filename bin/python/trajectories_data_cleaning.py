# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 12:36:05 2022

@author: Mahsa
"""
#import required packages
import pandas as pd


# UMTS
xls = pd.ExcelFile(r'../../data/raw/fuel-export-20130701-20220712.xlsx')
df_UMTS = pd.read_excel(xls, 'in')
df_UMTS['timestamp']= pd.to_datetime(df_UMTS['timestamp'])
df_UMTS.sort_values(by=['timestamp'], ascending=True, inplace=True)
df1['date'] = pd.to_datetime(df1['timestamp']).dt.date
df1['time'] = pd.to_datetime(df1['timestamp']).dt.time
df1['date']=df1['date'].astype(str)
df1['time_delta'] = df1.groupby(['bus_number', 'date'])['timestamp'].diff()
df1.time_delta=df1.time_delta.astype(str).str.replace('0 days ', '')
df1['time_delta']= pd.to_datetime(df1['time_delta'])

### CTF (Cottage Street)
xls = pd.ExcelFile(r'../../data/tidy/CTF Fuel Tickets FY22.xlsx')
df_CTF = pd.read_excel(xls, 'CTF')
df2['vehicle_year'] = df2['model'].str[:4]
df2['model'] = df2['model'].str[5:]
df2['model'].replace('GILLIG 30`', 'GILLIG', inplace=True)
df2['model'].replace('GILLIG 35`', 'GILLIG', inplace=True)
df2['model'].replace('GILLIG 40`', 'GILLIG', inplace=True)
df2['model'].replace('NEW FLYER XD35', 'NEW FLYER', inplace=True)
df2['model'].replace('NEW FLYER XD40', 'NEW FLYER', inplace=True)
df2['model'].replace('NEW FLYER XD40 (HYBRID)', 'NEW FLYER (HYBRID)', inplace=True)
df2['model'].replace('PROTERRA CATALYST BE-40 (ELECTRIC)', 'PROTERRA (ELECTRIC)', inplace=True)
df2['model'].replace('NEW FLYER XE35 (ELECTRIC)', 'NEW FLYER (ELECTRIC)', inplace=True)
df2['model'].replace('NEW FLYER XE40 (ELECTRIC)', 'NEW FLYER (ELECTRIC)', inplace=True)
df2['model'].replace('UMTS GILLIG', 'GILLIG', inplace=True)
df2['model'].replace('UMTS NEW FLYER', 'NEW FLYER', inplace=True)
df2['model'].replace('GILLIG', 'Gillig', inplace=True)
df2['model'].replace('NEW FLYER', 'New Flyer', inplace=True)
df2['model'].replace('NEW FLYER (HYBRID)', 'New Flyer (hybrid)', inplace=True)
df2['model'].replace('PROTERRA (ELECTRIC)', 'Proterra (electric)', inplace=True)
df2['model'].replace('NEW FLYER (ELECTRIC)', 'New Flyer (electric)', inplace=True)

### NTF (VATCo)
xls2 = pd.ExcelFile(r'../../data/tidy/NTF Fuel Tickets FY22.xlsx')
df_NTF = pd.read_excel(xls, 'NTF')
df2['vehicle_year'] = df2['Model'].str[:4]
df2.rename(columns = {'Model':'model'}, inplace = True)
df3['model'].replace('2007 GILLIG 40`', 'GILLIG', inplace=True)
df3['model'].replace('2008 GILLIG 35`', 'GILLIG', inplace=True)
df3['model'].replace('2009 GILLIG 40`', 'GILLIG', inplace=True)
df3['model'].replace('2010 GILLIG 35`', 'GILLIG', inplace=True)
df3['model'].replace('2011 NEW FLYER XD40', 'NEW FLYER', inplace=True)
df3['model'].replace('2012 NEW FLYER XD40', 'NEW FLYER', inplace=True)
df3['model'].replace('2013 NEW FLYER XDE60 (ARTIC) (HYBRID)', 'NEW FLYER (HYBRID)', inplace=True)
df3['model'].replace('2020 NEW FLYER XD35', 'NEW FLYER', inplace=True)
df3['model'].replace('2021 NEW FLYER XD40', 'NEW FLYER', inplace=True)
df3['model'].replace('2021 NEW FLYER XE40 (ELECTRIC)', 'NEW FLYER (ELECTRIC)', inplace=True)
df3['model'].replace('2022 NEW FLYER XD40', 'NEW FLYER', inplace=True)
df3['model'].replace('GILLIG', 'Gillig', inplace=True)


df3['model'].replace('NEW FLYER', 'New Flyer', inplace=True)
df3['model'].replace('NEW FLYER (HYBRID)', 'New Flyer (hybrid)', inplace=True)
df3['model'].replace('PROTERRA (ELECTRIC)', 'Proterra (electric)', inplace=True)
df3['model'].replace('NEW FLYER (ELECTRIC)', 'New Flyer (electric)', inplace=True)


### Mapping models
mydict = {
  'Gillig': 'conventional',
  'New Flyer': 'conventional',
  'New Flyer (hybrid)': 'hybrid',
  'New Flyer (electric)': 'electric',
  'Proterra (electric)': 'electric',
}
result['Powertrain'] = result['model'].map(mydict)


