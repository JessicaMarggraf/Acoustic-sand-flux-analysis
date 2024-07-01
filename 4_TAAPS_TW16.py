# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 14:17:37 2021

@author: jessica.laible

Time-averaged acoustics including physical samples

Input data: output data from BAAB-script, SPM data, sample data    
Output data: averaged acoustic data (AlphaSed, CelldB, AlphaW etc. and their standard deviations)
only S2_D50 now
    
    
"""

# Load packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta, datetime
import math as math
from sklearn.metrics import r2_score
import matplotlib.dates as md
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib import cm
from matplotlib.colors import ListedColormap as mpl_colors_ListedColormap 
import matplotlib as mpl
from matplotlib.legend_handler import HandlerTuple

from Theoretical_formulas import form_factor_function_ThorneMeral2008
from Theoretical_formulas import compute_model_lognorm_spherical
from Functions import user_input_path_freq, user_input_outpath, user_input_outpath_figures, user_input_path_data

#%% 1. LOAD DATA
# Acoustic data
# Define input and output path
print('============== SELECT PATH ==============')
print('Select path folder acoustic data')
path_folder = user_input_path_freq()
print('Select path folder of concurrent samplings, missing and deleted data and station data')
path_data = user_input_path_data()
print('Select outpath')
out_path = user_input_outpath()
print('Select outpath figures')
outpath_figures = user_input_outpath_figures()

#%%
print('Load data...')
missing_data = pd.read_csv(path_data + '\\Missing_data.csv', sep=';')
man_delete_data = pd.read_csv(path_data + '\\Manually_deleted_data.csv', sep=',')
samples = pd.read_csv(path_data + '\\Samples.csv', sep=';')
ISCO_data = pd.read_csv(path_data + '\\ISCO_data.csv', sep=';')
ISCO_GSD_data = pd.read_csv(path_data + '\\ISCO_GSD_data.csv', sep=';')
pump_data = pd.read_csv(path_data + '\\Pump_data.csv', sep=';')
size_classes_mum = pd.read_csv(path_data + '\\ISO_size_classes.csv', sep=';')
spm_data_raw = pd.read_csv(path_data + '\\Turbidity.csv', sep=';')
Q_data_raw = pd.read_csv(path_data + '\\Discharge.csv', sep=';')
stage_data_raw = pd.read_csv(path_data + '\\Water_stage.csv', sep=';')
RUTS_theo_freq1 = pd.read_csv(path_data + '\\RUTS_theo_freq1.csv', sep = ';')
RUTS_theo_freq2 = pd.read_csv(path_data + '\\RUTS_theo_freq2.csv', sep = ';')
events_dates = pd.read_csv(path_data + '\\Events.csv', sep=';')
spring_dates = pd.read_csv(path_data + '\\Spring_dates.csv', sep=';')

# Choose options
# Time-averaging
# 1h-window around temporal midpoint = True, all during sampling time = False
time_av = True
# B' correction
B_fines_correction = True

# Choose frequencies (in kHz)
freq1 = 400
freq2 = 1000

#%% Data preparation
freq1_Hz = freq1 * 1e3 #Hz
freq2_Hz = freq2 * 1e3 # Hz

# Define ADCP characteristics
k_freq1 = 2 * math.pi * freq1_Hz / 1500
k_freq2 = 2 * math.pi * freq2_Hz / 1500

# Acoustic ping duration (s)
t_p_freq1 = 0.000293333 # transmit pulse length in m /1500 m/s
t_p_freq2 = 0.00034

# Radius of the transducer (ceramics) (m)
a_T_freq1 = 0.145/2
a_T_freq2 = 0.045/2

# wavenumber
wavenum_freq1 = 2*np.pi/(1442.5/freq1_Hz)
wavenum_freq2 = 2*np.pi/(1442.5/freq2_Hz)

BeamAv_freq1 = pd.read_csv(path_folder + '\Beam_averaged_attenuation_backscatter_'+ str(freq1) + '.csv', sep= ';')
BeamAv_freq2 = pd.read_csv(path_folder + '\Beam_averaged_attenuation_backscatter_'+ str(freq2) + '.csv', sep= ';')
FluidCorrBackscatter_freq1 = pd.read_csv(path_folder + '\FluidCorrBackscatter_'+ str(freq1) + '.csv', sep= ';')
FluidCorrBackscatter_freq2 = pd.read_csv(path_folder + '\FluidCorrBackscatter_'+ str(freq2) + '.csv', sep= ';')
AveCount_db_freq1 = pd.read_csv(path_folder + '\AveCount_db_'+ str(freq1) + '.csv', sep= ';')
AveCount_db_freq2 = pd.read_csv(path_folder + '\AveCount_db_'+ str(freq2) + '.csv', sep= ';')
Time_datetime_AveCount_db_freq1 = pd.read_csv(path_folder + '\Time_datetime_AveCount_db_'+ str(freq1) + '.csv', sep= ';')
# Time_datetime_AveCount_db_freq2 = pd.read_csv(path_folder + '\Time_datetime_AveCount_db_'+ str(freq2) + '.csv', sep= ';')
CelldB_freq1 = pd.read_csv(path_folder + '\CelldB_'+ str(freq1) + '.csv', sep= ';')
CelldB_freq2 = pd.read_csv(path_folder + '\CelldB_'+ str(freq2) + '.csv', sep= ';')
celldist_along_beam_freq1 = pd.read_csv(path_folder + '\Celldist_along_beam_'+ str(freq1) + '.csv', sep= ';')
celldist_along_beam_freq2 = pd.read_csv(path_folder + '\Celldist_along_beam_'+ str(freq2) + '.csv', sep= ';')

samples = samples.drop(['Unnamed: 0'], axis = 1)
colnames_samples = list(samples.columns.values)
colnames_data_spm_data_raw = list(spm_data_raw.columns.values)
colnames_data_Q_data_raw = list(Q_data_raw.columns.values)
colnames_data_stage_data_raw = list(stage_data_raw.columns.values)

# Fontsizes
fontsize_axis = 14
fontsize_legend = 12
fontsize_legend_title = 14
fontsize_text = 12
fontsize_ticks = 12 

    
#%% 2. PREPARE DATA - All times in UTC
# Acoustic data
time_diff_h = timedelta(hours = 1)
BeamAvBS_freq1 = BeamAv_freq1 ['Beam-Averaged Backscatter (dB)']
BeamAvBS_freq2 = BeamAv_freq2 ['Beam-Averaged Backscatter (dB)']

Time_freq1_all = BeamAv_freq1['Date']
Time_list_freq1_all = list(Time_freq1_all)
Time_datetime_freq1_all = pd.to_datetime(Time_list_freq1_all)
Time_datetime_freq1_all = Time_datetime_freq1_all - time_diff_h

Time_freq2_all = BeamAv_freq2['Date']
Time_list_freq2_all = list(Time_freq2_all)
Time_datetime_freq2_all = pd.to_datetime(Time_list_freq2_all)
Time_datetime_freq2_all = Time_datetime_freq2_all - time_diff_h

CelldBAve_freq1 = [BeamAv_freq1['Beam-Averaged Backscatter (dB)'][i] for i in range(len(BeamAv_freq1))]
AlphaSed_freq1 = [BeamAv_freq1['Alpha Sediment (dB/m)'][i] for i in range(len(BeamAv_freq1))]
AlphaW_freq1 = [BeamAv_freq1['AlphaW'][i] for i in range(len(BeamAv_freq1))]

CelldBAve_freq2 = [BeamAv_freq2['Beam-Averaged Backscatter (dB)'][i] for i in range(len(BeamAv_freq2))]
AlphaSed_freq2 = [BeamAv_freq2['Alpha Sediment (dB/m)'][i] for i in range(len(BeamAv_freq2))]
AlphaW_freq2 = [BeamAv_freq2['AlphaW'][i] for i in range(len(BeamAv_freq2))]

# Prepare SPM data
spm_data = spm_data_raw.drop(spm_data_raw[spm_data_raw['Value'] == -9999].index, inplace = False)
spm_data.reset_index(inplace=True)
spm = spm_data['Value'].astype(float)
spm = spm.reset_index(drop = True)

#  Define spm_data timestamp
Time_spm = spm_data['DateHeure']
Time_spm_list = list(Time_spm)
Time_spm_datetime = pd.to_datetime(Time_spm_list,format='%d.%m.%Y %H:%M')

# Define timestamp in seconds after first amp_vel measurement
Timedelta_spm = [Time_spm_datetime[i] - Time_datetime_freq1_all[0] for i in range(0,len(Time_spm_datetime))] # 
Time_spm_sec = [int(Timedelta_spm[i].total_seconds()) for i in range(0,len(Time_spm_datetime))]

# Prepare Q data
Q_data = Q_data_raw.drop(Q_data_raw[Q_data_raw['Value'] == -9999].index, inplace = False)
Q_data.reset_index(inplace=True)
Q = Q_data['Value'].astype(float)
Q = Q.reset_index(drop = True)

#  Define Q_data timestamp
Time_Q = Q_data['DateHeure']
Time_Q_list = list(Time_Q)
Time_Q_datetime = pd.to_datetime(Time_Q_list,format='%d.%m.%Y %H:%M')

# Temperature
Temperature_freq1 = BeamAv_freq1['Temperature']
Temperature_freq2 = BeamAv_freq2['Temperature']

# Prepare stage data
stage_data = stage_data_raw.drop(stage_data_raw[stage_data_raw['Value'] == -9999].index, inplace = False)
stage_data.reset_index(inplace=True)
stage = stage_data['Value'].astype(float)
stage = stage.reset_index(drop = True)

#  Define stage_data timestamp
Time_stage = stage_data['DateHeure']
Time_stage_list = list(Time_stage)
Time_stage_datetime = pd.to_datetime(Time_stage_list,format='%d.%m.%Y %H:%M')

# Samples
date_sample_list = [str(samples['Date'][i]) for i in range(len(samples))]
date_sample_datetime = [datetime.strptime(date_sample_list[i],'%Y%m%d').date()
               for i in range(len(samples))]
start_time_sample_list = [str(samples['Start_sampling'][i]) for i in range(len(samples))]
start_time_sample_datetime = [datetime.strptime(start_time_sample_list[i],'%H:%M').time()
               for i in range(len(samples))]
samples_start_datetime = [datetime.combine(date_sample_datetime[i], start_time_sample_datetime[i])
                    for i in range(len(date_sample_datetime))]
end_time_sample_list = [str(samples['End_sampling'][i]) for i in range(len(samples))]
end_time_sample_datetime = [datetime.strptime(end_time_sample_list[i],'%H:%M').time()
               for i in range(len(samples))]
samples_end_datetime = [datetime.combine(date_sample_datetime[i], end_time_sample_datetime[i])
                    for i in range(len(date_sample_datetime))]

# ISCO
date_ISCO_list = [str(ISCO_data['Date'][i]) for i in range(len(ISCO_data))]
date_ISCO_datetime = [datetime.strptime(date_ISCO_list[i],'%d.%m.%Y').date()
               for i in range(len(ISCO_data))]
mid_time_ISCO_list = [str(ISCO_data['Mean_Time'][i]) for i in range(len(ISCO_data))]
mid_time_ISCO_datetime = [datetime.strptime(mid_time_ISCO_list[i],'%H:%M').time()
               for i in range(len(ISCO_data))]
ISCO_mid_datetime = [datetime.combine(date_ISCO_datetime[i], mid_time_ISCO_datetime[i])
                    for i in range(len(date_ISCO_datetime))]
Time_ISCO_mid_datetime = pd.to_datetime(ISCO_mid_datetime,format='%d.%m.%Y %H:%M')

# ISCO GSD
date_ISCO_GSD_list = [str(ISCO_GSD_data['Date'][i]) for i in range(len(ISCO_GSD_data))]
date_ISCO_GSD_datetime = [datetime.strptime(date_ISCO_GSD_list[i],'%d.%m.%Y').date()
               for i in range(len(ISCO_GSD_data))]
mid_time_ISCO_GSD_list = [str(ISCO_GSD_data['Hour'][i]) for i in range(len(ISCO_GSD_data))]
mid_time_ISCO_GSD_datetime = [datetime.strptime(mid_time_ISCO_GSD_list[i],'%H:%M').time()
               for i in range(len(ISCO_GSD_data))]
ISCO_GSD_mid_datetime = [datetime.combine(date_ISCO_GSD_datetime[i], mid_time_ISCO_GSD_datetime[i])
                    for i in range(len(date_ISCO_GSD_datetime))]
Time_ISCO_GSD_mid_datetime = pd.to_datetime(ISCO_GSD_mid_datetime,format='%d.%m.%Y %H:%M')


# pump
date_pump_list = [str(pump_data['Date'][i]) for i in range(len(pump_data))]
date_pump_datetime = [datetime.strptime(date_pump_list[i],'%d.%m.%Y').date()
               for i in range(len(pump_data))]
mid_time_pump_list = [str(pump_data['Mean_Time'][i]) for i in range(len(pump_data))]
mid_time_pump_datetime = [datetime.strptime(mid_time_pump_list[i],'%H:%M').time()
               for i in range(len(pump_data))]
pump_mid_datetime = [datetime.combine(date_pump_datetime[i], mid_time_pump_datetime[i])
                    for i in range(len(date_pump_datetime))]
Time_pump_mid_datetime = pd.to_datetime(pump_mid_datetime,format='%d.%m.%Y %H:%M')

# Determine sampling midpoint
midpoint_samples = [samples_start_datetime[i]+ (samples_end_datetime[i] - samples_start_datetime[i])/2
                    for i in range(len(samples_start_datetime))]

# Determine start and end point of a 1 hour window around midpoint
if time_av == True: 
    window = timedelta(minutes=30)
    samples_start_midpoint_datetime = [midpoint_samples[i] - window for i in range(len(midpoint_samples))]
    samples_end_midpoint_datetime = [midpoint_samples[i] + window for i in range(len(midpoint_samples))]
if time_av == False:
    samples_start_midpoint_datetime = samples_start_datetime 
    samples_end_midpoint_datetime = samples_end_datetime

# Calculate silt/sand ratio
samples['Sand_concentration_mg_l'] = samples['Sand_concentration_g_l']*1000
samples['Fine_concentration_mg_l'] = samples['Fine_concentration_g_l']*1000
ISCO_data['Sand_concentration_mg_l'] = ISCO_data['Concentration_sable']*1000
ISCO_data['Fine_concentration_mg_l'] = ISCO_data['Concentration_fine']*1000
ISCO_data['Sand_concentration_g_l'] = ISCO_data['Concentration_sable']
ISCO_data['Fine_concentration_g_l'] = ISCO_data['Concentration_fine']
pump_data['Sand_concentration_mg_l'] = pump_data['Sand_concentration_g_l']*1000

# Calculate log10 values
samples['log_sand'] = np.log10(samples['Sand_concentration_mg_l'])
samples['log_fine'] = np.log10(samples['Fine_concentration_mg_l'])
ISCO_data['log_sand'] = np.log10(ISCO_data['Sand_concentration_mg_l'])
ISCO_data['log_fine'] = np.log10(ISCO_data['Fine_concentration_mg_l'])
pump_data['log_sand'] = np.log10(pump_data['Sand_concentration_mg_l'])

# Correct Csand measured by BD
for i in range(len(samples)):
    if samples['Sampler'][i] == 'BD':
        samples['Sand_concentration_g_l'][i] = samples['Sand_concentration_g_l'][i]*2.37
        samples['Sand_flux_kg_s'][i] = samples['Sand_concentration_g_l'][i]*samples['Q_sampling_m3_s'][i]
    else: 
        samples['Sand_concentration_g_l'][i] = samples['Sand_concentration_g_l'][i]

#%% Filter acoustic data 
# delete measurements during maintenance or when only 1 frequency 
Time_datetime_freq1_all_int = [int(Time_datetime_freq1_all[i].timestamp()) for i in range(len(Time_datetime_freq1_all))]
Time_datetime_freq2_all_int = [int(Time_datetime_freq2_all[i].timestamp()) for i in range(len(Time_datetime_freq2_all))]


#  Define missing_data timestamp
Time_missing_start_datetime = pd.to_datetime(list(missing_data['Start_date']),format='%d.%m.%Y %H:%M')
Time_missing_end_datetime = pd.to_datetime(list(missing_data['End_date']),format='%d.%m.%Y %H:%M')
Time_missing_start_datetime_int = [int(pd.to_datetime(missing_data['Start_date'][i],format='%d.%m.%Y %H:%M').timestamp())  
                                   for i in range(len(missing_data))]
Time_missing_end_datetime_int = [int(pd.to_datetime(missing_data['End_date'][i],format='%d.%m.%Y %H:%M').timestamp())  
                                   for i in range(len(missing_data))]

# Define data to delete (e.g. aberrant values, filtering)
Time_man_delete_start_datetime = pd.to_datetime(list(man_delete_data['Start_date']),format='%d.%m.%Y %H:%M')
Time_man_delete_end_datetime = pd.to_datetime(list(man_delete_data['End_date']),format='%d.%m.%Y %H:%M')
Time_man_delete_start_datetime_int = [int(pd.to_datetime(man_delete_data['Start_date'][i],format='%d.%m.%Y %H:%M').timestamp())  
                                   for i in range(len(man_delete_data))]
Time_man_delete_end_datetime_int = [int(pd.to_datetime(man_delete_data['End_date'][i],format='%d.%m.%Y %H:%M').timestamp())  
                                   for i in range(len(man_delete_data))]

# combine both datasets
Time_delete_start_datetime_int = Time_man_delete_start_datetime_int + Time_missing_start_datetime_int
Time_delete_end_datetime_int = Time_man_delete_end_datetime_int + Time_missing_end_datetime_int
Time_delete_start_datetime_int.sort()
Time_delete_end_datetime_int.sort()

# Find index in freq timestamp
start_idx_missing_freq1 = [np.argmin([abs(Time_datetime_freq1_all_int[i] - Time_delete_start_datetime_int[j]) for i in range(len(Time_datetime_freq1_all))])
                       for j in range(len(Time_delete_start_datetime_int))]
end_idx_missing_freq1 = [np.argmin([abs(Time_datetime_freq1_all_int[i] - Time_delete_end_datetime_int[j]) for i in range(len(Time_datetime_freq1_all))])
                     for j in range(len(Time_delete_end_datetime_int))]
start_idx_missing_freq2 = [np.argmin([abs(Time_datetime_freq2_all_int[i] - Time_delete_start_datetime_int[j]) for i in range(len(Time_datetime_freq2_all))])
                       for j in range(len(Time_delete_start_datetime_int))]
end_idx_missing_freq2 = [np.argmin([abs(Time_datetime_freq2_all_int[i] - Time_delete_end_datetime_int[j]) for i in range(len(Time_datetime_freq2_all))])
                     for j in range(len(Time_delete_end_datetime_int))]
 
idxx_delete_freq1 = [np.arange(start_idx_missing_freq1[i], end_idx_missing_freq1[i]+1, 1).tolist()
                     for i in range(len(end_idx_missing_freq1))]
idxx_delete_freq1 = [item for sublist in idxx_delete_freq1 for item in sublist]
idxx_delete_freq2 = [np.arange(start_idx_missing_freq2[i], end_idx_missing_freq2[i]+1, 1).tolist()
                     for i in range(len(end_idx_missing_freq2))]
idxx_delete_freq2 = [item for sublist in idxx_delete_freq2 for item in sublist]


# Delete data
Time_datetime_freq1 = Time_datetime_freq1_all.delete([idxx_delete_freq1])
BeamAv_freq1 = BeamAv_freq1.drop(BeamAv_freq1.index[[idxx_delete_freq1]]).reset_index(drop = True)
BeamAvBS_freq1 = BeamAvBS_freq1.drop(BeamAvBS_freq1.index[[idxx_delete_freq1]],).reset_index(drop = True)
CelldBAve_freq1 = pd.Series(CelldBAve_freq1).drop(pd.Series(CelldBAve_freq1).index[[idxx_delete_freq1]]).reset_index(drop = True).tolist()
AlphaSed_freq1 = pd.Series(AlphaSed_freq1).drop(pd.Series(AlphaSed_freq1).index[[idxx_delete_freq1]]).reset_index(drop = True).tolist()
AlphaW_freq1 = pd.Series(AlphaW_freq1).drop(pd.Series(AlphaW_freq1).index[[idxx_delete_freq1]]).reset_index(drop = True).tolist()
# AveCount_db_cut_freq1 = AveCount_db_freq1_time_freq1.drop(AveCount_db_freq1_time_freq1.index[[idxx_delete_freq1]]).reset_index(drop = True)
Temperature_freq1 = pd.Series(Temperature_freq1).drop(pd.Series(Temperature_freq1).index[[idxx_delete_freq1]]).reset_index(drop = True).tolist()
FluidCorrBackscatter_freq1 = FluidCorrBackscatter_freq1.drop(FluidCorrBackscatter_freq1.index[idxx_delete_freq1])
CelldB_freq1 = CelldB_freq1.drop(CelldB_freq1.index[idxx_delete_freq1])

Time_datetime_freq2 = Time_datetime_freq2_all.delete([idxx_delete_freq2]) 
BeamAv_freq2 = BeamAv_freq2.drop(BeamAv_freq2.index[[idxx_delete_freq2]]).reset_index(drop = True)
BeamAvBS_freq2 = BeamAvBS_freq2.drop(BeamAvBS_freq2.index[[idxx_delete_freq2]],).reset_index(drop = True)
CelldBAve_freq2 = pd.Series(CelldBAve_freq2).drop(pd.Series(CelldBAve_freq2).index[[idxx_delete_freq2]]).reset_index(drop = True).tolist()
AlphaSed_freq2 = pd.Series(AlphaSed_freq2).drop(pd.Series(AlphaSed_freq2).index[[idxx_delete_freq2]]).reset_index(drop = True).tolist()
AlphaW_freq2 = pd.Series(AlphaW_freq2).drop(pd.Series(AlphaW_freq2).index[[idxx_delete_freq2]]).reset_index(drop = True).tolist()
# AveCount_db_cut_freq2 = AveCount_db_freq2_time_freq2.drop(AveCount_db_freq2_time_freq2.index[[idxx_delete_freq2]]).reset_index(drop = True)
Temperature_freq2 = pd.Series(Temperature_freq2).drop(pd.Series(Temperature_freq2).index[[idxx_delete_freq2]]).reset_index(drop = True).tolist()
FluidCorrBackscatter_freq2 = FluidCorrBackscatter_freq2.drop(FluidCorrBackscatter_freq2.index[idxx_delete_freq2])
CelldB_freq2 = CelldB_freq2.drop(CelldB_freq2.index[idxx_delete_freq2])

Time_datetime_freq1_int = [int(Time_datetime_freq1[i].timestamp()) for i in range(len(Time_datetime_freq1))]
Time_datetime_freq2_int = [int(Time_datetime_freq2[i].timestamp()) for i in range(len(Time_datetime_freq2))]

Time_spm_datetime_int = [int(Time_spm_datetime[i].timestamp()) for i in range(len(Time_spm_datetime))]
Time_Q_datetime_int = [int(Time_Q_datetime[i].timestamp()) for i in range(len(Time_Q_datetime))]

# Csand_TW16_time_freq2 = np.interp(Time_datetime_freq2, Time_datetime_TW16, Csand_TW16)

# Get spm data at Acoustic time
spm_time_freq1 = np.interp(Time_datetime_freq1, Time_spm_datetime,spm)
spm_time_freq2 = np.interp(Time_datetime_freq2, Time_spm_datetime,spm)

# Get Q data at Acoustic time
Q_time_freq1 = np.round(np.interp(Time_datetime_freq1, Time_Q_datetime,Q),1)
Q_time_freq2 = np.round(np.interp(Time_datetime_freq2, Time_Q_datetime,Q),1)

# Get stage data at Acoustic time
stage_time_freq1 = np.interp(Time_datetime_freq1, Time_stage_datetime,stage)
stage_time_freq2 = np.interp(Time_datetime_freq2, Time_stage_datetime,stage)

# Get spm data at Q time
spm_time_Q = np.interp(Time_Q_datetime, Time_spm_datetime,spm)
# Get Q data at spm time
Q_time_spm = np.interp(Time_spm_datetime, Time_Q_datetime,Q)

# Get acoustic data at SPM time
AlphaSed_freq1_time_spm = np.interp(Time_spm_datetime, Time_datetime_freq1,AlphaSed_freq1)
BeamAvBS_freq2_time_freq1 = np.interp(Time_datetime_freq1, Time_datetime_freq2, BeamAvBS_freq2)
AlphaSed_freq2_time_freq1 = np.interp(Time_datetime_freq1, Time_datetime_freq2, AlphaSed_freq2)
AlphaW_freq2_time_freq1 = np.interp(Time_datetime_freq1, Time_datetime_freq2, AlphaW_freq2)

# Get acoustic data at ISCO time
AlphaSed_freq1_time_ISCO = np.interp(Time_ISCO_mid_datetime, Time_datetime_freq1,AlphaSed_freq1)
AlphaSed_freq2_time_ISCO = np.interp(Time_ISCO_mid_datetime, Time_datetime_freq2,AlphaSed_freq2)
BeamAvBS_freq1_time_ISCO = np.interp(Time_ISCO_mid_datetime, Time_datetime_freq1,BeamAvBS_freq1)
BeamAvBS_freq2_time_ISCO = np.interp(Time_ISCO_mid_datetime, Time_datetime_freq2,BeamAvBS_freq2)
ISCO_data['AlphaSed_freq1'] = AlphaSed_freq1_time_ISCO
ISCO_data['AlphaSed_freq2'] = AlphaSed_freq2_time_ISCO
ISCO_data['BeamAvBS_freq1'] = BeamAvBS_freq1_time_ISCO
ISCO_data['BeamAvBS_freq2'] = BeamAvBS_freq2_time_ISCO

# Get acoustic data at ISCO_GSD time
AlphaSed_freq1_time_ISCO_GSD = np.interp(Time_ISCO_GSD_mid_datetime, Time_datetime_freq1,AlphaSed_freq1)
AlphaSed_freq2_time_ISCO_GSD = np.interp(Time_ISCO_GSD_mid_datetime, Time_datetime_freq2,AlphaSed_freq2)
BeamAvBS_freq1_time_ISCO_GSD = np.interp(Time_ISCO_GSD_mid_datetime, Time_datetime_freq1,BeamAvBS_freq1)
BeamAvBS_freq2_time_ISCO_GSD = np.interp(Time_ISCO_GSD_mid_datetime, Time_datetime_freq2,BeamAvBS_freq2)
ISCO_GSD_data['AlphaSed_freq1'] = AlphaSed_freq1_time_ISCO_GSD
ISCO_GSD_data['AlphaSed_freq2'] = AlphaSed_freq2_time_ISCO_GSD
ISCO_GSD_data['BeamAvBS_freq1'] = BeamAvBS_freq1_time_ISCO_GSD
ISCO_GSD_data['BeamAvBS_freq2'] = BeamAvBS_freq2_time_ISCO_GSD

# Get acoustic data at pump time
AlphaSed_freq1_time_pump = np.interp(Time_pump_mid_datetime, Time_datetime_freq1,AlphaSed_freq1)
AlphaSed_freq2_time_pump = np.interp(Time_pump_mid_datetime, Time_datetime_freq2,AlphaSed_freq2)
BeamAvBS_freq1_time_pump = np.interp(Time_pump_mid_datetime, Time_datetime_freq1,BeamAvBS_freq1)
BeamAvBS_freq2_time_pump = np.interp(Time_pump_mid_datetime, Time_datetime_freq2,BeamAvBS_freq2)
pump_data['AlphaSed_freq1'] = AlphaSed_freq1_time_pump
pump_data['AlphaSed_freq2'] = AlphaSed_freq2_time_pump
pump_data['BeamAvBS_freq1'] = BeamAvBS_freq1_time_pump
pump_data['BeamAvBS_freq2'] = BeamAvBS_freq2_time_pump
spm_time_pump = np.interp(Time_pump_mid_datetime, Time_spm_datetime,spm)
pump_data['spm_sampling_g_l'] = spm_time_pump

# Get Freq1 at Freq2 time
BeamAvBS_freq1_time_freq2 = np.interp(Time_datetime_freq2, Time_datetime_freq1,BeamAvBS_freq1)
AlphaSed_freq1_time_freq2 = np.interp(Time_datetime_freq2, Time_datetime_freq1,AlphaSed_freq1)

BeamAv_freq2['Q_m3_s'] = Q_time_freq2
BeamAv_freq1['Q_m3_s'] = Q_time_freq1
BeamAv_freq2['SPM'] = spm_time_freq2
BeamAv_freq1['SPM'] = spm_time_freq1
BeamAv_freq2['Stage'] = stage_time_freq2
BeamAv_freq1['Stage'] = stage_time_freq1
colnames_BeamAv_freq1 = list(BeamAv_freq1.columns.values)
colnames_BeamAv_freq2 = list(BeamAv_freq2.columns.values)

# Mean temperatures at the same time
Temperature_freq2_time_freq1 =np.interp(Time_datetime_freq1, Time_datetime_freq2,Temperature_freq2) 
Temperature = [(Temperature_freq2_time_freq1 [i] + Temperature_freq1[i])/2 
               for i in range(len(Temperature_freq1))]

spm_time_ISCO = np.interp(Time_ISCO_mid_datetime, Time_spm_datetime, spm)
ISCO_data['spm_sampling_g_l'] = spm_time_ISCO
   

#%% Define reference properties
# Water kinematic viscosity
nu_0 = 0.73 * 1e-6
# rho
rho_sed = 2650
siltDens = 2.65

# Concentration
C_sand_ref_g_l = 0.2 # 0.1

# D50 - sand 
D50_sand_ref_mum = 200
D50_sand_ref_phi = -np.log2(D50_sand_ref_mum/1000) 
sandD50ref = D50_sand_ref_mum/1000

# D50 - fines
D50_fines_ref_mum = 1
D50_fines_ref_phi = -np.log2(D50_fines_ref_mum/1000)
siltD50 = D50_fines_ref_mum/1000

# Sigma - sand 
sigma_sand_ref_mum = 0.59
# sigma_sand_ref_phi = -np.log2(sigma_sand_ref_mum/1000)
sortSand = np.log(2**sigma_sand_ref_mum)

# Sigma - fines 
sigma_fines_ref_mum = 1.4
# sigma_fines_ref_phi = -np.log2(sigma_fines_ref_mum/1000)
sortSilt = np.log(2**sigma_fines_ref_mum)

# Reference ranges
# D50 - sand 
D50_sand_ref_range_phi = [D50_sand_ref_phi - 0.4, D50_sand_ref_phi + 0.4] 
D50_sand_ref_range_mum = [2**(-D50_sand_ref_range_phi[i])*1000 for i in range(len(D50_sand_ref_range_phi))]


#%% Prepare GSD data
# Get cumsum size classes
size_classes_mum = size_classes_mum.iloc[:,1].tolist()
size_classes_mum = [float(size_classes_mum[i]) for i in range(len(size_classes_mum))]
size_classes_m = [size_classes_mum[i]*1e-6 for i in range(len(size_classes_mum))]

# For classified distribution
size_classes_inf = np.array(size_classes_mum[0:-1])* 1e-6 / 2
size_classes_sup = np.array(size_classes_mum[1:])* 1e-6 / 2
size_classes_center = [10**((np.log10(size_classes_mum[i]) + np.log10(size_classes_mum[i]))/2) 
                     for i in range(len(size_classes_mum))]
size_classes_center= np.array(size_classes_center)

# Convert to phi
size_classes_phi = [-np.log2(size_classes_mum[i]/1000) for i in range(len(size_classes_mum))]

#%% REFERENCE sand GSD
#Determine form factor
# Define boundaries, phi system and convert to mm (2**)   
gs = [2**(-35.25 + 0.0625*j) for j in range(0,672)]
gsbound = [2**(-35.2188 + 0.0625*j) for j in range(0,672)]

# Calculate ref sand f following TW16
volfracsand_ref = [(1/np.sqrt(2*math.pi*D50_sand_ref_mum))*np.exp((-(np.log(gs[j])-np.log(D50_sand_ref_mum*1e-3))**2)/(2*sigma_sand_ref_mum**2))       
                for j in range(len(gs))]
cumsumsand_ref = np.cumsum(volfracsand_ref)
proba_vol_sand_ref = volfracsand_ref/cumsumsand_ref[-1]

# Determine D50
cumsum_sand_ref = np.cumsum(proba_vol_sand_ref)
d500_ref = 0.5*cumsum_sand_ref[-1]

for j in range(len(cumsum_sand_ref)):
    if d500_ref >= cumsum_sand_ref[j-1] and d500_ref <= cumsum_sand_ref[j]:
        d50sand_ref=((d500_ref-cumsum_sand_ref[j-1])/(cumsum_sand_ref[j]-cumsum_sand_ref[j-1]))*(np.log(gsbound[j])-np.log(gsbound[j-1]))+np.log(gsbound[j-1])
d50_sand_ref = np.exp(d50sand_ref)/1000

# Computing number probability
ss = np.sum(proba_vol_sand_ref / np.array(gs)**3)
proba_num_sand_ref = proba_vol_sand_ref/np.array(gs)**3 / ss

# Calculate form function TM08
# Integrating over the distribution        
temp_a1 = 0
temp_a2f2_TM08_freq1 = 0
temp_a2f2_TM08_freq2 = 0
temp_a3 = 0  
# Summing the integrals
for l in range(len(proba_num_sand_ref)):
    a = np.array(gs)[l]/1000
    temp_a2f2_TM08_freq1 += (a/2)**2 * form_factor_function_ThorneMeral2008((a/2), freq1_Hz, 1500)**2 * proba_num_sand_ref[l]   
    temp_a2f2_TM08_freq2 += (a/2)**2 * form_factor_function_ThorneMeral2008((a/2), freq2_Hz, 1500)**2 * proba_num_sand_ref[l]    
    temp_a3 += (a/2)**3 * proba_num_sand_ref[l]

# computing output values   
f_TM08_freq1_sand_refTW16 = (((d50_sand_ref/2)*temp_a2f2_TM08_freq1)/temp_a3)**0.5
f_TM08_freq2_sand_refTW16 = (((d50_sand_ref/2)*temp_a2f2_TM08_freq2)/temp_a3)**0.5

formSandref_freq1 = f_TM08_freq1_sand_refTW16
formSandref_freq2 = f_TM08_freq2_sand_refTW16


#%% REFERENCE fines GSD
#Determine form factor
# Define boundaries, phi system and convert to mm (2**)   
gs = [2**(-35.25 + 0.0625*j) for j in range(0,672)]
gsbound = [2**(-35.2188 + 0.0625*j) for j in range(0,672)]

# Calculate ref fines f following TW16
volfracfines_ref = [(1/np.sqrt(2*math.pi*D50_fines_ref_mum))*np.exp((-(np.log(gs[j])-np.log(D50_fines_ref_mum*1e-3))**2)/(2*sigma_fines_ref_mum**2))       
                for j in range(len(gs))]
cumsumfines_ref = np.cumsum(volfracfines_ref)
proba_vol_fines_ref = volfracfines_ref/cumsumfines_ref[-1]

# Determine D50
cumsum_fines_ref = np.cumsum(proba_vol_fines_ref)
d500_ref = 0.5*cumsum_fines_ref[-1]

for j in range(len(cumsum_fines_ref)):
    if d500_ref >= cumsum_fines_ref[j-1] and d500_ref <= cumsum_fines_ref[j]:
        d50fines_ref=((d500_ref-cumsum_fines_ref[j-1])/(cumsum_fines_ref[j]-cumsum_fines_ref[j-1]))*(np.log(gsbound[j])-np.log(gsbound[j-1]))+np.log(gsbound[j-1])
d50_fines_ref = np.exp(d50fines_ref)/1000

# Computing number probability
ss = np.sum(proba_vol_fines_ref / np.array(gs)**3)
proba_num_fines_ref = proba_vol_fines_ref/np.array(gs)**3 / ss

# Calculate form function TM08
# Integrating over the distribution        
temp_a1 = 0
temp_a2f2_TM08_freq1 = 0
temp_a2f2_TM08_freq2 = 0
temp_a3 = 0  
# Summing the integrals
for l in range(len(proba_num_fines_ref)):
    a = np.array(gs)[l]/1000
    temp_a2f2_TM08_freq1 += (a/2)**2 * form_factor_function_ThorneMeral2008((a/2), freq1_Hz, 1500)**2 * proba_num_fines_ref[l]   
    temp_a2f2_TM08_freq2 += (a/2)**2 * form_factor_function_ThorneMeral2008((a/2), freq2_Hz, 1500)**2 * proba_num_fines_ref[l]    
    temp_a3 += (a/2)**3 * proba_num_fines_ref[l]

# computing output values   
f_TM08_freq1_fines_refTW16 = (((d50_fines_ref/2)*temp_a2f2_TM08_freq1)/temp_a3)**0.5
f_TM08_freq2_fines_refTW16 = (((d50_fines_ref/2)*temp_a2f2_TM08_freq2)/temp_a3)**0.5

formfinesref_freq1 = f_TM08_freq1_fines_refTW16
formfinesref_freq2 = f_TM08_freq2_fines_refTW16

# Compute zeta reference distributions
ref_dist_sand_freq1 = compute_model_lognorm_spherical(D50_sand_ref_mum*1e-6, sigma_sand_ref_mum, freq1_Hz, 1/20, rho_sed, nu_0)
ref_dist_sand_freq2 = compute_model_lognorm_spherical(D50_sand_ref_mum*1e-6, sigma_sand_ref_mum, freq2_Hz, 1/20, rho_sed, nu_0)
ref_dist_fines_freq1 = compute_model_lognorm_spherical(D50_fines_ref_mum*1e-6, sigma_fines_ref_mum, freq1_Hz, 1/2, rho_sed, nu_0)
ref_dist_fines_freq2 = compute_model_lognorm_spherical(D50_fines_ref_mum*1e-6, sigma_fines_ref_mum, freq2_Hz, 1/2, rho_sed, nu_0)
zeta_sand_freq1 = ref_dist_sand_freq1.zeta
zeta_sand_freq2 = ref_dist_sand_freq2.zeta
zeta_fines_freq1 = ref_dist_fines_freq1.zeta
zeta_fines_freq2 = ref_dist_fines_freq2.zeta


#%% STEP 8: FIND AND AVERAGE ACOUSTIC DATA DURING SAMPLING TIME
# Find index of first and last acoustic data within sampling range
# Frequency 1 
ind_first_beam_av_freq1 = []
for i in range(len(samples_start_datetime)):
    lli = next(x[0] for x in enumerate(Time_datetime_freq1) if x[1] > samples_start_midpoint_datetime[i])
    ind_first_beam_av_freq1.append(lli)
    
ind_last_beam_av_freq1 = []
for i in range(len(samples_start_datetime)):
    lli = next(x[0] for x in enumerate(Time_datetime_freq1) if samples_end_midpoint_datetime[i] < x[1])
    lli = lli # -1 
    ind_last_beam_av_freq1.append(lli)

# Calculate number of acoustic meas within sampling range
no_meas_freq1 = [ind_last_beam_av_freq1[i] - ind_first_beam_av_freq1[i]+1
           for i in range(len(ind_first_beam_av_freq1))]

# Calculate square root of acoustic meas number
sqr_no_meas_freq1 = [np.sqrt(i) for i in no_meas_freq1]

#-------------------------------------------------------------------------------------
# Frequency 2 
ind_first_beam_av_freq2 = []
for i in range(len(samples_start_datetime)):
    lli = next(x[0] for x in enumerate(Time_datetime_freq2) if x[1] > samples_start_midpoint_datetime[i])
    ind_first_beam_av_freq2.append(lli)
    
ind_last_beam_av_freq2 = []
for i in range(len(samples_start_datetime)):
    lli = next(x[0] for x in enumerate(Time_datetime_freq2) if samples_end_midpoint_datetime[i] < x[1])
    lli = lli # -1 
    ind_last_beam_av_freq2.append(lli)

# Calculate number of acoustic meas within sampling range
no_meas_freq2 = [ind_last_beam_av_freq2[i] - ind_first_beam_av_freq2[i]+1
           for i in range(len(ind_first_beam_av_freq2))]

# Calculate square root of acoustic meas number
sqr_no_meas_freq2 = [np.sqrt(i) for i in no_meas_freq2]

#-------------------------------------------------------------------------------------

# Average acoustic data during sampling time

# Frequency 1
# Average acoustic data
colnames_BeamAv_freq1 = BeamAv_freq1.columns
BAAB_mean_samples_freq1 = [np.nanmean(BeamAv_freq1.iloc[ind_first_beam_av_freq1[i]:ind_last_beam_av_freq1[i]+1,1:], axis = 0)
                     for i in range(len(ind_first_beam_av_freq1))]
BAAB_mean_samples_freq1 = pd.DataFrame(BAAB_mean_samples_freq1, columns = colnames_BeamAv_freq1[1:])
BAAB_mean_samples_freq1['Date'] = date_sample_datetime

# Calculate standard deviation
BAAB_std_samples_freq1 = [np.nanstd(BeamAv_freq1.iloc[ind_first_beam_av_freq1[i]:ind_last_beam_av_freq1[i]+1,1:], axis = 0)
                     for i in range(len(ind_first_beam_av_freq1))]
colnames_BeamAv_freq1_std = [colnames_BeamAv_freq1[i]+ str(' std') for i in range(len(colnames_BeamAv_freq1))]
BAAB_std_samples_freq1 = pd.DataFrame(BAAB_std_samples_freq1, columns = colnames_BeamAv_freq1_std[1:])
BAAB_std_samples_freq1['Date'] = date_sample_datetime

# Calculate standard error
BAAB_std_err_samples_freq1 = [BAAB_std_samples_freq1.iloc[i,0:-1]/sqr_no_meas_freq1[i]
                     for i in range(len(ind_first_beam_av_freq1))]
colnames_BeamAv_freq1_std_err = [colnames_BeamAv_freq1[i]+ str(' std err') for i in range(len(colnames_BeamAv_freq1))]
BAAB_std_err_samples_freq1 = pd.DataFrame(BAAB_std_err_samples_freq1)
BAAB_std_err_samples_freq1.columns = colnames_BeamAv_freq1_std_err[1:]
BAAB_std_err_samples_freq1['Date'] = date_sample_datetime

#-------------------------------------------------------------------------------------
# Frequency 2
# Average acoustic data
colnames_BeamAv_freq2 = BeamAv_freq2.columns
BAAB_mean_samples_freq2 = [np.nanmean(BeamAv_freq2.iloc[ind_first_beam_av_freq2[i]:ind_last_beam_av_freq2[i]+1,1:], axis = 0)
                     for i in range(len(ind_first_beam_av_freq2))]
BAAB_mean_samples_freq2 = pd.DataFrame(BAAB_mean_samples_freq2, columns = colnames_BeamAv_freq2[1:])
BAAB_mean_samples_freq2['Date'] = date_sample_datetime

# Calculate standard deviation
BAAB_std_samples_freq2 = [np.nanstd(BeamAv_freq2.iloc[ind_first_beam_av_freq2[i]:ind_last_beam_av_freq2[i]+1,1:], axis = 0)
                     for i in range(len(ind_first_beam_av_freq2))]
colnames_BeamAv_freq2_std = [colnames_BeamAv_freq2[i]+ str(' std') for i in range(len(colnames_BeamAv_freq2))]
BAAB_std_samples_freq2 = pd.DataFrame(BAAB_std_samples_freq2, columns = colnames_BeamAv_freq2_std[1:])
BAAB_std_samples_freq2['Date'] = date_sample_datetime

# Calculate standard error
BAAB_std_err_samples_freq2 = [BAAB_std_samples_freq2.iloc[i,0:-1]/sqr_no_meas_freq2[i]
                     for i in range(len(ind_first_beam_av_freq2))]
colnames_BeamAv_freq2_std_err = [colnames_BeamAv_freq2[i]+ str(' std err') for i in range(len(colnames_BeamAv_freq2))]
BAAB_std_err_samples_freq2 = pd.DataFrame(BAAB_std_err_samples_freq2)
BAAB_std_err_samples_freq2.columns = colnames_BeamAv_freq2_std_err[1:]
BAAB_std_err_samples_freq2['Date'] = date_sample_datetime

#
TAAPS_freq1 = pd.concat([pd.DataFrame(date_sample_datetime), samples.iloc[:,1:], BAAB_mean_samples_freq1, 
                         BAAB_std_samples_freq1.iloc[:,:-1], BAAB_std_err_samples_freq1.iloc[:,:-1]], axis = 1)
TAAPS_freq2 = pd.concat([pd.DataFrame(date_sample_datetime), samples.iloc[:,1:], BAAB_mean_samples_freq2, 
                         BAAB_std_samples_freq2.iloc[:,:-1], BAAB_std_err_samples_freq2.iloc[:,:-1]], axis = 1)

# CALCULATE S, THE RATIO BETWEEN C FINES AND C SAND
TAAPS_freq1['S'] = (TAAPS_freq1['Fine_concentration_mg_l'])/TAAPS_freq1['Sand_concentration_mg_l']
TAAPS_freq2['S'] = (TAAPS_freq2['Fine_concentration_mg_l'])/TAAPS_freq2['Sand_concentration_mg_l']

TAAPS_freq1['log10_S'] = np.log10(TAAPS_freq1['S'])
TAAPS_freq2['log10_S'] = np.log10(TAAPS_freq2['S'])


#%% Plot Fig2
fig, ax = plt.subplots(2, 1, figsize = (12,8), dpi=300)

# Q
p1, = ax[0].plot(Time_Q_datetime, Q,
                color = 'blue', ls = '-', lw = 0.5,
               zorder = 0, label = 'Q')

ax[0].set_ylabel(r'Q (m³/s)', fontsize=18, weight = 'bold')
ax[0].tick_params(axis='both', which='major', labelsize = 16)
ax[0].set_xlim(Time_datetime_freq1[0], Time_datetime_freq1[-1])
ax[0].set_ylim(0, 700)
ax[0].text(0.02, 0.9, '(a)', fontsize = 16, transform = ax[0].transAxes)
ax[0].xaxis.set_ticklabels([])

# Cfines
p2, = ax[1].plot(Time_spm_datetime, spm,
                color = 'tan', ls = '', markersize = 1, marker = 'o',
               zorder = 10)

p8, = ax[1].plot(TAAPS_freq1['Date'], TAAPS_freq1['spm_sampling_g_l'],
                color = 'darkorange', markersize = 8, marker = 'D', ls = '', markeredgewidth = 0.5,
                markeredgecolor='black', zorder = 20, label = 'Sampler')

p9, = ax[1].plot(Time_ISCO_mid_datetime[10], ISCO_data['spm_sampling_g_l'][10],
                color = 'yellowgreen', ls = '', markersize = 7, marker = 'o', markeredgewidth = 0.1,
                markeredgecolor='black',
                zorder = 10, label = ' ISCO')

ax[1].plot(Time_ISCO_mid_datetime, ISCO_data['spm_sampling_g_l'],
                color = 'yellowgreen', ls = '', markersize = 5, marker = 'o', markeredgewidth = 0.1,
                markeredgecolor='black',
                zorder = 10, label = 'ISCO')
p5, = ax[1].plot(Time_pump_mid_datetime, pump_data['spm_sampling_g_l'],
                color = 'mediumblue', markersize = 8, marker = 's', ls = '', markeredgewidth = 0.5,
               markeredgecolor='black', zorder = 20, label = 'Pump')

ax[1].text(0.02, 0.9, '(b)', fontsize = 16, transform = ax[1].transAxes)
ax[1].set_ylabel(r'$\mathregular{C_{tot}}$ (g/l)', fontsize=18, weight = 'bold')
ax[1].tick_params(axis='both', which='major', labelsize = 16)
ax[1].set_xlim(Time_datetime_freq1[1], Time_datetime_freq1[-1])
ax[1].set_ylim(0.01,20 )
ax[1].set_yscale('log')
ax[1].xaxis.set_major_locator(md.MonthLocator(interval=6))
ax[1].xaxis.set_major_formatter(md.DateFormatter('%d/%m/%Y'))

handles = [p8, p5, p9]
fig.legend(handles = handles, #labels=labels, 
          handler_map = {tuple: mpl.legend_handler.HandlerTuple(None)},
          fontsize = 16, loc = 'lower center', ncol = 4, bbox_to_anchor = (0.5, -0.07))

fig.supxlabel(r'Time', fontsize=18, weight = 'bold')

fig.tight_layout()
figname = 'Fig2'
fig.savefig(outpath_figures +'\\' +  figname + '.png', dpi = 300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight') 
# fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 300, bbox_inches='tight')

#%% STEP 9: REGRESSION BETWEEN ALPHASED AND SPM

start_datetime_int_nospm = int(pd.to_datetime('01.07.2022 00:00',format='%d.%m.%Y %H:%M').timestamp()) 
end_datetime_int_nospm = int(pd.to_datetime('28.11.2022 12:00',format='%d.%m.%Y %H:%M').timestamp())
start_freq1_nospm = np.argmin([abs(Time_datetime_freq1_int[i] - start_datetime_int_nospm) for i in range(len(Time_datetime_freq1))])
end_freq1_nospm = np.argmin([abs(Time_datetime_freq1_int[i] - end_datetime_int_nospm) for i in range(len(Time_datetime_freq1))])
start_freq2_nospm = np.argmin([abs(Time_datetime_freq2_int[i] - start_datetime_int_nospm) for i in range(len(Time_datetime_freq2))])
end_freq2_nospm = np.argmin([abs(Time_datetime_freq2_int[i] - end_datetime_int_nospm) for i in range(len(Time_datetime_freq2))])

start_datetime_int_nospm2 = int(pd.to_datetime('23.05.2023 15:00',format='%d.%m.%Y %H:%M').timestamp()) 
end_datetime_int_nospm2 = int(pd.to_datetime('28.06.2023 00:00',format='%d.%m.%Y %H:%M').timestamp())
start_freq1_nospm2 = np.argmin([abs(Time_datetime_freq1_int[i] - start_datetime_int_nospm2) for i in range(len(Time_datetime_freq1))])
end_freq1_nospm2 = np.argmin([abs(Time_datetime_freq1_int[i] - end_datetime_int_nospm2) for i in range(len(Time_datetime_freq1))])
start_freq2_nospm2 = np.argmin([abs(Time_datetime_freq2_int[i] - start_datetime_int_nospm2) for i in range(len(Time_datetime_freq2))])
end_freq2_nospm2 = np.argmin([abs(Time_datetime_freq2_int[i] - end_datetime_int_nospm2) for i in range(len(Time_datetime_freq2))])

x_range = np.linspace(0,10,100)
idx = np.isfinite(spm_time_freq1) & np.isfinite(AlphaSed_freq1) # only for valid data (no nan)
spm_time_freq12 = [spm_time_freq1[i] for i in range(len(spm_time_freq1)) if idx[i] == True]
del spm_time_freq12[start_freq1_nospm2:end_freq1_nospm2]
del spm_time_freq12[start_freq1_nospm:end_freq1_nospm]
AlphaSed_freq12 = [AlphaSed_freq1[i] for i in range(len(AlphaSed_freq1)) if idx[i] == True]
del AlphaSed_freq12[start_freq1_nospm2:end_freq1_nospm2]
del AlphaSed_freq12[start_freq1_nospm:end_freq1_nospm]
interp_alphaSed1_spm = np.polyfit(AlphaSed_freq12, spm_time_freq12, 1)
lin_model_alphaSed1_spm = [AlphaSed_freq12[i]*interp_alphaSed1_spm[0]+interp_alphaSed1_spm[1]
                            for i in range(len(AlphaSed_freq12))]

idx = np.isfinite(spm_time_freq2) & np.isfinite(AlphaSed_freq2)
spm_time_freq22 = [spm_time_freq2[i] for i in range(len(spm_time_freq2)) if idx[i] == True]
AlphaSed_freq22 = [AlphaSed_freq2[i] for i in range(len(AlphaSed_freq2)) if idx[i] == True]
del spm_time_freq22[start_freq2_nospm2:end_freq2_nospm2]
del spm_time_freq22[start_freq2_nospm:end_freq2_nospm]
AlphaSed_freq22 = [AlphaSed_freq2[i] for i in range(len(AlphaSed_freq2)) if idx[i] == True]
del AlphaSed_freq22[start_freq2_nospm2:end_freq2_nospm2]
del AlphaSed_freq22[start_freq2_nospm:end_freq2_nospm]
interp_alphaSed2_spm = np.polyfit(AlphaSed_freq22,spm_time_freq22, 1)
lin_model_alphaSed2_spm = [AlphaSed_freq22[i]*interp_alphaSed2_spm[0]+interp_alphaSed2_spm[1]
                            for i in range(len(AlphaSed_freq22))]

# R²
R2_time_freq1_AlphaSed_freq1_spm = r2_score(spm_time_freq12, lin_model_alphaSed1_spm)
R2_time_freq1_AlphaSed_freq2_spm = r2_score(spm_time_freq22, lin_model_alphaSed2_spm)

## Force origin = 0
# Regression between AlphaSed Frequence 1 & SPM
x = np.array(AlphaSed_freq12)
y = np.array(spm_time_freq12)
x = x[:,np.newaxis]
slope_AlphaSed_freq1_spm, _, _, _ = np.linalg.lstsq(x, y)
lin_model_alphaSed1_spm_origin = [AlphaSed_freq12[i]*slope_AlphaSed_freq1_spm
                            for i in range(len(AlphaSed_freq12))]
lin_model_alphaSed1_spm_origin_plot = x_range*slope_AlphaSed_freq1_spm
R2_AlphaSed_freq1_spm_origin = r2_score(spm_time_freq12, lin_model_alphaSed1_spm_origin)

# Regression between AlphaSed Frequence 2 & SPM
x = np.array(AlphaSed_freq22)
y = np.array(spm_time_freq22)
x = x[:,np.newaxis]
slope_AlphaSed_freq2_spm, _, _, _ = np.linalg.lstsq(x, y)
lin_model_alphaSed2_spm_origin = [AlphaSed_freq22[i]*slope_AlphaSed_freq2_spm
                            for i in range(len(AlphaSed_freq22))]
lin_model_alphaSed2_spm_origin_plot = x_range*slope_AlphaSed_freq2_spm
R2_AlphaSed_freq2_spm_origin = r2_score(spm_time_freq22, lin_model_alphaSed2_spm_origin)

# #--------------------
# # Regression between C fines & AlphaSed Frequence 1
TAAPS_freq1_fines = TAAPS_freq1.drop(TAAPS_freq1[TAAPS_freq1['Sampler'] == 'BD'].index, inplace = False)
TAAPS_freq1_fines.reset_index(drop = True, inplace = True)
# interp_C_fines_alphaSed1 = np.polyfit(TAAPS_freq1_fines['Alpha Sediment (dB/m)'],TAAPS_freq1_fines['Fine_concentration_g_l'], 1)
# lin_model_C_fines_alphaSed1 = [TAAPS_freq1_fines['Alpha Sediment (dB/m)'][i]*interp_C_fines_alphaSed1[0]+interp_C_fines_alphaSed1[1]
#                            for i in range(len(TAAPS_freq1_fines['Alpha Sediment (dB/m)']))]

# # # Regression between AlphaSed Frequence 2 & C fines
TAAPS_freq2_fines = TAAPS_freq2.drop(TAAPS_freq2[TAAPS_freq2['Sampler'] == 'BD'].index, inplace = False)
TAAPS_freq2_fines.reset_index(drop = True, inplace = True)


#%% Create AlphaSed - Cfines calibration dataset

Cfines_TAAPS_ISCO = TAAPS_freq2_fines['Fine_concentration_g_l'].append(ISCO_data['Fine_concentration_g_l'])
Cfines_TAAPS_ISCO.reset_index(drop = True, inplace = True)

AlphaSed_freq1_TAAPS_ISCO = TAAPS_freq1_fines['Alpha Sediment (dB/m)'].append(ISCO_data['AlphaSed_freq1'])
AlphaSed_freq1_TAAPS_ISCO.reset_index(drop = True, inplace = True)

AlphaSed_freq2_TAAPS_ISCO = TAAPS_freq2_fines['Alpha Sediment (dB/m)'].append(ISCO_data['AlphaSed_freq2'])
AlphaSed_freq2_TAAPS_ISCO.reset_index(drop = True, inplace = True)

# Regression freq1
x_range = np.linspace(0,10,100)
x = np.array(AlphaSed_freq1_TAAPS_ISCO)
y = np.array(Cfines_TAAPS_ISCO)
x = x[:,np.newaxis]
slope_alphaSed1_Cfines_TAAPS_ISCO, _, _, _ = np.linalg.lstsq(x, y)
lin_model_alphaSed1_Cfines_TAAPS_ISCO = [AlphaSed_freq1_TAAPS_ISCO[i]*slope_alphaSed1_Cfines_TAAPS_ISCO
                            for i in range(len(AlphaSed_freq1_TAAPS_ISCO))]
lin_model_alphaSed1_Cfines_TAAPS_ISCO_plot = x_range*slope_alphaSed1_Cfines_TAAPS_ISCO
R2_alphaSed1_Cfines_TAAPS_ISCO = r2_score(Cfines_TAAPS_ISCO, lin_model_alphaSed1_Cfines_TAAPS_ISCO)

# Regression freq2
x = np.array(AlphaSed_freq2_TAAPS_ISCO)
y = np.array(Cfines_TAAPS_ISCO)
x = x[:,np.newaxis]
slope_alphaSed2_Cfines_TAAPS_ISCO, _, _, _ = np.linalg.lstsq(x, y)
lin_model_alphaSed2_Cfines_TAAPS_ISCO = [AlphaSed_freq2_TAAPS_ISCO[i]*slope_alphaSed2_Cfines_TAAPS_ISCO
                            for i in range(len(AlphaSed_freq2_TAAPS_ISCO))]
lin_model_alphaSed2_Cfines_TAAPS_ISCO_plot = x_range*slope_alphaSed2_Cfines_TAAPS_ISCO
R2_alphaSed2_Cfines_TAAPS_ISCO = r2_score(Cfines_TAAPS_ISCO, lin_model_alphaSed2_Cfines_TAAPS_ISCO)

# #%% STEP 9: CALCULATE CFINES
# # Cfines = AlphaUnit/AlphaSed (origin)
# C_fines_est_freq1 = [slope_alphaSed1_Cfines_TAAPS_ISCO[0]* AlphaSed_freq1[i]
#                      for i in range(len(AlphaSed_freq1))]
# C_fines_est_freq2 = [slope_alphaSed2_Cfines_TAAPS_ISCO[0] * AlphaSed_freq2[i] 
#                      for i in range(len(AlphaSed_freq2))]
# C_fines_est_freq1_time_freq2 = np.interp(Time_datetime_freq2, Time_datetime_freq1, C_fines_est_freq1 )
# # Average estimated Cfines
# C_fines_est_time_freq2 =  ((C_fines_est_freq2 + C_fines_est_freq1_time_freq2)/2).tolist()


#%% STEP 9: Plot AlphaSed - C fines from ISCO, P6 solid gaugings
x_range = np.linspace(0,10,100)
fig, ax = plt.subplots(1, 2, figsize = (12,6), dpi=300)

# Freq1

m1, = ax[0].plot(ISCO_data['AlphaSed_freq1'], ISCO_data['Fine_concentration_g_l'],
    'o', color = 'yellowgreen', markersize = 6, markeredgecolor = 'black', markeredgewidth = 0.1,
    zorder = 10, label = r'400 kHz')
m4, = ax[0].plot(TAAPS_freq2['Alpha Sediment (dB/m)'][0], TAAPS_freq2['Fine_concentration_g_l'][0],
    'D', color = 'darkorange', markersize = 8, markeredgecolor = 'black', markeredgewidth = 0.1,
    zorder = 30, label = '1 MHz')
m2, = ax[0].plot(TAAPS_freq1['Alpha Sediment (dB/m)'], TAAPS_freq1['Fine_concentration_g_l'],
    'D', color = 'darkorange', markersize = 8, markeredgecolor = 'black', markeredgewidth = 0.1,
    zorder = 30, label = r'$\mathregular{\overline{C_{fines}}}$')
m3, = ax[0].plot(ISCO_data['AlphaSed_freq2'][0], ISCO_data['Fine_concentration_g_l'][0],
    'o', color = 'yellowgreen', markersize = 6, markeredgecolor = 'black', markeredgewidth = 0.1,
    zorder = 10, label = r'$\mathregular{C_{fines, ISCO}}$')
ax[0].plot(x_range, lin_model_alphaSed1_Cfines_TAAPS_ISCO_plot, color = 'black', ls = '-')

ax[0].text(0.05, 0.95, '400 kHz', fontsize = 16, transform = ax[0].transAxes, weight = 'bold')

ax[0].text(0.35, 0.95, ('y = ' + str(np.round(slope_alphaSed1_Cfines_TAAPS_ISCO[0],2)) + 'x'),
    color = 'black', fontsize = 16, transform = ax[0].transAxes)
ax[0].text(0.35, 0.89, ('R² = ' + str(float(np.round(R2_alphaSed1_Cfines_TAAPS_ISCO,2))) + ', n = ' + str(len(Cfines_TAAPS_ISCO))),
    color = 'black', fontsize = 16, transform = ax[0].transAxes)

ax[0].set_ylabel('$\mathregular{\overline{C_{fines}}}$ (g/l)', fontsize=18, weight = 'bold')
ax[0].tick_params(axis='both', which='major', labelsize = 16)
ax[0].set_xlim (0, 2)
ax[0].set_ylim(0,5)

# Freq2

ax[1].plot(TAAPS_freq2['Alpha Sediment (dB/m)'], TAAPS_freq2['Fine_concentration_g_l'],
    'D', color = 'darkorange', markersize = 8, markeredgecolor = 'black', markeredgewidth = 0.1,
    zorder = 30, label = 'Sampler')
m3, = ax[1].plot(ISCO_data['AlphaSed_freq2'], ISCO_data['Fine_concentration_g_l'],
    'o', color = 'yellowgreen', markersize = 6, markeredgecolor = 'black', markeredgewidth = 0.1,
    zorder = 10, label = r'ISCO')
m3, = ax[1].plot(ISCO_data['AlphaSed_freq2'][0], ISCO_data['Fine_concentration_g_l'][0],
    'o', color = 'yellowgreen', markersize = 10, markeredgecolor = 'black', markeredgewidth = 0.1,
    zorder = 10)

ax[1].text(0.05, 0.95, '1 MHz', fontsize = 16, transform = ax[1].transAxes, weight = 'bold')
ax[1].text(0.35, 0.95, ('y = ' + str(np.round(slope_alphaSed2_Cfines_TAAPS_ISCO[0],2)) + 'x'),
    color = 'black', fontsize = 16, transform = ax[1].transAxes)
ax[1].text(0.35, 0.89, ('R² = ' + str(float(np.round(R2_alphaSed2_Cfines_TAAPS_ISCO,2))) + ', n = ' + str(len(Cfines_TAAPS_ISCO))),
    color = 'black', fontsize = 16, transform = ax[1].transAxes)

ax[1].plot(x_range, lin_model_alphaSed2_Cfines_TAAPS_ISCO_plot, color = 'black', ls = '-')
ax[1].tick_params(axis='both', which='major', labelsize = 16)
ax[1].set_xlim (0, 4)
ax[1].set_ylim(0,5)
ax[1].yaxis.tick_right()
ax[1].legend(fontsize = 16, loc = 'lower right')

fig.supxlabel(r'$\mathregular{α_{sed}}$ (dB/m)', fontsize=18, weight = 'bold')

fig.tight_layout()
figname = 'AlphaSed_Cfines_TW16'
fig.savefig(outpath_figures +'\\' +  figname + '.png', dpi = 300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 300, bbox_inches='tight')


#%% Determine theoretical slope (alphaunit)

# using Richards viscous attenuation
# h = 1/2
# D50_fines_ref_Rich_mum = 1
# sigma_fines_ref_Rich_mum = 1
# ref_dist_fines_Rich_freq1 = compute_model_lognorm_spherical(D50_fines_ref_Rich_mum*1e-6, sigma_fines_ref_Rich_mum, freq1_Hz, h, rho_sed, nu_0)
# ref_dist_fines_Rich_freq2 = compute_model_lognorm_spherical(D50_fines_ref_Rich_mum*1e-6, sigma_fines_ref_Rich_mum, freq2_Hz, h, rho_sed, nu_0)
# zeta_Rich_fines_freq1 = ref_dist_fines_Rich_freq1.zeta_Rich
# zeta_Rich_fines_freq2 = ref_dist_fines_Rich_freq2.zeta_Rich

x_range = np.linspace(0,10,100)
# use Moate and Thorne for zetav
alphaunit_freq1 = 1/(zeta_fines_freq1*20/(np.log(10)))
alphaunit_freq2 = 1/(zeta_fines_freq2*20/(np.log(10)))

lin_model_alphaSed1_Cfines_theo_plot = x_range*alphaunit_freq1
lin_model_alphaSed1_Cfines_theo = [AlphaSed_freq1_TAAPS_ISCO[i]*alphaunit_freq1
                        for i in range(len(AlphaSed_freq1_TAAPS_ISCO))]
R2_alphaSed1_Cfines_theo = r2_score(Cfines_TAAPS_ISCO, lin_model_alphaSed1_Cfines_theo)

lin_model_alphaSed2_Cfines_theo_plot = x_range*alphaunit_freq2
lin_model_alphaSed2_Cfines_theo = [AlphaSed_freq2_TAAPS_ISCO[i]*alphaunit_freq2
                        for i in range(len(AlphaSed_freq2_TAAPS_ISCO))]
R2_alphaSed2_Cfines_theo = r2_score(Cfines_TAAPS_ISCO, lin_model_alphaSed2_Cfines_theo)

# use Richards for zetav
# alphaunit_Rich_freq1 = 1/(zeta_Rich_fines_freq1*20/(np.log(10)))
# alphaunit_Rich_freq2 = 1/(zeta_Rich_fines_freq2*20/(np.log(10)))

# lin_model_alphaSed1_Cfines_theo_Rich_plot = x_range*alphaunit_Rich_freq1
# lin_model_alphaSed1_Cfines_theo_Rich = [AlphaSed_freq1_TAAPS_ISCO[i]*alphaunit_Rich_freq1
#                         for i in range(len(AlphaSed_freq1_TAAPS_ISCO))]
# R2_alphaSed1_Cfines_theo_Rich = r2_score(Cfines_TAAPS_ISCO, lin_model_alphaSed1_Cfines_theo_Rich)

# lin_model_alphaSed2_Cfines_theo_Rich_plot = x_range*alphaunit_Rich_freq2
# lin_model_alphaSed2_Cfines_theo_Rich = [AlphaSed_freq2_TAAPS_ISCO[i]*alphaunit_Rich_freq2
#                         for i in range(len(AlphaSed_freq2_TAAPS_ISCO))]
# R2_alphaSed2_Cfines_theo_Rich = r2_score(Cfines_TAAPS_ISCO, lin_model_alphaSed2_Cfines_theo_Rich)

#%% STEP 9: Plot AlphaSed - C fines from ISCO, P6 solid gaugings including theoretical alphaunit
x_range = np.linspace(0,10,100)
fig, ax = plt.subplots(1, 2, figsize = (12,6), dpi=300)

# Freq1

m1, = ax[0].plot(ISCO_data['AlphaSed_freq1'], ISCO_data['Fine_concentration_g_l'],
    'o', color = 'yellowgreen', markersize = 6, markeredgecolor = 'black', markeredgewidth = 0.1,
    zorder = 10, label = r'400 kHz')
m4, = ax[0].plot(TAAPS_freq2['Alpha Sediment (dB/m)'][0], TAAPS_freq2['Fine_concentration_g_l'][0],
    'D', color = 'darkorange', markersize = 8, markeredgecolor = 'black', markeredgewidth = 0.1,
    zorder = 30, label = '1 MHz')
m2, = ax[0].plot(TAAPS_freq1['Alpha Sediment (dB/m)'], TAAPS_freq1['Fine_concentration_g_l'],
    'D', color = 'darkorange', markersize = 8, markeredgecolor = 'black', markeredgewidth = 0.1,
    zorder = 30, label = r'$\mathregular{\overline{C_{fines}}}$')
m3, = ax[0].plot(ISCO_data['AlphaSed_freq2'][0], ISCO_data['Fine_concentration_g_l'][0],
    'o', color = 'yellowgreen', markersize = 6, markeredgecolor = 'black', markeredgewidth = 0.1,
    zorder = 10, label = r'$\mathregular{C_{fines, ISCO}}$')
ax[0].plot(x_range, lin_model_alphaSed1_Cfines_theo_plot, color = 'black', ls = '-')
ax[0].plot(x_range, lin_model_alphaSed1_Cfines_TAAPS_ISCO_plot, color = 'grey', ls = '--')

# ax[0].text(0.05, 0.95, '400 kHz', fontsize = 16, transform = ax[0].transAxes, weight = 'bold')

ax[0].text(0.05, 0.95, '(a)', fontsize = 16, transform = ax[0].transAxes)
# ax[0].text(0.35, 0.95, ('y = ' + str(np.round(slope_alphaSed1_Cfines_TAAPS_ISCO[0],2)) + 'x'),
#     color = 'black', fontsize = 16, transform = ax[0].transAxes)
# ax[0].text(0.35, 0.89, ('R² = ' + str(float(np.round(R2_alphaSed1_Cfines_TAAPS_ISCO,2))) + ', n = ' + str(len(Cfines_TAAPS_ISCO))),
#     color = 'black', fontsize = 16, transform = ax[0].transAxes)

ax[0].text(0.7, 0.26, ('y = ' + str(np.round(alphaunit_freq1,2)) + 'x'),
    color = 'black', fontsize = 16, transform = ax[0].transAxes)
ax[0].text(0.7, 0.2, ('R² = ' + str(float(np.round(R2_alphaSed1_Cfines_theo,2)))),
    color = 'black', fontsize = 16, transform = ax[0].transAxes)

ax[0].set_ylabel('$\mathregular{\overline{C_{fines}}}$ (g/l)', fontsize=18, weight = 'bold')
ax[0].tick_params(axis='both', which='major', labelsize = 16)
ax[0].set_xlim (0, 2)
ax[0].set_ylim(0,5)

# Freq2
ax[1].text(0.05, 0.95, '(b)', fontsize = 16, transform = ax[1].transAxes)
ax[1].plot(TAAPS_freq2['Alpha Sediment (dB/m)'], TAAPS_freq2['Fine_concentration_g_l'],
    'D', color = 'darkorange', markersize = 8, markeredgecolor = 'black', markeredgewidth = 0.1,
    zorder = 30, label = 'Sampler')
m3, = ax[1].plot(ISCO_data['AlphaSed_freq2'], ISCO_data['Fine_concentration_g_l'],
    'o', color = 'yellowgreen', markersize = 6, markeredgecolor = 'black', markeredgewidth = 0.1,
    zorder = 10, label = r'ISCO')
m3, = ax[1].plot(ISCO_data['AlphaSed_freq2'][0], ISCO_data['Fine_concentration_g_l'][0],
    'o', color = 'yellowgreen', markersize = 10, markeredgecolor = 'black', markeredgewidth = 0.1,
    zorder = 10)

# ax[1].text(0.05, 0.95, '1 MHz', fontsize = 16, transform = ax[1].transAxes, weight = 'bold')
# ax[1].text(0.35, 0.95, ('y = ' + str(np.round(slope_alphaSed2_Cfines_TAAPS_ISCO[0],2)) + 'x'),
#     color = 'black', fontsize = 16, transform = ax[1].transAxes)
# ax[1].text(0.35, 0.89, ('R² = ' + str(float(np.round(R2_alphaSed2_Cfines_TAAPS_ISCO,2))) + ', n = ' + str(len(Cfines_TAAPS_ISCO))),
#     color = 'black', fontsize = 16, transform = ax[1].transAxes)

ax[1].text(0.7, 0.26, ('y = ' + str(np.round(alphaunit_freq2,2)) + 'x'),
    color = 'black', fontsize = 16, transform = ax[1].transAxes)
ax[1].text(0.7, 0.2, ('R² = ' + str(float(np.round(R2_alphaSed2_Cfines_theo,2)))),
    color = 'black', fontsize = 16, transform = ax[1].transAxes)

ax[1].plot(x_range, lin_model_alphaSed2_Cfines_theo_plot, color = 'black', ls = '-')
ax[1].plot(x_range, lin_model_alphaSed2_Cfines_TAAPS_ISCO_plot, color = 'grey', ls = '--')
ax[1].tick_params(axis='both', which='major', labelsize = 16)
ax[1].set_xlim (0, 4)
ax[1].set_ylim(0,5)
ax[1].yaxis.tick_right()
ax[1].legend(fontsize = 16, loc = 'lower right')

fig.supxlabel(r'$\mathregular{α_{sed}}$ (dB/m)', fontsize=18, weight = 'bold')

fig.tight_layout()
figname = 'AlphaSed_Cfines_theo_TW16'
fig.savefig(outpath_figures +'\\' +  figname + '_' + str(D50_fines_ref_mum) + '_' + str(sigma_fines_ref_mum)+ '.png', dpi = 300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 300, bbox_inches='tight')

#%% STEP 9: CALCULATE CFINES
# Cfines = AlphaUnit/AlphaSed (origin)
C_fines_est_freq1 = [alphaunit_freq1* AlphaSed_freq1[i]
                     for i in range(len(AlphaSed_freq1))]
C_fines_est_freq2 = [alphaunit_freq2 * AlphaSed_freq2[i] 
                     for i in range(len(AlphaSed_freq2))]
C_fines_est_freq1_time_freq2 = np.interp(Time_datetime_freq2, Time_datetime_freq1, C_fines_est_freq1 )
# Average estimated Cfines
C_fines_est_time_freq2 =  ((C_fines_est_freq2 + C_fines_est_freq1_time_freq2)/2).tolist()

#%% STEP 10: BBC-RELATION TO SOLVE FOR K1 AND K2

# Regression between beam-averaged backscatter Frequence 1 & log sand
x_range = np.arange(0,150,0.01)
interp_TAAPS1_logsand = np.polyfit(TAAPS_freq1['Beam-Averaged Backscatter (dB)'], TAAPS_freq1['log_sand'], 1)
lin_model_TAAPS1_logsand_R2 = [TAAPS_freq1['Beam-Averaged Backscatter (dB)'][i]*interp_TAAPS1_logsand[0]+interp_TAAPS1_logsand[1]
                            for i in range(len(TAAPS_freq1['Beam-Averaged Backscatter (dB)']))] # BBC - eq. 60 
lin_model_TAAPS1_logsand = [x_range[i]*interp_TAAPS1_logsand[0]+interp_TAAPS1_logsand[1]
                           for i in range(len(x_range))] # BBC - eq. 60 

# Regression between beam-averaged backscatter freq 2 & log sand
interp_TAAPS2_logsand = np.polyfit(TAAPS_freq2['Beam-Averaged Backscatter (dB)'], TAAPS_freq2['log_sand'], 1)
lin_model_TAAPS2_logsand_R2 = [TAAPS_freq2['Beam-Averaged Backscatter (dB)'][i]*interp_TAAPS2_logsand[0]+interp_TAAPS2_logsand[1]
                           for i in range(len(TAAPS_freq2['Beam-Averaged Backscatter (dB)']))] # BBC - eq. 60, "real equation", used for R2 
lin_model_TAAPS2_logsand = [x_range[i]*interp_TAAPS2_logsand[0]+interp_TAAPS2_logsand[1]
                           for i in range(len(x_range))] # BBC - eq. 60, for plot 

# R²
R2_TAAPS1_logsand = r2_score(TAAPS_freq1['log_sand'], lin_model_TAAPS1_logsand_R2)
R2_TAAPS2_logsand = r2_score(TAAPS_freq2['log_sand'], lin_model_TAAPS2_logsand_R2)

# Regression between beam-averaged backscatter Frequence 1 & C sand
x_range = np.arange(0,150,0.01)
interp_TAAPS1_Csand = np.polyfit(TAAPS_freq1['Beam-Averaged Backscatter (dB)'], np.log(TAAPS_freq1['Sand_concentration_g_l']), 1)
lin_model_TAAPS1_Csand_R2 = [TAAPS_freq1['Beam-Averaged Backscatter (dB)'][i]*interp_TAAPS1_Csand[0]+interp_TAAPS1_Csand[1]
                            for i in range(len(TAAPS_freq1['Beam-Averaged Backscatter (dB)']))] # BBC - eq. 60 
lin_model_TAAPS1_Csand = [x_range[i]*interp_TAAPS1_Csand[0]+interp_TAAPS1_Csand[1]
                           for i in range(len(x_range))]
# R²
R2_TAAPS1_Csand = r2_score(TAAPS_freq1['Sand_concentration_g_l'], np.exp(lin_model_TAAPS1_Csand_R2))

#-------------------------------------------------------------------------------------
# Only perform BBC on samples where S < 2
TAAPS_freq1_S2 = TAAPS_freq1[TAAPS_freq1['S'] < 3]
TAAPS_freq2_S2 = TAAPS_freq2[TAAPS_freq2['S'] < 3]
TAAPS_freq1_S2.reset_index(inplace=True)
TAAPS_freq2_S2.reset_index(inplace=True)
       
#-------------------------------------------------------------------------------------
# Only perform BBC on samples where S < 2 and D50_sand (phi) < 1/4 phi D50_sand_ref
TAAPS_freq1_S2_D50 = TAAPS_freq1_S2[TAAPS_freq1_S2['D50_mum'].le(D50_sand_ref_range_mum[0]) & TAAPS_freq1_S2['D50_mum'].ge(D50_sand_ref_range_mum[1])]
TAAPS_freq2_S2_D50 = TAAPS_freq2_S2[TAAPS_freq2_S2['D50_mum'].le(D50_sand_ref_range_mum[0]) & TAAPS_freq2_S2['D50_mum'].ge(D50_sand_ref_range_mum[1])]
TAAPS_freq1_S2_D50.reset_index(inplace=True)
TAAPS_freq2_S2_D50.reset_index(inplace=True)
      
interp_TAAPS1_logsand_S2_D50 = np.polyfit(TAAPS_freq1_S2_D50['Beam-Averaged Backscatter (dB)'], TAAPS_freq1_S2_D50['log_sand'], 1)
lin_model_TAAPS1_logsand_S2_D50_R2 = [TAAPS_freq1_S2_D50['Beam-Averaged Backscatter (dB)'][i]*interp_TAAPS1_logsand_S2_D50[0]+interp_TAAPS1_logsand_S2_D50[1]
                            for i in range(len(TAAPS_freq1_S2_D50['Beam-Averaged Backscatter (dB)']))] # BBC - eq. 60, "real equation", used for R2 
lin_model_TAAPS1_logsand_S2_D50 = [x_range[i]*interp_TAAPS1_logsand_S2_D50[0]+interp_TAAPS1_logsand_S2_D50[1]
                            for i in range(len(x_range))] 

interp_TAAPS2_logsand_S2_D50 = np.polyfit(TAAPS_freq2_S2_D50['Beam-Averaged Backscatter (dB)'], TAAPS_freq2_S2_D50['log_sand'], 1)
lin_model_TAAPS2_logsand_S2_D50 = [x_range[i]*interp_TAAPS2_logsand_S2_D50[0]+interp_TAAPS2_logsand_S2_D50[1]
                            for i in range(len(x_range))]

lin_model_TAAPS2_logsand_S2_D50_R2 = [TAAPS_freq2_S2_D50['Beam-Averaged Backscatter (dB)'][i]*interp_TAAPS2_logsand_S2_D50[0]+interp_TAAPS2_logsand_S2_D50[1]
                            for i in range(len(TAAPS_freq2_S2_D50['Beam-Averaged Backscatter (dB)']))] 

# R²
R2_TAAPS1_logsand_S2_D50 = r2_score(TAAPS_freq1_S2_D50['log_sand'], lin_model_TAAPS1_logsand_S2_D50_R2)
R2_TAAPS2_logsand_S2_D50 = r2_score(TAAPS_freq2_S2_D50['log_sand'], lin_model_TAAPS2_logsand_S2_D50_R2)


#Empirical determination
# Regression on samples with S < 2 and D50 range
K1_emp_freq1_S2_D50 = interp_TAAPS1_logsand_S2_D50[1]
K1_emp_freq2_S2_D50 = interp_TAAPS2_logsand_S2_D50[1]

K2_emp_freq1_S2_D50 = interp_TAAPS1_logsand_S2_D50[0]
K2_emp_freq2_S2_D50 = interp_TAAPS2_logsand_S2_D50[0]



#%% STEP 10: Plot BBC - color as D50 - MANUSCRIPT
nb_couleurs = 11  
viridis = cm.get_cmap('RdBu_r', nb_couleurs)  
echelle_colorimetrique = viridis(np.linspace(0, 1, nb_couleurs))  
vecteur_blanc = np.array([1, 1, 1, 1])
echelle_colorimetrique[5:6,:] = vecteur_blanc  
cmap = mpl_colors_ListedColormap(echelle_colorimetrique)  
cmap.set_under('black')  
cmap.set_over('saddlebrown')  
cbounds = [75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350]
norm = mpl.colors.BoundaryNorm(cbounds, cmap.N)
                       
fig, ax = plt.subplots(1, 2, figsize = (12,6), dpi=300)

# Freq1 
ax[0].plot(TAAPS_freq1['Beam-Averaged Backscatter (dB)'], TAAPS_freq1['log_sand'],
                color = 'grey', markersize = 5, marker = 'o', ls = '', markeredgewidth = 0.3,
               markeredgecolor='black', zorder = 0)
sc = ax[0].scatter(TAAPS_freq1['Beam-Averaged Backscatter (dB)'], TAAPS_freq1['log_sand'],
                c=TAAPS_freq1['D50_mum'], s= 60, marker = 'o',
                cmap = cmap,norm=norm, edgecolor='black', linewidth=0.2, zorder = 20, label = 'not used')
cax = ax[0].scatter(TAAPS_freq1_S2_D50['Beam-Averaged Backscatter (dB)'], TAAPS_freq1_S2_D50['log_sand'],
                c=TAAPS_freq1_S2_D50['D50_mum'], s= 200, marker = 'D',
                cmap = cmap,norm=norm, edgecolor='black', linewidth=0.4, zorder = 30, label = 'used')
ax[0].plot(x_range, lin_model_TAAPS1_logsand_S2_D50, color = 'black')

# cbar = fig.colorbar(cax, ax=ax, extend='both', ticks=[75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350])
# cbar.ax.set_yticklabels(['75','100','125','150','175','200','225','250','275','300', '325', '350'],ha='right')
# cbar.ax.yaxis.set_tick_params(pad=35) 
# cbar.set_label(r'$\mathregular{\overline{D_{50}} \; (\mu m)}$', labelpad= 10, fontsize = 16)
 
ax[0].text(0.05, 0.95, '(a)', fontsize = 16, transform = ax[0].transAxes) 
ax[0].text(0.3, 0.95, ('y = ' + str(np.round(interp_TAAPS1_logsand_S2_D50[0],2)) + 'x + (' + 
                      str(np.round(interp_TAAPS1_logsand_S2_D50[1],2)) + ')'),color = 'black', fontsize = 16,
           transform = ax[0].transAxes)
ax[0].text(0.3, 0.89, ('R² = ' + str(np.round(R2_TAAPS1_logsand_S2_D50,3)) + ', n = ' + 
                      str(len(TAAPS_freq1_S2_D50))), color = 'black', fontsize = 16,
           transform = ax[0].transAxes)

ax[0].set_ylabel(r'$\mathregular{{{log}_{10}}}$ ($\mathregular{\overline{C_{sand}}}$) (mg/l)', fontsize=18, weight = 'bold')
ax[0].tick_params(axis='both', which='major', labelsize = 16)
ax[0].set_xlim (95, 115)
ax[0].set_ylim(0, 3.5)

# Freq2
ax[1].plot(TAAPS_freq2['Beam-Averaged Backscatter (dB)'], TAAPS_freq2['log_sand'],
                color = 'grey', markersize = 5, marker = 'o', ls = '', markeredgewidth = 0.2,
               markeredgecolor='black', zorder = 0)
ax[1].scatter(TAAPS_freq2['Beam-Averaged Backscatter (dB)'], TAAPS_freq2['log_sand'],
                c=TAAPS_freq2['D50_mum'], s= 60, marker = 'o',
                cmap = cmap,norm=norm, edgecolor='black', linewidth=0.2, zorder = 20)
ax[1].scatter(TAAPS_freq2_S2_D50['Beam-Averaged Backscatter (dB)'], TAAPS_freq2_S2_D50['log_sand'],
                c=TAAPS_freq2_S2_D50['D50_mum'], s= 200, marker = 'D',
                cmap = cmap,norm=norm, edgecolor='black', linewidth=0.2, zorder = 30)
ax[1].plot(x_range, lin_model_TAAPS2_logsand_S2_D50, color = 'black')

# cbar = fig.colorbar(cax, ax=ax, extend='both', ticks=[75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350])
# cbar.ax.set_yticklabels(['75','100','125','150','175','200','225','250','275','300', '325', '350'],ha='right')
# cbar.ax.yaxis.set_tick_params(pad=35) 
# cbar.set_label(r'$\mathregular{\overline{D_{50}} \; (\mu m)}$', labelpad= 10, fontsize = 16)

# handles = [(m1, m3), (m2, m4)]
# _, labels = ax[0].get_legend_handles_labels()
# fig.legend(handles = handles, labels=labels, 
#           handler_map = {tuple: mpl.legend_handler.HandlerTuple(None)},
#           loc="lower right", ncol=2, bbox_to_anchor=(0.1, 0.13), 
#           markerscale = 5, fontsize = 18)

ax[1].text(0.05, 0.95, '(b)', fontsize = 16, transform = ax[1].transAxes) 
ax[1].text(0.3, 0.95,('y = ' + str(np.round(interp_TAAPS2_logsand_S2_D50[0],2)) + 'x + (' + 
                      str(np.round(interp_TAAPS2_logsand_S2_D50[1],2)) + ')'),color = 'black', fontsize = 16,
           transform = ax[1].transAxes)
ax[1].text(0.3, 0.89, ('R² = ' + str(np.round(R2_TAAPS2_logsand_S2_D50,3)) + ', n = ' + 
                      str(len(TAAPS_freq2_S2_D50))), color = 'black', fontsize = 16,
           transform = ax[1].transAxes)

ax[1].tick_params(axis='both', which='major', labelsize = 16)
ax[1].set_xlim (60,75)
ax[1].set_ylim(0, 3.5)
ax[1].yaxis.tick_right()

ax[0].legend(fontsize = 16, loc = 'lower right')

fig.supxlabel(r'$\mathregular{\overline{B}}$ (dB)', fontsize=18, weight = 'bold')

fig.tight_layout()
figname = 'Base_backscatter_calibration_D50'
fig.savefig(outpath_figures +'\\' +  figname + '.png', dpi = 300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight') 
# fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 300, bbox_inches='tight')


#%% STEP 10: Plot BBC   
x_range = np.arange(0,150,0.01)                     
fig, ax = plt.subplots(1, 2, figsize = (12,6), dpi=300)

# Freq1 
ax[0].plot(TAAPS_freq1['Beam-Averaged Backscatter (dB)'], TAAPS_freq1['log_sand'],
                color = 'darkorange', markersize = 10, marker = 'D', ls = '', markeredgewidth = 0.3,
               markeredgecolor='black', zorder = 40)
ax[0].plot(pump_data['BeamAvBS_freq1'], pump_data['log_sand'],
                color = 'mediumblue', markersize = 8, marker = 's', ls = '', markeredgewidth = 0.3,
               markeredgecolor='black', zorder = 30)
ax[0].plot(ISCO_data['BeamAvBS_freq1'], ISCO_data['log_sand'],
                color = 'yellowgreen', markersize = 5, marker = 'o', ls = '', markeredgewidth = 0.3,
               markeredgecolor='black', zorder = 0)
ax[0].plot(x_range, lin_model_TAAPS1_logsand_S2_D50, color = 'black')
 
ax[0].text(0.05, 0.95, '(a)', fontsize = 16, transform = ax[0].transAxes) 
ax[0].text(0.15, 0.95, ('y = ' + str(np.round(interp_TAAPS1_logsand_S2_D50[0],2)) + 'x + (' + 
                      str(np.round(interp_TAAPS1_logsand_S2_D50[1],2)) + ')'),color = 'black', fontsize = 16,
            transform = ax[0].transAxes)
ax[0].text(0.15, 0.89, ('R² = ' + str(np.round(R2_TAAPS1_logsand_S2_D50,3)) + ', n = ' + 
                      str(len(TAAPS_freq1_S2_D50))), color = 'black', fontsize = 16,
            transform = ax[0].transAxes)

ax[0].set_ylabel(r'$\mathregular{{{log}_{10}}}$ ($\mathregular{\overline{C_{sand}}}$) (mg/l)', fontsize=18, weight = 'bold')
ax[0].tick_params(axis='both', which='major', labelsize = 16)
ax[0].set_xlim (94, 115)
ax[0].set_ylim(0, 3.5)

# Freq2
ax[1].plot(TAAPS_freq2['Beam-Averaged Backscatter (dB)'], TAAPS_freq2['log_sand'],
                color = 'darkorange', markersize = 10, marker = 'D', ls = '', markeredgewidth = 0.2,
               markeredgecolor='black', zorder = 40)
ax[1].plot(pump_data['BeamAvBS_freq2'], pump_data['log_sand'],
                color = 'mediumblue', markersize = 8, marker = 's', ls = '', markeredgewidth = 0.3,
               markeredgecolor='black', zorder = 30)
ax[1].plot(ISCO_data['BeamAvBS_freq2'], ISCO_data['log_sand'],
                color = 'yellowgreen', markersize = 5, marker = 'o', ls = '', markeredgewidth = 0.3,
               markeredgecolor='black', zorder = 0)
ax[1].plot(x_range, lin_model_TAAPS2_logsand_S2_D50, color = 'black')

ax[1].text(0.05, 0.95, '(b)', fontsize = 16, transform = ax[1].transAxes) 
ax[1].text(0.15, 0.95,('y = ' + str(np.round(interp_TAAPS2_logsand_S2_D50[0],2)) + 'x + (' + 
                      str(np.round(interp_TAAPS2_logsand_S2_D50[1],2)) + ')'),color = 'black', fontsize = 16,
           transform = ax[1].transAxes)
ax[1].text(0.15, 0.89, ('R² = ' + str(np.round(R2_TAAPS2_logsand_S2_D50,3)) + ', n = ' + 
                      str(len(TAAPS_freq2_S2_D50))), color = 'black', fontsize = 16,
           transform = ax[1].transAxes)

ax[1].tick_params(axis='both', which='major', labelsize = 16)
ax[1].set_xlim (60, 75)
ax[1].set_ylim(0, 3.5)
ax[1].yaxis.tick_right()

# ax[0].legend(fontsize = 16, loc = 'lower right')

fig.supxlabel(r'$\mathregular{\overline{B}}$ (dB)', fontsize=18, weight = 'bold')

fig.tight_layout()
figname = 'Base_backscatter_calibration'
fig.savefig(outpath_figures +'\\' +  figname + '.png', dpi = 300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight') 
# fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 300, bbox_inches='tight')

#%% STEP 10: Plot BBC2
# x_range = np.arange(0,150,0.01)                     
# fig, ax = plt.subplots(1, 2, figsize = (12,6), dpi=300)

# # Freq1 
# ax[0].plot(TAAPS_freq1_S2_D50['Beam-Averaged Backscatter (dB)'], TAAPS_freq1_S2_D50['log_sand'],
#                 color = 'darkorange', markersize = 10, marker = 'D', ls = '', markeredgewidth = 0.3,
#                markeredgecolor='black', zorder = 40)
# ax[0].plot(TAAPS_freq1['Beam-Averaged Backscatter (dB)'], TAAPS_freq1['log_sand'],
#                 color = 'lightgrey', markersize = 10, marker = 'D', ls = '', markeredgewidth = 0.3,
#                markeredgecolor='black', zorder = 30, alpha = 0.7)
# ax[0].plot(pump_data['BeamAvBS_freq1'], pump_data['log_sand'],
#                 color = 'lightgrey', markersize = 8, marker = 's', ls = '', markeredgewidth = 0.3,
#                markeredgecolor='black', zorder = 30, alpha = 0.7)
# ax[0].plot(ISCO_data['BeamAvBS_freq1'], ISCO_data['log_sand'],
#                 color = 'lightgrey', markersize = 5, marker = 'o', ls = '', markeredgewidth = 0.3,
#                markeredgecolor='black', zorder = 0, alpha = 0.7)
# ax[0].plot(x_range, lin_model_TAAPS1_logsand_S2_D50, color = 'black')
 
# ax[0].text(0.05, 0.95, '(a)', fontsize = 16, transform = ax[0].transAxes) 
# # ax[0].text(0.05, 0.95, '400 kHz', fontsize = 16, transform = ax[0].transAxes, weight = 'bold') 
# ax[0].text(0.05, 0.85, ('y = ' + str(np.round(interp_TAAPS1_logsand_S2_D50[0],2)) + 'x + (' + 
#                       str(np.round(interp_TAAPS1_logsand_S2_D50[1],2)) + ')'),color = 'black', fontsize = 16,
#             transform = ax[0].transAxes)
# ax[0].text(0.05, 0.79, ('R² = ' + str(np.round(R2_TAAPS1_logsand_S2_D50,3)) + ', n = ' + 
#                       str(len(TAAPS_freq1_S2_D50))), color = 'black', fontsize = 16,
#             transform = ax[0].transAxes)

# ax[0].set_ylabel(r'$\mathregular{{{log}_{10}}}$ ($\mathregular{\overline{C_{sand}}}$) (mg/l)', fontsize=18, weight = 'bold')
# ax[0].tick_params(axis='both', which='major', labelsize = 16)
# ax[0].set_xlim (94, 115)
# ax[0].set_ylim(0, 3.5)

# # Freq2
# ax[1].plot(TAAPS_freq2_S2_D50['Beam-Averaged Backscatter (dB)'], TAAPS_freq2_S2_D50['log_sand'],
#                 color = 'darkorange', markersize = 10, marker = 'D', ls = '', markeredgewidth = 0.3,
#                markeredgecolor='black', zorder = 40)
# ax[1].plot(TAAPS_freq2['Beam-Averaged Backscatter (dB)'], TAAPS_freq2['log_sand'],
#                 color = 'lightgrey', markersize = 10, marker = 'D', ls = '', markeredgewidth = 0.3,
#                markeredgecolor='black', zorder = 30, alpha = 0.7)
# ax[1].plot(pump_data['BeamAvBS_freq2'], pump_data['log_sand'],
#                 color = 'lightgrey', markersize = 8, marker = 's', ls = '', markeredgewidth = 0.3,
#                markeredgecolor='black', zorder = 30, alpha = 0.7)
# ax[1].plot(ISCO_data['BeamAvBS_freq2'], ISCO_data['log_sand'],
#                 color = 'lightgrey', markersize = 5, marker = 'o', ls = '', markeredgewidth = 0.3,
#                markeredgecolor='black', zorder = 0, alpha = 0.7)
# ax[1].plot(x_range, lin_model_TAAPS2_logsand_S2_D50, color = 'black')

# ax[1].text(0.05, 0.95, '(b)', fontsize = 16, transform = ax[1].transAxes)
# # ax[1].text(0.05, 0.95, '1 MHz', fontsize = 16, transform = ax[1].transAxes, weight = 'bold') 
# ax[1].text(0.05, 0.85,('y = ' + str(np.round(interp_TAAPS2_logsand_S2_D50[0],2)) + 'x + (' + 
#                       str(np.round(interp_TAAPS2_logsand_S2_D50[1],2)) + ')'),color = 'black', fontsize = 16,
#             transform = ax[1].transAxes)
# ax[1].text(0.05, 0.79, ('R² = ' + str(np.round(R2_TAAPS2_logsand_S2_D50,3)) + ', n = ' + 
#                       str(len(TAAPS_freq2_S2_D50))), color = 'black', fontsize = 16,
#             transform = ax[1].transAxes)

# ax[1].tick_params(axis='both', which='major', labelsize = 16)
# ax[1].set_xlim (60, 75)
# ax[1].set_ylim(0, 3.5)
# ax[1].yaxis.tick_right()

# # ax[0].legend(fontsize = 16, loc = 'lower right')

# fig.supxlabel(r'$\mathregular{\overline{B}}$ (dB)', fontsize=18, weight = 'bold')

# fig.tight_layout()
# figname = 'Base_backscatter_calibration2'
# fig.savefig(outpath_figures +'\\' +  figname + '.png', dpi = 300, bbox_inches='tight')
# # fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight') 
# # fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 300, bbox_inches='tight')

#%% STEP 10: Plot BBC2b
x_range = np.arange(0,150,0.01)                     
fig, ax = plt.subplots(2, 1, figsize = (6,10), dpi=300)

# Freq1 
ax[0].plot(TAAPS_freq1_S2_D50['Beam-Averaged Backscatter (dB)'], TAAPS_freq1_S2_D50['log_sand'],
                color = 'darkorange', markersize = 10, marker = 'D', ls = '', markeredgewidth = 0.3,
               markeredgecolor='black', zorder = 40, label = 'used')
ax[0].plot(TAAPS_freq1['Beam-Averaged Backscatter (dB)'], TAAPS_freq1['log_sand'],
                color = 'lightgrey', markersize = 10, marker = 'D', ls = '', markeredgewidth = 0,
               markeredgecolor='black', zorder = 30, label = 'not used')
# ax[0].plot(pump_data['BeamAvBS_freq1'], pump_data['log_sand'],
#                 color = 'lightgrey', markersize = 8, marker = 's', ls = '', markeredgewidth = 0,
#                markeredgecolor='black', zorder = 30)
# ax[0].plot(ISCO_data['BeamAvBS_freq1'], ISCO_data['log_sand'],
#                 color = 'lightgrey', markersize = 5, marker = 'o', ls = '', markeredgewidth = 0,
#                markeredgecolor='black', zorder = 0)
ax[0].plot(x_range, lin_model_TAAPS1_logsand_S2_D50, color = 'black')
 
ax[0].text(0.05, 0.95, '400 kHz', fontsize = 16, transform = ax[0].transAxes, weight = 'bold') 
ax[0].text(0.5, 0.15, ('y = ' + str(np.round(interp_TAAPS1_logsand_S2_D50[0],2)) + 'x + (' + 
                      str(np.round(interp_TAAPS1_logsand_S2_D50[1],2)) + ')'),color = 'black', fontsize = 16,
            transform = ax[0].transAxes, zorder = 50)
ax[0].text(0.5, 0.08, ('R² = ' + str(np.round(R2_TAAPS1_logsand_S2_D50,3)) + ', n = ' + 
                      str(len(TAAPS_freq1_S2_D50))), color = 'black', fontsize = 16,
            transform = ax[0].transAxes, zorder = 50)

ax[0].tick_params(axis='both', which='major', labelsize = 16)
ax[0].set_xlim (94, 115)
ax[0].set_ylim(0, 3.5)
ax[0].legend(loc = 'lower left', fontsize = 16)

# Freq2
ax[1].plot(TAAPS_freq2_S2_D50['Beam-Averaged Backscatter (dB)'], TAAPS_freq2_S2_D50['log_sand'],
                color = 'darkorange', markersize = 10, marker = 'D', ls = '', markeredgewidth = 0.3,
               markeredgecolor='black', zorder = 40)
ax[1].plot(TAAPS_freq2['Beam-Averaged Backscatter (dB)'], TAAPS_freq2['log_sand'],
                color = 'lightgrey', markersize = 10, marker = 'D', ls = '', markeredgewidth = 0,
               markeredgecolor='black', zorder = 30)
# ax[1].plot(pump_data['BeamAvBS_freq2'], pump_data['log_sand'],
#                 color = 'lightgrey', markersize = 8, marker = 's', ls = '', markeredgewidth = 0,
#                markeredgecolor='black', zorder = 30)
# ax[1].plot(ISCO_data['BeamAvBS_freq2'], ISCO_data['log_sand'],
#                 color = 'lightgrey', markersize = 5, marker = 'o', ls = '', markeredgewidth = 0,
#                markeredgecolor='black', zorder = 0)
ax[1].plot(x_range, lin_model_TAAPS2_logsand_S2_D50, color = 'black')

ax[1].text(0.05, 0.95, '1 MHz', fontsize = 16, transform = ax[1].transAxes, weight = 'bold') 
ax[1].text(0.5, 0.15,('y = ' + str(np.round(interp_TAAPS2_logsand_S2_D50[0],2)) + 'x + (' + 
                      str(np.round(interp_TAAPS2_logsand_S2_D50[1],2)) + ')'),color = 'black', fontsize = 16,
            transform = ax[1].transAxes)
ax[1].text(0.5, 0.08, ('R² = ' + str(np.round(R2_TAAPS2_logsand_S2_D50,3)) + ', n = ' + 
                      str(len(TAAPS_freq2_S2_D50))), color = 'black', fontsize = 16,
            transform = ax[1].transAxes)

ax[1].tick_params(axis='both', which='major', labelsize = 16)
ax[1].set_xlim (60, 75)
ax[1].set_ylim(0, 3.5)

# ax[0].legend(fontsize = 16, loc = 'lower right')

ax[1].set_xlabel(r'$\mathregular{\overline{B}}$ (dB)', fontsize=18, weight = 'bold')
fig.supylabel(r'$\mathregular{{{log}_{10}}}$ ($\mathregular{\overline{C_{sand}}}$) (mg/l)', fontsize=18, weight = 'bold')

fig.tight_layout()
figname = 'Base_backscatter_calibration2b'
fig.savefig(outpath_figures +'\\' +  figname + '.png', dpi = 300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight') 
# fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 300, bbox_inches='tight')


#%% STEP 10: Plot BBC   S
x_range = np.arange(0,150,0.01)                     
fig, ax = plt.subplots(1, 2, figsize = (12,6), dpi=300)

# Freq1 
ax[0].plot(TAAPS_freq1['Beam-Averaged Backscatter (dB)'][TAAPS_freq1['S']<=2], TAAPS_freq1['log_sand'][TAAPS_freq1['S']<=2],
                color = 'black', markersize = 10, marker = 'D', ls = '', markeredgewidth = 0.3,
               markeredgecolor='black', zorder = 40)
ax[0].plot(TAAPS_freq1['Beam-Averaged Backscatter (dB)'][TAAPS_freq1['S']>2], TAAPS_freq1['log_sand'][TAAPS_freq1['S']>2],
                color = 'red', markersize = 10, marker = 'D', ls = '', markeredgewidth = 0.3,
               markeredgecolor='black', zorder = 30)
ax[0].plot(TAAPS_freq1['Beam-Averaged Backscatter (dB)'][TAAPS_freq1['S']>10], TAAPS_freq1['log_sand'][TAAPS_freq1['S']>10],
                color = 'grey', markersize = 10, marker = 'D', ls = '', markeredgewidth = 0.3,
               markeredgecolor='black', zorder = 30)
ax[0].plot(x_range, lin_model_TAAPS1_logsand_S2_D50, color = 'black')
 
ax[0].text(0.05, 0.95, '(a)', fontsize = 16, transform = ax[0].transAxes) 
ax[0].text(0.15, 0.95, ('y = ' + str(np.round(interp_TAAPS1_logsand_S2_D50[0],2)) + 'x + (' + 
                      str(np.round(interp_TAAPS1_logsand_S2_D50[1],2)) + ')'),color = 'black', fontsize = 16,
            transform = ax[0].transAxes)
ax[0].text(0.15, 0.89, ('R² = ' + str(np.round(R2_TAAPS1_logsand_S2_D50,3)) + ', n = ' + 
                      str(len(TAAPS_freq1_S2_D50))), color = 'black', fontsize = 16,
            transform = ax[0].transAxes)

ax[0].set_ylabel(r'$\mathregular{{{log}_{10}}}$ ($\mathregular{\overline{C_{sand}}}$) (mg/l)', fontsize=18, weight = 'bold')
ax[0].tick_params(axis='both', which='major', labelsize = 16)
ax[0].set_xlim (94, 115)
ax[0].set_ylim(0, 3.5)

# Freq2
ax[1].plot(TAAPS_freq2['Beam-Averaged Backscatter (dB)'][TAAPS_freq2['S']<=2], TAAPS_freq2['log_sand'][TAAPS_freq2['S']<=2],
                color = 'black', markersize = 10, marker = 'D', ls = '', markeredgewidth = 0.3,
               markeredgecolor='black', zorder = 40, label = 'S < 2')
ax[1].plot(TAAPS_freq2['Beam-Averaged Backscatter (dB)'][TAAPS_freq2['S']>2], TAAPS_freq2['log_sand'][TAAPS_freq2['S']>2],
                color = 'red', markersize = 10, marker = 'D', ls = '', markeredgewidth = 0.3,
               markeredgecolor='black', zorder = 30, label = '2 < S < 10')
ax[1].plot(TAAPS_freq2['Beam-Averaged Backscatter (dB)'][TAAPS_freq2['S']>10], TAAPS_freq2['log_sand'][TAAPS_freq2['S']>10],
                color = 'grey', markersize = 10, marker = 'D', ls = '', markeredgewidth = 0.3,
               markeredgecolor='black', zorder = 30, label = '10 < S')
ax[1].plot(x_range, lin_model_TAAPS2_logsand_S2_D50, color = 'black')

ax[1].text(0.05, 0.95, '(b)', fontsize = 16, transform = ax[1].transAxes) 
ax[1].text(0.15, 0.95,('y = ' + str(np.round(interp_TAAPS2_logsand_S2_D50[0],2)) + 'x + (' + 
                      str(np.round(interp_TAAPS2_logsand_S2_D50[1],2)) + ')'),color = 'black', fontsize = 16,
           transform = ax[1].transAxes)
ax[1].text(0.15, 0.89, ('R² = ' + str(np.round(R2_TAAPS2_logsand_S2_D50,3)) + ', n = ' + 
                      str(len(TAAPS_freq2_S2_D50))), color = 'black', fontsize = 16,
           transform = ax[1].transAxes)

ax[1].tick_params(axis='both', which='major', labelsize = 16)
ax[1].set_xlim (60, 75)
ax[1].set_ylim(0, 3.5)
ax[1].yaxis.tick_right()

ax[1].legend(fontsize = 16, loc = 'lower right')

fig.supxlabel(r'$\mathregular{\overline{B}}$ (dB)', fontsize=18, weight = 'bold')

fig.tight_layout()
figname = 'Base_backscatter_calibration_S'
fig.savefig(outpath_figures +'\\' +  figname + '.png', dpi = 300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight') 
# fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 300, bbox_inches='tight')


#%% Calculate theoretical B' (model

sRatio = [10**(-1.1 + 0.1*j) for j in range(0,81)]
densitySed = [(2.65+sRatio[i] *siltDens)/(sRatio[i]+1) for i in range(len(sRatio))]
sRatioSample = TAAPS_freq1_S2_D50['S'].mean()

ExcessBackScat_freq1 = []
d50Bimodal_freq1 = []
formBimodal_freq1 = []
for k in range(len(sRatio)):
        
    # Define boundaries, phi system and convert to mm (2**)   
    gs = [(-35.25 + 0.0625*j) for j in range(0,672)]
    gs = [2**gs[i] for i in range(len(gs))]
    gsbound = [2**(-35.2188 + 0.0625*j) for j in range(0,672)]

    # Calculate ref sand f following TW16
    volfracSand = [(1/np.sqrt(2*math.pi*sortSand))*np.exp((-(np.log(gs[j])-np.log(sandD50ref))**2)/(2*sortSand**2))       
                   for j in range(len(gs))]
    volfracSilt = [(1/np.sqrt(2*math.pi*sortSilt))*np.exp((-(np.log(gs[j])-np.log(siltD50))**2)/(2*sortSilt**2))       
                   for j in range(len(gs))]
    vol = [(4/3)*math.pi*(gs[i]/2)**3 for i in range(len(gs))]
    
    volsumSand = np.cumsum(volfracSand)
    volsumSilt = np.cumsum(volfracSilt)
    
    volfracSand = [volfracSand[i]/volsumSand[-1] for i in range(len(volsumSand))]
    volfracSilt = [sRatio[k]*volfracSilt[i]/volsumSilt[-1] for i in range(len(volsumSilt))]
    
    volcomb = [volfracSand[i] + volfracSilt[i] for i in range(len(volfracSilt))]
    volsum = np.cumsum(volcomb)
    
    d50 = 0.5*volsum[-1]
    for j in range(1,672):
        if d50 >= volsum[j-1] and d50 <= volsum[j]:
            # print(j)
            d50bimodal=((d50 - volsum[j])/(volsum[j] - volsum[j-1]))*(np.log(gsbound[j])-np.log(gsbound[j-1]))+np.log(gsbound[j-1])
    d50bimodal = np.exp(d50bimodal)
    
    gs = [gs[i]/1000 for i in range(len(gs))]
    volrat = [volcomb[i]/vol[i] for i in range(len(vol))]
    
    volsum = np.cumsum(volrat)
    numFrac = [volrat[i]/volsum[-1] for i in range(len(vol))]
    d50bimodal_j = d50bimodal/1000
    
    denom3_1 = (gs[0]/2)**3*numFrac[0]
    denom3_j = denom3_1
    denom3 = []
    for i in range(1,672):  
        denom3_i = 0.5*((gs[i-1]/2)**3*numFrac[i-1]+(gs[i]/2)**3*numFrac[i]) + denom3_j
        denom3_j = denom3_i
        denom3.append(denom3_i)  
    denom3.insert(0,denom3_1)
    
    ka = [wavenum_freq1*(gs[i]/2) for i in range(len(gs))]
    formf = [ka[i]**2*(1-0.35*np.exp(-((ka[i]-1.5)/0.7)**2))*(1+0.5*np.exp(-((ka[i]-1.8)/2.2)**2))/(1+0.9*ka[i]**2)
             for i in range(len(ka))]
    
    forma3_1 = (gs[0]/2)**2*numFrac[0]*formf[0]**2
    forma3_j = forma3_1
    forma3 = []
    for i in range(1,672):  
        forma3_i = 0.5*((gs[i-1]/2)**2*numFrac[i-1]*formf[i-1]**2+(gs[i]/2)**2*numFrac[i]*formf[i]**2)+forma3_j
        forma3_j = forma3_i
        forma3.append(forma3_i)  
    forma3.insert(0,forma3_1)
    
    # total sediment form factor
    formBimodal_j = ((d50bimodal_j/2)*forma3[-1]/denom3[-1])**0.5
    
    # calculate B', excessive backscatter, Topping and Wright 2016, equation 66
    ExcessBackScat_j = (1/K2_emp_freq1_S2_D50)*np.log10(((formBimodal_j/formSandref_freq1)**2)*(sandD50ref/(d50bimodal))*(2.65/densitySed[k])*(1+sRatio[k]))
    
    # append    
    ExcessBackScat_freq1.append(ExcessBackScat_j)
    d50Bimodal_freq1.append(d50bimodal_j)
    formBimodal_freq1.append(formBimodal_j)
 
# Determine offset
for k in range(1,81):
    if sRatioSample >= sRatio[k-1] and sRatioSample <= sRatio[k]:
        offset_freq1 = ((np.log10(sRatioSample)-np.log10(sRatio[k-1]))/(np.log10(sRatio[k])-np.log10(sRatio[k-1])))*(ExcessBackScat_freq1[k]-ExcessBackScat_freq1[k-1])+ExcessBackScat_freq1[k-1]
    
ExcessBackScat_freq1 = [ExcessBackScat_freq1[i]-offset_freq1 for i in range(len(ExcessBackScat_freq1))]


ExcessBackScat_freq2 = []
d50Bimodal_freq2 = []
formBimodal_freq2 = []
for k in range(len(sRatio)):
        
    # Define boundaries, phi system and convert to mm (2**)   
    gs = [(-35.25 + 0.0625*j) for j in range(0,672)]
    gs = [2**gs[i] for i in range(len(gs))]
    gsbound = [2**(-35.2188 + 0.0625*j) for j in range(0,672)]

    # Calculate ref sand f following TW16
    volfracSand = [(1/np.sqrt(2*math.pi*sortSand))*np.exp((-(np.log(gs[j])-np.log(sandD50ref))**2)/(2*sortSand**2))       
                   for j in range(len(gs))]
    volfracSilt = [(1/np.sqrt(2*math.pi*sortSilt))*np.exp((-(np.log(gs[j])-np.log(siltD50))**2)/(2*sortSilt**2))       
                   for j in range(len(gs))]
    vol = [(4/3)*math.pi*(gs[i]/2)**3 for i in range(len(gs))]
    
    volsumSand = np.cumsum(volfracSand)
    volsumSilt = np.cumsum(volfracSilt)
    
    volfracSand = [volfracSand[i]/volsumSand[-1] for i in range(len(volsumSand))]
    volfracSilt = [sRatio[k]*volfracSilt[i]/volsumSilt[-1] for i in range(len(volsumSilt))]
    
    volcomb = [volfracSand[i] + volfracSilt[i] for i in range(len(volfracSilt))]
    volsum = np.cumsum(volcomb)
    
    d50 = 0.5*volsum[-1]
    for j in range(1,672):
        if d50 >= volsum[j-1] and d50 <= volsum[j]:
            # print(j)
            d50bimodal=((d50 - volsum[j])/(volsum[j] - volsum[j-1]))*(np.log(gsbound[j])-np.log(gsbound[j-1]))+np.log(gsbound[j-1])
    d50bimodal = np.exp(d50bimodal)
    
    gs = [gs[i]/1000 for i in range(len(gs))]
    volrat = [volcomb[i]/vol[i] for i in range(len(vol))]
    
    volsum = np.cumsum(volrat)
    numFrac = [volrat[i]/volsum[-1] for i in range(len(vol))]
    d50bimodal_j = d50bimodal/1000
    
    denom3_1 = (gs[0]/2)**3*numFrac[0]
    denom3_j = denom3_1
    denom3 = []
    for i in range(1,672):  
        denom3_i = 0.5*((gs[i-1]/2)**3*numFrac[i-1]+(gs[i]/2)**3*numFrac[i]) + denom3_j
        denom3_j = denom3_i
        denom3.append(denom3_i)  
    denom3.insert(0,denom3_1)
    
    ka = [wavenum_freq2*(gs[i]/2) for i in range(len(gs))]
    formf = [ka[i]**2*(1-0.35*np.exp(-((ka[i]-1.5)/0.7)**2))*(1+0.5*np.exp(-((ka[i]-1.8)/2.2)**2))/(1+0.9*ka[i]**2)
             for i in range(len(ka))]
    
    forma3_1 = (gs[0]/2)**2*numFrac[0]*formf[0]**2
    forma3_j = forma3_1
    forma3 = []
    for i in range(1,672):  
        forma3_i = 0.5*((gs[i-1]/2)**2*numFrac[i-1]*formf[i-1]**2+(gs[i]/2)**2*numFrac[i]*formf[i]**2)+forma3_j
        forma3_j = forma3_i
        forma3.append(forma3_i)  
    forma3.insert(0,forma3_1)
    
    # total sediment form factor
    formBimodal_j = ((d50bimodal_j/2)*forma3[-1]/denom3[-1])**0.5
    
    # calculate B', excessive backscatter, Topping and Wright 2016, equation 66
    ExcessBackScat_j = (1/K2_emp_freq2_S2_D50)*np.log10(((formBimodal_j/formSandref_freq2)**2)*(sandD50ref/(d50bimodal))*(2.65/densitySed[k])*(1+sRatio[k]))
    
    # append    
    ExcessBackScat_freq2.append(ExcessBackScat_j)
    d50Bimodal_freq2.append(d50bimodal_j)
    formBimodal_freq2.append(formBimodal_j)
 
# Determine offset
for k in range(1,81):
    if sRatioSample >= sRatio[k-1] and sRatioSample <= sRatio[k]:
        offset_freq2 = ((np.log10(sRatioSample)-np.log10(sRatio[k-1]))/(np.log10(sRatio[k])-np.log10(sRatio[k-1])))*(ExcessBackScat_freq2[k]-ExcessBackScat_freq2[k-1])+ExcessBackScat_freq2[k-1]
    
ExcessBackScat_freq2 = [ExcessBackScat_freq2[i]-offset_freq2 for i in range(len(ExcessBackScat_freq2))]

#%% Determine sand form factors of samples (TAAPS)

# Define boundaries, phi system and convert to mm (2**)   
gs = [2**(-35.25 + 0.0625*j) for j in range(0,672)]
gsbound = [2**(-35.2188 + 0.0625*j) for j in range(0,672)]

f_TM08_freq1_sand_samples = []
f_TM08_freq2_sand_samples = []
for i in range(len(TAAPS_freq1)):
    # Calculate sand f following TW16
    volfracsand_i = [(1/np.sqrt(2*math.pi*TAAPS_freq1['D50_mum'][i]))*np.exp((-(np.log(gs[j])-np.log(TAAPS_freq1['D50_mum'][i]*1e-3))**2)/(2*TAAPS_freq1['sigma_mum'][i]**2))       
                    for j in range(len(gs))]
    cumsumsand_i = np.cumsum(volfracsand_i)
    proba_vol_sand_i = volfracsand_i/cumsumsand_i[-1]
    
    # Determine D50
    cumsum_sand_i = np.cumsum(proba_vol_sand_i)
    d500_i = 0.5*cumsum_sand_i[-1]
    
    for j in range(len(cumsum_sand_i)):
        if d500_i >= cumsum_sand_i[j-1] and d500_i <= cumsum_sand_i[j]:
            d50sand_i=((d500_i-cumsum_sand_i[j-1])/(cumsum_sand_i[j]-cumsum_sand_i[j-1]))*(np.log(gsbound[j])-np.log(gsbound[j-1]))+np.log(gsbound[j-1])
    d50_sand_i = np.exp(d50sand_i)/1000
    
    # Computing number probability
    ss = np.sum(proba_vol_sand_i / np.array(gs)**3)
    proba_num_sand_i = proba_vol_sand_i/np.array(gs)**3 / ss
    
    # Calculate form function TM08
    # Integrating over the distribution        
    temp_a1 = 0
    temp_a2f2_TM08_freq1 = 0
    temp_a2f2_TM08_freq2 = 0
    temp_a3 = 0  
    # Summing the integrals
    for l in range(len(proba_num_sand_i)):
        a = np.array(gs)[l]/1000
        temp_a2f2_TM08_freq1 += (a/2)**2 * form_factor_function_ThorneMeral2008((a/2), freq1_Hz, 1500)**2 * proba_num_sand_i[l]   
        temp_a2f2_TM08_freq2 += (a/2)**2 * form_factor_function_ThorneMeral2008((a/2), freq2_Hz, 1500)**2 * proba_num_sand_i[l]    
        temp_a3 += (a/2)**3 * proba_num_sand_i[l]
    
    # computing output values   
    f_TM08_freq1_sand_iTW16 = (((d50_sand_i/2)*temp_a2f2_TM08_freq1)/temp_a3)**0.5
    f_TM08_freq2_sand_iTW16 = (((d50_sand_i/2)*temp_a2f2_TM08_freq2)/temp_a3)**0.5

    f_TM08_freq1_sand_samples.append(f_TM08_freq1_sand_iTW16)
    f_TM08_freq2_sand_samples.append(f_TM08_freq2_sand_iTW16)

TAAPS_freq1['f_sand_TW16'] = f_TM08_freq1_sand_samples
TAAPS_freq2['f_sand_TW16'] = f_TM08_freq2_sand_samples

# convert phi sigma to ln sigma
TAAPS_freq1['sigma_ln'] = [np.log(2**TAAPS_freq1['sigma_mum'][i]) for i in range(len(TAAPS_freq1))]
TAAPS_freq2['sigma_ln'] = [np.log(2**TAAPS_freq2['sigma_mum'][i]) for i in range(len(TAAPS_freq2))]
TAAPS_freq1['sigma_ln_fines'] = [np.log(2**TAAPS_freq1['sigma_mum_fines'][i]) for i in range(len(TAAPS_freq1))]
TAAPS_freq2['sigma_ln_fines'] = [np.log(2**TAAPS_freq2['sigma_mum_fines'][i]) for i in range(len(TAAPS_freq2))]

TAAPS_freq1_fines_GSD = TAAPS_freq1_fines.dropna(subset = ['D50_mum_fines'], how = 'any') 
TAAPS_freq1_fines_GSD.reset_index(drop = True, inplace = True)
TAAPS_freq2_fines_GSD = TAAPS_freq2_fines.dropna(subset = ['D50_mum_fines'], how = 'any') 
TAAPS_freq2_fines_GSD.reset_index(drop = True, inplace = True)

TAAPS_freq1_fines_GSD['sigma_ln'] = [np.log(2**TAAPS_freq1_fines_GSD['sigma_mum'][i]) for i in range(len(TAAPS_freq1_fines_GSD))]
TAAPS_freq2_fines_GSD['sigma_ln'] = [np.log(2**TAAPS_freq2_fines_GSD['sigma_mum'][i]) for i in range(len(TAAPS_freq2_fines_GSD))]
TAAPS_freq1_fines_GSD['sigma_ln_fines'] = [np.log(2**TAAPS_freq1_fines_GSD['sigma_mum_fines'][i]) for i in range(len(TAAPS_freq1_fines_GSD))]
TAAPS_freq2_fines_GSD['sigma_ln_fines'] = [np.log(2**TAAPS_freq2_fines_GSD['sigma_mum_fines'][i]) for i in range(len(TAAPS_freq2_fines_GSD))]


#%% Determine fines form factors of samples (TAAPS)

# Define boundaries, phi system and convert to mm (2**)   
gs = [2**(-35.25 + 0.0625*j) for j in range(0,672)]
gsbound = [2**(-35.2188 + 0.0625*j) for j in range(0,672)]

f_TM08_freq1_fines_samples = []
f_TM08_freq2_fines_samples = []
for i in range(len(TAAPS_freq1_fines_GSD)):
    # Calculate fines f following TW16
    volfracfines_i = [(1/np.sqrt(2*math.pi*TAAPS_freq1_fines_GSD['D50_mum_fines'][i]))*np.exp((-(np.log(gs[j])-np.log(TAAPS_freq1_fines_GSD['D50_mum_fines'][i]*1e-3))**2)/(2*TAAPS_freq1_fines_GSD['sigma_mum_fines'][i]**2))       
                    for j in range(len(gs))]
    cumsumfines_i = np.cumsum(volfracfines_i)
    proba_vol_fines_i = volfracfines_i/cumsumfines_i[-1]
    
    # Determine D50
    cumsum_fines_i = np.cumsum(proba_vol_fines_i)
    d500_i = 0.5*cumsum_fines_i[-1]
    
    for j in range(len(cumsum_fines_i)):
        if d500_i >= cumsum_fines_i[j-1] and d500_i <= cumsum_fines_i[j]:
            d50fines_i=((d500_i-cumsum_fines_i[j-1])/(cumsum_fines_i[j]-cumsum_fines_i[j-1]))*(np.log(gsbound[j])-np.log(gsbound[j-1]))+np.log(gsbound[j-1])
    d50_fines_i = np.exp(d50fines_i)/1000
    
    # Computing number probability
    ss = np.sum(proba_vol_fines_i / np.array(gs)**3)
    proba_num_fines_i = proba_vol_fines_i/np.array(gs)**3 / ss
    
    # Calculate form function TM08
    # Integrating over the distribution        
    temp_a1 = 0
    temp_a2f2_TM08_freq1 = 0
    temp_a2f2_TM08_freq2 = 0
    temp_a3 = 0  
    # Summing the integrals
    for l in range(len(proba_num_fines_i)):
        a = np.array(gs)[l]/1000
        temp_a2f2_TM08_freq1 += (a/2)**2 * form_factor_function_ThorneMeral2008((a/2), freq1_Hz, 1500)**2 * proba_num_fines_i[l]   
        temp_a2f2_TM08_freq2 += (a/2)**2 * form_factor_function_ThorneMeral2008((a/2), freq2_Hz, 1500)**2 * proba_num_fines_i[l]    
        temp_a3 += (a/2)**3 * proba_num_fines_i[l]
    
    # computing output values   
    f_TM08_freq1_fines_iTW16 = (((d50_fines_i/2)*temp_a2f2_TM08_freq1)/temp_a3)**0.5
    f_TM08_freq2_fines_iTW16 = (((d50_fines_i/2)*temp_a2f2_TM08_freq2)/temp_a3)**0.5

    f_TM08_freq1_fines_samples.append(f_TM08_freq1_fines_iTW16)
    f_TM08_freq2_fines_samples.append(f_TM08_freq2_fines_iTW16)

TAAPS_freq1_fines_GSD['f_fines_TW16'] = f_TM08_freq1_fines_samples
TAAPS_freq1_fines_GSD['f_fines_TW16'] = f_TM08_freq2_fines_samples

#%% Determine B' of samples (TAAPS) using BBC relation and meas (as TW16)
#Freq1
ExcessBackScat_samples_freq11 = [TAAPS_freq1['Beam-Averaged Backscatter (dB)'][i]-
                                 ((TAAPS_freq1['log_sand'][i]-K1_emp_freq1_S2_D50)/K2_emp_freq1_S2_D50)
                                 for i in range(len(TAAPS_freq1))]
ExcessBackScat_samples_freq21 = [TAAPS_freq2['Beam-Averaged Backscatter (dB)'][i]-
                                 ((TAAPS_freq1['log_sand'][i]-K1_emp_freq2_S2_D50)/K2_emp_freq2_S2_D50)
                                 for i in range(len(TAAPS_freq2))]


#%% Determine B' of samples (TAAPS) theoretically
#Freq1
ExcessBackScat_samples_freq1 = []
d50Bimodal_samples_freq1 = []
formBimodal_samples_freq1 = []
for k in range(len(TAAPS_freq1_fines_GSD)):
        
    # Define boundaries, phi system and convert to mm (2**)   
    gs = [(-35.25 + 0.0625*j) for j in range(0,672)]
    gs = [2**gs[i] for i in range(len(gs))]
    gsbound = [2**(-35.2188 + 0.0625*j) for j in range(0,672)]

    # Calculate ref sand f following TW16
    volfracSand = [(1/np.sqrt(2*math.pi*TAAPS_freq1_fines_GSD['sigma_ln'][k]))*np.exp((-(np.log(gs[j])-np.log(TAAPS_freq1_fines_GSD['D50_mum'][k]/1000))**2)/(2*TAAPS_freq1_fines_GSD['sigma_ln'][k]**2))       
                    for j in range(len(gs))]
    volfracSilt = [(1/np.sqrt(2*math.pi*TAAPS_freq1_fines_GSD['sigma_ln_fines'][k]))*np.exp((-(np.log(gs[j])-np.log(TAAPS_freq1_fines_GSD['D50_mum_fines'][k]/1000))**2)/(2*TAAPS_freq1_fines_GSD['sigma_ln_fines'][k]**2))       
                    for j in range(len(gs))]
    vol = [(4/3)*math.pi*(gs[i]/2)**3 for i in range(len(gs))]
    
    volsumSand = np.cumsum(volfracSand)
    volsumSilt = np.cumsum(volfracSilt)
    
    volfracSand = [volfracSand[i]/volsumSand[-1] for i in range(len(volsumSand))]
    volfracSilt = [TAAPS_freq1_fines_GSD['S'][k]*volfracSilt[i]/volsumSilt[-1] for i in range(len(volsumSilt))]
    
    volcomb = [volfracSand[i] + volfracSilt[i] for i in range(len(volfracSilt))]
    volsum = np.cumsum(volcomb)
    
    d50 = 0.5*volsum[-1]
    for j in range(1,672):
        if d50 >= volsum[j-1] and d50 <= volsum[j]:
            # print(j)
            d50bimodal=((d50 - volsum[j])/(volsum[j] - volsum[j-1]))*(np.log(gsbound[j])-np.log(gsbound[j-1]))+np.log(gsbound[j-1])
    d50bimodal = np.exp(d50bimodal)
    
    gs = [gs[i]/1000 for i in range(len(gs))]
    volrat = [volcomb[i]/vol[i] for i in range(len(vol))]
    
    volsum = np.cumsum(volrat)
    numFrac = [volrat[i]/volsum[-1] for i in range(len(vol))]
    d50bimodal_j = d50bimodal/1000
    
    denom3_1 = (gs[0]/2)**3*numFrac[0]
    denom3_j = denom3_1
    denom3 = []
    for i in range(1,672):  
        denom3_i = 0.5*((gs[i-1]/2)**3*numFrac[i-1]+(gs[i]/2)**3*numFrac[i]) + denom3_j
        denom3_j = denom3_i
        denom3.append(denom3_i)  
    denom3.insert(0,denom3_1)
    
    ka = [wavenum_freq1*(gs[i]/2) for i in range(len(gs))]
    formf = [ka[i]**2*(1-0.35*np.exp(-((ka[i]-1.5)/0.7)**2))*(1+0.5*np.exp(-((ka[i]-1.8)/2.2)**2))/(1+0.9*ka[i]**2)
              for i in range(len(ka))]
    
    forma3_1 = (gs[0]/2)**2*numFrac[0]*formf[0]**2
    forma3_j = forma3_1
    forma3 = []
    for i in range(1,672):  
        forma3_i = 0.5*((gs[i-1]/2)**2*numFrac[i-1]*formf[i-1]**2+(gs[i]/2)**2*numFrac[i]*formf[i]**2)+forma3_j
        forma3_j = forma3_i
        forma3.append(forma3_i)  
    forma3.insert(0,forma3_1)
    
    # total sediment form factor
    formBimodal_j = ((d50bimodal_j/2)*forma3[-1]/denom3[-1])**0.5
    
    # calculate B', excessive backscatter, Topping and Wright 2016, equation 66
    ExcessBackScat_j = (1/K2_emp_freq1_S2_D50)*np.log10(((formBimodal_j/formSandref_freq1)**2)*(sandD50ref/(d50bimodal))*(2.65/2.65)*(1+TAAPS_freq1_fines_GSD['S'][k]))
    
    # append    
    ExcessBackScat_samples_freq1.append(ExcessBackScat_j)
    d50Bimodal_samples_freq1.append(d50bimodal_j)
    formBimodal_samples_freq1.append(formBimodal_j)

TAAPS_freq1_fines_GSD['B_fines'] =  ExcessBackScat_samples_freq1
TAAPS_freq1_fines_GSD['D50_bimodal'] =  d50Bimodal_samples_freq1
TAAPS_freq1_fines_GSD['f_bimodal'] =  formBimodal_samples_freq1

# Freq2 
ExcessBackScat_samples_freq2 = []
d50Bimodal_samples_freq2 = []
formBimodal_samples_freq2 = []
for k in range(len(TAAPS_freq2_fines_GSD)):
        
    # Define boundaries, phi system and convert to mm (2**)   
    gs = [(-35.25 + 0.0625*j) for j in range(0,672)]
    gs = [2**gs[i] for i in range(len(gs))]
    gsbound = [2**(-35.2188 + 0.0625*j) for j in range(0,672)]

    # Calculate ref sand f following TW16
    volfracSand = [(1/np.sqrt(2*math.pi*TAAPS_freq2_fines_GSD['sigma_ln'][k]))*np.exp((-(np.log(gs[j])-np.log(TAAPS_freq2_fines_GSD['D50_mum'][k]/1000))**2)/(2*TAAPS_freq2_fines_GSD['sigma_ln'][k]**2))       
                    for j in range(len(gs))]
    volfracSilt = [(1/np.sqrt(2*math.pi*TAAPS_freq2_fines_GSD['sigma_ln_fines'][k]))*np.exp((-(np.log(gs[j])-np.log(TAAPS_freq2_fines_GSD['D50_mum_fines'][k]/1000))**2)/(2*TAAPS_freq2_fines_GSD['sigma_ln_fines'][k]**2))       
                    for j in range(len(gs))]
    vol = [(4/3)*math.pi*(gs[i]/2)**3 for i in range(len(gs))]
    
    volsumSand = np.cumsum(volfracSand)
    volsumSilt = np.cumsum(volfracSilt)
    
    volfracSand = [volfracSand[i]/volsumSand[-1] for i in range(len(volsumSand))]
    volfracSilt = [TAAPS_freq2_fines_GSD['S'][k]*volfracSilt[i]/volsumSilt[-1] for i in range(len(volsumSilt))]
    
    volcomb = [volfracSand[i] + volfracSilt[i] for i in range(len(volfracSilt))]
    volsum = np.cumsum(volcomb)
    
    d50 = 0.5*volsum[-1]
    for j in range(1,672):
        if d50 >= volsum[j-1] and d50 <= volsum[j]:
            # print(j)
            d50bimodal=((d50 - volsum[j])/(volsum[j] - volsum[j-1]))*(np.log(gsbound[j])-np.log(gsbound[j-1]))+np.log(gsbound[j-1])
    d50bimodal = np.exp(d50bimodal)
    
    gs = [gs[i]/1000 for i in range(len(gs))]
    volrat = [volcomb[i]/vol[i] for i in range(len(vol))]
    
    volsum = np.cumsum(volrat)
    numFrac = [volrat[i]/volsum[-1] for i in range(len(vol))]
    d50bimodal_j = d50bimodal/1000
    
    denom3_1 = (gs[0]/2)**3*numFrac[0]
    denom3_j = denom3_1
    denom3 = []
    for i in range(1,672):  
        denom3_i = 0.5*((gs[i-1]/2)**3*numFrac[i-1]+(gs[i]/2)**3*numFrac[i]) + denom3_j
        denom3_j = denom3_i
        denom3.append(denom3_i)  
    denom3.insert(0,denom3_1)
    
    ka = [wavenum_freq2*(gs[i]/2) for i in range(len(gs))]
    formf = [ka[i]**2*(1-0.35*np.exp(-((ka[i]-1.5)/0.7)**2))*(1+0.5*np.exp(-((ka[i]-1.8)/2.2)**2))/(1+0.9*ka[i]**2)
              for i in range(len(ka))]
    
    forma3_1 = (gs[0]/2)**2*numFrac[0]*formf[0]**2
    forma3_j = forma3_1
    forma3 = []
    for i in range(1,672):  
        forma3_i = 0.5*((gs[i-1]/2)**2*numFrac[i-1]*formf[i-1]**2+(gs[i]/2)**2*numFrac[i]*formf[i]**2)+forma3_j
        forma3_j = forma3_i
        forma3.append(forma3_i)  
    forma3.insert(0,forma3_1)
    
    # total sediment form factor
    formBimodal_j = ((d50bimodal_j/2)*forma3[-1]/denom3[-1])**0.5
    
    # calculate B', excessive backscatter, Topping and Wright 2016, equation 66
    ExcessBackScat_j = (1/K2_emp_freq2_S2_D50)*np.log10(((formBimodal_j/formSandref_freq2)**2)*(sandD50ref/(d50bimodal))*(2.65/2.65)*(1+TAAPS_freq2_fines_GSD['S'][k]))
    
    # append    
    ExcessBackScat_samples_freq2.append(ExcessBackScat_j)
    d50Bimodal_samples_freq2.append(d50bimodal_j)
    formBimodal_samples_freq2.append(formBimodal_j)

TAAPS_freq2_fines_GSD['B_fines'] =  ExcessBackScat_samples_freq2
TAAPS_freq2_fines_GSD['D50_bimodal'] =  d50Bimodal_samples_freq2
TAAPS_freq2_fines_GSD['f_bimodal'] =  formBimodal_samples_freq2

#%% STEP10: Calculate Backscattering cross section of a unit reverberating volume sv and reverberating volume V
# sv
TAAPS_freq1['s_v_sand'] = [TAAPS_freq1['f_sand_TW16'].iloc[i]**2 *(3*TAAPS_freq1['Sand_concentration_g_l'].iloc[i])/(16*math.pi*TAAPS_freq1['D50_mum'].iloc[i]*1e-6 * rho_sed*1**2)
             for i in range(len(TAAPS_freq1))] # in 1/m³, sizes (a_s) in m
TAAPS_freq2['s_v_sand']  = [TAAPS_freq2['f_sand_TW16'].iloc[i]**2 *(3*TAAPS_freq2['Sand_concentration_g_l'].iloc[i])/(16*math.pi*TAAPS_freq2['D50_mum'].iloc[i]*1e-6 * rho_sed*1**2)
             for i in range(len(TAAPS_freq2))] # in 1/m³, sizes (a_s) in m
# # sv
# TAAPS_freq1['s_v_fines'] = [TAAPS_freq1['f_int_fines'].iloc[i]**2 *(3*TAAPS_freq1['Fine_concentration_g_l'].iloc[i])/(16*math.pi*TAAPS_freq1['D50_mum'].iloc[i]*1e-6 * rho_sed*1**2)
#              for i in range(len(TAAPS_freq1))] # in 1/m³, sizes (a_s) in m
# TAAPS_freq2_fines_GSD['s_v_fines']  = [TAAPS_freq2_fines_GSD['f_int_fines'].iloc[i]**2 *(3*TAAPS_freq2_fines_GSD['Fine_concentration_g_l'].iloc[i])/(16*math.pi*TAAPS_freq2_fines_GSD['D50_mum'].iloc[i]*1e-6 * rho_sed*1**2)
#              for i in range(len(TAAPS_freq2))] # in 1/m³, sizes (a_s) in m

# Calculate the reverberating volume V
V_freq1 = [2*t_p_freq1*1500*math.pi*(0.96/(k_freq1 * a_T_freq1))**2 *celldist_along_beam_freq1.iloc[i,0]
           for i in range(len(celldist_along_beam_freq1))]
V_freq2 = [2*t_p_freq2*1500*math.pi*(0.96/(k_freq2 * a_T_freq2))**2 *celldist_along_beam_freq2.iloc[i,0]
           for i in range(len(celldist_along_beam_freq2))]

#%% Plot D50sand - f sand -- check with results Topping

D50_f_freq1 = pd.DataFrame([TAAPS_freq1['D50_mum'],TAAPS_freq1['f_sand_TW16']])


Form_TS = pd.read_csv(r'C:\Users\jessi\OneDrive\Dokumente\Soutenance\Form_TS.csv', sep = ';')
Form_TS['D50_mum'] = [Form_TS['D50(mm) of lab grain-size distribution'].iloc[i]*1000 for i in range(len(Form_TS))]



#%% STEP10: Calculate the target strength TS, UTS and RUTS
# TS
TS_freq1 = 10*np.log10(TAAPS_freq1['s_v_sand']) + 10*np.log10(2*t_p_freq1*1500*math.pi*(0.96/(k_freq1 * a_T_freq1))**2) + 20*np.log10(celldist_along_beam_freq1.iloc[-1,0])
TS_freq2 = 10*np.log10(TAAPS_freq2['s_v_sand']) + 10*np.log10(2*t_p_freq2*1500*math.pi*(0.96/(k_freq2 * a_T_freq2))**2) + 20*np.log10(celldist_along_beam_freq2.iloc[-1,0])
TS_sand_ref_freq1 = 10*np.log10(C_sand_ref_g_l) + 10*np.log10(2*t_p_freq1*1500*math.pi*(0.96/(k_freq1 * a_T_freq1))**2) + 20*np.log10(celldist_along_beam_freq1.iloc[-1,0])
TS_sand_ref_freq2 = 10*np.log10(C_sand_ref_g_l) + 10*np.log10(2*t_p_freq2*1500*math.pi*(0.96/(k_freq2 * a_T_freq2))**2) + 20*np.log10(celldist_along_beam_freq2.iloc[-1,0])

# UTS SED
TAAPS_freq1['UTS_sand'] = [10*np.log10(TAAPS_freq1['f_sand_TW16'].iloc[i]**2 *3/(16*math.pi*TAAPS_freq1['D50_mum'].iloc[i]*1e-6 * rho_sed*1**2))
             for i in range(len(TAAPS_freq1))] # in dB, sizes (a_s) in m
TAAPS_freq2['UTS_sand']  = [10*np.log10(TAAPS_freq2['f_sand_TW16'].iloc[i]**2 *3/(16*math.pi*TAAPS_freq2['D50_mum'].iloc[i]*1e-6 * rho_sed*1**2))
             for i in range(len(TAAPS_freq2))] # in dB, sizes (a_s) in m

# UTS sand ref
UTS_sand_ref_freq1 = 10*np.log10(f_TM08_freq1_sand_refTW16**2 *3/(16*math.pi*D50_sand_ref_mum*1e-6 * rho_sed*1**2))
UTS_sand_ref_freq2  = 10*np.log10(f_TM08_freq2_sand_refTW16**2 *3/(16*math.pi*D50_sand_ref_mum*1e-6 * rho_sed*1**2))
             
# UTS beam
UTS_beam_freq1 = 10*np.log10(2*t_p_freq1*1500*math.pi*(0.96/(k_freq1 * a_T_freq1))**2) # x ~ 0.001 (for UTS beam = 30)
UTS_beam_freq2 = 10*np.log10(2*t_p_freq2*1500*math.pi*(0.96/(k_freq2 * a_T_freq2))**2)

# UTS
UTS_ref_freq1 = UTS_sand_ref_freq1+ UTS_beam_freq1
UTS_ref_freq2 = UTS_sand_ref_freq2+ UTS_beam_freq2


#%% Compare theoretical with empirical AlphaUnit Annex 5
# empirical relation is slope in AlphaSed - Cfines (mg/l) plot is AlphaUnit in (dB l)/(m mg))
# calculate AlphaUnit for each gauging: AlphaSed/Cfines
# theoretical relation calculates AlphaUnit for a D50, sigma G and rho s (all to be defined) 
h = 0.5
D50_fines_range_mum = np.arange(1,63,1)
# freq1
zeta_freq1_fines_range = []
for i in range(len(D50_fines_range_mum)):
    ref_dist_fines_freq1_range1 = compute_model_lognorm_spherical(D50_fines_range_mum[i]*1e-6, 2.4, freq1_Hz, h, rho_sed, nu_0)
    zz1 = ref_dist_fines_freq1_range1.zeta      
    zeta_freq1_fines_range.append(zz1)

# Freq2
zeta_freq2_fines_range = []
for i in range(len(D50_fines_range_mum)):
    ref_dist_fines_freq2_range1 = compute_model_lognorm_spherical(D50_fines_range_mum[i]*1e-6, 2.4, freq2_Hz, h, rho_sed, nu_0)
    zz1 = ref_dist_fines_freq2_range1.zeta      
    zeta_freq2_fines_range.append(zz1)
    
D50_sand_range_mum = np.arange(63,500,5)
# freq1
zeta_freq1_sand_range = []
for i in range(len(D50_sand_range_mum)):
    ref_dist_sand_freq1_range1 = compute_model_lognorm_spherical(D50_sand_range_mum[i]*1e-6, 0.59, freq1_Hz, h, rho_sed, nu_0)
    zz1 = ref_dist_sand_freq1_range1.zeta      
    zeta_freq1_sand_range.append(zz1)

# Freq2
zeta_freq2_sand_range = []
for i in range(len(D50_sand_range_mum)):
    ref_dist_sand_freq2_range1 = compute_model_lognorm_spherical(D50_sand_range_mum[i]*1e-6, 0.59, freq2_Hz, h, rho_sed, nu_0)
    zz1 = ref_dist_sand_freq2_range1.zeta      
    zeta_freq2_sand_range.append(zz1)

#%% CALCULATE INTEGRATED FORM FACTOR F (THORNE & MERAL 2008) AND ZETA FOR SAMPLINGS
# sand
f_TM08_freq1_sand = []
f_TM08_freq2_sand = []
zeta_freq1_sand = []
zeta_freq2_sand = []
zetas_freq1_sand = []
zetas_freq2_sand = []
zetav_freq1_sand = []
zetav_freq2_sand = []
for i in range(len(TAAPS_freq1)):
    if TAAPS_freq1['D50_mum'].iloc[i] >= 1:
        ff1 = compute_model_lognorm_spherical(TAAPS_freq1['D50_mum'].iloc[i]*1e-6, TAAPS_freq1['sigma_mum'].iloc[i], freq1_Hz, h, rho_sed, nu_0)
        f1 = ff1.f_TM08
        z1 = ff1.zeta
        zs1 = ff1.zetas
        zv1 = ff1.zetav
        ff2 = compute_model_lognorm_spherical(TAAPS_freq2['D50_mum'].iloc[i]*1e-6, TAAPS_freq2['sigma_mum'].iloc[i], freq2_Hz, h, rho_sed, nu_0)
        f2 = ff2.f_TM08
        z2 = ff2.zeta
        zs2 = ff2.zetas
        zv2 = ff2.zetav
    else:
        f1 = None
        f2 = None
        z1 = None
        zs1 = None
        zv1 = None
        z2 = None
        zs2 = None
        zv2 = None
    f_TM08_freq1_sand.append(f1)
    f_TM08_freq2_sand.append(f2)
    zeta_freq1_sand.append(z1)
    zeta_freq2_sand.append(z2)
    zetas_freq1_sand.append(zs1)
    zetas_freq2_sand.append(zs2)
    zetav_freq1_sand.append(zv1)
    zetav_freq2_sand.append(zv2)
TAAPS_freq1['f_int_sand'] = f_TM08_freq1_sand
TAAPS_freq1['zeta_sand'] = zeta_freq1_sand # m-1
TAAPS_freq1['zeta_s_sand'] = zetas_freq1_sand # m-1
TAAPS_freq1['zeta_v_sand'] = zetav_freq1_sand # m-1
TAAPS_freq2['f_int_sand'] = f_TM08_freq2_sand # m-1
TAAPS_freq2['zeta_sand'] = zeta_freq2_sand # m-1
TAAPS_freq2['zeta_s_sand'] = zetas_freq2_sand # m-1
TAAPS_freq2['zeta_v_sand'] = zetav_freq2_sand # m-1

# fines
f_TM08_freq1_fines = []
f_TM08_freq2_fines = []
zeta_freq1_fines = []
zeta_freq2_fines = []
zetas_freq1_fines = []
zetas_freq2_fines = []
zetav_freq1_fines = []
zetav_freq2_fines = []
for i in range(len(TAAPS_freq1)):
    if TAAPS_freq1['D50_mum_fines'].iloc[i] >= 1:
        ff1 = compute_model_lognorm_spherical(TAAPS_freq1['D50_mum_fines'].iloc[i]*1e-6, TAAPS_freq1['sigma_mum_fines'].iloc[i], freq1_Hz, h, rho_sed, nu_0)
        f1 = ff1.f_TM08
        z1 = ff1.zeta
        zs1 = ff1.zetas
        zv1 = ff1.zetav
        ff2 = compute_model_lognorm_spherical(TAAPS_freq2['D50_mum_fines'].iloc[i]*1e-6, TAAPS_freq2['sigma_mum_fines'].iloc[i], freq2_Hz, h, rho_sed, nu_0)
        f2 = ff2.f_TM08
        z2 = ff2.zeta
        zs2 = ff2.zetas
        zv2 = ff2.zetav
    else:
        f1 = None
        f2 = None
        z1 = None
        zs1 = None
        zv1 = None
        z2 = None
        zs2 = None
        zv2 = None
    f_TM08_freq1_fines.append(f1)
    f_TM08_freq2_fines.append(f2)
    zeta_freq1_fines.append(z1)
    zeta_freq2_fines.append(z2)
    zetas_freq1_fines.append(zs1)
    zetas_freq2_fines.append(zs2)
    zetav_freq1_fines.append(zv1)
    zetav_freq2_fines.append(zv2)
TAAPS_freq1['f_int_fines'] = f_TM08_freq1_fines
TAAPS_freq1['zeta_fines'] = zeta_freq1_fines # m-1
TAAPS_freq1['zeta_s_fines'] = zetas_freq1_fines # m-1
TAAPS_freq1['zeta_v_fines'] = zetav_freq1_fines # m-1
TAAPS_freq2['f_int_fines'] = f_TM08_freq2_fines # m-1
TAAPS_freq2['zeta_fines'] = zeta_freq2_fines # m-1
TAAPS_freq2['zeta_s_fines'] = zetas_freq2_fines # m-1
TAAPS_freq2['zeta_v_fines'] = zetav_freq2_fines # m-1

#%% 5) Calculate alpha samples ('meas')
TAAPS_freq1['Alpha_fines_1_m'] = TAAPS_freq1['zeta_fines'] *TAAPS_freq1['Fine_concentration_g_l']
TAAPS_freq2['Alpha_fines_1_m'] = TAAPS_freq2['zeta_fines'] *TAAPS_freq2['Fine_concentration_g_l']
# TAAPS_freq1['Alphas_fines_1_m'] = TAAPS_freq1['zetas_fines'] *TAAPS_freq1['Fine_concentration_g_l']
# TAAPS_freq2['Alphas_fines_1_m'] = TAAPS_freq2['zetas_fines'] *TAAPS_freq2['Fine_concentration_g_l']
# TAAPS_freq1['Alphav_fines_1_m'] = TAAPS_freq1['zetav_fines'] *TAAPS_freq1['Fine_concentration_g_l']
# TAAPS_freq2['Alphav_fines_1_m'] = TAAPS_freq2['zetav_fines'] *TAAPS_freq2['Fine_concentration_g_l']
TAAPS_freq1['Alpha_fines_dB_m'] = 20/(np.log(10))*TAAPS_freq1['Alpha_fines_1_m']
TAAPS_freq2['Alpha_fines_dB_m'] = 20/(np.log(10))*TAAPS_freq2['Alpha_fines_1_m']

TAAPS_freq1['Alpha_sand_dB_m'] = 20/(np.log(10))*TAAPS_freq1['zeta_sand'] *TAAPS_freq1['Sand_concentration_g_l']
TAAPS_freq2['Alpha_sand_dB_m'] = 20/(np.log(10))*TAAPS_freq2['zeta_sand'] *TAAPS_freq2['Sand_concentration_g_l']
TAAPS_freq1['Alpha_dB_m'] = TAAPS_freq1['Alpha_sand_dB_m'] + TAAPS_freq1['Alpha_fines_dB_m']
TAAPS_freq2['Alpha_dB_m'] = TAAPS_freq2['Alpha_sand_dB_m'] + TAAPS_freq2['Alpha_fines_dB_m']

#%% STEP 11: Fig4 Plot D50- AlphaUnit and log(S) - B' theoretical - empirical - MANUSCRIPT
D50_sand_range_mum = np.arange(63,500,5)
Alphaunit_freq1_fines_range = [zeta_freq1_fines_range[i]*20/np.log(10)/1000 for i in range(len(zeta_freq1_fines_range))]
Alphaunit_freq2_fines_range = [zeta_freq2_fines_range[i]*20/np.log(10)/1000 for i in range(len(zeta_freq2_fines_range))]
Alphaunit_freq1_sand_range = [zeta_freq1_sand_range[i]*20/np.log(10)/1000 for i in range(len(zeta_freq1_sand_range))]
Alphaunit_freq2_sand_range = [zeta_freq2_sand_range[i]*20/np.log(10)/1000 for i in range(len(zeta_freq2_sand_range))]
 
fig, ax = plt.subplots(1, 2, figsize = (12,6), dpi=300)

# Plot 1
m1, = ax[0].plot(TAAPS_freq1['D50_mum_fines'], TAAPS_freq1['zeta_fines']*20/np.log(10)/1000, 
        'D', color = 'olive', markersize = 8, markeredgecolor = 'black',markeredgewidth = 0.3,  
        label = '400 kHz', zorder = 30)
m2, = ax[0].plot(TAAPS_freq1['D50_mum_fines'], TAAPS_freq2['zeta_fines']*20/np.log(10)/1000, 
        's', color = 'mediumorchid', markersize = 8, markeredgecolor = 'black', markeredgewidth = 0.3,
        label = '1 MHz', zorder = 30)
ax[0].plot(TAAPS_freq1['D50_mum'], TAAPS_freq1['zeta_sand']*20/np.log(10)/1000, 
        'D', color = 'olive', markersize = 8, markeredgecolor = 'black',markeredgewidth = 0.3, zorder = 30, label = 'calculated')
ax[0].plot(TAAPS_freq1['D50_mum'], TAAPS_freq2['zeta_sand']*20/np.log(10)/1000, 
        's', color = 'mediumorchid', markersize = 8, markeredgecolor = 'black', markeredgewidth = 0.3,zorder = 30, label = 'modeled')

m3, = ax[0].plot(D50_fines_range_mum, Alphaunit_freq1_fines_range, 
        color = 'olive')
m4, = ax[0].plot(D50_fines_range_mum, Alphaunit_freq2_fines_range, 
        color = 'mediumorchid')
ax[0].plot(D50_sand_range_mum, Alphaunit_freq1_sand_range, 
        color = 'olive')
ax[0].plot(D50_sand_range_mum, Alphaunit_freq2_sand_range, 
        color = 'mediumorchid')

ax[0].text(0.05, 0.95, '(a)', fontsize = 16, transform = ax[0].transAxes)
ax[0].grid(linewidth = 0.2)
ax[0].grid(which = 'minor', axis = 'x', linewidth = 0.1)
ax[0].set_xscale('log')
ax[0].set_xlim(5, 500)
ax[0].set_ylim(-0.0001, 0.003)
ax[0].set_ylabel(r'$\mathregular{\alpha_{unit} \; \left(\frac{dB \; L}{m \; mg}\right)}$', fontsize=18, weight = 'bold')
ax[0].set_xlabel(r'$\mathregular{D_{50}}$ ($\mathregular{\mu}$m)', fontsize=18, weight = 'bold')
ax[0].tick_params(axis='both', which='major', labelsize = 16)
ax[0].vlines(63,-1,1, ls = '--', lw = 1, color = 'black')
ax[0].text(0.5, 0.6, 'silt-sand break', fontsize = 14, transform = ax[0].transAxes, rotation = 90)

# Plot 2
ax[1].semilogx(sRatio, ExcessBackScat_freq1, ls = '-', lw = 1, color = 'olive', label = 'Model 400 kHz')
           
ax[1].semilogx(sRatio, ExcessBackScat_freq2, ls = '-', lw = 1, color = 'mediumorchid', label = 'Model 1 MHz')

# sc = ax[1].semilogx(TAAPS_freq1_fines_GSD['S'], ExcessBackScat_samples_freq1,
#             'D', color = 'olive', markersize = 8, markeredgecolor = 'black', markeredgewidth = 0.3, label = 'calculated 400 kHz')
# ax[1].semilogx(TAAPS_freq2_fines_GSD['S'], ExcessBackScat_samples_freq2,'s', color = 'mediumorchid',
#                 markersize = 8, markeredgecolor = 'black', markeredgewidth = 0.3, label = 'calculated 1 MHz')


sc = ax[1].semilogx(TAAPS_freq1['S'], ExcessBackScat_samples_freq11,
            'D', color = 'olive', markersize = 8, markeredgecolor = 'black', markeredgewidth = 0.3, label = 'calculated 400 kHz')
ax[1].semilogx(TAAPS_freq2['S'], ExcessBackScat_samples_freq21,'s', color = 'mediumorchid',
                markersize = 8, markeredgecolor = 'black', markeredgewidth = 0.3, label = 'calculated 1 MHz')

ax[1].text(0.05, 0.95, '(b)', fontsize = 16, transform = ax[1].transAxes)
ax[1].grid(linewidth = 0.2)
ax[1].grid(which = 'minor', axis = 'x', linewidth = 0.1)
#ax.legend(fontsize = 14)
ax[1].set_ylabel(r'$\mathregular{B\' (dB)}$', fontsize=18, weight = 'bold')
ax[1].set_xlabel(r'S', fontsize=18, weight = 'bold')
ax[1].tick_params(axis='both', which='major', labelsize = 16)
ax[1].set_xlim (0.1, 100)
ax[1].set_ylim (-10, 25)
ax[1].set_xscale('log')

x_formatter = FixedFormatter(["10", "63", "100", "500", "1000"])
x_locator = FixedLocator([10, 63, 100, 500, 1000])
ax[0].xaxis.set_major_formatter(x_formatter)
ax[0].xaxis.set_major_locator(x_locator)
ax[1].legend(loc = 'upper right', fontsize = 14, framealpha = 1)
# handles = [(m1, m3), (m2, m4), (m1, m2), (m3, m4)]
# _, labels = ax[0].get_legend_handles_labels()
# fig.legend(handles = handles, labels=labels, 
#           handler_map = {tuple: mpl.legend_handler.HandlerTuple(None)},
#           loc="upper right", framealpha = 1,bbox_to_anchor=(0.97, 0.95), 
#           fontsize = 14)

fig.tight_layout()
figname = 'Fig4'
fig.savefig(outpath_figures +'\\' +  figname + '.png', dpi = 300, bbox_inches='tight')
# fig.savefig(outpath_figures +'\\' +  figname + '.eps',format = 'eps', dpi = 400, bbox_inches='tight')

#%% ------------- STEP 11 B: Calcute f sands TAAPS calculating f following TW16

# f_TM08_freq1_sand_samples = []
# f_TM08_freq2_sand_samples = []
# for i in range(len(TAAPS_freq1)):
#     volfracsand = [(1/np.sqrt(2*math.pi*TAAPS_freq1['D50_mum'][i]))*np.exp((-(np.log(gs[j])-np.log(TAAPS_freq1['D50_mum'][i]*1e-3))**2)/(2*TAAPS_freq1['sigma_mum'][i]**2))       
#                    for j in range(len(gs))]
#     cumsumsand = np.cumsum(volfracsand)
#     proba_vol_sand = volfracsand/cumsumsand[-1]
    
#     # Determine D50
#     cumsum_sand = np.cumsum(proba_vol_sand)
#     d500 = 0.5*cumsum_sand[-1]
    
#     for j in range(len(cumsum_sand)):
#         if d500 >= cumsum_sand[j-1] and d500 <= cumsum_sand[j]:
#             d50sand=((d500-cumsum_sand[j-1])/(cumsum_sand[j]-cumsum_sand[j-1]))*(np.log(gsbound[j])-np.log(gsbound[j-1]))+np.log(gsbound[j-1])
#     d50_sand = np.exp(d50sand)/1000
    
#     # Computing number probability
#     ss = np.sum(proba_vol_sand / np.array(gs)**3)
#     proba_num_sand = proba_vol_sand/np.array(gs)**3 / ss
    
#     # Calculate form function TM08
#     # Integrating over the distribution        
#     temp_a1 = 0
#     temp_a2f2_TM08_freq1 = 0
#     temp_a2f2_TM08_freq2 = 0
#     temp_a3 = 0  
#     # Summing the integrals
#     for l in range(len(proba_num_sand)):
#         a = np.array(gs)[l]/1000
#         temp_a2f2_TM08_freq1 += (a/2)**2 * form_factor_function_ThorneMeral2008((a/2), freq1_Hz, 1500)**2 * proba_num_sand[l]   
#         temp_a2f2_TM08_freq2 += (a/2)**2 * form_factor_function_ThorneMeral2008((a/2), freq2_Hz, 1500)**2 * proba_num_sand[l]    
#         temp_a3 += (a/2)**3 * proba_num_sand[l]
    
#     # computing output values   
#     f_TM08_freq1_sandi = (((d50_sand/2)*temp_a2f2_TM08_freq1)/temp_a3)**0.5
#     f_TM08_freq2_sandi = (((d50_sand/2)*temp_a2f2_TM08_freq2)/temp_a3)**0.5
#     f_TM08_freq1_sand_samples.append(f_TM08_freq1_sandi)
#     f_TM08_freq2_sand_samples.append(f_TM08_freq2_sandi)

# TAAPS_freq1['f_sand_TW16'] = f_TM08_freq1_sand_samples
# TAAPS_freq2['f_sand_TW16'] = f_TM08_freq2_sand_samples


#%% ------------ STEP 11 B: CALCULATE B': CALCULATE empirical B' (Backscatter originating from fine sediments) on TAAPS

# B_fines_freq1_calc = []
# f_TM08_freq1 = []
# B_fines_freq2_calc = []
# f_TM08_freq2 = []
# for i in range(len(TAAPS_freq1)):
#     if TAAPS_freq1['D50_mum_fines'][i] is not None and TAAPS_freq1['D50_mum_fines'][i] <= 50:
#         # Create volfractions fines based on log-normal distribution 
#         volfracfines_i = [(1/np.sqrt(2*math.pi*TAAPS_freq1['sigma_mum_fines'][i]))*np.exp((-(np.log(gs[j])-np.log(TAAPS_freq1['D50_mum_fines'][i]*1e-3))**2)/(2*TAAPS_freq1['sigma_mum_fines'][i]**2)) 
#                           for j in range(0,672)]
#         cumsumfines_i = np.cumsum(volfracfines_i)        
        
#         proba_vol_fines = [TAAPS_freq1['S'][0]*volfracfines_1[j]/cumsumfines[-1] for j in range(len(volfracfines_1))]
#         proba_vol_sed = [proba_vol_sand_ref[j] + proba_vol_fines[j] for j in range(len(proba_vol_fines))]
    
#         # Determine D50
#         cumsum_sed = np.cumsum(proba_vol_sed)
#         d500 = 0.5*cumsum_sed[-1]
        
#         for j in range(len(cumsum_sed)):
#             if d500 >= cumsum_sed[j-1] and d500 <= cumsum_sed[j]:
#                 d50Bimodal=((d500-cumsum_sed[j-1])/(cumsum_sed[j]-cumsum_sed[j-1]))*(np.log(gsbound[j])-np.log(gsbound[j-1]))+np.log(gsbound[j-1])
#         d50_sed_i = np.exp(d50Bimodal)/1000
    
#         # Computing number probability
#         ss = np.sum(proba_vol_sed / np.array(gs)**3)
#         proba_num_sed = proba_vol_sed/np.array(gs)**3 / ss
    
#         # Calculate form function TM08
#         # Integrating over the distribution        
#         temp_a1 = 0
#         temp_a2f2_TM08_freq1 = 0
#         temp_a2f2_TM08_freq2 = 0
#         temp_a3 = 0  
#         # Summing the integrals
#         for l in range(len(proba_num_sed)):
#             a = np.array(gs)[l]/1000
#             temp_a2f2_TM08_freq1 += (a/2)**2 * form_factor_function_ThorneMeral2008((a/2), freq1_Hz, 1500)**2 * proba_num_sed[l]   
#             temp_a2f2_TM08_freq2 += (a/2)**2 * form_factor_function_ThorneMeral2008((a/2), freq2_Hz, 1500)**2 * proba_num_sed[l]    
#             temp_a3 += (a/2)**3 * proba_num_sed[l]
    
#         # computing output values   
#         f_TM08_freq1_i = (((d50_sed_i/2)*temp_a2f2_TM08_freq1)/temp_a3)**0.5
#         f_TM08_freq2_i = (((d50_sed_i/2)*temp_a2f2_TM08_freq2)/temp_a3)**0.5
    
#         B_fines_freq1_calc_i = 1/K2_emp_freq1_S2_D50*np.log10((f_TM08_freq1_i/f_TM08_freq1_sand_refTW16)**2*(
#                             (D50_sand_ref_mum*1e-6)/(d50_sed_i)) *(
#                             2650/rho_sed)*(1 + TAAPS_freq1['S'][i]))
#         B_fines_freq2_calc_i = 1/K2_emp_freq2_S2_D50*np.log10((f_TM08_freq2_i/f_TM08_freq2_sand_refTW16)**2*(
#                             (D50_sand_ref_mum*1e-6)/(d50_sed_i)) *(
#                             2650/rho_sed)*(1 + TAAPS_freq1['S'][i]))
#     else: 
#         B_fines_freq1_calc_i = np.nan
#         B_fines_freq2_calc_i = np.nan
#         f_TM08_freq1_i = np.nan
#         f_TM08_freq2_i = np.nan 
        
#     B_fines_freq1_calc.append(B_fines_freq1_calc_i)          
#     B_fines_freq2_calc.append(B_fines_freq2_calc_i)   
#     f_TM08_freq1.append(f_TM08_freq1_i)
#     f_TM08_freq2.append(f_TM08_freq2_i)   

# TAAPS_freq1['B_fines'] = B_fines_freq1_calc
# TAAPS_freq2['B_fines'] = B_fines_freq2_calc

#%% STEP 11 D: CALCULATE B'
# # Determine S for samples using acoustic and reference data
# C_sand_S2_D50_g_l_time_freq1 = np.interp(Time_datetime_freq1, Time_datetime_freq2, C_sand_S2_D50_g_l)
# S_freq11 = [C_fines_est_freq1[i]/C_sand_S2_D50_g_l_time_freq1[i] for i in range(len(C_fines_est_freq1))]
# S_freq21 = [C_fines_est_freq2[i]/C_sand_S2_D50_g_l[i] for i in range(len(C_fines_est_freq2))]
S_freq11 = [C_fines_est_freq1[i]/C_sand_ref_g_l for i in range(len(C_fines_est_freq1))]
S_freq21 = [C_fines_est_freq2[i]/C_sand_ref_g_l for i in range(len(C_fines_est_freq2))]

B_fines_freq1 = np.interp(S_freq11, sRatio, ExcessBackScat_freq1)
B_fines_freq2 = np.interp(S_freq21, sRatio, ExcessBackScat_freq2)

#%% STEP 11 E: CALCULATE B SAND (CORRECTED FOR B')

# Bsand = B - B'
if B_fines_correction == True:
    BBase_freq1 = BeamAvBS_freq1 - B_fines_freq1
    BBase_freq2 = BeamAvBS_freq2 - B_fines_freq2
if B_fines_correction == False:
    BBase_freq1 = BeamAvBS_freq1
    BBase_freq2 = BeamAvBS_freq2
    
#%% STEP 14+: CALCULATE C SAND
#%% STEP15+: CALCULATE C SAND
# Empirical
C_sand_S2_D50_freq1_g_l = [10**(K1_emp_freq1_S2_D50 + K2_emp_freq1_S2_D50*BBase_freq1.iloc[i])/1000
                         for i in range(len(BBase_freq1))]
C_sand_S2_D50_freq2_g_l = [10**(K1_emp_freq2_S2_D50 + K2_emp_freq2_S2_D50*BBase_freq2.iloc[i])/1000
                         for i in range(len(BBase_freq2))]


#%%#############################################################################################

# RUTS - BASED DUAL FREQUENCY METHOD

##############################################################################################
#%% STEP 1: Calculate theoretical RUTS relation
# Define D50 range and sigma
max_D50_sand_range = TAAPS_freq1['D50_mum'].max()
D50_sand_range_mum = np.arange(63, 1500, 1)

#Determine form factor
# Define boundaries, phi system and convert to mm (2**)   
gs = [2**(-35.25 + 0.0625*j) for j in range(0,672)]
gsbound = [2**(-35.2188 + 0.0625*j) for j in range(0,672)]

f_TM08_freq1_sand_range = []
f_TM08_freq2_sand_range = []
for i in range(len(D50_sand_range_mum)):
    D50_sand_i_mum = D50_sand_range_mum[i]
    
    # Calculate i sand f following TW16
    volfracsand_i = [(1/np.sqrt(2*math.pi*D50_sand_i_mum))*np.exp((-(np.log(gs[j])-np.log(D50_sand_i_mum*1e-3))**2)/(2*sigma_sand_ref_mum**2))       
                    for j in range(len(gs))]
    cumsumsand_i = np.cumsum(volfracsand_i)
    proba_vol_sand_i = volfracsand_i/cumsumsand_i[-1]
    
    # Determine D50
    cumsum_sand_i = np.cumsum(proba_vol_sand_i)
    d500_i = 0.5*cumsum_sand_i[-1]
    
    for j in range(len(cumsum_sand_i)):
        if d500_i >= cumsum_sand_i[j-1] and d500_i <= cumsum_sand_i[j]:
            d50sand_i=((d500_i-cumsum_sand_i[j-1])/(cumsum_sand_i[j]-cumsum_sand_i[j-1]))*(np.log(gsbound[j])-np.log(gsbound[j-1]))+np.log(gsbound[j-1])
    d50_sand_i = np.exp(d50sand_i)/1000
    
    # Computing number probability
    ss = np.sum(proba_vol_sand_i / np.array(gs)**3)
    proba_num_sand_i = proba_vol_sand_i/np.array(gs)**3 / ss
    
    # Calculate form function TM08
    # Integrating over the distribution        
    temp_a1 = 0
    temp_a2f2_TM08_freq1 = 0
    temp_a2f2_TM08_freq2 = 0
    temp_a3 = 0  
    # Summing the integrals
    for l in range(len(proba_num_sand_i)):
        a = np.array(gs)[l]/1000
        temp_a2f2_TM08_freq1 += (a/2)**2 * form_factor_function_ThorneMeral2008((a/2), freq1_Hz, 1500)**2 * proba_num_sand_i[l]   
        temp_a2f2_TM08_freq2 += (a/2)**2 * form_factor_function_ThorneMeral2008((a/2), freq2_Hz, 1500)**2 * proba_num_sand_i[l]    
        temp_a3 += (a/2)**3 * proba_num_sand_i[l]
    
    # computing output values   
    f_TM08_freq1_sand_iTW16 = (((d50_sand_i/2)*temp_a2f2_TM08_freq1)/temp_a3)**0.5
    f_TM08_freq2_sand_iTW16 = (((d50_sand_i/2)*temp_a2f2_TM08_freq2)/temp_a3)**0.5
    
    f_TM08_freq1_sand_range.append(f_TM08_freq1_sand_iTW16)
    f_TM08_freq2_sand_range.append(f_TM08_freq2_sand_iTW16)

#%%
# f_TM08_freq2_sed_range use only variation of D50, keeping sigma constant
# calculate UTS theoretical 
UTS_theo_freq1 = [10*np.log10(f_TM08_freq1_sand_range[i]**2 *3/(16*math.pi*D50_sand_range_mum[i]*1e-6 * rho_sed*1**2)) + UTS_beam_freq1
                  for i in range(len(f_TM08_freq1_sand_range))]
UTS_theo_freq2 = [10*np.log10(f_TM08_freq2_sand_range[i]**2 *3/(16*math.pi*D50_sand_range_mum[i]*1e-6 * rho_sed*1**2)) + UTS_beam_freq2
                  for i in range(len(f_TM08_freq2_sand_range))]
UTS_SED_theo_freq1 = [10*np.log10(f_TM08_freq1_sand_range[i]**2 *3/(16*math.pi*D50_sand_range_mum[i]*1e-6 * rho_sed*1**2))
                  for i in range(len(f_TM08_freq1_sand_range))]
UTS_SED_theo_freq2 = [10*np.log10(f_TM08_freq2_sand_range[i]**2 *3/(16*math.pi*D50_sand_range_mum[i]*1e-6 * rho_sed*1**2))
                  for i in range(len(f_TM08_freq2_sand_range))]

# Calculate theoretical RUTS
# using D50 sand ref and sigma G
RUTS_theo_freq1 = [UTS_theo_freq1[i] - UTS_ref_freq1 for i in range(len(UTS_theo_freq1))]
RUTS_theo_freq2 = [UTS_theo_freq2[i] - UTS_ref_freq2 for i in range(len(UTS_theo_freq2))]

#%% STEP 2: C_sand_freq2 as initial state
C_sand_S2_D50_freq1_time_freq2_g_l = np.interp(Time_datetime_freq2, Time_datetime_freq1, C_sand_S2_D50_freq1_g_l)

#%% STEP 3A: # Determine meas, where C_sand_est_freq1 < / = / > C_sand_freq2_g_l
idx_C_freq1_smaller_C_freq2_S2_D50 = [i for i in range(len(C_sand_S2_D50_freq2_g_l))
           if np.round(C_sand_S2_D50_freq1_time_freq2_g_l[i],3) < np.round(C_sand_S2_D50_freq2_g_l[i],3)] 
            # D50 sand < D50 sand ref, log_C_sand_S2_D50_freq2_g_l is increased using the RUTS and BBC relations for the two frequencies
            
idx_C_freq1_equal_C_freq2_S2_D50 = [i for i in range(len(C_sand_S2_D50_freq2_g_l))
           if np.round(C_sand_S2_D50_freq1_time_freq2_g_l[i],3) == np.round(C_sand_S2_D50_freq2_g_l[i],3)] 
            # D50 sand = D50 sand ref
            
idx_C_freq1_greater_C_freq2_S2_D50 = [i for i in range(len(C_sand_S2_D50_freq2_g_l))
           if np.round(C_sand_S2_D50_freq1_time_freq2_g_l[i],3) > np.round(C_sand_S2_D50_freq2_g_l[i],3)] 
        # D50 sand > D50 sand ref, log_C_sand_S2_D50_freq2_g_l is reduced using the RUTS and BBC relations for the two frequencies


#%% STEP 3B: Determine effective beam-averaged BS for freq1
BeamAvBS_effective_1_freq1_S2_D50 = [(np.log10(C_sand_S2_D50_freq1_time_freq2_g_l[i]*1000) - K1_emp_freq1_S2_D50)/K2_emp_freq1_S2_D50
                            for i in range(len(C_sand_S2_D50_freq1_time_freq2_g_l))]
BeamAvBS_effective_2_freq1_S2_D50 = [(np.log10(C_sand_S2_D50_freq2_g_l[i]*1000) - K1_emp_freq1_S2_D50)/K2_emp_freq1_S2_D50
                            for i in range(len(C_sand_S2_D50_freq2_g_l))]
B_defect_freq1_S2_D50 = [BeamAvBS_effective_1_freq1_S2_D50[i] - BeamAvBS_effective_2_freq1_S2_D50[i]
                  for i in range(len(C_sand_S2_D50_freq2_g_l))]

#%% STEP 3C: Determine D50 and B_defect_freq2 using RUTS
# Determine idx(RUTS_freq1) at B_defect_freq1
idx_est_S2_D50 = []
for i in range(len(B_defect_freq1_S2_D50)):
    difference_array = np.absolute(np.array(RUTS_theo_freq1)-B_defect_freq1_S2_D50[i])                    
    index = difference_array.argmin()
    idx_est_S2_D50.append(index)

# Determine D50_xs_sand at RUTS_freq1
D50_est_S2_D50 = D50_sand_range_mum[idx_est_S2_D50]

# Determine B_defect_freq2_theo at D50_xs_sand 
B_defect_freq2_S2_D50 = [RUTS_theo_freq2[idx_est_S2_D50[i]]
                  for i in range(len(idx_est_S2_D50))]

# Determine effective beam-averaged BS for freq2 
BeamAvBS_effective_freq2_S2_D50 = [(np.log10(C_sand_S2_D50_freq2_g_l[i]*1000) - K1_emp_freq2_S2_D50)/K2_emp_freq2_S2_D50
                            for i in range(len(C_sand_S2_D50_freq2_g_l))]

# Determine sand concentration
C_sand_S2_D50_g_l = [10**(K1_emp_freq2_S2_D50 + K2_emp_freq2_S2_D50*(BeamAvBS_effective_freq2_S2_D50[i] - B_defect_freq2_S2_D50[i]))/1000
                    for i in range(len(B_defect_freq2_S2_D50))]


#%% #########################################################################################################


# VISUALISATION


#############################################################################################################

#%% Get CSAND and CFINES and D50, RUTS during sampling time 

# # Get D50_xs_sand during sampling time
D50_S2_D50_samples_g_l = [np.nanmean(D50_est_S2_D50[ind_first_beam_av_freq2[i]:ind_last_beam_av_freq2[i]+1])
                     for i in range(len(ind_first_beam_av_freq2))]

# Get C sand est during sampling time
C_sand_S2_D50_samples_g_l = [np.nanmean(C_sand_S2_D50_g_l[ind_first_beam_av_freq2[i]:ind_last_beam_av_freq2[i]+1])
                     for i in range(len(ind_first_beam_av_freq2))]

# Get Cfines est during sampling time
C_fines_samples_g_l = [np.nanmean(C_fines_est_time_freq2[ind_first_beam_av_freq2[i]:ind_last_beam_av_freq2[i]+1])
                     for i in range(len(ind_first_beam_av_freq2))]
C_fines_samples_g_l = C_fines_samples_g_l[0:19] #19 TW16 and 20 for TW16
TAAPS_freq1_fines['C_fines_samples_g_l'] = np.round(C_fines_samples_g_l,3)
TAAPS_freq2_fines['C_fines_samples_g_l'] = np.round(C_fines_samples_g_l,3)  

# RUTS for meas
RUTS_samples_S2_D50_freq1 = TAAPS_freq1['Beam-Averaged Backscatter (dB)'] - ((TAAPS_freq1['log_sand']-K1_emp_freq1_S2_D50)/K2_emp_freq1_S2_D50)
RUTS_samples_S2_D50_freq2 = TAAPS_freq2['Beam-Averaged Backscatter (dB)'] - ((TAAPS_freq2['log_sand']-K1_emp_freq2_S2_D50)/K2_emp_freq2_S2_D50)

RUTS_pump_S2_D50_freq1 = pump_data['BeamAvBS_freq1'] - ((pump_data['log_sand']-K1_emp_freq1_S2_D50)/K2_emp_freq1_S2_D50)
RUTS_pump_S2_D50_freq2 = pump_data['BeamAvBS_freq2'] - ((pump_data['log_sand']-K1_emp_freq2_S2_D50)/K2_emp_freq2_S2_D50)


#%% Plot Sensitivity RUTS pres

D50_sand_range_mum = np.arange(63, 1500, 1)
fig, ax = plt.subplots(1,1, figsize = (8,6), dpi=300)

# 1 
m3, = ax.plot(D50_sand_range_mum,RUTS_theo_freq1, 
         ls = '-', lw = 2, color = 'olive', label = 'Model 400 kHz')
m5 = ax.vlines(1/k_freq1/1e-6*2, -15,15, color = 'olive',ls = '--', lw = 2)
# m6 = ax.vlines(1/k_freq2/1e-6*2, -15,15, color = 'mediumorchid',ls = ':', lw = 2)
# m4, = ax.plot(D50_sand_range_mum, RUTS_theo_freq2, 
#          ls = '--', lw = 2, color = 'mediumorchid', label = 'Model 1 MHz')
m1, = ax.plot(TAAPS_freq1['D50_mum'],RUTS_samples_S2_D50_freq1, 
          'd', markersize = 9, ls = '', color = 'olive', markeredgecolor = 'black', markeredgewidth = 0.2, label = 'Calculated 400 kHz')
# m2, = ax.plot(TAAPS_freq2['D50_mum'],RUTS_samples_S2_D50_freq2, 
#           'p', markersize = 9, ls = '', color = 'mediumorchid', markeredgecolor = 'black', markeredgewidth = 0.1, label = 'Calculated 1 MHz')

# # ax.text(0.59, 0.05, '1 MHz Rayleigh -', fontsize = 12, rotation = 90, transform = ax.transAxes)
# ax.text(0.65, 0.05, 'geometric regime limit', fontsize = 12, rotation = 90, transform = ax.transAxes)

ax.text(0.88, 0.05, '400 kHz Rayleigh -', fontsize = 14, rotation = 90, transform = ax.transAxes)
ax.text(0.94, 0.05, 'geometric regime limit', fontsize = 14, rotation = 90, transform = ax.transAxes)

ax.tick_params(axis='both', which='major', labelsize = 16)
ax.set_xlim (63, 1500)
ax.set_ylim (-15, 15)
ax.set_xscale('log')
ax.set_ylabel(r'$\mathregular{RUTS}}$ (dB)', fontsize=18, weight = 'bold')
x_formatter = FixedFormatter(["100", "200", "500", "1000"])
x_locator = FixedLocator([100, 200, 500, 1000])
ax.xaxis.set_major_formatter(x_formatter)
ax.xaxis.set_major_locator(x_locator)
ax.axvspan(100, 350, 0,1500, color = 'lightgrey', alpha = 0.7, zorder = 0, label = 'Grain sizes Isère River')
ax.legend(loc = 'upper left', fontsize = 14, framealpha = 1)
ax.set_xlabel(r'$\mathregular{\overline{D_{50, sand}}}$ ($\mathregular{\mu}$m)', fontsize=18, weight = 'bold')


fig.tight_layout()
figname = 'Sensitivity_RUTS'
fig.savefig(outpath_figures +'\\' +  figname + '.png', dpi = 300, bbox_inches='tight')
# fig.savefig(outpath_figures +'\\' +  figname + '.eps', format = 'eps', dpi = 300, bbox_inches='tight')

#%%############################################################################################################

# VALIDATION

#############################################################################################################

# Validation data
# 1) Suspended sand measurements using the pump
# 2) Corrected automatic pumping samples (Csand and Cfines)
# 3) Calculated suspended sand concentration using turbidity


#%% Prepare validation data         

# 1) Pumping validation data
# Find nearest acoustic measurement to pump sample
Csand_S2_D50_pump_valid = np.interp(Time_pump_mid_datetime, Time_datetime_freq2, C_sand_S2_D50_g_l)
Cfines_pump_valid = np.interp(Time_pump_mid_datetime, Time_datetime_freq2, C_fines_est_time_freq2)
Q_time_pump = np.interp(Time_pump_mid_datetime, Time_Q_datetime,Q)

# Find nearest 1freq acoustic measurement to pump sample
Csand_S2_D50_freq1_pump_valid = np.interp(Time_pump_mid_datetime, Time_datetime_freq1, C_sand_S2_D50_freq1_g_l)
Csand_S2_D50_freq2_pump_valid = np.interp(Time_pump_mid_datetime, Time_datetime_freq2, C_sand_S2_D50_freq2_g_l)


# 2) ISCO validation data
# Find nearest acoustic measurement to ISCO sample
Csand_S2_D50_ISCO_valid = np.interp(Time_ISCO_mid_datetime, Time_datetime_freq2, C_sand_S2_D50_g_l)
D50_S2_D50_ISCO_valid = np.interp(Time_ISCO_GSD_mid_datetime, Time_datetime_freq2, D50_est_S2_D50)
Cfines_ISCO_valid = np.interp(Time_ISCO_mid_datetime, Time_datetime_freq2, C_fines_est_time_freq2)
Q_time_ISCO = np.interp(Time_ISCO_mid_datetime, Time_Q_datetime,Q)  
Cfines_ISCO_valid_freq1 = np.interp(Time_ISCO_mid_datetime, Time_datetime_freq1, C_fines_est_freq1) 
Cfines_ISCO_valid_freq2 = np.interp(Time_ISCO_mid_datetime, Time_datetime_freq2, C_fines_est_freq2)                  
                                   
# Correct ISCO sand xx
# Csand_S2_ISCO_corr = [43.987*Csand_S2_ISCO_valid[i]**(3.7037) for i in range(len(Csand_S2_ISCO_valid))]

# Find nearest 1freq acoustic measurement to ISCO sample
Csand_S2_D50_freq1_ISCO_valid = np.interp(Time_ISCO_mid_datetime, Time_datetime_freq1, C_sand_S2_D50_freq1_g_l)
Csand_S2_D50_freq2_ISCO_valid = np.interp(Time_ISCO_mid_datetime, Time_datetime_freq2, C_sand_S2_D50_freq2_g_l)

# other
# Get C sand est 1freq during sampling time
C_sand_S2_D50_freq1_samples_g_l = [np.nanmean(C_sand_S2_D50_freq1_g_l[ind_first_beam_av_freq1[i]:ind_last_beam_av_freq1[i]+1])
                     for i in range(len(ind_first_beam_av_freq1))]
C_sand_S2_D50_freq2_samples_g_l = [np.nanmean(C_sand_S2_D50_freq2_g_l[ind_first_beam_av_freq2[i]:ind_last_beam_av_freq2[i]+1])
                     for i in range(len(ind_first_beam_av_freq2))] 

# Get C sand est 1freq during sampling time
C_fines_freq1_samples_g_l = [np.nanmean(C_fines_est_freq1[ind_first_beam_av_freq1[i]:ind_last_beam_av_freq1[i]+1])
                     for i in range(len(ind_first_beam_av_freq1))]
C_fines_freq2_samples_g_l = [np.nanmean(C_fines_est_freq2[ind_first_beam_av_freq2[i]:ind_last_beam_av_freq2[i]+1])
                     for i in range(len(ind_first_beam_av_freq2))]      
     

#%% Validation Fines ##############

#%% Regress Cfines,ISCO and Cfines,meas with Cfines,HADCP
x_range_fines = np.linspace(0,20,10000)
# force through origin (and loglog plot adapted)
# meas
from scipy.optimize import curve_fit
x_range = np.logspace(-3, 1, base=10)
def myExpFunc(x, a, b):
    return a * np.power(x, b)

# samples 
interp_Cmeas_CHADCP_fines = np.polyfit(np.log10(TAAPS_freq1_fines['Fine_concentration_g_l']),np.log10(C_fines_samples_g_l), 1)
lin_model_Cmeas_CHADCP_fines = [10**(np.log10(TAAPS_freq1_fines['Fine_concentration_g_l'][i])*interp_Cmeas_CHADCP_fines[0]+interp_Cmeas_CHADCP_fines[1])
                           for i in range(len(C_fines_samples_g_l))]
lin_model_Cmeas_CHADCP_fines_plot = [10**(np.log10(x_range_fines[i])*interp_Cmeas_CHADCP_fines[0]+interp_Cmeas_CHADCP_fines[1])
                           for i in range(len(x_range_fines))]
R2_time_Cmeas_CHADCP_fines = r2_score(C_fines_samples_g_l, lin_model_Cmeas_CHADCP_fines)

# ISCO 
interp_CISCO_CHADCP_fines = np.polyfit(np.log10(ISCO_data['Fine_concentration_g_l']),np.log10(Cfines_ISCO_valid), 1)
lin_model_CISCO_CHADCP_fines = [10**(np.log10(ISCO_data['Fine_concentration_g_l'][i])*interp_CISCO_CHADCP_fines[0]+interp_CISCO_CHADCP_fines[1])
                           for i in range(len(Cfines_ISCO_valid))]
lin_model_CISCO_CHADCP_fines_plot = [10**(np.log10(x_range_fines[i])*interp_CISCO_CHADCP_fines[0]+interp_CISCO_CHADCP_fines[1])
                           for i in range(len(x_range_fines))]
R2_CISCO_CHADCP_fines = r2_score(Cfines_ISCO_valid, lin_model_CISCO_CHADCP_fines)

# spm 
interp_Cspm_CHADCP_fines = np.polyfit(np.log10(spm_time_freq2),np.log10(C_fines_est_time_freq2), 1)
lin_model_Cspm_CHADCP_fines = [10**(np.log10(spm_time_freq2[i])*interp_Cspm_CHADCP_fines[0]+interp_Cspm_CHADCP_fines[1])
                           for i in range(len(C_fines_est_time_freq2))]
lin_model_Cspm_CHADCP_fines_plot = [10**(np.log10(x_range_fines[i])*interp_Cspm_CHADCP_fines[0]+interp_Cspm_CHADCP_fines[1])
                           for i in range(len(x_range_fines))]
R2_Cspm_CHADCP_fines = r2_score(C_fines_est_time_freq2, lin_model_Cspm_CHADCP_fines)


#%% Regress Cfines,ISCO and Cfines,meas with Cfines,HADCP - Single freq
x_range_fines = np.linspace(0,20,10000)

# freq1
# samples 
interp_Cmeas_CHADCP_fines_freq1 = np.polyfit(np.log10(TAAPS_freq1['Fine_concentration_g_l']),np.log10(C_fines_freq1_samples_g_l), 1)
lin_model_Cmeas_CHADCP_fines_freq1 = [10**(np.log10(TAAPS_freq1['Fine_concentration_g_l'][i])*interp_Cmeas_CHADCP_fines_freq1[0]+interp_Cmeas_CHADCP_fines_freq1[1])
                           for i in range(len(C_fines_freq1_samples_g_l))]
lin_model_Cmeas_CHADCP_fines_freq1_plot = [10**(np.log10(x_range_fines[i])*interp_Cmeas_CHADCP_fines_freq1[0]+interp_Cmeas_CHADCP_fines_freq1[1])
                           for i in range(len(x_range_fines))]
R2_time_Cmeas_CHADCP_fines_freq1 = r2_score(C_fines_freq1_samples_g_l, lin_model_Cmeas_CHADCP_fines_freq1)

# ISCO 
interp_CISCO_CHADCP_fines_freq1 = np.polyfit(np.log10(ISCO_data['Fine_concentration_g_l']),np.log10(Cfines_ISCO_valid_freq1), 1)
lin_model_CISCO_CHADCP_fines_freq1 = [10**(np.log10(ISCO_data['Fine_concentration_g_l'][i])*interp_CISCO_CHADCP_fines_freq1[0]+interp_CISCO_CHADCP_fines_freq1[1])
                           for i in range(len(Cfines_ISCO_valid_freq1))]
lin_model_CISCO_CHADCP_fines_freq1_plot = [10**(np.log10(x_range[i])*interp_CISCO_CHADCP_fines_freq1[0]+interp_CISCO_CHADCP_fines_freq1[1])
                           for i in range(len(x_range))]
R2_CISCO_CHADCP_fines_freq1 = r2_score(Cfines_ISCO_valid_freq1, lin_model_CISCO_CHADCP_fines_freq1)

# spm 
interp_Cspm_CHADCP_fines_freq1 = np.polyfit(np.log10(spm_time_freq1),np.log10(C_fines_est_freq1), 1)
lin_model_Cspm_CHADCP_fines_freq1 = [10**(np.log10(spm_time_freq1[i])*interp_Cspm_CHADCP_fines_freq1[0]+interp_Cspm_CHADCP_fines_freq1[1])
                           for i in range(len(C_fines_est_freq1))]
lin_model_Cspm_CHADCP_fines_freq1_plot = [10**(np.log10(x_range_fines[i])*interp_Cspm_CHADCP_fines[0]+interp_Cspm_CHADCP_fines_freq1[1])
                           for i in range(len(x_range_fines))]
R2_Cspm_CHADCP_fines_freq1 = r2_score(C_fines_est_freq1, lin_model_Cspm_CHADCP_fines_freq1)


# freq2
# samples 
interp_Cmeas_CHADCP_fines_freq2 = np.polyfit(np.log10(TAAPS_freq2['Fine_concentration_g_l']),np.log10(C_fines_freq2_samples_g_l), 1)
lin_model_Cmeas_CHADCP_fines_freq2 = [10**(np.log10(TAAPS_freq2['Fine_concentration_g_l'][i])*interp_Cmeas_CHADCP_fines_freq2[0]+interp_Cmeas_CHADCP_fines_freq2[1])
                           for i in range(len(C_fines_freq2_samples_g_l))]
lin_model_Cmeas_CHADCP_fines_freq2_plot = [10**(np.log10(x_range_fines[i])*interp_Cmeas_CHADCP_fines_freq2[0]+interp_Cmeas_CHADCP_fines_freq2[1])
                           for i in range(len(x_range_fines))]
R2_time_Cmeas_CHADCP_fines_freq2 = r2_score(C_fines_freq2_samples_g_l, lin_model_Cmeas_CHADCP_fines_freq2)

# ISCO 
interp_CISCO_CHADCP_fines_freq2 = np.polyfit(np.log10(ISCO_data['Fine_concentration_g_l']),np.log10(Cfines_ISCO_valid_freq2), 1)
lin_model_CISCO_CHADCP_fines_freq2 = [10**(np.log10(ISCO_data['Fine_concentration_g_l'][i])*interp_CISCO_CHADCP_fines_freq2[0]+interp_CISCO_CHADCP_fines_freq2[1])
                           for i in range(len(Cfines_ISCO_valid_freq2))]
lin_model_CISCO_CHADCP_fines_freq2_plot = [10**(np.log10(x_range[i])*interp_CISCO_CHADCP_fines_freq2[0]+interp_CISCO_CHADCP_fines_freq2[1])
                           for i in range(len(x_range))]
R2_CISCO_CHADCP_fines_freq2 = r2_score(Cfines_ISCO_valid_freq2, lin_model_CISCO_CHADCP_fines_freq2)

# spm 
interp_Cspm_CHADCP_fines_freq2 = np.polyfit(np.log10(spm_time_freq2),np.log10(C_fines_est_freq2), 1)
lin_model_Cspm_CHADCP_fines_freq2 = [10**(np.log10(spm_time_freq2[i])*interp_Cspm_CHADCP_fines_freq2[0]+interp_Cspm_CHADCP_fines_freq2[1])
                           for i in range(len(C_fines_est_freq2))]
lin_model_Cspm_CHADCP_fines_freq2_plot = [10**(np.log10(x_range_fines[i])*interp_Cspm_CHADCP_fines[0]+interp_Cspm_CHADCP_fines_freq2[1])
                           for i in range(len(x_range_fines))]
R2_Cspm_CHADCP_fines_freq2 = r2_score(C_fines_est_freq2, lin_model_Cspm_CHADCP_fines_freq2)



#%% Plot C est fines with C meas - HADCP
fig, ax = plt.subplots(1, 1, figsize = (8,6), dpi=300)

ax.plot(TAAPS_freq2_fines['Fine_concentration_g_l'], C_fines_samples_g_l,  marker = 'D', 
        ls = '', markersize = 8, color = 'darkorange', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 40,
        label = 'Sampler')
ax.plot(ISCO_data['Fine_concentration_g_l'], Cfines_ISCO_valid, marker = 'o',              
        ls = '', markersize = 5, color = 'yellowgreen', markeredgecolor = 'black', markeredgewidth = 0.1, zorder = 30) 
ax.plot(ISCO_data['Fine_concentration_g_l'][10], Cfines_ISCO_valid[10], marker = 'o',              
        ls = '', markersize = 7, color = 'yellowgreen', markeredgecolor = 'black', markeredgewidth = 0.1, zorder = 30,
        label = 'ISCO') 

ax.plot(5*x_range, x_range,  zorder = 21,
         ls = (0, (1, 10)), lw = 1, color = 'black') 
ax.plot(2*x_range, x_range,  zorder = 21,
        ls = ':', lw = 1, color = 'black')
p4, = ax.plot(x_range, x_range,  zorder = 21,
        ls = '-', lw = 1, color = 'black', label = 'Perfect agreement')
p5, = ax.plot(x_range, 2*x_range,  zorder = 21,
        ls = ':', lw = 1, color = 'black', label = 'Error of a factor of 2') 
p6, = ax.plot(x_range, 5*x_range,  zorder = 21,
         ls = (0, (1, 10)), lw = 1, color = 'black', label = 'Error of a factor of 5') 
 
ax.text(0.05, 0.95, '(a)', fontsize = 16, transform = ax.transAxes)
# ax.legend(fontsize = 16, loc = 'upper left')
ax.set_xlabel('$\mathregular{C_{fines, meas}}$ (g/l)', fontsize=20, weight = 'bold')
ax.set_ylabel('$\mathregular{\overline{C_{fines, HADCP}}}$ (g/l)', fontsize=20, weight = 'bold')
ax.tick_params(axis='both', which='major', labelsize = 16)
ax.set_xlim(0.01,10)
ax.set_ylim(0.01,10)
ax.set_xscale('log')
ax.set_yscale('log')

fig.tight_layout()
figname = 'Fig_A2'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 300, bbox_inches='tight')
fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight') 
fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 300, bbox_inches='tight')


#%% Validation Single-frequency ##############

#%% Regressions Single-frequency 

# Freq1 
x_range = np.linspace(0,10,10000)
interp_Cmeas_CHADCP_sand_freq1 = np.polyfit(np.log10(TAAPS_freq1['Sand_concentration_g_l']),np.log10(C_sand_S2_D50_freq1_samples_g_l), 1)
lin_model_Cmeas_CHADCP_sand_freq1 = [10**(np.log10(TAAPS_freq1['Sand_concentration_g_l'][i])*interp_Cmeas_CHADCP_sand_freq1[0]+interp_Cmeas_CHADCP_sand_freq1[1])
                           for i in range(len(C_sand_S2_D50_freq1_samples_g_l))]
lin_model_Cmeas_CHADCP_sand_freq1_plot = [10**(np.log10(x_range[i])*interp_Cmeas_CHADCP_sand_freq1[0]+interp_Cmeas_CHADCP_sand_freq1[1])
                           for i in range(len(x_range))]
R2_Cmeas_CHADCP_sand_freq1 = r2_score(C_sand_S2_D50_freq1_samples_g_l, lin_model_Cmeas_CHADCP_sand_freq1)

interp_CISCO_CHADCP_sand_freq1 = np.polyfit(np.log10(ISCO_data['ISCO_sand_concentration_corr_g_l']),np.log10(Csand_S2_D50_freq1_ISCO_valid), 1)
lin_model_CISCO_CHADCP_sand_freq1 = [10**(np.log10(ISCO_data['ISCO_sand_concentration_corr_g_l'][i])*interp_CISCO_CHADCP_sand_freq1[0]+interp_CISCO_CHADCP_sand_freq1[1])
                           for i in range(len(Csand_S2_D50_freq1_ISCO_valid))]
lin_model_CISCO_CHADCP_sand_freq1_plot = [10**(np.log10(x_range[i])*interp_CISCO_CHADCP_sand_freq1[0]+interp_CISCO_CHADCP_sand_freq1[1])
                           for i in range(len(x_range))]
R2_CISCO_CHADCP_sand_freq1 = r2_score(Csand_S2_D50_freq1_ISCO_valid, lin_model_CISCO_CHADCP_sand_freq1)

interp_Cpump_CHADCP_sand_freq1 = np.polyfit(np.log10(pump_data['Sand_concentration_g_l']),np.log10(Csand_S2_D50_freq1_pump_valid), 1)
lin_model_Cpump_CHADCP_sand_freq1 = [10**(np.log10(pump_data['Sand_concentration_g_l'][i])*interp_Cpump_CHADCP_sand_freq1[0]+interp_Cpump_CHADCP_sand_freq1[1])
                           for i in range(len(Csand_S2_D50_freq1_pump_valid))]
lin_model_Cpump_CHADCP_sand_freq1_plot = [10**(np.log10(x_range[i])*interp_Cpump_CHADCP_sand_freq1[0]+interp_Cpump_CHADCP_sand_freq1[1])
                           for i in range(len(x_range))]
R2_Cpump_CHADCP_sand_freq1 = r2_score(Csand_S2_D50_freq1_pump_valid, lin_model_Cpump_CHADCP_sand_freq1)

# Freq2
interp_Cmeas_CHADCP_sand_freq2 = np.polyfit(np.log10(TAAPS_freq2['Sand_concentration_g_l']),np.log10(C_sand_S2_D50_freq2_samples_g_l), 1)
lin_model_Cmeas_CHADCP_sand_freq2 = [10**(np.log10(TAAPS_freq2['Sand_concentration_g_l'][i])*interp_Cmeas_CHADCP_sand_freq2[0]+interp_Cmeas_CHADCP_sand_freq2[1])
                           for i in range(len(C_sand_S2_D50_freq2_samples_g_l))]
lin_model_Cmeas_CHADCP_sand_freq2_plot = [10**(np.log10(x_range[i])*interp_Cmeas_CHADCP_sand_freq2[0]+interp_Cmeas_CHADCP_sand_freq2[1])
                           for i in range(len(x_range))]
R2_Cmeas_CHADCP_sand_freq2 = r2_score(C_sand_S2_D50_freq2_samples_g_l, lin_model_Cmeas_CHADCP_sand_freq2)

interp_CISCO_CHADCP_sand_freq2 = np.polyfit(np.log10(ISCO_data['ISCO_sand_concentration_corr_g_l']),np.log10(Csand_S2_D50_freq2_ISCO_valid), 1)
lin_model_CISCO_CHADCP_sand_freq2 = [10**(np.log10(ISCO_data['ISCO_sand_concentration_corr_g_l'][i])*interp_CISCO_CHADCP_sand_freq2[0]+interp_CISCO_CHADCP_sand_freq2[1])
                           for i in range(len(Csand_S2_D50_freq2_ISCO_valid))]
lin_model_CISCO_CHADCP_sand_freq2_plot = [10**(np.log10(x_range[i])*interp_CISCO_CHADCP_sand_freq2[0]+interp_CISCO_CHADCP_sand_freq2[1])
                           for i in range(len(x_range))]
R2_CISCO_CHADCP_sand_freq2 = r2_score(Csand_S2_D50_freq2_ISCO_valid, lin_model_CISCO_CHADCP_sand_freq2)

interp_Cpump_CHADCP_sand_freq2 = np.polyfit(np.log10(pump_data['Sand_concentration_g_l']),np.log10(Csand_S2_D50_freq2_pump_valid), 1)
lin_model_Cpump_CHADCP_sand_freq2 = [10**(np.log10(pump_data['Sand_concentration_g_l'][i])*interp_Cpump_CHADCP_sand_freq2[0]+interp_Cpump_CHADCP_sand_freq2[1])
                           for i in range(len(Csand_S2_D50_freq2_pump_valid))]
lin_model_Cpump_CHADCP_sand_freq2_plot = [10**(np.log10(x_range[i])*interp_Cpump_CHADCP_sand_freq2[0]+interp_Cpump_CHADCP_sand_freq2[1])
                           for i in range(len(x_range))]
R2_Cpump_CHADCP_sand_freq2 = r2_score(Csand_S2_D50_freq2_pump_valid, lin_model_Cpump_CHADCP_sand_freq2)


#%% Plot Fig7
fig, ax = plt.subplots(2, 2, figsize = (12,12), dpi=300)

# Freq1 
p1 = ax[0,0].errorbar(TAAPS_freq1['Sand_concentration_g_l'], C_sand_S2_D50_freq1_samples_g_l,  marker = 'D', 
            xerr = TAAPS_freq1['U_C']*TAAPS_freq1['Sand_concentration_g_l']/100, elinewidth = 1, capsize = 1, zorder = 10,
        ls = '', markersize = 8, color = 'darkorange', markeredgecolor = 'black', markeredgewidth = 0.2, label = 'Sampler')


ax[0,0].plot(5*x_range, x_range,    
        ls = (0, (1, 10)), lw = 1, color = 'black')
ax[0,0].plot(2*x_range, x_range,  
        ls = ':', lw = 1, color = 'black')
ax[0,0].plot(x_range, x_range,  
        ls = '-', lw = 1, color = 'black')
ax[0,0].plot(x_range, 2*x_range,  
        ls = ':', lw = 1, color = 'black')
ax[0,0].plot(x_range, 5*x_range,    
        ls = (0, (1, 10)), lw = 1, color = 'black')

ax[0,0].text(0.02, 0.95, '(a)', transform = ax[0,0].transAxes, fontsize = 20)
ax[0,0].set_ylabel('$\mathregular{\overline{C_{sand, 400 kHz}}}$ (g/l)', fontsize=24, weight = 'bold')
#ax[0,0].set_xlabel('$\mathregular{\overline{C_{sand, cal}}}$ (g/l)', fontsize=24, weight = 'bold')
ax[0,0].tick_params(axis='both', which='major', labelsize = 20)
ax[0,0].set_xlim(0.02,1)
ax[0,0].set_ylim(0.02,1)
ax[0,0].set_xscale('log')
ax[0,0].set_yscale('log')
ax[0,0].xaxis.set_ticklabels([])

# Freq1 
p2, = ax[0,1].plot(pump_data['Sand_concentration_g_l'], Csand_S2_D50_freq1_pump_valid, marker = 's',              
        ls = '', markersize = 8, color = 'mediumblue',zorder = 24, markeredgecolor = 'black', markeredgewidth = 0.01, 
        label = 'Pump')
p3, = ax[0,1].plot(ISCO_data['ISCO_sand_concentration_corr_g_l'], Csand_S2_D50_freq1_ISCO_valid, marker = 'o',              
        ls = '', markersize = 5, color = 'yellowgreen', #markeredgecolor = 'black', markeredgewidth = 0.01,
        label = 'ISCO')

ax[0,1].plot(2*x_range, x_range,  
        ls = ':', lw = 1, color = 'black')
ax[0,1].plot(x_range, x_range,  
        ls = '-', lw = 1, color = 'black')
ax[0,1].plot(x_range, 2*x_range,  
        ls = ':', lw = 1, color = 'black')
ax[0,1].plot(5*x_range, x_range,    
        ls = (0, (1, 10)), lw = 1, color = 'black')
ax[0,1].plot(x_range, 5*x_range,    
        ls = (0, (1, 10)), lw = 1, color = 'black')

ax[0,1].text(0.02, 0.95, '(b)', transform = ax[0,1].transAxes, fontsize = 20)
#ax[0,1].set_xlabel('$\mathregular{\overline{C_{sand, val}}}$ (g/l)', fontsize=24, weight = 'bold')
#ax[0,1].set_ylabel('$\mathregular{\overline{C_{sand, HADCP, 400 kHz}}}$ (g/l)', fontsize=24, weight = 'bold')
ax[0,1].tick_params(axis='both', which='major', labelsize = 20)
ax[0,1].set_xlim(0.02,1)
ax[0,1].set_ylim(0.02,1)
ax[0,1].set_xscale('log')
ax[0,1].set_yscale('log')

# Freq2 
p1 = ax[1,0].errorbar(TAAPS_freq2['Sand_concentration_g_l'], C_sand_S2_D50_freq2_samples_g_l,  marker = 'D', 
            xerr = TAAPS_freq2['U_C']*TAAPS_freq2['Sand_concentration_g_l']/100, elinewidth = 1, capsize = 1, zorder = 10,
        ls = '', markersize = 8, color = 'darkorange', markeredgecolor = 'black', markeredgewidth = 0.2, label = 'Sampler')

ax[1,0].plot(2*x_range, x_range,  
        ls = ':', lw = 1, color = 'black')
ax[1,0].plot(x_range, x_range,  
        ls = '-', lw = 1, color = 'black')
ax[1,0].plot(x_range, 2*x_range,  
        ls = ':', lw = 1, color = 'black')
ax[1,0].plot(5*x_range, x_range,    
        ls = (0, (1, 10)), lw = 1, color = 'black')
ax[1,0].plot(x_range, 5*x_range,    
        ls = (0, (1, 10)), lw = 1, color = 'black')

ax[1,0].text(0.02, 0.95, '(c)', transform = ax[1,0].transAxes, fontsize = 20)
ax[1,0].set_xlabel('$\mathregular{\overline{C_{sand, meas}}}$ (g/l)', fontsize=24, weight = 'bold')
ax[1,0].set_ylabel('$\mathregular{\overline{C_{sand, 1MHz}}}$ (g/l)', fontsize=24, weight = 'bold')
ax[1,0].tick_params(axis='both', which='major', labelsize = 20)
ax[1,0].set_xlim(0.02,1)
ax[1,0].set_ylim(0.02,1)
ax[1,0].set_xscale('log')
ax[1,0].set_yscale('log')


# Freq2 
p2, = ax[1,1].plot(pump_data['Sand_concentration_g_l'], Csand_S2_D50_freq2_pump_valid, marker = 's',              
        ls = '', markersize = 8, color = 'mediumblue',zorder = 24,markeredgecolor = 'black', markeredgewidth = 0.1, 
        label = 'Pump')
p3, = ax[1,1].plot(ISCO_data['ISCO_sand_concentration_corr_g_l'][10], Csand_S2_D50_freq2_ISCO_valid[10], marker = 'o',              
        ls = '', markersize = 7, color = 'yellowgreen', #markeredgecolor = 'black', markeredgewidth = 0.1,
        label = 'ISCO')
ax[1,1].plot(ISCO_data['ISCO_sand_concentration_corr_g_l'], Csand_S2_D50_freq2_ISCO_valid, marker = 'o',              
        ls = '', markersize = 5, color = 'yellowgreen', #markeredgecolor = 'black', markeredgewidth = 0.01,
        label = 'Validation - ISCO')

ax[1,1].plot(2*x_range, x_range,  
        ls = ':', lw = 1, color = 'black')
p4, = ax[1,1].plot(x_range, x_range,  
        ls = '-', lw = 1, color = 'black', label = 'Perfect agreement')
p5, = ax[1,1].plot(x_range, 2*x_range,  
        ls = ':', lw = 1, color = 'black', label = 'Error of a factor of 2')
p6, = ax[1,1].plot(5*x_range, x_range,    
        ls = (0, (1, 10)), lw = 1, color = 'black', label = 'Error of a factor of 5')
ax[1,1].plot(x_range, 5*x_range,    
        ls = (0, (1, 10)), lw = 1, color = 'black')
 
ax[1,1].text(0.02, 0.95, '(d)', transform = ax[1,1].transAxes, fontsize = 20)
ax[1,1].set_xlabel('$\mathregular{\overline{C_{sand, meas}}}$ (g/l)', fontsize=24, weight = 'bold')
#ax[1,1].set_ylabel('$\mathregular{\overline{C_{sand, HADCP, 1MHz}}}$ (g/l)', fontsize=24, weight = 'bold')
ax[1,1].tick_params(axis='both', which='major', labelsize = 20)
ax[1,1].set_xlim(0.02,1)
ax[1,1].set_ylim(0.02,1)
ax[1,1].set_xscale('log')
ax[1,1].set_yscale('log')

handles = [p1, p2, p3, p4 , p5, p6]
#_, labels = ax.get_legend_handles_labels()
fig.legend(handles = handles, #labels=labels, 
          handler_map = {tuple: mpl.legend_handler.HandlerTuple(None)}, framealpha = 1, 
          fontsize = 20, loc = 'lower center', ncol = 3, bbox_to_anchor = (0.5, -0.09))

fig.tight_layout()
figname = 'Fig7'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 300, bbox_inches='tight')
fig.savefig(outpath_figures + '\\' + figname + '.eps', dpi = 300, bbox_inches='tight')

#%% Plot Fig7 - Pres
fig, ax = plt.subplots(1, 1, figsize = (7,6), dpi=300)

# Freq2 
p1 = ax.errorbar(TAAPS_freq2['Sand_concentration_g_l'], C_sand_S2_D50_freq2_samples_g_l,  marker = 'D', 
            xerr = TAAPS_freq2['U_C']*TAAPS_freq2['Sand_concentration_g_l']/100, elinewidth = 1, capsize = 1, zorder = 40,
        ls = '', markersize = 9, color = 'darkorange', markeredgecolor = 'black', markeredgewidth = 0.2, label = 'Sampler')
p2, = ax.plot(pump_data['Sand_concentration_g_l'], Csand_S2_D50_freq2_pump_valid, marker = 's',              
        ls = '', markersize = 9, color = 'mediumblue', markeredgecolor = 'black', markeredgewidth = 0.1, zorder = 24,
        label = 'Pump')

ax.plot(2*x_range, x_range,  
        ls = ':', lw = 1, color = 'black')
ax.plot(x_range, x_range,  
        ls = '-', lw = 1, color = 'black')
ax.plot(x_range, 2*x_range,  
        ls = ':', lw = 1, color = 'black')
ax.plot(5*x_range, x_range,    
        ls = (0, (1, 10)), lw = 1, color = 'black')
ax.plot(x_range, 5*x_range,    
        ls = (0, (1, 10)), lw = 1, color = 'black')

ax.set_xlabel('$\mathregular{\overline{C_{sand, meas}}}$ (g/l)', fontsize=20, weight = 'bold')
ax.set_ylabel('$\mathregular{\overline{C_{sand, 1MHz}}}$ (g/l)', fontsize=20, weight = 'bold')
ax.tick_params(axis='both', which='major', labelsize = 18)
ax.set_xlim(0.01,2)
ax.set_ylim(0.01,2)
ax.set_xscale('log')
ax.set_yscale('log')

fig.tight_layout()
figname = 'Fig7_pres_1000'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 300, bbox_inches='tight')

#%% Plot Fig7 400 kHz - Pres
fig, ax = plt.subplots(1, 1, figsize = (7,6), dpi=300)

# Freq1 
p1 = ax.errorbar(TAAPS_freq1['Sand_concentration_g_l'], C_sand_S2_D50_freq1_samples_g_l,  marker = 'D', 
            xerr = TAAPS_freq1['U_C']*TAAPS_freq1['Sand_concentration_g_l']/100, elinewidth = 1, capsize = 1, zorder = 40,
        ls = '', markersize = 9, color = 'darkorange', markeredgecolor = 'black', markeredgewidth = 0.2, label = 'Sampler')

p2, = ax.plot(pump_data['Sand_concentration_g_l'], Csand_S2_D50_freq1_pump_valid, marker = 's',              
        ls = '', markersize = 9, color = 'mediumblue', markeredgecolor = 'black', markeredgewidth = 0.1, zorder = 24,
        label = 'Pump')

ax.plot(5*x_range, x_range,    
        ls = (0, (1, 10)), lw = 1, color = 'black')
ax.plot(2*x_range, x_range,  
        ls = ':', lw = 1, color = 'black')
ax.plot(x_range, x_range,  
        ls = '-', lw = 1, color = 'black')
ax.plot(x_range, 2*x_range,  
        ls = ':', lw = 1, color = 'black')
ax.plot(x_range, 5*x_range,    
        ls = (0, (1, 10)), lw = 1, color = 'black')

ax.set_ylabel('$\mathregular{\overline{C_{sand, 400 kHz}}}$ (g/l)', fontsize=20, weight = 'bold')
ax.set_xlabel('$\mathregular{\overline{C_{sand, meas}}}$ (g/l)', fontsize=20, weight = 'bold')
ax.tick_params(axis='both', which='major', labelsize = 18)
ax.set_xlim(0.01,2)
ax.set_ylim(0.01,2)
ax.set_xscale('log')
ax.set_yscale('log')

fig.tight_layout()
figname = 'Fig7_pres_400'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 300, bbox_inches='tight')

#%% Validation Dual-frequency ##############

#%% Regressions Dual-frequency 
 
x_range_sand = np.linspace(0,10,10000)

interp_Cmeas_CHADCP_sand = np.polyfit(np.log10(TAAPS_freq2['Sand_concentration_g_l']),np.log10(C_sand_S2_D50_samples_g_l), 1)
lin_model_Cmeas_CHADCP_sand = [10**(np.log10(TAAPS_freq2['Sand_concentration_g_l'][i])*interp_Cmeas_CHADCP_sand[0]+interp_Cmeas_CHADCP_sand[1])
                           for i in range(len(C_sand_S2_D50_samples_g_l))]
lin_model_Cmeas_CHADCP_sand_plot = [10**(np.log10(x_range_sand[i])*interp_Cmeas_CHADCP_sand[0]+interp_Cmeas_CHADCP_sand[1])
                           for i in range(len(x_range_sand))]
R2_Cmeas_CHADCP_sand = r2_score(C_sand_S2_D50_samples_g_l, lin_model_Cmeas_CHADCP_sand)

interp_CISCO_CHADCP_sand = np.polyfit(np.log10(ISCO_data['ISCO_sand_concentration_corr_g_l']),np.log10(Csand_S2_D50_ISCO_valid), 1)
lin_model_CISCO_CHADCP_sand = [10**(np.log10(ISCO_data['ISCO_sand_concentration_corr_g_l'][i])*interp_CISCO_CHADCP_sand[0]+interp_CISCO_CHADCP_sand[1])
                           for i in range(len(Csand_S2_D50_ISCO_valid))]
lin_model_CISCO_CHADCP_sand_plot = [10**(np.log10(x_range_sand[i])*interp_CISCO_CHADCP_sand[0]+interp_CISCO_CHADCP_sand[1])
                           for i in range(len(x_range_sand))]
R2_CISCO_CHADCP_sand = r2_score(Csand_S2_D50_ISCO_valid, lin_model_CISCO_CHADCP_sand)

interp_Cpump_CHADCP_sand = np.polyfit(np.log10(pump_data['Sand_concentration_g_l']),np.log10(Csand_S2_D50_pump_valid), 1)
lin_model_Cpump_CHADCP_sand = [10**(np.log10(pump_data['Sand_concentration_g_l'][i])*interp_Cpump_CHADCP_sand[0]+interp_Cpump_CHADCP_sand[1])
                           for i in range(len(Csand_S2_D50_pump_valid))]
lin_model_Cpump_CHADCP_sand_plot = [10**(np.log10(x_range_sand[i])*interp_Cpump_CHADCP_sand[0]+interp_Cpump_CHADCP_sand[1])
                           for i in range(len(x_range_sand))]
R2_Cpump_CHADCP_sand = r2_score(Csand_S2_D50_pump_valid, lin_model_Cpump_CHADCP_sand)


#%% Plot Fig 5
fig, ax = plt.subplots(1, 3, figsize = (12,4), dpi=300)

# Cal 
p1 = ax[0].errorbar(TAAPS_freq1['Sand_concentration_g_l'], C_sand_S2_D50_samples_g_l,  marker = 'D', 
            xerr = TAAPS_freq1['U_C']*TAAPS_freq1['Sand_concentration_g_l']/100, elinewidth = 1, capsize = 1.5, zorder = 40,
        ls = '', markersize = 7, color = 'darkorange', markeredgecolor = 'black', markeredgewidth = 0.2, label = 'Sampler')

ax[0].plot(5*x_range, x_range,  zorder = 41,
         ls = (0, (1, 10)), lw = 1, color = 'black') 
ax[0].plot(2*x_range, x_range,  zorder = 40,
        ls = ':', lw = 1, color = 'black')
ax[0].plot(x_range, x_range,  zorder = 40,
        ls = '-', lw = 1, color = 'black')
ax[0].plot(x_range, 2*x_range,  zorder = 40,
        ls = ':', lw = 1, color = 'black') 
ax[0].plot(x_range, 5*x_range,  zorder = 41,
         ls = (0, (1, 10)), lw = 1, color = 'black') 
ax[0].text(0.02, 0.95, '(a)', transform = ax[0].transAxes, fontsize = 12)

ax[0].set_xlabel('$\mathregular{\overline{C_{sand, meas}}}$ (g/l)', fontsize=14, weight = 'bold')
ax[0].set_ylabel('$\mathregular{\overline{C_{sand, HADCP}}}$ (g/l)', fontsize=14, weight = 'bold')
ax[0].tick_params(axis='both', which='major', labelsize = 12)
ax[0].set_xlim(0.01,2)
ax[0].set_ylim(0.01,2)
ax[0].set_xscale('log')
ax[0].set_yscale('log')

# Val 
p2, = ax[1].plot(pump_data['Sand_concentration_g_l'], Csand_S2_D50_pump_valid, marker = 's',       zorder = 31,       
        ls = '', markersize = 7, color = 'mediumblue', markeredgecolor = 'black', markeredgewidth = 0.1,
        label = 'Pump')
p3, = ax[1].plot(ISCO_data['ISCO_sand_concentration_corr_g_l'][10], Csand_S2_D50_ISCO_valid[10], marker = 'o', zorder = 21,             
        ls = '', markersize = 7, color = 'yellowgreen', markeredgecolor = 'black', markeredgewidth = 0.1,
        label = 'ISCO') 
ax[1].plot(ISCO_data['ISCO_sand_concentration_corr_g_l'], Csand_S2_D50_ISCO_valid, marker = 'o', zorder = 21,             
        ls = '', markersize = 4, color = 'yellowgreen', markeredgecolor = 'black', markeredgewidth = 0.1,
        label = '$\mathregular{C_{sand, ISCO, corr}}$') 

ax[1].plot(5*x_range, x_range,  zorder = 41,
         ls = (0, (1, 10)), lw = 1, color = 'black') 
ax[1].plot(2*x_range, x_range,  zorder = 41,
        ls = ':', lw = 1, color = 'black')
p4, = ax[1].plot(x_range, x_range,  zorder = 41,
        ls = '-', lw = 1, color = 'black', label = 'Perfect agreement')
p5, = ax[1].plot(x_range, 2*x_range,  zorder = 41,
        ls = ':', lw = 1, color = 'black', label = 'Error of a factor of 2') 
p6, = ax[1].plot(x_range, 5*x_range,  zorder = 41,
         ls = (0, (1, 10)), lw = 1, color = 'black', label = 'Error of a factor of 5') 

ax[1].text(0.02, 0.95, '(b)', transform = ax[1].transAxes, fontsize = 12)

ax[1].set_xlabel('$\mathregular{\overline{C_{sand, meas}}}$ (g/l)', fontsize=14, weight = 'bold')
ax[1].set_ylabel('$\mathregular{\overline{C_{sand, HADCP}}}$ (g/l)', fontsize=14, weight = 'bold')
ax[1].tick_params(axis='both', which='major', labelsize = 12)
ax[1].set_xlim(0.01,2)
ax[1].set_ylim(0.01,2)
ax[1].set_xscale('log')
ax[1].set_yscale('log')

# D50 
x_range_d50 = np.arange(63,1000,10)
ax[2].plot(TAAPS_freq2['D50_mum'], D50_S2_D50_samples_g_l,  marker = 'D',             
        ls = '', markersize = 7, color = 'maroon', markeredgecolor = 'black', markeredgewidth = 0.1, zorder = 40, label = 'XS')

ax[2].plot(5*x_range_d50, x_range_d50,    
         ls = (0, (1, 10)), lw = 1, color = 'black')
ax[2].plot(2*x_range_d50, x_range_d50,  
        ls = ':', lw = 1, color = 'black')
ax[2].plot(np.arange(63,1000,10), np.arange(63,1000,10),  
        ls = '-', lw = 1, color = 'black')
ax[2].plot(x_range_d50, 2*x_range_d50,    
        ls = ':', lw = 1, color = 'black')
ax[2].plot(x_range_d50, 5*x_range_d50,    
        ls = (0, (1, 10)), lw = 1, color = 'black')
ax[2].text(0.02, 0.95, '(c)', transform = ax[2].transAxes, fontsize = 12)

ax[2].set_xlabel('$\mathregular{\overline{D_{50, meas}}\; (\mu m)}$', fontsize=14, weight = 'bold')
ax[2].set_ylabel('$\mathregular{\overline{D_{50, HADCP}}\; (\mu m)}$', fontsize=14, weight = 'bold')
ax[2].tick_params(axis='both', which='major', labelsize = 12)
ax[2].set_xlim(63,800)
ax[2].set_ylim(63,800)
ax[2].set_xscale('log')
ax[2].set_yscale('log')
x_formatter = FixedFormatter(["100", "200", "500", "1000"])
x_locator = FixedLocator([100, 200, 500, 1000])
ax[2].xaxis.set_major_formatter(x_formatter)
ax[2].xaxis.set_major_locator(x_locator)
ax[2].yaxis.set_major_formatter(x_formatter)
ax[2].yaxis.set_major_locator(x_locator)

# handles = [p1, p2, p3, p4, p5, p6]
# #_, labels = ax.get_legend_handles_labels()
# fig.legend(handles = handles, #labels=labels, 
#           handler_map = {tuple: mpl.legend_handler.HandlerTuple(None)},
#           fontsize = 12, loc = 'lower center', ncol = 6, bbox_to_anchor = (0.5, -0.13))

# fig.tight_layout()
# figname = 'Fig6_legend'
# fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight') 
# fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 300, bbox_inches='tight')

fig.tight_layout()
figname = 'Fig6_300'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 300, bbox_inches='tight')


#%% Plot Fig 5 samples
fig, ax = plt.subplots(1, 1, figsize = (7,6), dpi=300)

# Cal 
p1 = ax.errorbar(TAAPS_freq1['Sand_concentration_g_l'], C_sand_S2_D50_samples_g_l,  marker = 'D', 
            xerr = TAAPS_freq1['U_C']*TAAPS_freq1['Sand_concentration_g_l']/100, elinewidth = 1, capsize = 1.5, zorder = 40,
        ls = '', markersize = 7, color = 'darkorange', markeredgecolor = 'black', markeredgewidth = 0.2, label = 'Sampler')

ax.plot(5*x_range, x_range,  zorder = 41,
         ls = (0, (1, 10)), lw = 1, color = 'black') 
ax.plot(2*x_range, x_range,  zorder = 40,
        ls = ':', lw = 1, color = 'black')
ax.plot(x_range, x_range,  zorder = 40,
        ls = '-', lw = 1, color = 'black')
ax.plot(x_range, 2*x_range,  zorder = 40,
        ls = ':', lw = 1, color = 'black') 
ax.plot(x_range, 5*x_range,  zorder = 41,
         ls = (0, (1, 10)), lw = 1, color = 'black') 

ax.set_xlabel('$\mathregular{\overline{C_{sand, meas}}}$ (g/l)', fontsize=18, weight = 'bold')
ax.set_ylabel('$\mathregular{\overline{C_{sand, TW16}}}$ (g/l)', fontsize=18, weight = 'bold')
ax.tick_params(axis='both', which='major', labelsize = 16)
ax.set_xlim(0.01,2)
ax.set_ylim(0.01,2)
ax.set_xscale('log')
ax.set_yscale('log')

# handles = [p1, p2, p3, p4, p5, p6]
# #_, labels = ax.get_legend_handles_labels()
# fig.legend(handles = handles, #labels=labels, 
#           handler_map = {tuple: mpl.legend_handler.HandlerTuple(None)},
#           fontsize = 12, loc = 'lower center', ncol = 6, bbox_to_anchor = (0.5, -0.13))

fig.tight_layout()
figname = 'Fig6_samples_TW16'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 300, bbox_inches='tight')


#%% Plot Fig 5 samples pump
fig, ax = plt.subplots(1, 1, figsize = (7,6), dpi=300)

# Cal 
p1 = ax.errorbar(TAAPS_freq1['Sand_concentration_g_l'], C_sand_S2_D50_samples_g_l,  marker = 'D', 
            xerr = TAAPS_freq1['U_C']*TAAPS_freq1['Sand_concentration_g_l']/100, elinewidth = 1, capsize = 1.5, zorder = 40,
        ls = '', markersize = 9, color = 'darkorange', markeredgecolor = 'black', markeredgewidth = 0.2, label = 'Sampler')

p2, = ax.plot(pump_data['Sand_concentration_g_l'], Csand_S2_D50_pump_valid, marker = 's',       zorder = 31,       
        ls = '', markersize = 9, color = 'mediumblue', markeredgecolor = 'black', markeredgewidth = 0.1,
        label = 'Pump')

ax.plot(5*x_range, x_range,  zorder = 41,
         ls = (0, (1, 10)), lw = 1, color = 'black') 
ax.plot(2*x_range, x_range,  zorder = 40,
        ls = ':', lw = 1, color = 'black')
ax.plot(x_range, x_range,  zorder = 40,
        ls = '-', lw = 1, color = 'black')
ax.plot(x_range, 2*x_range,  zorder = 40,
        ls = ':', lw = 1, color = 'black') 
ax.plot(x_range, 5*x_range,  zorder = 41,
         ls = (0, (1, 10)), lw = 1, color = 'black') 

ax.set_xlabel('$\mathregular{\overline{C_{sand, meas}}}$ (g/l)', fontsize=18, weight = 'bold')
ax.set_ylabel('$\mathregular{\overline{C_{sand, TW16}}}$ (g/l)', fontsize=18, weight = 'bold')
ax.tick_params(axis='both', which='major', labelsize = 16)
ax.set_xlim(0.01,2)
ax.set_ylim(0.01,2)
ax.set_xscale('log')
ax.set_yscale('log')
# ax.legend('lowe')

fig.tight_layout()
figname = 'Fig6_samples_pump_TW16'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 300, bbox_inches='tight')


#%% Plot Fig 5 samples pump grain size
fig, ax = plt.subplots(1, 1, figsize = (7,6), dpi=300)

x_range_d50 = np.arange(63,1000,10)
ax.plot(TAAPS_freq2['D50_mum'], D50_S2_D50_samples_g_l,  marker = 'D',             
        ls = '', markersize = 9, color = 'maroon', markeredgecolor = 'black', markeredgewidth = 0.1, zorder = 40, label = 'XS')

ax.plot(5*x_range_d50, x_range_d50,    
         ls = (0, (1, 10)), lw = 1, color = 'black')
ax.plot(2*x_range_d50, x_range_d50,  
        ls = ':', lw = 1, color = 'black')
ax.plot(np.arange(63,1000,10), np.arange(63,1000,10),  
        ls = '-', lw = 1, color = 'black')
ax.plot(x_range_d50, 2*x_range_d50,    
        ls = ':', lw = 1, color = 'black')
ax.plot(x_range_d50, 5*x_range_d50,    
        ls = (0, (1, 10)), lw = 1, color = 'black')

ax.set_xlabel('$\mathregular{\overline{D_{50, meas}}\; (\mu m)}$', fontsize=18, weight = 'bold')
ax.set_ylabel('$\mathregular{\overline{D_{50, TW16}}\; (\mu m)}$', fontsize=18, weight = 'bold')
ax.tick_params(axis='both', which='major', labelsize = 16)
ax.set_xlim(63,800)
ax.set_ylim(63,800)
ax.set_xscale('log')
ax.set_yscale('log')
x_formatter = FixedFormatter(["100", "200", "500", "1000"])
x_locator = FixedLocator([100, 200, 500, 1000])
ax.xaxis.set_major_formatter(x_formatter)
ax.xaxis.set_major_locator(x_locator)
ax.yaxis.set_major_formatter(x_formatter)
ax.yaxis.set_major_locator(x_locator)

fig.tight_layout()
figname = 'Fig6_samples_pump_grain_size_TW16'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 300, bbox_inches='tight')


#%% Plot Fig 5 validation
fig, ax = plt.subplots(1, 1, figsize = (7,6), dpi=300)

# Cal 
p2, = ax.plot(pump_data['Sand_concentration_g_l'], Csand_S2_D50_pump_valid, marker = 's',       zorder = 31,       
        ls = '', markersize = 7, color = 'mediumblue', markeredgecolor = 'black', markeredgewidth = 0.1,
        label = 'Pump')
ax.plot(ISCO_data['ISCO_sand_concentration_corr_g_l'], Csand_S2_D50_ISCO_valid, marker = 'o', zorder = 21,             
        ls = '', markersize = 4, color = 'yellowgreen', markeredgecolor = 'black', markeredgewidth = 0.1,
        label = '$\mathregular{C_{sand, ISCO, corr}}$') 

ax.plot(5*x_range, x_range,  zorder = 41,
         ls = (0, (1, 10)), lw = 1, color = 'black') 
ax.plot(2*x_range, x_range,  zorder = 40,
        ls = ':', lw = 1, color = 'black')
ax.plot(x_range, x_range,  zorder = 40,
        ls = '-', lw = 1, color = 'black')
ax.plot(x_range, 2*x_range,  zorder = 40,
        ls = ':', lw = 1, color = 'black') 
ax.plot(x_range, 5*x_range,  zorder = 41,
         ls = (0, (1, 10)), lw = 1, color = 'black') 

ax.set_xlabel('$\mathregular{\overline{C_{sand, meas}}}$ (g/l)', fontsize=18, weight = 'bold')
ax.set_ylabel('$\mathregular{\overline{C_{sand, TW16}}}$ (g/l)', fontsize=18, weight = 'bold')
ax.tick_params(axis='both', which='major', labelsize = 16)
ax.set_xlim(0.01,2)
ax.set_ylim(0.01,2)
ax.set_xscale('log')
ax.set_yscale('log')

# handles = [p1, p2, p3, p4, p5, p6]
# #_, labels = ax.get_legend_handles_labels()
# fig.legend(handles = handles, #labels=labels, 
#           handler_map = {tuple: mpl.legend_handler.HandlerTuple(None)},
#           fontsize = 12, loc = 'lower center', ncol = 6, bbox_to_anchor = (0.5, -0.13))

fig.tight_layout()
figname = 'Fig6_val_TW16'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 300, bbox_inches='tight')

#%%##########################################################################################

# EVENTS & METHODS COMPARISON

#############################################################################################

ISCO_GSD_data['Time_datetime'] = ISCO_GSD_mid_datetime
ISCO_GSD_data_fines = ISCO_GSD_data[ISCO_GSD_data['Fraction'] == 'fines'] 
ISCO_GSD_data_fines.reset_index(drop = True, inplace = True)
ISCO_GSD_data_total = ISCO_GSD_data[ISCO_GSD_data['Fraction'] == 'sediments'] 
ISCO_GSD_data_total.reset_index(drop = True, inplace = True)
ISCO_GSD_data_sand = ISCO_GSD_data[ISCO_GSD_data['Fraction'] == 'sand'] 
ISCO_GSD_data_sand.reset_index(drop = True, inplace = True)
ISCO_GSD_data_sand_mean = ISCO_GSD_data_sand[3::4]   
ISCO_GSD_data_total_mean = ISCO_GSD_data_total[3::4] 
ISCO_GSD_data_total_mean.reset_index(drop = True, inplace = True)

# Correct acoustic time
time_diff_min = timedelta(minutes = 49)
# time_diff_sec = timedelta(seconds = 14)
Time_datetime_freq2_corr = Time_datetime_freq2 - time_diff_min
Time_datetime_freq1_corr = Time_datetime_freq1 - time_diff_min
        
# ISCO GSD
date_ISCO_GSD_total_mean_list = [str(ISCO_GSD_data_total_mean['Date'][i]) for i in range(len(ISCO_GSD_data_total_mean))]
date_ISCO_GSD_total_mean_datetime = [datetime.strptime(date_ISCO_GSD_total_mean_list[i],'%d.%m.%Y').date()
           for i in range(len(ISCO_GSD_data_total_mean))]
mid_time_ISCO_GSD_total_mean_list = [str(ISCO_GSD_data_total_mean['Hour'][i]) for i in range(len(ISCO_GSD_data_total_mean))]
mid_time_ISCO_GSD_total_mean_datetime = [datetime.strptime(mid_time_ISCO_GSD_total_mean_list[i],'%H:%M').time()
           for i in range(len(ISCO_GSD_data_total_mean))]
ISCO_GSD_total_mean_mid_datetime = [datetime.combine(date_ISCO_GSD_total_mean_datetime[i], mid_time_ISCO_GSD_total_mean_datetime[i])
                for i in range(len(date_ISCO_GSD_total_mean_datetime))]
Time_ISCO_GSD_total_mean_mid_datetime = pd.to_datetime(ISCO_GSD_total_mean_mid_datetime,format='%d.%m.%Y %H:%M')

    
#%% Prepare dates using spreadsheet

start_int_freq1 = []
end_int_freq1 = []
start_int_freq2 = []
end_int_freq2 = []
start_int_spm = []
end_int_spm = []
start_int_Q = []
end_int_Q = []
start_int_TW16 = []
end_int_TW16 = []
for i in range(len(events_dates)):
    start_i = int(pd.to_datetime(events_dates['Start_date'].iloc[i],format='%d.%m.%Y %H:%M').timestamp())
    end_i = int(pd.to_datetime(events_dates['End_date'][i],format='%d.%m.%Y %H:%M').timestamp())
    start_int_freq1_i = np.argmin([abs(Time_datetime_freq1_int[i] - start_i) for i in range(len(Time_datetime_freq1))])
    end_int_freq1_i = np.argmin([abs(Time_datetime_freq1_int[i] - end_i) for i in range(len(Time_datetime_freq1))])
    start_int_freq2_i = np.argmin([abs(Time_datetime_freq2_int[i] - start_i) for i in range(len(Time_datetime_freq2))])
    end_int_freq2_i = np.argmin([abs(Time_datetime_freq2_int[i] - end_i) for i in range(len(Time_datetime_freq2))])
    start_int_spm_i = np.argmin([abs(Time_spm_datetime_int[i] - start_i) for i in range(len(Time_spm_datetime_int))])
    end_int_spm_i = np.argmin([abs(Time_spm_datetime_int[i] - end_i) for i in range(len(Time_spm_datetime_int))])
    start_int_Q_i = np.argmin([abs(Time_Q_datetime_int[i] - start_i) for i in range(len(Time_Q_datetime_int))])
    end_int_Q_i = np.argmin([abs(Time_Q_datetime_int[i] - end_i) for i in range(len(Time_Q_datetime_int))])        
    # start_int_TW16_i = np.argmin([abs(Time_datetime_TW16_int[i] - start_i) for i in range(len(Time_datetime_TW16))])
    # end_int_TW16_i = np.argmin([abs(Time_datetime_TW16_int[i] - end_i) for i in range(len(Time_datetime_TW16))])
    start_int_freq1.append(start_int_freq1_i)
    end_int_freq1.append(end_int_freq1_i)
    start_int_freq2.append(start_int_freq2_i)
    end_int_freq2.append(end_int_freq2_i)
    start_int_spm.append(start_int_spm_i)
    end_int_spm.append(end_int_spm_i)
    start_int_Q.append(start_int_Q_i)
    end_int_Q.append(end_int_Q_i)
    # start_int_TW16.append(start_int_TW16_i)
    # end_int_TW16.append(end_int_TW16_i)
events_dates['Start_int_freq1'] = start_int_freq1  
events_dates['End_int_freq1'] = end_int_freq1 
events_dates['Start_int_freq2'] = start_int_freq2  
events_dates['End_int_freq2'] = end_int_freq2   
events_dates['Start_int_spm'] = start_int_spm  
events_dates['End_int_spm'] = end_int_spm       
events_dates['Start_int_Q'] = start_int_Q  
events_dates['End_int_Q'] = end_int_Q 
# events_dates['Start_int_TW16'] = start_int_TW16  
# events_dates['End_int_TW16'] = end_int_TW16    
  
colors_ev = ['royalblue', 'cyan', 
             'red', 'sienna', 'orange',
              'darkmagenta', 'violet',                  
              'yellowgreen', 'forestgreen']
    

#%%###################################################################################

# Compare methods and determine cumulated fluxes

#%%###################################################################################


#%% Calculate Flux per time step (C * Q)
# ISCO point Csand corr in flux
# Q_time_ISCO = np.interp(Time_ISCO_mid_datetime, Time_Q_datetime, Q)
# ISCO_data['Q_sampling_m3_s'] = Q_time_ISCO
# ISCO_data['Phi_sand_kg_s'] = ISCO_data['ISCO_sand_concentration_corr_g_l']*ISCO_data['Q_sampling_m3_s']
# ISCO_data['Phi_fines_kg_s'] = ISCO_data['Fine_concentration_g_l']*ISCO_data['Q_sampling_m3_s']

# Determine TW16 results
Phi_sand_S2_D50_kg_s = C_sand_S2_D50_g_l *Q_time_freq2

# Determine time intervall between HADCP meas
Time_interval_freq2 = [(Time_datetime_freq2[i+1] -Time_datetime_freq2[i]).total_seconds() for i in range(len(Time_datetime_freq2)-1)]

total_Phi_sand_S2_D50_kg = np.nansum(Phi_sand_S2_D50_kg_s[1:]*Time_interval_freq2)
total_Phi_sand_S2_D50_t = total_Phi_sand_S2_D50_kg/1000
Phi_sand_S2_D50_kg_s_cumsum = np.nancumsum(Phi_sand_S2_D50_kg_s[1:]*Time_interval_freq2)
Phi_sand_S2_D50_t_s_cumsum = Phi_sand_S2_D50_kg_s_cumsum/1000
Q_time_freq2_cumsum = np.nancumsum(Q_time_freq2[1:]*Time_interval_freq2)

# for SFCP 400 kHz
Time_interval_freq1 = [(Time_datetime_freq1[i+1] -Time_datetime_freq1[i]).total_seconds() for i in range(len(Time_datetime_freq1)-1)]
Phi_sand_freq1_kg_s = C_sand_S2_D50_freq1_g_l *Q_time_freq1

total_Phi_sand_freq1_kg = np.nansum(Phi_sand_freq1_kg_s[1:]*Time_interval_freq1)
total_Phi_sand_freq1_t = total_Phi_sand_freq1_kg/1000
Phi_sand_freq1_kg_s_cumsum = np.nancumsum(Phi_sand_freq1_kg_s[1:]*Time_interval_freq1)
Phi_sand_freq1_t_s_cumsum = Phi_sand_freq1_kg_s_cumsum/1000
Q_time_freq1_cumsum = np.nancumsum(Q_time_freq1[1:]*Time_interval_freq1)

aa_freq1 = pd.DataFrame([Time_datetime_freq1, Phi_sand_freq1_t_s_cumsum]).transpose()

# for SFCP 1 MHz
Phi_sand_freq2_kg_s = C_sand_S2_D50_freq2_g_l *Q_time_freq2

total_Phi_sand_freq2_kg = np.nansum(Phi_sand_freq2_kg_s[1:]*Time_interval_freq2)
total_Phi_sand_freq2_t = total_Phi_sand_freq2_kg/1000
Phi_sand_freq2_kg_s_cumsum = np.nancumsum(Phi_sand_freq2_kg_s[1:]*Time_interval_freq2)
Phi_sand_freq2_t_s_cumsum = Phi_sand_freq2_kg_s_cumsum/1000
Q_time_freq2_cumsum = np.nancumsum(Q_time_freq2[1:]*Time_interval_freq2)

# for fines
Phi_fines_kg_s = C_fines_est_time_freq2 *Q_time_freq2

total_Phi_fines_kg = np.nansum(Phi_fines_kg_s[1:]*Time_interval_freq2)
total_Phi_fines_t = total_Phi_fines_kg/1000
Phi_fines_kg_s_cumsum = np.nancumsum(Phi_fines_kg_s[1:]*Time_interval_freq2)   

        
    
#%% Determine suspended sediment flux using turbidity data
    
# turbidity (phi_spm)
Time_interval_spm = [(Time_spm_datetime[i+1] -Time_spm_datetime[i]).total_seconds() for i in range(len(Time_spm_datetime)-1)]
Phi_spm_kg_s = spm *Q_time_spm

total_Phi_spm_kg = np.nansum(np.array(Phi_spm_kg_s[1:])*Time_interval_spm)
total_Phi_spm_t = total_Phi_spm_kg/1000
Phi_spm_kg_s_cumsum = np.nancumsum(Phi_spm_kg_s[1:]*Time_interval_spm)
Q_time_spm_cumsum = np.nancumsum(np.array(Q_time_spm[1:])*Time_interval_spm)

# Turbidity fines using the index concentration method
slope_spm_fines = 0.83
Cfines_turbidity_calc = [slope_spm_fines*spm[i] for i in range(len(spm))]
Phi_fines_T_calc_kg_s = Cfines_turbidity_calc *Q_time_spm
Cfines_turbidity_calc_time_freq2 = np.interp(Time_datetime_freq2, Time_spm_datetime, Cfines_turbidity_calc)  
total_Phi_fines_T_calc_kg = np.nansum(np.array(Phi_fines_T_calc_kg_s[1:])*Time_interval_spm)
total_Phi_fines_T_calc_t = total_Phi_fines_T_calc_kg/1000

#%% Sediment rating curve       
# Use power law with critical Q

Q_range = np.linspace(0,1000, 500)  

a_power_cr = 0.0003
b_power_cr = 2.3
Q_cr = 50
Qss_range_power_cr = [a_power_cr *(Q_range[i]-Q_cr)**b_power_cr for i in range(len(Q_range))]
Q_range_Cr = [Q_range[i] for i in range(len(Q_range))]

# Determine R2
Qss_power_cr_samples = [a_power_cr *(TAAPS_freq2['Q_sampling_m3_s'][i]-Q_cr)**b_power_cr for i in range(len(TAAPS_freq2))]
R2_power_cr = r2_score(TAAPS_freq2['Sand_flux_kg_s'], Qss_power_cr_samples)

# Plot 
err_Q = TAAPS_freq2['U_Q']/100*TAAPS_freq2['Q_sampling_m3_s']
err_Phi = TAAPS_freq2['U_F']/100*TAAPS_freq2['Sand_flux_kg_s']

TAAPS_freq2['err_Q'] = err_Q
TAAPS_freq2['err_Phi'] = err_Phi

Q_time_ISCO = np.interp(Time_ISCO_mid_datetime, Time_Q_datetime, Q)
ISCO_data['Q_sampling_m3_s'] = Q_time_ISCO
ISCO_data['Phi_sand_kg_s'] = ISCO_data['ISCO_sand_concentration_corr_g_l']*ISCO_data['Q_sampling_m3_s']
ISCO_data['Phi_fines_kg_s'] = ISCO_data['Fine_concentration_g_l']*ISCO_data['Q_sampling_m3_s']
spm_time_ISCO = np.interp(Time_ISCO_mid_datetime, Time_spm_datetime, spm)
ISCO_data['spm_sampling_g_l'] = spm_time_ISCO

# apply relations 
Phi_power_cr_time_Q = [a_power_cr *(Q[i]-Q_cr)**b_power_cr for i in range(len(Q))]

# total flux
Phi_power_cr_total_kg = np.nansum(Phi_power_cr_time_Q*30*60)
Phi_power_cr_total_t = Phi_power_cr_total_kg/1000

# Calculate Phi sand cumsum 
# Determine time intervall between Q meas
Time_interval_Q = [(Time_Q_datetime[i+1] -Time_Q_datetime[i]).total_seconds() for i in range(len(Time_Q_datetime)-1)]
Q_cumsum = np.nancumsum(Q[1:]*Time_interval_Q)

total_Phi_power_cr_time_Q_kg = np.nansum(np.array(Phi_power_cr_time_Q[1:])*Time_interval_Q)
total_Phi_power_cr_time_Q_t = total_Phi_power_cr_time_Q_kg/1000
Phi_power_cr_time_Q_kg_s_cumsum = np.nancumsum(np.array(Phi_power_cr_time_Q[1:])*Time_interval_Q)
Phi_power_cr_time_Q_t_s_cumsum = Phi_power_cr_time_Q_kg_s_cumsum/1000    

# Calculate C
Csand_power_cr_time_Q = Phi_power_cr_time_Q/Q
 
#%% Plot Fig3
fig, ax = plt.subplots(1, 1, figsize = (8, 6), dpi=300)

p1, = ax.plot(Q_range_Cr, Qss_range_power_cr,
        lw = 2, color = 'darkorange', label = r'Rating curve', zorder = 40) #$\mathregular{\Phi_{sand,cr} = a_{cr}(Q-Q_{cr})^{b_{cr}}}$
# $\mathregular{\Phi_{cr}}$
p2 = ax.errorbar(TAAPS_freq2['Q_sampling_m3_s'], TAAPS_freq2['Sand_flux_kg_s'],
           ls=' ', marker= 'D', markersize = '7', color='darkorange', markeredgecolor = 'black', markeredgewidth=0.5,                 
           xerr = TAAPS_freq2['err_Q'], 
           yerr = TAAPS_freq2['err_Phi'], elinewidth = 0.7, capsize = 1.5,
           label = r'Sampler', zorder = 30)        
      
# ax.plot(950, 1527,
#         ls=' ', marker= 's', markersize = '7', color='mediumblue',  markeredgecolor = 'black', markeredgewidth=0.5, zorder = 30)
# ax.plot(820, 1347,
#         ls=' ',marker= 's', markersize = '7', color='mediumblue',  markeredgecolor = 'black', markeredgewidth=0.5, zorder = 30)
p3, = ax.plot(pump_data['Q_sampling_m3_s'], pump_data['Sand_flux_kg_s'],              
        ls=' ', marker= 's', markersize = '7', color='mediumblue', markeredgecolor = 'black', markeredgewidth=0.5,
        label = r'Pump', zorder = 30)

p4, = ax.plot(ISCO_data['Q_sampling_m3_s'][10], ISCO_data['Phi_sand_kg_s'][10], marker = 'o',              
            ls = '', markersize = 7, color = 'yellowgreen',markeredgecolor = 'black', markeredgewidth=0.5,
            label = 'ISCO', zorder = 0)

ax.plot(ISCO_data['Q_sampling_m3_s'], ISCO_data['Phi_sand_kg_s'], marker = 'o',             
            ls = '', markersize = 5, color = 'yellowgreen', markeredgecolor = 'black', markeredgewidth=0.1, 
            zorder = 0)

ax.text(0.02, 0.95, '(c)', fontsize = 16, transform = ax.transAxes)
#ax.legend(fontsize = 16,loc = 'lower right', framealpha = 1)
# ax.set_xlim(0, 1000)
# ax.set_ylim(1,5000)
ax.set_xlim(0, 700)
ax.set_ylim(1,1000)
ax.set_yscale('log')
ax.tick_params(axis='both', which='major', labelsize = 16)
ax.set_xlabel(r'Q (m³/s)', fontsize=18, weight = 'bold')
ax.set_ylabel(r'$\mathregular{\Phi_{sand}}$ (kg/s)', fontsize=18, weight = 'bold')

handles = [p2, p3, p4, p1]
fig.legend(handles = handles, #labels=labels, 
          handler_map = {tuple: mpl.legend_handler.HandlerTuple(None)},
          fontsize = 16, loc = 'lower right', bbox_to_anchor = (0.95, 0.12))


fig.tight_layout()
figname = 'Fig3_1'
fig.savefig(outpath_figures+ '\\' + figname +  '.png', dpi = 300, bbox_inches='tight')
fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight')
 

#%% Cumulated fluxes

methods = ['turbidity', 'fines_t_calc', 
           'TW16, fines', 'TW16, sand', 'TW16', 'TW16, 400 kHz', 'TW16, 1 MHz',
            'rating_cr',]

# Phi_sand_cum_t = pd.DataFrame([total_Phi_spm_t, total_Phi_fines_T_calc_t,
#                   total_Phi_fines_t, total_phi_TW16_t, total_Phi_sand_S2_D50_t, total_Phi_sand_freq1_t, total_Phi_sand_freq2_t, 
#                  Phi_power_cr_total_t]).transpose()
# Phi_sand_cum_t.columns = methods



#%% 1) Spring snow melt contribution - Load data
    # Determine dates    
start_int_freq1 = []
end_int_freq1 = []
start_int_freq2 = []
end_int_freq2 = []
start_int_spm = []
end_int_spm = []
start_int_Q = []
end_int_Q = []
for i in range(len(spring_dates)):
    start_i = int(pd.to_datetime(spring_dates['Start_date'].iloc[i],format='%d.%m.%Y %H:%M').timestamp())
    end_i = int(pd.to_datetime(spring_dates['End_date'][i],format='%d.%m.%Y %H:%M').timestamp())
    start_int_freq1_i = np.argmin([abs(Time_datetime_freq1_int[i] - start_i) for i in range(len(Time_datetime_freq1))])
    end_int_freq1_i = np.argmin([abs(Time_datetime_freq1_int[i] - end_i) for i in range(len(Time_datetime_freq1))])
    start_int_freq2_i = np.argmin([abs(Time_datetime_freq2_int[i] - start_i) for i in range(len(Time_datetime_freq2))])
    end_int_freq2_i = np.argmin([abs(Time_datetime_freq2_int[i] - end_i) for i in range(len(Time_datetime_freq2))])
    start_int_spm_i = np.argmin([abs(Time_spm_datetime_int[i] - start_i) for i in range(len(Time_spm_datetime_int))])
    end_int_spm_i = np.argmin([abs(Time_spm_datetime_int[i] - end_i) for i in range(len(Time_spm_datetime_int))])
    start_int_Q_i = np.argmin([abs(Time_Q_datetime_int[i] - start_i) for i in range(len(Time_Q_datetime_int))])
    end_int_Q_i = np.argmin([abs(Time_Q_datetime_int[i] - end_i) for i in range(len(Time_Q_datetime_int))])
    start_int_freq1.append(start_int_freq1_i)
    end_int_freq1.append(end_int_freq1_i)
    start_int_freq2.append(start_int_freq2_i)
    end_int_freq2.append(end_int_freq2_i)
    start_int_spm.append(start_int_spm_i)
    end_int_spm.append(end_int_spm_i)
    start_int_Q.append(start_int_Q_i)
    end_int_Q.append(end_int_Q_i)
spring_dates['Start_int_freq1'] = start_int_freq1  
spring_dates['End_int_freq1'] = end_int_freq1 
spring_dates['Start_int_freq2'] = start_int_freq2  
spring_dates['End_int_freq2'] = end_int_freq2   
spring_dates['Start_int_spm'] = start_int_spm  
spring_dates['End_int_spm'] = end_int_spm       
spring_dates['Start_int_Q'] = start_int_Q  
spring_dates['End_int_Q'] = end_int_Q    


#%% Plot Fig8
fig, ax = plt.subplots(2, 1, figsize = (12,10), dpi=300,gridspec_kw={'height_ratios': [1, 3]})

ax[0].plot(Time_Q_datetime, Q, 
        lw = 1, alpha = 0.7, color = 'blue', zorder = 10)
ax[0].tick_params(axis='both', which='major', labelsize = 14)
ax[0].set_ylabel('Q (m³/s)', fontsize=16, weight = 'bold')
ax[0].set_xticklabels([])
ax[0].xaxis.set_major_locator(md.MonthLocator(interval=6))
ax[0].xaxis.set_minor_locator(md.MonthLocator(interval=1))
ax[0].set_xlim(Time_datetime_freq2[0], Time_datetime_freq2[-1])
ax[0].set_ylim(0,)
ax2 = ax[1].twinx()
ax2.tick_params(axis='both', which='major', labelsize = 14)    

# spm
# ax[1].plot(Time_datetime_freq2[1:], Phi_fines_kg_s_cumsum, 
#         ls = ':', lw = 2, color = 'midnightblue', label = '$\mathregular{{\Phi_{fines}}}$') #darkgoldenrod

# Rating curve
ax[1].plot(Time_Q_datetime[1:], Phi_power_cr_time_Q_kg_s_cumsum, 
        lw = 2, color = 'lightseagreen', ls = '--', label = 'Rating curve')

# Acoustic      
# ax[1].plot(Time_datetime_TW16, phi_TW16_cumsum, 
#         ls = '-.', lw = 2, color = 'orangered', label = 'TW16')
ax[1].plot(Time_datetime_freq2[1:], Phi_sand_S2_D50_kg_s_cumsum, 
        ls = '-', lw = 2, color = 'mediumslateblue', label = 'TW16')
# ax[1].plot(Time_datetime_TW16_B, phi_TW16_B_cumsum, 
#         ls = '-.', lw = 2, color = 'green', label = 'TW16-B')
ax[1].plot(Time_datetime_freq1[1:], Phi_sand_freq1_kg_s_cumsum, 
        ls = '-', lw = 2, color = 'brown', label = 'TW16, 400 kHz')
ax[1].plot(Time_datetime_freq2[1:], Phi_sand_freq2_kg_s_cumsum, 
        ls = '-', lw = 2, color = 'palevioletred', label = 'TW16, 1 MHz')
 

ax[1].hlines(2.67*1e9, Time_datetime_freq2[spring_dates['Start_int_freq2']][0], Time_datetime_freq2[spring_dates['End_int_freq2']][0],
            colors = 'yellowgreen', lw = 15, zorder = 40)#, label = 'Snowmelt period')
ax[1].hlines(2.67*1e9, Time_datetime_freq2[spring_dates['Start_int_freq2']][1], Time_datetime_freq2[spring_dates['End_int_freq2']][1],
            colors = 'yellowgreen', lw = 15, zorder = 40)
ax[1].hlines(2.67*1e9, Time_datetime_freq2[spring_dates['Start_int_freq2']][2], Time_datetime_freq2[spring_dates['End_int_freq2']][2],
            colors = 'yellowgreen', lw = 15, zorder = 40)
ax[1].arrow(Time_datetime_freq2[events_dates['Start_int_freq2'][0]], 1.8*1e9, 0, -0.3*1e9, 
          head_width = 12, head_length = 0.1*1e9, width = 3, ec ='None', fc = 'royalblue', zorder = 40) #, label = 'Floods')
ax[1].arrow(Time_datetime_freq2[events_dates['Start_int_freq2'][1]], 2.1*1e9, 0, -0.3*1e9, 
          head_width = 12, head_length = 0.1*1e9, width = 3, ec ='None', fc = 'royalblue', zorder = 40)
ax[1].arrow(Time_datetime_freq2[events_dates['Start_int_freq2'][2]], 0.6*1e9, 0, -0.3*1e9, 
          head_width = 12, head_length = 0.1*1e9, width = 3, ec ='None', fc = 'darkviolet', zorder = 40)

ax[1].set_xlabel('Time', fontsize=16, weight = 'bold')
ax[1].set_ylabel('Cumulative sand mass (kg)', fontsize=16, weight = 'bold')
ax[1].tick_params(axis='both', which='major', labelsize = 14)

ax[1].xaxis.set_major_locator(md.MonthLocator(interval=6))
ax[1].xaxis.set_minor_locator(md.MonthLocator(interval=1))
ax[1].xaxis.set_major_formatter(md.DateFormatter('%d/%m/%Y'))
ax[1].set_xlim(Time_datetime_freq2[0], Time_datetime_freq2[-1])
ax[1].set_ylim(0,2.7*1e9)
ax2.set_ylim(0,2.7*1e9)
ax[1].legend(fontsize = 16, loc = 'lower right', ncol = 2)#, bbox_to_anchor = (0.5, -0.23))

ax[0].text(0.02, 0.92, '(a)', fontsize = 14, transform = ax[0].transAxes)
ax[1].text(0.02, 0.95, '(b)', fontsize = 14, transform = ax[1].transAxes)


ax[1].text(0.06, 0.95, 'Spring', fontsize = 14, transform = ax[1].transAxes)
ax[1].text(0.47, 0.95, 'Spring', fontsize = 14, transform = ax[1].transAxes)
ax[1].text(0.92, 0.95, 'Spring', fontsize = 14, transform = ax[1].transAxes)
ax[1].text(0.31, 0.75, 'Flood', fontsize = 14, rotation = 90, transform = ax[1].transAxes)
ax[1].text(0.75, 0.87, 'Flood', fontsize = 14, rotation = 90, transform = ax[1].transAxes)
ax[1].text(0.015, 0.1, 'Flood (Figure 9)', fontsize = 14, rotation = 90, transform = ax[1].transAxes)

fig.tight_layout()
figname = 'Fig8'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight') 
# fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 300, bbox_inches='tight')


#%% Plot Fig8 TW16 A dual and single
fig, ax = plt.subplots(2, 1, figsize = (12,10), dpi=300,gridspec_kw={'height_ratios': [1, 3]})

ax[0].plot(Time_Q_datetime, Q, 
        lw = 1, alpha = 0.7, color = 'blue', zorder = 10)
ax[0].tick_params(axis='both', which='major', labelsize = 18)
ax[0].set_ylabel('Q (m³/s)', fontsize=20, weight = 'bold')
ax[0].set_xticklabels([])
ax[0].xaxis.set_major_locator(md.MonthLocator(interval=6))
ax[0].xaxis.set_minor_locator(md.MonthLocator(interval=1))
ax[0].set_xlim(Time_datetime_freq2[0], Time_datetime_freq2[-1])
ax[0].set_ylim(0,)
ax2 = ax[1].twinx()
ax2.tick_params(axis='both', which='major', labelsize = 18)    

# spm
# ax[1].plot(Time_datetime_freq2[1:], Phi_fines_kg_s_cumsum, 
#         ls = ':', lw = 2, color = 'midnightblue', label = '$\mathregular{{\Phi_{fines}}}$') #darkgoldenrod

# Rating curve
ax[1].plot(Time_Q_datetime[1:], Phi_power_cr_time_Q_kg_s_cumsum, 
        lw = 3, color = 'lightseagreen', ls = '--', label = 'Rating curve')

# Acoustic      
# ax[1].plot(Time_datetime_TW20, phi_TW20_cumsum, 
#         ls = '-.', lw = 2, color = 'orangered', label = 'TW20')
ax[1].plot(Time_datetime_freq2[1:], Phi_sand_S2_D50_kg_s_cumsum, 
        ls = '-', lw = 3, color = 'sienna', label = 'TW16')
# ax[1].plot(Time_datetime_TW16_B, phi_TW16_B_cumsum, 
#         ls = ':', lw = 3, color = 'darkviolet', label = 'TW20-B')
ax[1].plot(Time_datetime_freq1[1:], Phi_sand_freq1_kg_s_cumsum, 
        ls = ':', lw = 3, color = 'forestgreen', label = 'TW20-A, 400 kHz')
ax[1].plot(Time_datetime_freq2[1:], Phi_sand_freq2_kg_s_cumsum, 
        ls = '-.', lw = 3, color = 'darkviolet', label = 'TW20-A, 1 MHz')
 
ax[1].hlines(2.65*1e9, Time_datetime_freq2[spring_dates['Start_int_freq2']][0], Time_datetime_freq2[spring_dates['End_int_freq2']][0],
            colors = 'yellowgreen', lw = 15, zorder = 40)#, label = 'Snowmelt period')
ax[1].hlines(2.65*1e9, Time_datetime_freq2[spring_dates['Start_int_freq2']][1], Time_datetime_freq2[spring_dates['End_int_freq2']][1],
            colors = 'yellowgreen', lw = 15, zorder = 40)
ax[1].hlines(2.65*1e9, Time_datetime_freq2[spring_dates['Start_int_freq2']][2], Time_datetime_freq2[spring_dates['End_int_freq2']][2],
            colors = 'yellowgreen', lw = 15, zorder = 40)
# ax[1].arrow(Time_datetime_freq2[events_dates['Start_int_freq2'][0]], 1.8*1e9, 0, -0.3*1e9, 
#           head_width = 12, head_length = 0.1*1e9, width = 3, ec ='None', fc = 'royalblue', zorder = 40) #, label = 'Floods')
# ax[1].arrow(Time_datetime_freq2[events_dates['Start_int_freq2'][1]], 2.1*1e9, 0, -0.3*1e9, 
#           head_width = 12, head_length = 0.1*1e9, width = 3, ec ='None', fc = 'royalblue', zorder = 40)
# ax[1].arrow(Time_datetime_freq2[events_dates['Start_int_freq2'][2]], 0.6*1e9, 0, -0.3*1e9, 
#           head_width = 12, head_length = 0.1*1e9, width = 3, ec ='None', fc = 'darkviolet', zorder = 40)

ax[1].text(0.17, 0.6, 'Dual-frequency', fontsize = 20, transform = ax[1].transAxes)
ax[1].text(0.17, 0.07, '400 kHz', fontsize = 20, transform = ax[1].transAxes)
ax[1].text(0.17, 0.253, '1 MHz', fontsize = 20, transform = ax[1].transAxes)
ax[1].text(0.15, 0.39, 'Rating curve', fontsize = 20, transform = ax[1].transAxes)

ax[1].set_xlabel('Time', fontsize=20, weight = 'bold')
ax[1].set_ylabel('Cumulative sand mass (Mt)', fontsize=20, weight = 'bold')
ax[1].tick_params(axis='both', which='major', labelsize = 18)

ax[1].xaxis.set_major_locator(md.MonthLocator(interval=6))
ax[1].xaxis.set_minor_locator(md.MonthLocator(interval=1))
ax[1].xaxis.set_major_formatter(md.DateFormatter('%d/%m/%Y'))
ax[1].set_xlim(Time_datetime_freq2[0], Time_datetime_freq2[-1])
ax[1].set_ylim(0,2.7*1e9)
ax2.set_ylim(0,2.7*1e9)
# ax[1].legend(fontsize = 20, loc = 'lower right', ncol = 3)#, bbox_to_anchor = (0.5, -0.23))

# ax[0].text(0.02, 0.9, '(a)', fontsize = 18, transform = ax[0].transAxes)
# ax[1].text(0.02, 0.95, '(b)', fontsize = 18, transform = ax[1].transAxes)

ax[1].text(0.06, 0.95, 'Spring', fontsize = 18, transform = ax[1].transAxes)
ax[1].text(0.47, 0.95, 'Spring', fontsize = 18, transform = ax[1].transAxes)
ax[1].text(0.9, 0.95, 'Spring', fontsize = 18, transform = ax[1].transAxes)
# ax[1].text(0.31, 0.71, 'Flood', fontsize = 18, rotation = 90, transform = ax[1].transAxes)
# ax[1].text(0.75, 0.85, 'Flood', fontsize = 18, rotation = 90, transform = ax[1].transAxes)
# ax[1].text(0.015, 0.1, 'Flood (Figure 9)', fontsize = 18, rotation = 90, transform = ax[1].transAxes)

fig.tight_layout()
figname = 'Fig8_TW16_dual_single'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight') 
# fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 300, bbox_inches='tight')


#%% Plot C est S2_D50 sands
fig, ax = plt.subplots(2, 1, figsize = (10,6), dpi=300)

# Csand
p6, = ax[0].plot(Time_datetime_freq2[0], C_sand_S2_D50_g_l[0], '.',markersize = 10,          
        color = 'sienna', label = 'TW16')
ax[0].plot(Time_datetime_freq2, C_sand_S2_D50_g_l, '.',markersize = 1,
        color = 'sienna', label = '400 kHz')
p1, = ax[0].plot(TAAPS_freq1['Date'], TAAPS_freq1['Sand_concentration_g_l'], 'D', 
        color = 'darkorange', markersize = 7, zorder = 40,
        markeredgecolor = 'black', markeredgewidth = 0.2, label = 'Sampler')
p2, = ax[0].plot(Time_pump_mid_datetime, pump_data['Sand_concentration_g_l'], 's', 
        color = 'mediumblue', markersize = 7,
        markeredgecolor = 'black', markeredgewidth = 0.2, label = 'Pump')

ax[0].set_xlim(Time_datetime_freq1[0], Time_datetime_freq1[-1])
ax[0].tick_params(axis='y', which='major', labelsize = 14)
ax[0].set_ylim(0.001,5)
ax[0].set_yscale('log')
ax[0].xaxis.set_ticklabels([])
# ax[0].legend(fontsize = 18, loc = 'upper right')

# D50
p5, = ax[1].plot(Time_datetime_freq2[0], D50_est_S2_D50[0], '.', markersize = 10,
        color = 'tan')
ax[1].plot(Time_datetime_freq2, D50_est_S2_D50, '.', markersize = 1,
        color = 'tan')
ax[1].plot(TAAPS_freq1['Date'], TAAPS_freq1['D50_mum'], 'D', 
        color = 'darkorange', markersize = 7,
        markeredgecolor = 'black', markeredgewidth = 0.2, label = 'XS')

ax[1].set_xlabel('Time', fontsize=16, weight = 'bold')
ax[1].xaxis.set_major_locator(md.MonthLocator(interval=6))
ax[1].xaxis.set_major_formatter(md.DateFormatter('%d/%m/%Y'))
ax[1].xaxis.set_minor_locator(md.MonthLocator(interval=1))
ax[1].set_xlim(Time_datetime_freq1[0], Time_datetime_freq1[-1])
ax[1].tick_params(axis='both', which='major', labelsize = 14)
ax[1].set_ylim(63,400)

ax[0].set_ylabel('$\mathregular{\overline{C_{sand}}}$ (g/l)', fontsize=20, weight = 'bold')
ax[1].set_ylabel('$\mathregular{\overline{D_{50}} \; (\mu m)}$', fontsize=20, weight = 'bold')
ax[1].set_xlabel('Time', fontsize=20, weight = 'bold')
ax[0].hlines(0.2, Time_datetime_freq1[0], Time_datetime_freq1[-1], lw = 1, ls = '--', color = 'black',
             label = r'$\mathregular{\overline{C_{sand,ref}}}$')
ax[1].hlines(200, Time_datetime_freq1[0], Time_datetime_freq1[-1], lw = 1, ls = '--', color = 'black',
             label = r'$\mathregular{\overline{D_{50,sand,ref}}}$')

p3 = ax[0].hlines(0.2, Time_datetime_freq1[0], Time_datetime_freq1[-1], lw = 1, ls = '--', color = 'black',
             label = 'Reference sand') #label = r'$\mathregular{\overline{C_{sand,ref}}}$')
p4 = ax[1].hlines(200, Time_datetime_freq1[0], Time_datetime_freq1[-1], lw = 1, ls = '--', color = 'black',
             label = 'Reference sand') #label = r'$\mathregular{\overline{D_{50,sand,ref}}}$')

ax[0].text(0.02, 0.9, '(a)', fontsize = 14, transform = ax[0].transAxes)
ax[1].text(0.02, 0.9, '(b)', fontsize = 14, transform = ax[1].transAxes)
l = fig.legend([(p6, p5), p1, p2, (p3, p4)], ['TW16', 'Sampler', 'Pump PP36', 'Reference sand'],
               handler_map={tuple: HandlerTuple(ndivide=None)},
               fontsize = 16, loc = 'lower center', ncol = 4, bbox_to_anchor = (0.5, -0.07))

fig.tight_layout()
figname = 'Time_Csand_D50'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 400, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight') 
# fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 300, bbox_inches='tight')


#%% Plot Time -  C sand
fig, ax = plt.subplots(1, 1, figsize = (10,4), dpi=300)

ax.plot(Time_datetime_freq2, C_sand_S2_D50_g_l, '.', markersize = 2,
        color = 'peru', label = 'fines')
ax.plot(TAAPS_freq2['Date'], TAAPS_freq2['Sand_concentration_g_l'], 'D', 
        color = 'black', label = 'sand')

ax.set_xlabel('Time', fontsize=16, weight = 'bold')
ax.set_ylabel('$\mathregular{\overline{C_{sand}}}$ (g/l)', fontsize=16, weight = 'bold')
ax.xaxis.set_major_locator(md.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(md.DateFormatter('%d.%m.%Y'))
ax.set_xlim(Time_datetime_freq2[0], Time_datetime_freq2[-1])
ax.tick_params(axis='both', which='major', labelsize = 14)
ax.set_ylim(0.001,10)
ax.set_yscale('log')

fig.tight_layout()
figname = 'Time_C_sand_S2_D50'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 400, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight') 
fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 300, bbox_inches='tight')


#%% Plot Time -  Phi sand
fig, ax = plt.subplots(1, 1, figsize = (10,4), dpi=300)

ax.plot(Time_datetime_freq2, Phi_sand_S2_D50_kg_s, '.', markersize = 2,
        color = 'peru', label = 'fines')
ax.plot(TAAPS_freq2['Date'], TAAPS_freq2['Sand_flux_kg_s'], 'D', 
        color = 'black', label = 'sand')

ax.set_xlabel('Time', fontsize=16, weight = 'bold')
ax.set_ylabel('$\mathregular{\overline{\Phi_{sand}}}$ (kg/s)', fontsize=16, weight = 'bold')
ax.xaxis.set_major_locator(md.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(md.DateFormatter('%d.%m.%Y'))
ax.set_xlim(Time_datetime_freq2[0], Time_datetime_freq2[-1])
ax.tick_params(axis='both', which='major', labelsize = 14)
ax.set_ylim(0.1,2000)
ax.set_yscale('log')

fig.tight_layout()
figname = 'Time_Phisand_S2_D50'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 400, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight') 
# fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 300, bbox_inches='tight')


#%% Plot Time - D50 sand

fig, ax = plt.subplots(1, 1, figsize = (8,4), dpi=300)

ax.plot(Time_datetime_freq2, D50_est_S2_D50, '.', markersize = 2,  
        color = 'peru', label = 'fines')
ax.plot(TAAPS_freq2['Date'], TAAPS_freq2['D50_mum'], 'D', 
        color = 'black', label = 'sand')

ax.set_xlabel('Time', fontsize=16, weight = 'bold')
ax.set_ylabel('$\mathregular{\overline{D_{50}} \; (\mu m)}$', fontsize=16, weight = 'bold')
ax.xaxis.set_major_locator(md.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(md.DateFormatter('%d.%m.%Y'))
ax.set_xlim(Time_datetime_freq2[0], Time_datetime_freq2[-1])
ax.tick_params(axis='both', which='major', labelsize = 14)
ax.set_ylim(63,700)

fig.tight_layout()
figname = 'Time_D50_sand_dual'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 400, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight') 
fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 300, bbox_inches='tight')


#%% Plot Events timeseries i = 0 Flood Dec 2021
i = 0
fig, ax = plt.subplots(4, 1, figsize = (12,12), dpi=300)

# Fines and Q 

ax1 = ax[0].twinx()
p10, = ax1.plot(Time_datetime_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]],
         Q_time_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]],
        ls = '-', lw = 2, color = 'blue', label = 'Q', zorder = 0)

p4, = ax[0].plot(Time_datetime_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]], 
               C_fines_est_time_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]],
               lw = 2, ls = '-', 
          color = 'darkorange', label = r'$\mathregular{\overline{C_{\rm{fines, TW16}}}}$', zorder = 10)
p7, = ax[0].plot(Time_datetime_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]], 
               spm_time_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]],
        ls = '-', lw = 2, color = 'tan', zorder = 0, label = r'$\mathregular{C_{\rm{turbidity}}}$')    
p18, = ax[0].plot(Time_ISCO_GSD_total_mean_mid_datetime, ISCO_GSD_data_total_mean['Concentration_g_l'], marker = '*',              
        ls = '', markersize = 8, color = 'orangered', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 40,
        label = r'$\mathregular{C_{\rm{tot, ISCO}}}$')
p19, = ax[0].plot(Time_datetime_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]], 
               Cfines_turbidity_calc_time_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]],
               lw = 2, ls = '-', 
          color = 'sienna', label = r'$\mathregular{\overline{C_{\rm{fines, T, calc}}}}$', zorder = 10)

ax[0].set_ylabel('$\mathregular{\overline{C_{fines}}}$ (g/l)', fontsize=16, weight = 'bold')
ax[0].set_ylim(0,) 
ax[0].xaxis.set_major_locator(md.DayLocator(interval=1))
ax[0].xaxis.set_major_formatter(md.DateFormatter('%d.%m.%Y'))
ax[0].xaxis.set_minor_locator(md.HourLocator(interval=1))
ax[0].set_xlim(Time_datetime_freq2[events_dates['Start_int_freq2'][i]], Time_datetime_freq2[events_dates['End_int_freq2'][i]])
ax[0].tick_params(axis='both', which='major', labelsize = 14)
ax[0].xaxis.set_ticklabels([])
    
ax1.spines['right'].set_visible(True)
ax1.spines['left'].set_visible(False)
ax1.yaxis.set_label_position('right')
ax1.yaxis.set_ticks_position('right')
ax1.tick_params(axis='both', which='major', labelsize = 14)
ax1.set_ylim(0,)
ax1.set_ylabel('Q (m³/s)', fontsize=16, weight = 'bold', color = 'blue')  
ax1.set_ylim(0,)  

# CSands
# p20, = ax[1].plot(Time_datetime_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]], 
#               Csand_TW16_time_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]], 
#               ls = '-', lw = 2, 
#         color = 'indianred', label = r'$\mathregular{\overline{C_{\rm{sand, TW16}}}}$', zorder = 10)
p1, = ax[1].plot(Time_datetime_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]], 
              C_sand_S2_D50_g_l[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]], 
              ls = '-', lw = 2, 
        color = 'peru', label = r'$\mathregular{\overline{C_{\rm{sand, TW16}}}}$', zorder = 10)
p12, = ax[1].plot(Time_datetime_freq1[events_dates['Start_int_freq1'][i]:events_dates['End_int_freq1'][i]], 
              C_sand_S2_D50_freq1_g_l[events_dates['Start_int_freq1'][i]:events_dates['End_int_freq1'][i]], 
              ls = '-', lw = 2, 
        color = 'navy', label = r'$\mathregular{\overline{C_{\rm{sand, TW16, 400 kHz}}}}$', zorder = 10)
p13, = ax[1].plot(Time_datetime_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]], 
              C_sand_S2_D50_freq2_g_l[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]], 
              ls = '-', lw = 2, 
        color = 'darkred', label = r'$\mathregular{\overline{C_{\rm{sand,TW16, 1 MHz}}}}$', zorder = 10)      

ax[1].set_ylabel('$\mathregular{\overline{C_{sand}}}$ (g/l)', fontsize=16, weight = 'bold')
ax[1].set_ylim(0,2) 
ax[1].xaxis.set_major_locator(md.DayLocator(interval=1))
ax[1].xaxis.set_ticklabels([])
ax[1].xaxis.set_minor_locator(md.HourLocator(interval=1))
ax[1].set_xlim(Time_datetime_freq2[events_dates['Start_int_freq2'][i]], Time_datetime_freq2[events_dates['End_int_freq2'][i]])
ax[1].tick_params(axis='both', which='major', labelsize = 14)

# Csand 
p16, = ax[2].plot(Time_Q_datetime, Csand_power_cr_time_Q, 
              ls = ':', lw = 2, color = 'lightseagreen', label = r'$\mathregular{\overline{C_{\rm{sand, cr}}}}$', zorder = 10)
ax[2].tick_params(axis='both', which='major', labelsize = 14)
ax[2].set_ylim(0,2)
ax[2].set_ylabel('$\mathregular{\overline{C_{sand}}}$ (g/l)', fontsize=16, weight = 'bold') 
ax[2].xaxis.set_major_locator(md.DayLocator(interval=1))
ax[2].xaxis.set_ticklabels([])
ax[2].xaxis.set_minor_locator(md.HourLocator(interval=1))
ax[2].set_xlim(Time_datetime_freq2[events_dates['Start_int_freq2'][i]], Time_datetime_freq2[events_dates['End_int_freq2'][i]])
ax[2].tick_params(axis='both', which='major', labelsize = 14)         

# D50, sand
p8, = ax[3].plot(Time_datetime_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]],
               D50_est_S2_D50[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]],
        color = 'darkviolet', lw = 2, ls = '-', label = r'$\mathregular{\overline{D_{50,sand, TW16}}}$')

ax[3].tick_params(axis='both', which='major', labelsize = 14)
ax[3].set_ylim(0,800)
ax[3].set_ylabel('$\mathregular{\overline{D_{50, sand}}}$ ($\mathregular{\mu}$m)', fontsize=16, weight = 'bold') 
ax[3].set_xlabel('Time', fontsize=16, weight = 'bold') 
ax[3].xaxis.set_major_locator(md.DayLocator(interval=1))
ax[3].xaxis.set_major_formatter(md.DateFormatter('%d.%m.%Y'))
ax[3].xaxis.set_minor_locator(md.HourLocator(interval=1))
ax[3].set_xlim(Time_datetime_freq2[events_dates['Start_int_freq2'][i]], Time_datetime_freq2[events_dates['End_int_freq2'][i]])
ax[3].tick_params(axis='both', which='major', labelsize = 14)   

ax[0].text(0.02, 0.9, '(a)', fontsize = 14, transform = ax[0].transAxes)
ax[1].text(0.02, 0.9, '(b)', fontsize = 14, transform = ax[1].transAxes)
ax[2].text(0.02, 0.9, '(c)', fontsize = 14, transform = ax[2].transAxes)
ax[3].text(0.02, 0.9, '(d)', fontsize = 14, transform = ax[3].transAxes)
ax[1].text(0.05, 0.9, 'Acoustic measurements', fontsize = 14, transform = ax[1].transAxes)
ax[2].text(0.05, 0.9, 'Rating curve ', fontsize = 14, transform = ax[2].transAxes)
   
# handles = [p7, p18, p19, p4, p10, p20, p1, p12, p13, p16, p8]
handles = [p7, p18, p19, p4, p10, p1, p12, p13, p16, p8]
#_, labels = ax.get_legend_handles_labels()
fig.legend(handles = handles, #labels=labels, 
          handler_map = {tuple: mpl.legend_handler.HandlerTuple(None)},
          fontsize = 16, loc = 'lower center', ncol = 5, bbox_to_anchor = (0.5, -0.13))
#ax.legend(fontsize = 14, loc = 'lower center', ncol = 4, bbox_to_anchor = (0.5, -0.23))

fig.tight_layout()
figname = 'Fig_A4'
fig.savefig(outpath_figures + '\\' + figname + str(events_dates['Name_date'][i]) + '.png', dpi = 100, bbox_inches='tight')
# fig.savefig(outpath_figures + '\\' + figname + str(events_dates['Name_date'][i]) + '.eps', dpi = 300, bbox_inches='tight')
# fig.savefig(outpath_figures + '\\' + figname + str(events_dates['Name_date'][i]) + '.pdf', dpi = 300, bbox_inches='tight')
    

#%% Plot Events timeseries i = 1 Flood Dec 2022

i = 1
fig, ax = plt.subplots(4, 1, figsize = (12,12), dpi=300)

# Fines and Q     
ax1 = ax[0].twinx()
p10, = ax1.plot(Time_datetime_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]],
         Q_time_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]],
        ls = '-', lw = 2, color = 'blue', label = 'Q', zorder = 0)

p4, = ax[0].plot(Time_datetime_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]], 
               C_fines_est_time_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]],
               ls = '-', lw = 2,
         color = 'darkorange', label = r'$\mathregular{\overline{C_{\rm{fines, TW16}}}}$', zorder = 10)
p5 = ax[0].hlines(TAAPS_freq2['Fine_concentration_g_l'][0], samples_start_datetime[0], samples_end_datetime[0], 
       lw = 2.5, color = 'yellowgreen', label = r'$\mathregular{\overline{C_{\rm{fines}}}}$', zorder = 40)
for j in range(len(TAAPS_freq2)):
    ax[0].hlines(TAAPS_freq2['Fine_concentration_g_l'][j], samples_start_datetime[j], samples_end_datetime[j], 
          lw = 2.5, color = 'yellowgreen', zorder = 40)
p6, = ax[0].plot(Time_ISCO_mid_datetime, ISCO_data['Fine_concentration_g_l'], marker = 'o',              
        ls = '', markersize = 6, color = 'forestgreen', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 40,
        label = r'$\mathregular{C_{\rm{fines, ISCO}}}$')
p7, = ax[0].plot(Time_datetime_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]], 
               spm_time_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]],
        ls = '-', lw = 2, color = 'tan', zorder = 0, label = r'$\mathregular{C_{\rm{turbidity}}}$')
p19, = ax[0].plot(Time_datetime_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]], 
               Cfines_turbidity_calc_time_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]],
               lw = 2, ls = '-', 
          color = 'sienna', label = r'$\mathregular{\overline{C_{\rm{fines, T, calc}}}}$', zorder = 10)

ax[0].set_ylabel('$\mathregular{\overline{C_{fines}}}$ (g/l)', fontsize=16, weight = 'bold')
ax[0].set_ylim(0,) 
ax[0].xaxis.set_major_locator(md.DayLocator(interval=1))
ax[0].xaxis.set_major_formatter(md.DateFormatter('%d.%m.%Y'))
ax[0].xaxis.set_minor_locator(md.HourLocator(interval=1))
ax[0].set_xlim(Time_datetime_freq2[events_dates['Start_int_freq2'][i]], Time_datetime_freq2[events_dates['End_int_freq2'][i]])
ax[0].tick_params(axis='both', which='major', labelsize = 14)
ax[0].xaxis.set_ticklabels([])
    
ax1.spines['right'].set_visible(True)
ax1.spines['left'].set_visible(False)
ax1.yaxis.set_label_position('right')
ax1.yaxis.set_ticks_position('right')
ax1.tick_params(axis='both', which='major', labelsize = 14)
ax1.set_ylim(0,)
ax1.set_ylabel('Q (m³/s)', fontsize=16, weight = 'bold', color = 'blue')  
ax1.set_ylim(0,)  

# CSand
# p20, = ax[1].plot(Time_datetime_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]], 
#               Csand_TW16_time_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]], 
#               ls = '-', lw = 2, 
#         color = 'indianred', label = r'$\mathregular{\overline{C_{\rm{sand, TW16}}}}$', zorder = 10)
p1, = ax[1].plot(Time_datetime_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]], 
              C_sand_S2_D50_g_l[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]], 
              ls = '-', lw = 2, 
        color = 'peru', label = r'$\mathregular{\overline{C_{\rm{sand, TW16}}}}$', zorder = 10)
p12, = ax[1].plot(Time_datetime_freq1[events_dates['Start_int_freq1'][i]:events_dates['End_int_freq1'][i]], 
              C_sand_S2_D50_freq1_g_l[events_dates['Start_int_freq1'][i]:events_dates['End_int_freq1'][i]], 
              ls = '-', lw = 2, 
        color = 'navy', label = r'$\mathregular{\overline{C_{\rm{sand, TW16, 400 kHz}}}}$', zorder = 10)
p13, = ax[1].plot(Time_datetime_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]], 
              C_sand_S2_D50_freq2_g_l[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]], 
              ls = '-', lw = 2, 
        color = 'darkred', label = r'$\mathregular{\overline{C_{\rm{sand, TW16, 1 MHz}}}}$', zorder = 10)
   
# p2 =  ax[1].hlines(TAAPS_freq2['Sand_concentration_g_l'][0], samples_start_datetime[0], samples_end_datetime[0], 
#        lw = 2.5, color = 'firebrick', label = r'$\mathregular{\overline{C_{\rm{sand}}}}$', zorder = 40)
# for j in range(len(TAAPS_freq2)):
#     ax[1].hlines(TAAPS_freq2['Sand_concentration_g_l'][j], samples_start_datetime[j], samples_end_datetime[j], 
#           lw = 2.5, color = 'firebrick', zorder = 40)
p3, =  ax[1].plot(Time_ISCO_mid_datetime, ISCO_data['ISCO_sand_concentration_corr_g_l'], marker = 'o',              
        ls = '', markersize = 6, color = 'red', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 40,
        label = r'$\mathregular{C_{\rm{sand, ISCO, corr}}}$')

ax[1].set_ylabel('$\mathregular{\overline{C_{sand}}}$ (g/l)', fontsize=16, weight = 'bold')
ax[1].set_ylim(0,1.5) 
# ax[1].xaxis.set_major_locator(md.DayLocator(interval=1))
# ax[1].xaxis.set_major_formatter(md.DateFormatter('%d.%m.%Y'))
ax[1].tick_params(axis='both', which='major', labelsize = 14)
ax[1].set_xlim(Time_datetime_freq2[events_dates['Start_int_freq2'][i]], Time_datetime_freq2[events_dates['End_int_freq2'][i]])
ax[1].xaxis.set_ticklabels([])

# Csand 
p16, = ax[2].plot(Time_Q_datetime, Csand_power_cr_time_Q, 
              ls = ':', lw = 2, color = 'lightseagreen', label = r'$\mathregular{\overline{C_{\rm{sand, cr}}}}$', zorder = 10)
p3, =  ax[2].plot(Time_ISCO_mid_datetime, ISCO_data['ISCO_sand_concentration_corr_g_l'], marker = 'o',              
        ls = '', markersize = 6, color = 'red', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 40,
        label = r'$\mathregular{C_{\rm{sand, ISCO, corr}}}$')

ax[2].tick_params(axis='both', which='major', labelsize = 14)
ax[2].set_ylim(0,1.2)
ax[2].set_ylabel('$\mathregular{\overline{C_{sand}}}$ (g/l)', fontsize=16, weight = 'bold')     
ax[2].xaxis.set_ticklabels([])
ax[2].set_xlim(Time_datetime_freq2[events_dates['Start_int_freq2'][i]], Time_datetime_freq2[events_dates['End_int_freq2'][i]])
  
# D50, sand
p8, = ax[3].plot(Time_datetime_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]],
               D50_est_S2_D50[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]],
        color = 'darkviolet', ls = '-', lw = 2, label = r'$\mathregular{\overline{D_{50,sand, TW16}}}$')    
p11, = ax[3].plot(ISCO_GSD_data_sand_mean['Time_datetime'],ISCO_GSD_data_sand_mean['D50'],
          color = 'magenta', markersize = 6, ls = '', marker = 'o', 
          markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 40, label = r'$\mathregular{D_{50,sand, ISCO}}$')
 
ax[3].tick_params(axis='both', which='major', labelsize = 14)
ax[3].set_ylim(0,)
ax[3].set_ylabel('$\mathregular{\overline{D_{50, sand}}}$ ($\mathregular{\mu}$m)', fontsize=16, weight = 'bold') 
ax[3].set_xlabel('Time', fontsize=16, weight = 'bold')      
ax[3].xaxis.set_major_locator(md.HourLocator(interval=12))
ax[3].xaxis.set_major_formatter(md.DateFormatter('%d.%m.%Y %H:%M'))
ax[3].xaxis.set_minor_locator(md.HourLocator(interval=1))
ax[3].set_xlim(Time_datetime_freq2[events_dates['Start_int_freq2'][i]], Time_datetime_freq2[events_dates['End_int_freq2'][i]])

ax[0].text(0.02, 0.9, '(a)', fontsize = 14, transform = ax[0].transAxes)
ax[1].text(0.02, 0.9, '(b)', fontsize = 14, transform = ax[1].transAxes)
ax[2].text(0.02, 0.9, '(c)', fontsize = 14, transform = ax[2].transAxes)
ax[3].text(0.02, 0.9, '(d)', fontsize = 14, transform = ax[3].transAxes)
ax[1].text(0.05, 0.9, 'Acoustic measurements', fontsize = 14, transform = ax[1].transAxes)
ax[2].text(0.05, 0.9, 'Rating curve ', fontsize = 14, transform = ax[2].transAxes)

# handles = [p5, p6, p7, p19, p4, p10, p3, p20, p1, p12, p13, p8, p11]
handles = [p5, p6, p7, p19, p4, p10, p3, p1, p12, p13, p8, p11]
#_, labels = ax.get_legend_handles_labels()
fig.legend(handles = handles, #labels=labels, 
          handler_map = {tuple: mpl.legend_handler.HandlerTuple(None)},
          fontsize = 16, loc = 'lower center', ncol = 5, bbox_to_anchor = (0.5, -0.15))
#ax.legend(fontsize = 14, loc = 'lower center', ncol = 4, bbox_to_anchor = (0.5, -0.23))

fig.tight_layout()
figname = 'Fig_A5'
fig.savefig(outpath_figures + '\\' + figname + str(events_dates['Name_date'][i]) + '.png', dpi = 100, bbox_inches='tight')
# fig.savefig(outpath_figures + '\\' + figname + str(events_dates['Name_date'][i]) + '.eps', dpi = 300, bbox_inches='tight')
# fig.savefig(outpath_figures + '\\' + figname + str(events_dates['Name_date'][i]) + '.pdf', dpi = 300, bbox_inches='tight')      


#%% Plot Fig9  Events timeseries i = 2 Spring flood May 2021
i = 2
fig, ax = plt.subplots(3, 1, figsize = (14,10), dpi=300)

# Fines and Q     
ax1 = ax[0].twinx()
p10, = ax1.plot(Time_datetime_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]],
         Q_time_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]],
        ls = '-', lw = 3, color = 'blue', label = 'Q', zorder = 0)

for j in range(len(TAAPS_freq2)):
    ax[0].hlines(TAAPS_freq2['Fine_concentration_g_l'][j], samples_start_datetime[j], samples_end_datetime[j], 
          lw = 6, color = 'darkorange', zorder = 40)    
p5 = ax[0].hlines(TAAPS_freq2['Fine_concentration_g_l'][0], samples_start_datetime[0], samples_end_datetime[0], 
       lw = 6, color = 'darkorange', label = r'Sampler', zorder = 40)
p6, = ax[0].plot(Time_ISCO_mid_datetime, ISCO_data['Fine_concentration_g_l'], marker = 'o',              
        ls = '', markersize = 12, color = 'yellowgreen', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 40,
        label = r'ISCO')

p4, = ax[0].plot(Time_datetime_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]], 
               C_fines_est_time_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]],
               ls = '-', lw = 3, 
         color = 'sienna', label = r'TW20-A', zorder = 10)


ax[0].set_ylabel('$\mathregular{\overline{C_{fines}}}$ (g/l)', fontsize=20, weight = 'bold')
ax[0].set_ylim(0,6) 
ax[0].xaxis.set_major_locator(md.DayLocator(interval=1))
ax[0].xaxis.set_major_formatter(md.DateFormatter('%d.%m.%Y'))
ax[0].xaxis.set_minor_locator(md.HourLocator(interval=1))
ax[0].set_xlim(Time_datetime_freq2[events_dates['Start_int_freq2'][i]], Time_datetime_freq2[events_dates['End_int_freq2'][i]])
ax[0].tick_params(axis='both', which='major', labelsize = 18)
ax[0].xaxis.set_ticklabels([]) 
    
ax1.spines['right'].set_visible(True)
ax1.spines['left'].set_visible(False)
ax1.yaxis.set_label_position('right')
ax1.yaxis.set_ticks_position('right')
ax1.tick_params(axis='both', which='major', labelsize = 18)
ax1.set_ylim(300,550)
ax1.set_ylabel('Q (m³/s)', fontsize=20, weight = 'bold', color = 'blue')  

# CSands
p2 =  ax[1].hlines(TAAPS_freq2['Sand_concentration_g_l'][0], samples_start_datetime[0], samples_end_datetime[0], 
       lw = 6, color = 'darkorange', label = r'Sampler', zorder = 40)
for j in range(len(TAAPS_freq2)):
    ax[1].hlines(TAAPS_freq2['Sand_concentration_g_l'][j], samples_start_datetime[j], samples_end_datetime[j], 
          lw = 6, color = 'darkorange', zorder = 40)
p3, =  ax[1].plot(Time_ISCO_mid_datetime, ISCO_data['ISCO_sand_concentration_corr_g_l'], marker = 'o',              
        ls = '', markersize = 12, color = 'yellowgreen', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 40,
        label = r'ISCO')

p1, = ax[1].plot(Time_datetime_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]], 
              C_sand_S2_D50_g_l[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]], 
              ls = '-', lw = 3, 
        color = 'sienna', label = r'TW16', zorder = 20)  

p12, = ax[1].plot(Time_datetime_freq1[events_dates['Start_int_freq1'][i]:events_dates['End_int_freq1'][i]], 
              C_sand_S2_D50_freq1_g_l[events_dates['Start_int_freq1'][i]:events_dates['End_int_freq1'][i]], 
              ls = '-', lw = 3, 
        color = 'olive', label = r'TW16, 400 kHz', zorder = 10)
p13, = ax[1].plot(Time_datetime_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]], 
              C_sand_S2_D50_freq2_g_l[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]], 
              ls = '-', lw = 3, 
        color = 'mediumorchid', label = r'TW16, 1 MHz', zorder = 10) 
 
p20, = ax[1].plot(Time_Q_datetime, Csand_power_cr_time_Q, 
              ls = '--', lw = 3, color = 'lightseagreen', label = r'Rating curve', zorder = 15)

   
ax[1].set_ylabel('$\mathregular{\overline{C_{sand}}}$ (g/l)', fontsize=20, weight = 'bold')
ax[1].set_ylim(0,1.5)
# ax[1].set_yscale('log') 
ax[1].xaxis.set_major_locator(md.HourLocator(interval=12))
ax[1].xaxis.set_ticklabels([])
ax[1].xaxis.set_minor_locator(md.HourLocator(interval=1))
ax[1].set_xlim(Time_datetime_freq2[events_dates['Start_int_freq2'][i]], Time_datetime_freq2[events_dates['End_int_freq2'][i]])
ax[1].tick_params(axis='both', which='major', labelsize = 18)
# ax[1].legend(fontsize = 18, loc = 'lower center', framealpha = 1, facecolor = 'white', bbox_to_anchor = (0.5, -0.12), ncol= 4)  

# D50,sands
p8, = ax[2].plot(Time_datetime_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]],
               D50_est_S2_D50[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]],
        color = 'sienna', lw = 3, ls = '-',  label = r'TW20-A')
p9 = ax[2].hlines(TAAPS_freq2['D50_mum'][0], samples_start_datetime[0], samples_end_datetime[0], 
        lw = 6, color = 'darkorange', zorder = 30, label = r'Sampler')
for j in range(len(TAAPS_freq2)):
    ax[2].hlines(TAAPS_freq2['D50_mum'][j], samples_start_datetime[j], samples_end_datetime[j], 
          lw = 6, color = 'darkorange', zorder = 30)
    
ax[2].set_ylim(50,350) 
ax[2].xaxis.set_major_locator(md.HourLocator(interval=12))
ax[2].xaxis.set_major_formatter(md.DateFormatter('%d/%m/%Y %H:%M'))
ax[2].xaxis.set_minor_locator(md.HourLocator(interval=1))
ax[2].set_xlim(Time_datetime_freq2[events_dates['Start_int_freq2'][i]], Time_datetime_freq2[events_dates['End_int_freq2'][i]])
ax[2].tick_params(axis='both', which='major', labelsize = 18)
ax[2].set_ylabel('$\mathregular{\overline{D_{50, sand}}}$ ($\mathregular{\mu}$m)', fontsize=20, weight = 'bold') 
ax[2].set_xlabel('Time', fontsize=22, weight = 'bold')     

ax[0].text(0.02, 0.9, '(a)', fontsize = 18, transform = ax[0].transAxes)
ax[1].text(0.02, 0.9, '(b)', fontsize = 18, transform = ax[1].transAxes)
ax[2].text(0.02, 0.9, '(c)', fontsize = 18, transform = ax[2].transAxes)
ax[0].text(0.07, 0.9, 'Fine sediment concentration', fontsize = 18, transform = ax[0].transAxes)
ax[1].text(0.07, 0.9, 'Sand concentration', fontsize = 18, transform = ax[1].transAxes)
ax[2].text(0.07, 0.9, 'Sand grain size', fontsize = 18, transform = ax[2].transAxes)

handles = [p2, p3, p1, p12, p13, p20]
#_, labels = ax.get_legend_handles_labels()
fig.legend(handles = handles, #labels=labels, 
          handler_map = {tuple: mpl.legend_handler.HandlerTuple(None)},
          fontsize = 20, loc = 'lower center', ncol = 3, bbox_to_anchor = (0.5, -0.11))

fig.tight_layout()
figname = 'Fig9'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 300, bbox_inches='tight')
fig.savefig(outpath_figures + '\\' + figname + '.eps', dpi = 300, bbox_inches='tight')                   
           
#%% Plot Fig9_A Events timeseries i = 2 Spring flood May 2021
i = 2
fig, ax = plt.subplots(3, 1, figsize = (14,10), dpi=300)

# Fines and Q     
p10, = ax[0].plot(Time_datetime_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]],
         Q_time_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]],
        ls = '-', lw = 3, color = 'blue', label = 'Q', zorder = 0)

ax[0].set_ylabel('Q (m³/s)', fontsize=22, weight = 'bold')
ax[0].set_ylim(200,500) 
ax[0].xaxis.set_major_locator(md.DayLocator(interval=1))
ax[0].xaxis.set_major_formatter(md.DateFormatter('%d.%m.%Y'))
ax[0].xaxis.set_minor_locator(md.HourLocator(interval=1))
ax[0].set_xlim(Time_datetime_freq2[events_dates['Start_int_freq2'][i]], Time_datetime_freq2[events_dates['End_int_freq2'][i]])
ax[0].tick_params(axis='both', which='major', labelsize = 20)
ax[0].xaxis.set_ticklabels([]) 
    
# CSands
p2 =  ax[1].hlines(TAAPS_freq2['Sand_concentration_g_l'][0], samples_start_datetime[0], samples_end_datetime[0], 
       lw = 6, color = 'darkorange', label = r'Sampler', zorder = 40)
for j in range(len(TAAPS_freq2)):
    ax[1].hlines(TAAPS_freq2['Sand_concentration_g_l'][j], samples_start_datetime[j], samples_end_datetime[j], 
          lw = 6, color = 'darkorange', zorder = 40)
p3, =  ax[1].plot(Time_ISCO_mid_datetime, ISCO_data['ISCO_sand_concentration_corr_g_l'], marker = 'o',              
        ls = '', markersize = 12, color = 'yellowgreen', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 40,
        label = r'ISCO')

p1, = ax[1].plot(Time_datetime_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]], 
              C_sand_S2_D50_g_l[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]], 
              ls = '-', lw = 3, 
        color = 'sienna', label = r'TW16', zorder = 10)  
   
p22, = ax[1].plot(Time_Q_datetime, Csand_power_cr_time_Q, 
              ls = '--', lw = 3, color = 'lightseagreen', label = r'Rating curve', zorder = 10)

   
ax[1].set_ylabel('$\mathregular{\overline{C_{sand}}}$ (g/l)', fontsize=22, weight = 'bold')
ax[1].set_ylim(0,1.5)
# ax[1].set_yscale('log') 
ax[1].xaxis.set_major_locator(md.HourLocator(interval=12))
ax[1].xaxis.set_ticklabels([])
ax[1].xaxis.set_minor_locator(md.HourLocator(interval=1))
ax[1].set_xlim(Time_datetime_freq2[events_dates['Start_int_freq2'][i]], Time_datetime_freq2[events_dates['End_int_freq2'][i]])
ax[1].tick_params(axis='both', which='major', labelsize = 20)
# ax[1].legend(fontsize = 20, loc = 'lower center', framealpha = 1, facecolor = 'white', bbox_to_anchor = (0.5, -0.12), ncol= 4)  

# D50,sands
p8, = ax[2].plot(Time_datetime_freq2[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]],
               D50_est_S2_D50[events_dates['Start_int_freq2'][i]:events_dates['End_int_freq2'][i]],
        color = 'sienna', lw = 3, ls = '-',  label = r'TW22-A')
p9 = ax[2].hlines(TAAPS_freq2['D50_mum'][0], samples_start_datetime[0], samples_end_datetime[0], 
        lw = 6, color = 'darkorange', zorder = 30, label = r'Sampler')
for j in range(len(TAAPS_freq2)):
    ax[2].hlines(TAAPS_freq2['D50_mum'][j], samples_start_datetime[j], samples_end_datetime[j], 
          lw = 6, color = 'darkorange', zorder = 30)
    
ax[2].set_ylim(0,350) 
ax[2].xaxis.set_major_locator(md.DayLocator(interval=1))
ax[2].xaxis.set_major_formatter(md.DateFormatter('%d/%m/%Y'))
ax[2].xaxis.set_minor_locator(md.HourLocator(interval=1))
ax[2].set_xlim(Time_datetime_freq2[events_dates['Start_int_freq2'][i]], Time_datetime_freq2[events_dates['End_int_freq2'][i]])
ax[2].tick_params(axis='both', which='major', labelsize = 20)
ax[2].set_ylabel('$\mathregular{\overline{D_{50, sand}}}$ ($\mathregular{\mu}$m)', fontsize=22, weight = 'bold') 
ax[2].set_xlabel('Time', fontsize=22, weight = 'bold')     

# ax[0].text(0.02, 0.9, '(a)', fontsize = 20, transform = ax[0].transAxes)
# ax[1].text(0.02, 0.9, '(b)', fontsize = 20, transform = ax[1].transAxes)
# ax[2].text(0.02, 0.9, '(c)', fontsize = 20, transform = ax[2].transAxes)
ax[0].text(0.07, 0.9, 'Discharge', fontsize = 20, transform = ax[0].transAxes)
ax[1].text(0.07, 0.9, 'Sand concentration', fontsize = 20, transform = ax[1].transAxes)
ax[2].text(0.07, 0.9, 'Sand grain size', fontsize = 20, transform = ax[2].transAxes)

handles = [p2, p3, p1, p22]
#_, labels = ax.get_legend_handles_labels()
fig.legend(handles = handles, #labels=labels, 
          handler_map = {tuple: mpl.legend_handler.HandlerTuple(None)},
          fontsize = 20, loc = 'lower center', ncol = 5, bbox_to_anchor = (0.5, -0.07))

fig.tight_layout()
figname = 'Fig9_A'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 300, bbox_inches='tight')

#%%##########################################################################################

# Export data

#%% ############################################################################################
## Results_samples
results_samples = pd.concat([pd.Series(np.round(C_sand_S2_D50_samples_g_l, 3)), 
                             pd.Series(np.round(C_sand_S2_D50_freq1_samples_g_l, 3)), 
                             pd.Series(np.round(C_sand_S2_D50_freq2_samples_g_l,3))] , axis = 1)
results_samples.columns = ['C_sand_S2_D50_samples_g_l', 'C_sand_S2_D50_freq1_samples_g_l', 'C_sand_S2_D50_freq2_samples_g_l']
results_samples.to_csv(out_path + '\Results_samples.csv', sep = ';', index=False)

##
TAAPS_freq1.to_csv(out_path + '\Time_averaged_acoustics_physical_samples_' + str(freq1)+ '.csv', sep = ';', index=False)
TAAPS_freq2.to_csv(out_path + '\Time_averaged_acoustics_physical_samples_' + str(freq2)+ '.csv', sep = ';', index=False)
TAAPS_freq1_fines.to_csv(out_path + '\TAAPS_freq1_fines.csv', sep = ';', index=False)
TAAPS_freq2_fines.to_csv(out_path + '\TAAPS_freq2_fines.csv', sep = ';', index=False)

C_est_sand_S2_D50_time_freq2_export = pd.DataFrame([Time_datetime_freq2, C_sand_S2_D50_g_l]).transpose()
C_est_sand_S2_D50_time_freq2_export.columns = ['Date', 'C_sand_g_l']
C_est_sand_S2_D50_time_freq2_export.to_csv(out_path + '\C_sand_g_l.csv', sep = ';', index=False)

Phi_sand_S2_D50_kg_s_export = pd.DataFrame([Time_datetime_freq2, Phi_sand_S2_D50_kg_s, Phi_sand_S2_D50_kg_s_cumsum]).transpose()
Phi_sand_S2_D50_kg_s_export.columns = ['Date', 'Phi_sand_S2_D50_kg_s', 'Phi_sand_S2_D50_kg_s_cumsum']
# pd.DataFrame(Phi_sand_S2_D50_kg_s_cumsum).to_csv(out_path + '\Phi_sand_kg_s_cumsum.csv', sep = ';', index=False)
Phi_sand_S2_D50_kg_s_export.to_csv(out_path + '\Phi_sand_kg_s.csv', sep = ';', index=False)

D50_time_freq2_export = pd.DataFrame([Time_datetime_freq2, D50_est_S2_D50]).transpose()
D50_time_freq2_export.columns = ['Date', 'D50_mum']
D50_time_freq2_export.to_csv(out_path + '\D50_est.csv', sep = ';', index=False)

# Create df with HADCP data and C and D50
Beff_Bdefect_freq2_S2_D50 = [BeamAvBS_effective_freq2_S2_D50[i] - B_defect_freq2_S2_D50[i]
                             for i in range(len(B_defect_freq2_S2_D50))]
S_S2_D50 = [C_fines_est_time_freq2[i]/C_sand_S2_D50_g_l[i] for i in range(len(C_fines_est_time_freq2))]
results  = pd.concat([BeamAv_freq2, pd.Series(BeamAvBS_freq1_time_freq2), pd.Series(AlphaSed_freq1_time_freq2), 
                      pd.Series(BeamAvBS_effective_1_freq1_S2_D50), pd.Series(BeamAvBS_effective_2_freq1_S2_D50),
                      pd.Series(B_defect_freq1_S2_D50), pd.Series(B_defect_freq2_S2_D50), pd.Series(BeamAvBS_effective_freq2_S2_D50),
                      pd.Series(Beff_Bdefect_freq2_S2_D50),
                      pd.Series(C_sand_S2_D50_g_l), pd.Series(D50_est_S2_D50), pd.Series(C_fines_est_time_freq2),
                      pd.Series(S_S2_D50)], axis = 1)
colnames_results = list(results.columns)
colnames_results = ['Date',  'Alpha Sediment (dB/m) Freq2',  'Beam-Averaged Backscatter (dB) Freq2',  'AlphaW Freq2',  'Temperature',
                    'Instrument background (dB)',  'Effective background (dB)',  'Q_m3_s',  'SPM',  'Stage',
                    'Beam-Averaged Backscatter (dB) Freq1', 
                    'Alpha Sediment (dB/m) Freq1', 'BeamAvBS_effective_1_freq1_S2_D50', 'BeamAvBS_effective_2_freq1_S2_D50', 
                    'B_defect_freq1_S2_D50', 'B_defect_freq2_S2_D50', 'BeamAvBS_effective_freq2_S2_D50', 'Beff_Bdefect_freq2_S2_D50',
                    'Csand_S2_D50_g_l', 'D50_est_S2_D50', 'Cfines', 'S_S2_D50']
results.columns = colnames_results



#%%######################################################################################
   
TAAPS_freq1_fines_GSDsand = TAAPS_freq1_fines.dropna(subset = ['D50_mum'], how = 'any') 
TAAPS_freq1_fines_GSDsand.reset_index(drop = True, inplace = True)
TAAPS_freq2_fines_GSDsand = TAAPS_freq2_fines.dropna(subset = ['D50_mum'], how = 'any') 
TAAPS_freq2_fines_GSDsand.reset_index(drop = True, inplace = True)
ISCO_GSD_data_fines = ISCO_GSD_data[ISCO_GSD_data['Fraction'] == 'fines'] 
ISCO_GSD_data_fines.reset_index(drop = True, inplace = True)
ISCO_GSD_data_total = ISCO_GSD_data[ISCO_GSD_data['Fraction'] == 'sediments'] 
ISCO_GSD_data_total.reset_index(drop = True, inplace = True)
ISCO_GSD_data_mean = ISCO_GSD_data[3::4]
ISCO_GSD_data_mean.reset_index(drop = True, inplace = True)
    
#%% Define acoustic models

from math import pi, sqrt, asin, atan # maths
    
def water_velocity(T):  #originally: T = 15 
    """Computing sound speed in m/s from Bilaniuk and Wong 1993"""
    C = 1.40238744*1e3 + 5.03836171*T - 5.81172916*1e-2 * T**2 + 3.34638117*1e-4 * T**3 - \
    1.48259672*1e-6 * T**4 + 3.16585020*1e-9*T**5
    return(C)

water_vel = water_velocity(BeamAv_freq2['Temperature'])

def xi_s_function_ThorneMeral2008(a_s, freq, C= np.nanmean(water_vel)): # originally: C = 1500, here: mean
    """This function computes the total normalized scattering cross
    section, following the equation of Thorne and Meral (2008)"""
    # computing the wave number
    k = 2*pi*freq / C 
    x = k*a_s
    Xi_s = 0.29*x**4 / (0.95 + 1.28*x**2 + 0.25*x**4)  
    return(Xi_s)


def xi_s_function_MoateThorne2012(a_s, freq, C= np.nanmean(water_vel), mica ='false',rho_s = 2650): # originally: C = 1500, here: mean
    """This function computes the total normalized scattering cross
    section, following the equation of Moate and Thorne (2012)
    If parameter <mica> is set to "false", the generic formula is used. 
    If parameter <mica> is set to "true", the formula developped for mica particles is used.
    /!\ it computes the ratio Xi/rho_s"""
    
    # computing the wave number
    k = 2*pi*freq / C
    
    x = k*a_s
    if mica == 'false':
        Xi_s = 0.09 * x**4 / (1380 + 560 * x**2 + 150 * x**4)
    
    elif mica == 'true':
        c1 = 0.3
        c2 = 1.46
        c3 = 0.95
        c4 = 0.19
        Xi_s = c1*x**4 / (c2 + c3*x**2 + c4*x**4)/rho_s
    return(Xi_s)


def xi_v_function_Urick(a_s, freq, rho_s=2650, rho_0=1000, nu_0=1.2*1e-6, C=1500):
    """This function computes the total normalized viscous absorption cross
    section, following the equation of Urick (1948)"""
    # angular frequencies
    omega = 2*pi*freq
    
    # wave number
    k = omega / C
    
    # coefficient b (inverse of the visquous boundary layer thickness)
    b = np.sqrt(omega / (2*nu_0))
    
    # coefficient delta
    delta = 0.5*(1 + 9/(2*b*a_s))
    
    # coefficient g (density ratio)
    g = rho_s / rho_0
    
    # coefficient s
    s = (9 / (4*b*a_s))*(1 + 1/(b*a_s))
    
    # Computing the cross section
    Xi_v = (2/3)*k*a_s*(g-1)**2*(s/(s**2 + (g + delta)**2))
    
    return(Xi_v)

def xi_v_function_richards(a_s, h, freq, rho_s=2650, rho_0=1000, nu_0=1.2*1e-6, C=1500):
    """This function computes the total normalized viscous absorption cross
    section in the case of oblate spheroid particles, following the equation of Richards et al (2003)"""
    # angular frequencies
    omega = 2 * pi * freq
    
    # wave number
    k = omega / C
    
    # Case where h=1: spherical particles
    if h ==1:
        # coefficient b (inverse of the visquous boundary layer thickness)
        b = np.sqrt(omega / (2 * nu_0))

        # coefficient delta
        delta = 0.5 * (1 + 9/(2 * b * a_s))

        # coefficient g (density ratio)
        g = rho_s / rho_0

        # coefficient s
        s = (9 / (4 * b * a_s)) * (1 + 1 / (b * a_s))

        # Computing the cross section
        Xi_v = (2 / 3) * k * a_s * (g - 1)**2 * (s / (s**2 + (g + delta)**2))
    
    # Case where h < 1: oblate spheroids
    else:
        # coefficient b (inverse of the visquous boundary layer thickness)
        beta = np.sqrt(omega / (2 * nu_0))

        # coefficient g (density ratio)
        g = rho_s / rho_0

        # aspect ratio
        epsilon = sqrt(1 - h**2)

        ###### Particles oriented paralel to the direction of sound propagation #####
        alpha0 = (2 / epsilon**2) * ( 1 - sqrt(1 - epsilon**2) * asin(epsilon) / epsilon)
        Li_pa = alpha0 / (2 - alpha0)
        K_pa = (8 / 3) * (2 * h /(1 - h**2) + 2 * (1 - 2 * h**2) / (1 - h**2)**(3/2) * atan((1 - h**2)**0.5 / h))**(-1)

        s_pa = 9 / (4 * h * a_s * beta) * K_pa**2 * (1 + 1 / (K_pa * beta * a_s))
        delta_pa = Li_pa + 9 / (4 * h * a_s * beta) * K_pa**2
        Xi_v_pa = (2 / 3) * k * a_s * (g - 1)**2 * (s_pa / (s_pa**2 + (g + delta_pa)**2))

        ###### Particles oriented perpendicular to the direction of sound propagation #####
        gamma0 = sqrt(1 - epsilon**2) / epsilon**3 * asin(epsilon) - (1 - epsilon**2) / epsilon**2
        Li_ort = gamma0 / (2 - gamma0)
        K_ort = (8 / 3) * (- h /(1 - h**2) - (2 * h**2 - 3) / (1 - h**2)**(3/2) * asin((1 - h**2)**0.5))**(-1)

        s_ort = 9 / (4 * h * a_s * beta) * K_ort**2 * (1 + 1 / (K_ort * beta * a_s))
        delta_ort = Li_ort + 9 / (4 * h * a_s * beta) * K_ort**2
        Xi_v_ort = (2 / 3) * k * a_s * (g - 1)**2 * (s_ort / (s_ort**2 + (g + delta_ort)**2))

        # Computing the cross section, considering two third of the particles being oriented perpendicularly to the 
        # direction of sound propagation
        Xi_v = (2/3) * Xi_v_pa + (1/3) * Xi_v_ort

    return(Xi_v)


def plot_cross_section(freq1, freq2, A):
    plt.plot(A, xi_v_function_Urick(A, freq1) * 3 / (4 * A * 2650), color='navy', label = 'viscous')
    plt.plot(A, xi_v_function_Urick(A, freq2) * 3 / (4 * A * 2650), color='darkred')
    plt.plot(A, xi_s_function_ThorneMeral2008(A, freq1) * 3 / (4 * A * 2650), color='navy', linestyle='--')
    plt.plot(A, xi_s_function_ThorneMeral2008(A, freq2)*3/(4 * A * 2650), color='darkred', linestyle='--')
    
    plt.xlabel(r'Radius ($\mu$m)')
    plt.ylabel(r'Sed. atten. const. $\zeta$')
    plt.xscale('log')
    plt.legend(['%1.1f MHz' %(freq1 * 1e-6), '%1.1f MHz' %(freq2 * 1e-6)], loc=2)
    # x_formatter = FixedFormatter(["10","45","100"])
    # x_locator = FixedLocator([0.0010,0.00045,0.0000100])
    # plt.major_formatter(x_formatter)
    # plt.major_locator(x_locator)



#%% 5) Compare the forward and inverse method

# #%% CALCULATE INTEGRATED FORM FACTOR F (THORNE & MERAL 2008) AND ZETA FOR SAMPLINGS
# # sand
# f_TM08_freq1_sand = []
# f_TM08_freq2_sand = []
# zeta_freq1_sand = []
# zeta_freq2_sand = []
# zetas_freq1_sand = []
# zetas_freq2_sand = []
# zetav_freq1_sand = []
# zetav_freq2_sand = []
# for i in range(len(TAAPS_freq1)):
#     if TAAPS_freq1['D50_mum'].iloc[i] >= 1:
#         ff1 = compute_model_lognorm_spherical(TAAPS_freq1['D50_mum'].iloc[i]*1e-6, TAAPS_freq1['sigma_mum'].iloc[i], freq1_Hz,h,  rho_sed, nu_0)
#         f1 = ff1.f_TM08
#         z1 = ff1.zeta
#         zs1 = ff1.zetas
#         zv1 = ff1.zetav
#         ff2 = compute_model_lognorm_spherical(TAAPS_freq2['D50_mum'].iloc[i]*1e-6, TAAPS_freq2['sigma_mum'].iloc[i], freq2_Hz, h, rho_sed, nu_0)
#         f2 = ff2.f_TM08
#         z2 = ff2.zeta
#         zs2 = ff2.zetas
#         zv2 = ff2.zetav
#     else:
#         f1 = None
#         f2 = None
#         z1 = None
#         zs1 = None
#         zv1 = None
#         z2 = None
#         zs2 = None
#         zv2 = None
#     f_TM08_freq1_sand.append(f1)
#     f_TM08_freq2_sand.append(f2)
#     zeta_freq1_sand.append(z1)
#     zeta_freq2_sand.append(z2)
#     zetas_freq1_sand.append(zs1)
#     zetas_freq2_sand.append(zs2)
#     zetav_freq1_sand.append(zv1)
#     zetav_freq2_sand.append(zv2)
# TAAPS_freq1['f_int_sand'] = f_TM08_freq1_sand
# TAAPS_freq1['zeta_sand'] = zeta_freq1_sand # m-1
# TAAPS_freq1['zeta_s_sand'] = zetas_freq1_sand # m-1
# TAAPS_freq1['zeta_v_sand'] = zetav_freq1_sand # m-1
# TAAPS_freq2['f_int_sand'] = f_TM08_freq2_sand # m-1
# TAAPS_freq2['zeta_sand'] = zeta_freq2_sand # m-1
# TAAPS_freq2['zeta_s_sand'] = zetas_freq2_sand # m-1
# TAAPS_freq2['zeta_v_sand'] = zetav_freq2_sand # m-1

# # fines
# f_TM08_freq1_fines = []
# f_TM08_freq2_fines = []
# zeta_freq1_fines = []
# zeta_freq2_fines = []
# zetas_freq1_fines = []
# zetas_freq2_fines = []
# zetav_freq1_fines = []
# zetav_freq2_fines = []
# for i in range(len(TAAPS_freq1)):
#     if TAAPS_freq1['D50_mum_fines'].iloc[i] >= 1:
#         ff1 = compute_model_lognorm_spherical(TAAPS_freq1['D50_mum_fines'].iloc[i]*1e-6, TAAPS_freq1['sigma_mum_fines'].iloc[i], freq1_Hz, h, rho_sed, nu_0)
#         f1 = ff1.f_TM08
#         z1 = ff1.zeta
#         zs1 = ff1.zetas
#         zv1 = ff1.zetav
#         ff2 = compute_model_lognorm_spherical(TAAPS_freq2['D50_mum_fines'].iloc[i]*1e-6, TAAPS_freq2['sigma_mum_fines'].iloc[i], freq2_Hz, h, rho_sed, nu_0)
#         f2 = ff2.f_TM08
#         z2 = ff2.zeta
#         zs2 = ff2.zetas
#         zv2 = ff2.zetav
#     else:
#         f1 = None
#         f2 = None
#         z1 = None
#         zs1 = None
#         zv1 = None
#         z2 = None
#         zs2 = None
#         zv2 = None
#     f_TM08_freq1_fines.append(f1)
#     f_TM08_freq2_fines.append(f2)
#     zeta_freq1_fines.append(z1)
#     zeta_freq2_fines.append(z2)
#     zetas_freq1_fines.append(zs1)
#     zetas_freq2_fines.append(zs2)
#     zetav_freq1_fines.append(zv1)
#     zetav_freq2_fines.append(zv2)
# TAAPS_freq1['f_int_fines'] = f_TM08_freq1_fines
# TAAPS_freq1['zeta_fines'] = zeta_freq1_fines # m-1
# TAAPS_freq1['zeta_s_fines'] = zetas_freq1_fines # m-1
# TAAPS_freq1['zeta_v_fines'] = zetav_freq1_fines # m-1
# TAAPS_freq2['f_int_fines'] = f_TM08_freq2_fines # m-1
# TAAPS_freq2['zeta_fines'] = zeta_freq2_fines # m-1
# TAAPS_freq2['zeta_s_fines'] = zetas_freq2_fines # m-1
# TAAPS_freq2['zeta_v_fines'] = zetav_freq2_fines # m-1

# #%% 5) Calculate alpha samples ('meas')
# TAAPS_freq1['Alpha_fines_1_m'] = TAAPS_freq1['zeta_fines'] *TAAPS_freq1['Fine_concentration_g_l']
# TAAPS_freq2['Alpha_fines_1_m'] = TAAPS_freq2['zeta_fines'] *TAAPS_freq2['Fine_concentration_g_l']
# # TAAPS_freq1['Alphas_fines_1_m'] = TAAPS_freq1['zetas_fines'] *TAAPS_freq1['Fine_concentration_g_l']
# # TAAPS_freq2['Alphas_fines_1_m'] = TAAPS_freq2['zetas_fines'] *TAAPS_freq2['Fine_concentration_g_l']
# # TAAPS_freq1['Alphav_fines_1_m'] = TAAPS_freq1['zetav_fines'] *TAAPS_freq1['Fine_concentration_g_l']
# # TAAPS_freq2['Alphav_fines_1_m'] = TAAPS_freq2['zetav_fines'] *TAAPS_freq2['Fine_concentration_g_l']
# TAAPS_freq1['Alpha_fines_dB_m'] = 20/(np.log(10))*TAAPS_freq1['Alpha_fines_1_m']
# TAAPS_freq2['Alpha_fines_dB_m'] = 20/(np.log(10))*TAAPS_freq2['Alpha_fines_1_m']

# TAAPS_freq1['Alpha_sand_dB_m'] = 20/(np.log(10))*TAAPS_freq1['zeta_sand'] *TAAPS_freq1['Sand_concentration_g_l']
# TAAPS_freq2['Alpha_sand_dB_m'] = 20/(np.log(10))*TAAPS_freq2['zeta_sand'] *TAAPS_freq2['Sand_concentration_g_l']
# TAAPS_freq1['Alpha_dB_m'] = TAAPS_freq1['Alpha_sand_dB_m'] + TAAPS_freq1['Alpha_fines_dB_m']
# TAAPS_freq2['Alpha_dB_m'] = TAAPS_freq2['Alpha_sand_dB_m'] + TAAPS_freq2['Alpha_fines_dB_m']



#%% 5) Calculate Alpha on ISCO meas
# fines
zeta_freq1_ISCO_fines = []
zeta_freq2_ISCO_fines = []
for i in range(len(ISCO_GSD_data_total)):
    if ISCO_GSD_data_total['D50'].iloc[i] >= 1 and ISCO_GSD_data_total['D50'].iloc[i] <= 63:
        ff1 = compute_model_lognorm_spherical(ISCO_GSD_data_total['D50'].iloc[i]*1e-6, 2.4, freq1_Hz, h, rho_sed, nu_0)
        # f1 = ff1.f_TM08
        z1 = ff1.zeta
        # zs1 = ff1.zetas
        # zv1 = ff1.zetav
        ff2 = compute_model_lognorm_spherical(ISCO_GSD_data_total['D50'].iloc[i]*1e-6, 2.4, freq2_Hz, h, rho_sed, nu_0)
        # f2 = ff2.f_TM08
        z2 = ff2.zeta
        # zs2 = ff2.zetas
        # zv2 = ff2.zetav
    else:
        f1 = None
        f2 = None
        z1 = None
        zs1 = None
        zv1 = None
        z2 = None
        zs2 = None
        zv2 = None
    # f_TM08_freq1_fines.append(f1)
    # f_TM08_freq2_fines.append(f2)
    zeta_freq1_ISCO_fines.append(z1)
    zeta_freq2_ISCO_fines.append(z2)
    # zetas_freq1_fines.append(zs1)
    # zetas_freq2_fines.append(zs2)
    # zetav_freq1_fines.append(zv1)
    # zetav_freq2_fines.append(zv2)
# TAAPS_freq1['f_int_fines'] = f_TM08_freq1_fines
ISCO_GSD_data_total['zeta_total_freq1'] = zeta_freq1_ISCO_fines # m-1
# TAAPS_freq1['zeta_s_fines'] = zetas_freq1_fines # m-1
# TAAPS_freq1['zeta_v_fines'] = zetav_freq1_fines # m-1
# TAAPS_freq2['f_int_fines'] = f_TM08_freq2_fines # m-1
ISCO_GSD_data_total['zeta_total_freq2'] = zeta_freq2_ISCO_fines # m-1
# TAAPS_freq2['zeta_s_fines'] = zetas_freq2_fines # m-1
# TAAPS_freq2['zeta_v_fines'] = zetav_freq2_fines # m-1

ISCO_GSD_data_total['Alpha_total_freq1_dB_m'] = 20/(np.log(10))*ISCO_GSD_data_total['zeta_total_freq1']*ISCO_GSD_data_total['Concentration_g_l']
ISCO_GSD_data_total['Alpha_total_freq2_dB_m'] = 20/(np.log(10))*ISCO_GSD_data_total['zeta_total_freq2']*ISCO_GSD_data_total['Concentration_g_l']

ISCO_GSD_data_total_mean = ISCO_GSD_data_total[3::4]

#%% 5) Regression between AlphaSed HADCP & AlphaSed sample 
# Meas
# freq1       
idx = np.isfinite(TAAPS_freq1['Alpha Sediment (dB/m)']) & np.isfinite(TAAPS_freq1['Alpha_dB_m'])
AlphaSed_freq1_HADCP = [TAAPS_freq1['Alpha Sediment (dB/m)'][i] for i in range(len(TAAPS_freq1['Alpha Sediment (dB/m)'])) if idx[i] == True]
AlphaSed_freq1_meas = [TAAPS_freq1['Alpha_dB_m'][i] for i in range(len(TAAPS_freq1['Alpha_dB_m'])) if idx[i] == True]
x = np.array(AlphaSed_freq1_HADCP)
y = np.array(AlphaSed_freq1_meas)
x = x[:,np.newaxis]
slope_AlphaSed_HADCP_sample_freq1, _, _, _ = np.linalg.lstsq(x, y)
lin_model_AlphaSed_HADCP_sample_freq1_origin = [AlphaSed_freq1_HADCP[i]*slope_AlphaSed_HADCP_sample_freq1
                           for i in range(len(AlphaSed_freq1_HADCP))]
lin_model_AlphaSed_HADCP_sample_freq1_origin_plot = x_range*slope_AlphaSed_HADCP_sample_freq1
R2_AlphaSed_HADCP_sample_freq1_origin = r2_score(AlphaSed_freq1_meas, lin_model_AlphaSed_HADCP_sample_freq1_origin)

# Freq2
idx = np.isfinite(TAAPS_freq2['Alpha Sediment (dB/m)']) & np.isfinite(TAAPS_freq2['Alpha_dB_m'])
AlphaSed_freq2_HADCP = [TAAPS_freq2['Alpha Sediment (dB/m)'][i] for i in range(len(TAAPS_freq2['Alpha Sediment (dB/m)'])) if idx[i] == True]
AlphaSed_freq2_meas = [TAAPS_freq2['Alpha_dB_m'][i] for i in range(len(TAAPS_freq2['Alpha_dB_m'])) if idx[i] == True]
x = np.array(AlphaSed_freq2_HADCP)
y = np.array(AlphaSed_freq2_meas)
x = x[:,np.newaxis]
slope_AlphaSed_HADCP_sample_freq2, _, _, _ = np.linalg.lstsq(x, y)
lin_model_AlphaSed_HADCP_sample_freq2_origin = [AlphaSed_freq2_HADCP[i]*slope_AlphaSed_HADCP_sample_freq2
                           for i in range(len(AlphaSed_freq2_HADCP))]
lin_model_AlphaSed_HADCP_sample_freq2_origin_plot = x_range*slope_AlphaSed_HADCP_sample_freq2
R2_AlphaSed_HADCP_sample_freq2_origin = r2_score(AlphaSed_freq2_meas, lin_model_AlphaSed_HADCP_sample_freq2_origin)

# ISCO
# freq1       
idx_ISCO = np.isfinite(ISCO_GSD_data_total['AlphaSed_freq1']) & np.isfinite(ISCO_GSD_data_total_mean['Alpha_total_freq1_dB_m'])
AlphaSed_freq1_HADCP_ISCO = [ISCO_GSD_data_total_mean['AlphaSed_freq1'][i] for i in range(len(ISCO_GSD_data_total_mean['AlphaSed_freq1'])) if idx_ISCO[i] == True]
AlphaSed_freq1_meas_ISCO = [ISCO_GSD_data_total_mean['Alpha_total_freq1_dB_m'][i] for i in range(len(ISCO_GSD_data_total_mean['Alpha_total_freq1_dB_m'])) if idx_ISCO[i] == True]
x = np.array(AlphaSed_freq1_HADCP_ISCO)
y = np.array(AlphaSed_freq1_meas_ISCO)
x = x[:,np.newaxis]
slope_AlphaSed_HADCP_sample_freq1_ISCO, _, _, _ = np.linalg.lstsq(x, y)
lin_model_AlphaSed_HADCP_sample_freq1_origin_ISCO = [AlphaSed_freq1_HADCP_ISCO[i]*slope_AlphaSed_HADCP_sample_freq1_ISCO
                           for i in range(len(AlphaSed_freq1_HADCP_ISCO))]
lin_model_AlphaSed_HADCP_sample_freq1_origin_plot_ISCO = x_range*slope_AlphaSed_HADCP_sample_freq1_ISCO
R2_AlphaSed_HADCP_sample_freq1_origin_ISCO = r2_score(AlphaSed_freq1_meas_ISCO, lin_model_AlphaSed_HADCP_sample_freq1_origin_ISCO)

# Freq2
idx_ISCO = np.isfinite(ISCO_GSD_data_total['AlphaSed_freq2']) & np.isfinite(ISCO_GSD_data_total['Alpha_total_freq2_dB_m'])
AlphaSed_freq2_HADCP_ISCO = [ISCO_GSD_data_total_mean['AlphaSed_freq2'].iloc[i] for i in range(len(ISCO_GSD_data_total_mean['AlphaSed_freq2'])) if idx_ISCO[i] == True]
AlphaSed_freq2_meas_ISCO = [ISCO_GSD_data_total_mean['Alpha_total_freq2_dB_m'].iloc[i] for i in range(len(ISCO_GSD_data_total_mean['Alpha_total_freq2_dB_m'])) if idx_ISCO[i] == True]
x = np.array(AlphaSed_freq2_HADCP_ISCO)
y = np.array(AlphaSed_freq2_meas_ISCO)
x = x[:,np.newaxis]
slope_AlphaSed_HADCP_sample_freq2_ISCO, _, _, _ = np.linalg.lstsq(x, y)
lin_model_AlphaSed_HADCP_sample_freq2_origin_ISCO = [AlphaSed_freq2_HADCP_ISCO[i]*slope_AlphaSed_HADCP_sample_freq2_ISCO
                           for i in range(len(AlphaSed_freq2_HADCP_ISCO))]
lin_model_AlphaSed_HADCP_sample_freq2_origin_plot_ISCO = x_range*slope_AlphaSed_HADCP_sample_freq2_ISCO
R2_AlphaSed_HADCP_sample_freq2_origin_ISCO = r2_score(AlphaSed_freq2_meas_ISCO, lin_model_AlphaSed_HADCP_sample_freq2_origin_ISCO)

#%% 5) Determine s_v_dB
TAAPS_freq1['s_v_sand_db'] = TAAPS_freq1['s_v_sand']*20/(np.log(10))
TAAPS_freq2['s_v_sand_db'] = TAAPS_freq2['s_v_sand']*20/(np.log(10)) # sv in 1/m3
TAAPS_freq1['Beam-Averaged Backscatter (m)'] = TAAPS_freq1['Beam-Averaged Backscatter (dB)']/20*(np.log(10))
TAAPS_freq2['Beam-Averaged Backscatter (m)'] = TAAPS_freq2['Beam-Averaged Backscatter (dB)']/20*(np.log(10)) 

# meas 
# kt_freq1 = 1.8*1e6 #1000000000
# kt_freq2 = 1.3*1e4
kt_freq1 = 3.7*1e6 #1000000000
kt_freq2 = 1.8*1e4
TAAPS_freq2['s_v_sand_db_meas'] = 2*(TAAPS_freq2['Beam-Averaged Backscatter (dB)']- 0.5*np.log((16*math.pi*kt_freq2**2)/3))

TAAPS_freq1['B_model'] = 0.5*np.log((16*math.pi*kt_freq1**2)/3 *TAAPS_freq1['s_v_sand'])*20/(np.log(10))
TAAPS_freq2['B_model'] = 0.5*np.log(((16*math.pi*kt_freq2**2)/3) *TAAPS_freq2['s_v_sand'])*20/(np.log(10))

 #%% Plot Fig11
 
x_range = np.arange(0.005,120,10)
x_range_B = np.arange(60,150, 10)

fig, ax = plt.subplots(2, 2, figsize = (14,12), dpi=300, layout = 'constrained')
# Freq1
ax[0,0].scatter(TAAPS_freq1['Alpha_dB_m'], TAAPS_freq1['Alpha Sediment (dB/m)'], marker = 'D', 
   c = TAAPS_freq1['Sand_concentration_g_l']+TAAPS_freq1['Fine_concentration_g_l'],
   s = 100, lw = 0.2, edgecolor = 'black', zorder = 10, cmap = 'plasma',
   norm = mpl.colors.Normalize(vmin=np.min(ISCO_GSD_data_total_mean['Concentration_g_l']), vmax=np.max(ISCO_GSD_data_total_mean['Concentration_g_l'])),
   label = '400 kHz')

ax[0,0].scatter(ISCO_GSD_data_total_mean['Alpha_total_freq1_dB_m'], ISCO_GSD_data_total_mean['AlphaSed_freq1'], marker = 'o', 
  c = ISCO_GSD_data_total_mean['Concentration_g_l'],
  s = 50, lw = 0.2, edgecolor = 'black', zorder = 10, cmap = 'plasma',
  norm = mpl.colors.Normalize(vmin=np.min(ISCO_GSD_data_total_mean['Concentration_g_l']), vmax=np.max(ISCO_GSD_data_total_mean['Concentration_g_l'])),
  label = '400 kHz')

ax[0,0].plot(x_range, x_range, color = 'black')
ax[0,0].plot(x_range, 5*x_range, color = 'black', ls = (0, (1, 10)), label= 'Error of a factor of 1.1')
ax[0,0].plot(5*x_range, x_range, color = 'black', ls = (0, (1, 10)))
ax[0,0].plot(x_range, 2*x_range, color = 'black', ls = ':')
ax[0,0].plot(2*x_range, x_range, color = 'black', ls = ':')

ax[0,0].text(0.05, 0.95, '(a)', fontsize = 18, transform = ax[0,0].transAxes)
ax[0,0].set_ylabel(r'$\mathregular{\alpha_{sed, 400 kHz} \; (dB/m)}$', fontsize=20, weight = 'bold')
ax[0,0].set_xlabel(r'$\mathregular{\alpha_{sed, model} \; (dB/m)}$', fontsize=20, weight = 'bold')
ax[0,0].tick_params(axis='both', which='major', labelsize = 18)
ax[0,0].set_xlim(0.01,10)
ax[0,0].set_ylim(0.01,10)
ax[0,0].set_xscale('log')
ax[0,0].set_yscale('log')

# Freq2
p1, = ax[0,1].plot(TAAPS_freq2['Alpha_dB_m'][0], TAAPS_freq2['Alpha Sediment (dB/m)'][0], marker = 'D', 
   color = 'black', markersize = 7, lw = 0,zorder = 0,
   label = 'Sampler')
sc = ax[0,1].scatter(TAAPS_freq2['Alpha_dB_m'], TAAPS_freq2['Alpha Sediment (dB/m)'], marker = 'D', 
   c = TAAPS_freq2['Sand_concentration_g_l']+TAAPS_freq2['Fine_concentration_g_l'],
   s = 100, lw = 0.2, edgecolor = 'black', zorder = 10, cmap = 'plasma', 
   norm = mpl.colors.Normalize(vmin=np.min(ISCO_GSD_data_total_mean['Concentration_g_l']), vmax=np.max(ISCO_GSD_data_total_mean['Concentration_g_l'])))
ax[0,1].scatter(ISCO_GSD_data_total_mean['Alpha_total_freq2_dB_m'], ISCO_GSD_data_total_mean['AlphaSed_freq2'], marker = 'o', 
  c = ISCO_GSD_data_total_mean['Concentration_g_l'],
  s = 50, lw = 0.2, edgecolor = 'black', zorder = 10, cmap = 'plasma',
  norm = mpl.colors.Normalize(vmin=np.min(ISCO_GSD_data_total_mean['Concentration_g_l']), vmax=np.max(ISCO_GSD_data_total_mean['Concentration_g_l'])))    
p2, = ax[0,1].plot( ISCO_GSD_data_total_mean['Alpha_total_freq2_dB_m'], ISCO_GSD_data_total_mean['AlphaSed_freq2'], marker = 'o', 
   color = 'black', markersize = 7, lw = 0, zorder = 0,
   label = 'ISCO')    

p3, = ax[0,1].plot(x_range, x_range, color = 'black', label = 'Perfect agreement')
ax[0,1].plot(x_range, 5*x_range, color = 'black', ls = (0, (1, 10)), label= 'Error of a factor of 1.1')
ax[0,1].plot(5*x_range, x_range, color = 'black', ls = (0, (1, 10)))
p5, = ax[0,1].plot(x_range, 2*x_range, color = 'black', ls = ':', label = 'Error of a factor of 2')
ax[0,1].plot(2*x_range, x_range, color = 'black', ls = ':')

ax[0,1].text(0.05, 0.95, '(b)', fontsize = 18, transform = ax[0,1].transAxes)
ax[0,1].set_ylabel(r'$\mathregular{\alpha_{sed, 1 MHz} \; (dB/m)}$', fontsize=20, weight = 'bold')
ax[0,1].set_xlabel(r'$\mathregular{\alpha_{sed, model} \; (dB/m)}$', fontsize=20, weight = 'bold')
ax[0,1].tick_params(axis='both', which='major', labelsize = 18)
ax[0,1].set_xlim(0.01,10)
ax[0,1].set_ylim(0.01,10)
ax[0,1].set_xscale('log')
ax[0,1].set_yscale('log')
#ax[0,1].legend(fontsize = 18, loc = 'lower right')
# ax[0,1].yaxis.tick_right()

cbar = fig.colorbar(sc, shrink = 0.9, ax=ax[0,1], extend='both')
cbar.ax.yaxis.set_tick_params(pad=10, labelsize = 18)
cbar.set_label(r'$\mathregular{C_{tot}}$ (g/l)', labelpad=10, fontsize = 20, weight = 'bold')

# BS Freq1
ax[1,0].scatter(TAAPS_freq1['B_model'], TAAPS_freq1['Beam-Averaged Backscatter (dB)'], marker = 'D',
  c = TAAPS_freq1['Sand_concentration_g_l'],
s = 80, lw = 0.2, edgecolor = 'black', zorder = 10, cmap = 'magma',
label = '400 kHz')

ax[1,0].text(0.05, 0.95, '(c)', fontsize = 18, transform = ax[1,0].transAxes)
ax[1,0].set_xlabel(r'$\mathregular{\overline{B_{model}} \; (dB)}$', fontsize=20, weight = 'bold')
ax[1,0].set_ylabel(r'$\mathregular{\overline{B_{400 kHz}} \; (dB)}$', fontsize=20, weight = 'bold')
ax[1,0].tick_params(axis='both', which='major', labelsize = 18)
ax[1,0].set_xlim(95,110)
ax[1,0].set_ylim(95,110)
ax[1,0].plot(x_range_B,x_range_B,
   lw = 1, color = 'black')
p4, = ax[1,0].plot(x_range_B, 1.1*x_range_B, color = 'black', ls = '--', label= 'Error of a factor of 1.1')
ax[1,0].plot(1.1*x_range_B, x_range_B, color = 'black', ls = '--')
p5, = ax[1,0].plot(x_range, 2*x_range, color = 'black', ls = ':', label = 'Error of a factor of 2')
p6, = ax[1,0].plot(x_range, 5*x_range, color = 'black', ls = (0, (1, 10)), label= 'Error of a factor of 5')

# BS Freq2
sc = ax[1,1].scatter(TAAPS_freq2['B_model'], TAAPS_freq2['Beam-Averaged Backscatter (dB)'], marker = 'D', 
  c = TAAPS_freq2['Sand_concentration_g_l'],
  s = 80, lw = 0.2, edgecolor = 'black', zorder = 10, cmap = 'magma',
label = '1000 kHz')

ax[1,1].plot(x_range_B, x_range_B, lw = 1, color = 'black')
ax[1,1].plot(x_range_B, 1.1*x_range_B, color = 'black', ls = '--')
ax[1,1].plot(1.1*x_range_B, x_range_B, color = 'black', ls = '--')

ax[1,1].text(0.05, 0.95, '(d)', fontsize = 18, transform = ax[1,1].transAxes)
ax[1,1].set_ylabel(r'$\mathregular{\overline{B_{1 MHz}} \; (dB)}$', fontsize=20, weight = 'bold')
ax[1,1].set_xlabel(r'$\mathregular{\overline{B_{model}} \; (dB)}$', fontsize=20, weight = 'bold')
ax[1,1].tick_params(axis='both', which='major', labelsize = 18)
ax[1,1].set_xlim(60,75)
ax[1,1].set_ylim(60,75)

cbar = fig.colorbar(sc, shrink = 0.9, ax=ax[1,1], extend='both')
cbar.ax.yaxis.set_tick_params(pad=10, labelsize = 18)
cbar.set_label(r'$\mathregular{\overline{C_{sand}}}$ (g/l)', labelpad=10, fontsize = 20, weight = 'bold')

handles = [p1, p2, p3, p4, p5, p6]
fig.legend(handles = handles, #labels=labels, 
          handler_map = {tuple: mpl.legend_handler.HandlerTuple(None)},
          fontsize = 18, loc = 'lower center', ncol = 3, bbox_to_anchor = (0.5, -0.09))

# fig.tight_layout()
# figname = 'Fig11_150'
# fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 150, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight') 
# fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 300, bbox_inches='tight')

figname = 'Fig11'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 300, bbox_inches='tight')
fig.savefig(outpath_figures + '\\' + figname + '.eps', dpi = 300, bbox_inches='tight')
   

#%% Plot Q - C est S2_D50 sand
C_power_cr = [Qss_range_power_cr[i]/Q_range_Cr[i] for i in range(len(Qss_range_power_cr))]

fig, ax = plt.subplots(1, 1, figsize = (8,6), dpi=300)

ax.plot(Q_time_freq2[0], C_sand_S2_D50_g_l[0], '.', markersize = 8, alpha = 0.8,
        color = 'tan', label = 'TW16')
ax.plot(Q_time_freq2, C_sand_S2_D50_g_l, '.', markersize = 3, alpha = 0.8,
        color = 'tan')
ax.plot(TAAPS_freq2['Q_sampling_m3_s'], TAAPS_freq2['Sand_concentration_g_l'], 'D', 
        color = 'darkorange', markeredgecolor = 'black', markeredgewidth = 0.2, label = 'Sampler', zorder = 30)
ax.plot(Q_time_pump, pump_data['Sand_concentration_g_l'], marker = 's',              
        ls = '', markersize = 6, color = 'mediumblue', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 20,
        label = 'Pump')
ax.plot(Q_time_ISCO, ISCO_data['ISCO_sand_concentration_corr_g_l'], marker = 'o',              
        ls = '', markersize = 5, color = 'yellowgreen', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 10,
        label = 'ISCO')

p1, = ax.plot(Q_range_Cr, C_power_cr,
        lw = 2, color = 'darkorange', label = r'Rating curve', zorder = 40) #$\mathregular{\Phi_{sand,cr} = a_{cr}(Q-Q_{cr})^{b_{cr}}}$
 

ax.text(0.05, 0.95, '(a)', fontsize = 16, transform = ax.transAxes)
ax.set_xlabel('Q (m³/s)', fontsize=20, weight = 'bold')
ax.set_ylabel('$\mathregular{\overline{C_{sand}}}$ (g/l)', fontsize=20, weight = 'bold')
ax.tick_params(axis='both', which='major', labelsize = 16)
ax.set_ylim(0.001,7)
ax.set_xlim(0,700)
ax.set_yscale('log')
ax.legend(fontsize = 14, loc = 'lower right')

fig.tight_layout()
figname = 'Fig_A3_TW16'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight') 
# fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 300, bbox_inches='tight')

    
#%% Time - Csand_freq
fig, ax = plt.subplots(2, 1, figsize = (10,10), dpi=300)

ax[0].plot(Time_datetime_freq1, C_sand_S2_D50_freq1_g_l, lw = 0, marker = '.', markersize = 1, 
        color = 'darkgoldenrod')

ax[0].text(0.03, 0.95, '400 kHz', fontsize = 16, transform = ax[0].transAxes, weight = 'bold') 
# ax.set_xlabel('Time', fontsize=16, weight = 'bold')
# ax.set_ylabel('$\mathregular{C_{sand}}$ (g/l)', fontsize=16, weight = 'bold')
# ax.xaxis.set_major_locator(md.MonthLocator(interval=6))
# #ax.xaxis.set_minor_locator(md.MonthLocator(interval=1))
# ax.xaxis.set_major_formatter(md.DateFormatter('%d/%m/%Y'))
# ax.set_xlim(Time_datetime_freq1[0], Time_datetime_freq1[-1])
ax[0].tick_params(axis='both', which='major', labelsize = 16)
ax[0].set_xticks([])
ax[0].set_ylim(0,1.4)


ax[1].plot(Time_datetime_freq2, C_sand_S2_D50_freq2_g_l, lw = 0, marker = '.', markersize = 1, 
        color = 'darkgoldenrod')

ax[1].text(0.03, 0.95, '1 MHz', fontsize = 16, transform = ax[1].transAxes, weight = 'bold') 
ax[1].set_xlabel('Time', fontsize=18, weight = 'bold')
fig.supylabel('$\mathregular{\overline{C_{sand, freq}}}$ (g/l)', fontsize=18, weight = 'bold')
ax[1].xaxis.set_major_locator(md.MonthLocator(interval=6))
#ax.xaxis.set_minor_locator(md.MonthLocator(interval=1))
ax[1].xaxis.set_major_formatter(md.DateFormatter('%d/%m/%Y'))
ax[1].set_xlim(Time_datetime_freq1[0], Time_datetime_freq1[-1])
ax[1].tick_params(axis='both', which='major', labelsize = 16)
ax[1].set_ylim(0,1.4)

fig.tight_layout()
figname = 'Time_Csand_single_freq'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 300, bbox_inches='tight')


#%%
s_time_freq2 = [C_sand_S2_D50_g_l[i]/C_fines_est_time_freq2[i] for i in range(len(C_sand_S2_D50_g_l))]
S_smaller_2 = sum(1 for x in s_time_freq2 if x <= 2)
perc_S_smaller_2 =  S_smaller_2/len(s_time_freq2)*100
S_larger_10 = sum(1 for x in s_time_freq2 if x > 10)
perc_S_larger_10 =  S_larger_10/len(s_time_freq2)*100
S_larger_20 = sum(1 for x in s_time_freq2 if x > 20)
perc_S_larger_20 =  S_larger_20/len(s_time_freq2)*100

    
#%% Time - S
fig, ax = plt.subplots(1, 1, figsize = (8,4), dpi=300)

ax.plot(Time_datetime_freq2, s_time_freq2, lw = 0, marker = '.', markersize = 1, 
        color = 'darkgoldenrod')

ax.set_xlabel('Time', fontsize=16, weight = 'bold')
ax.set_ylabel('S', fontsize=16, weight = 'bold')
ax.xaxis.set_major_locator(md.MonthLocator(interval=6))
#ax.xaxis.set_minor_locator(md.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(md.DateFormatter('%d/%m/%Y'))
ax.set_xlim(Time_datetime_freq1[0], Time_datetime_freq1[-1])
ax.tick_params(axis='both', which='major', labelsize = 14)
ax.set_ylim(0,10)

fig.tight_layout()
figname = 'Time_S_10'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 300, bbox_inches='tight')

    
#%% Csand - S
fig, ax = plt.subplots(1, 1, figsize = (6,6), dpi=300)

ax.plot(C_sand_S2_D50_g_l, s_time_freq2, lw = 0, marker = '.', markersize = 1, 
        color = 'darkgoldenrod')

ax.set_xlabel('$\mathregular{\overline{C_{sand}}}$ (g/l)', fontsize=16, weight = 'bold')
ax.set_ylabel('S', fontsize=16, weight = 'bold')
ax.tick_params(axis='both', which='major', labelsize = 14)
ax.set_xlim(0,3)
ax.set_ylim(0,50)

fig.tight_layout()
figname = 'Csand_S'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 300, bbox_inches='tight')

    #%% Hist S
fig, ax = plt.subplots(1, 1, figsize = (8, 6), dpi=300)

# freq1
ax.hist(s_time_freq2, bins = 100, density = True,
        color = 'darkgoldenrod')
ax.tick_params(axis='both', which='major', labelsize = 14)
ax.set_ylabel('Frequency', fontsize=16, weight = 'bold')
ax.set_xlim(0,20)

fig.tight_layout()
figname = 'Histogram_S'
fig.savefig(outpath_figures+ '\\' + figname +  '.png', dpi = 300, bbox_inches='tight')


#%% Calculate sigma ISCO gsd
# Calculate sigma 
sigma = [np.exp(np.nanstd(ISCO_GSD_data.iloc[i,16:-6]))/2 #VERIFY!!!! PROBABLY WRONG!
          for i in range(len(ISCO_GSD_data))]

indexx = ISCO_GSD_data.index
ll = [0]*len(ISCO_GSD_data)
for i in range(len(indexx)):
    ll[indexx[i]] = np.round(sigma[i],3)
ll = [i if i != 0 else None for i in ll]

ISCO_GSD_data['sigma_mum'] = ll


sigma = [np.exp(np.nanstd(ISCO_GSD_data_total.iloc[i,16:-6]))/2 #VERIFY!!!! PROBABLY WRONG!
          for i in range(len(ISCO_GSD_data_total))]

indexx = ISCO_GSD_data_total.index
ll = [0]*len(ISCO_GSD_data_total)
for i in range(len(indexx)):
    ll[indexx[i]] = np.round(sigma[i],3)
ll = [i if i != 0 else None for i in ll]

ISCO_GSD_data_total['sigma_mum'] = ll


#%% Plot Q - S

fig, ax = plt.subplots(1, 1, figsize = (8,6), dpi=300)

ax.plot(TAAPS_freq1['Q_sampling_m3_s'], TAAPS_freq1['S'],
                color = 'darkorange', markersize = 10, marker = 'D', ls = '', markeredgewidth = 0.3,
                markeredgecolor='black', zorder = 40, label = 'Sampler') 
ax.plot(ISCO_data['Q_sampling_m3_s'], ISCO_data['S'],
                color = 'yellowgreen', markersize = 5, marker = 'o', ls = '', markeredgewidth = 0.2,
                markeredgecolor='black', zorder = 0, label = 'ISCO')

ax.set_ylabel(r'S', fontsize=18, weight = 'bold')
ax.set_xlabel(r'Q (m³/s)', fontsize=18, weight = 'bold')
ax.tick_params(axis='both', which='major', labelsize = 16)
ax.set_xlim (0, 700)
ax.set_ylim (0, 25)
ax.legend(fontsize = 16, loc = 'upper right')

fig.tight_layout()
figname = 'Q_S'
fig.savefig(outpath_figures +'\\' +  figname + '.png', dpi = 300, bbox_inches='tight')


#%% Q - D50 sand

fig, ax = plt.subplots(1, 1, figsize=(7,6), dpi=100)

# Sand 
# ax.scatter(TAAPS_freq1['Q_sampling_m3_s'], TAAPS_freq1['D50_mum'],
#         marker = 'D', s = 60, c = TAAPS_freq1['S'], ec = 'black', lw = 0.5)

ax.plot(TAAPS_freq1['Q_sampling_m3_s'][TAAPS_freq1['S']< 2], TAAPS_freq1['D50_mum'][TAAPS_freq1['S']< 2],
        marker = 'D', markersize = 12, color = 'darkorange', markeredgecolor = 'black', markeredgewidth = 0.5, lw = 0)

ax.plot(TAAPS_freq1['Q_sampling_m3_s'][TAAPS_freq1['S']> 3], TAAPS_freq1['D50_mum'][TAAPS_freq1['S']> 3],
        marker = 'D', markersize = 12, color = 'grey', markeredgecolor = 'black', markeredgewidth = 0.5, lw = 0)


ax.hlines(D50_sand_ref_range_mum[0], 0, 500, lw = 1,ls = '--', color = 'black')
ax.hlines(D50_sand_ref_range_mum[1], 0, 500, lw = 1, ls = '--', color = 'black')
ax.hlines(200, 0, 500, lw = 1, color = 'black')

ax.set_xlabel('Q (m³/s)', fontsize=fontsize_axis, weight = 'bold')
ax.set_ylabel('$\mathregular{\overline{D_{50,sand}}}$ ($\mathregular{\mu}$m)', fontsize=fontsize_axis, weight = 'bold')
ax.tick_params(axis='both', which='major', labelsize = fontsize_ticks)
ax.set_xlim(0,500)
ax.set_ylim(0,350)

fig.tight_layout()
figname = 'Q_D50sand'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 300, bbox_inches='tight')
fig.savefig(outpath_figures + '\\' + figname + '.eps', dpi = 300, bbox_inches='tight')


#%% Csand - D50 sand

fig, ax = plt.subplots(1, 1, figsize=(7,6), dpi=100)

ax.plot(TAAPS_freq1['Sand_concentration_g_l'], TAAPS_freq1['D50_mum'],
        marker = 'D', markersize = 8, color = 'darkorange', markeredgecolor = 'black', markeredgewidth = 0.5, lw = 0, label = 'Sampler')

ax.plot(C_sand_S2_D50_g_l, D50_est_S2_D50,
        marker = '.', markersize = 2, color = 'burlywood', markeredgewidth = 0, lw = 0, zorder = 0)

ax.plot(C_sand_S2_D50_g_l[10], D50_est_S2_D50[10],
        marker = '.', markersize = 20, color = 'burlywood', markeredgewidth = 0, lw = 0, zorder = 0, label = 'TW16')

ax.set_xlabel('$\mathregular{\overline{C_{sand}}}$ (g/l)', fontsize=20, weight = 'bold')
ax.set_ylabel('$\mathregular{\overline{D_{50,sand}}}$ ($\mathregular{\mu}$m)', fontsize=20, weight = 'bold')
ax.tick_params(axis='both', which='major', labelsize = 18)
ax.set_xlim(0.01,2)
ax.set_ylim(63,600)
ax.set_xscale('log')
ax.legend(fontsize = 18, loc = 'upper right')

fig.tight_layout()
figname = 'Csand_D50sand'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 300, bbox_inches='tight')
fig.savefig(outpath_figures + '\\' + figname + '.eps', dpi = 300, bbox_inches='tight')

#%% find cell with highest FCB

idx_max_cell_FCB_freq1 = [np.argmax(FluidCorrBackscatter_freq1.iloc[i,:])
                          for i in range(len(FluidCorrBackscatter_freq1))]



#%% 1) Plots max cell
# cell
idx_max_freq1 = [np.argmax(FluidCorrBackscatter_freq1.iloc[i,:])
                 for i in range(len(FluidCorrBackscatter_freq1))]
idx_max_freq2 = [np.argmax(FluidCorrBackscatter_freq2.iloc[i,:])
                 for i in range(len(FluidCorrBackscatter_freq2))]
# Cell distance
max_cell_freq1 = [celldist_along_beam_freq1.iloc[idx_max_freq1[i],0]
            for i in range(len(idx_max_freq1))]
max_cell_freq2 = [celldist_along_beam_freq2.iloc[idx_max_freq2[i],0]
            for i in range(len(idx_max_freq2))]

# no cells with maximum
no_cells_max_freq1 = np.unique(idx_max_freq1)
no_cells_max_freq2 = np.unique(idx_max_freq2)

#
aa_df_freq1 = pd.DataFrame([idx_max_freq1, spm_time_freq1]).transpose()
aa_df_freq2 = pd.DataFrame([idx_max_freq2, spm_time_freq2]).transpose()

#%% Plots max cell
fig, ax = plt.subplots(3, 2, figsize = (12, 12), dpi=300)

#freq1 
ax[0,0].hist(max_cell_freq1, bins = len(no_cells_max_freq1), density = True,
        color = 'olive')
ax[0,0].set_xlim(celldist_along_beam_freq1.iloc[0,0],celldist_along_beam_freq1.iloc[-1,0] )
ax[0,0].tick_params(axis='both', which='major', labelsize = 16)
ax[0,0].set_ylabel('Frequency', fontsize=18, weight = 'bold')
ax[0,0].text(0.1,0.9, '(a)', fontsize = 18, transform = ax[0,0].transAxes)
ax[0,0].set_xticks([])

# freq2
ax[0,1].hist(max_cell_freq2, bins = len(no_cells_max_freq2), density = True,
        color = 'mediumorchid')
ax[0,1].set_xlim(celldist_along_beam_freq2.iloc[0,0],celldist_along_beam_freq2.iloc[-1,0])
ax[0,1].set_xticks([])
ax[0,1].tick_params(axis='both', which='major', labelsize = 16)
# ax[0,1].yaxis.set_label_position("right")
# ax[0,1].yaxis.tick_right()
ax[0,1].set_xticks([])
ax[0,1].text(0.1,0.9, '(b)', fontsize = 18, transform = ax[0,1].transAxes)

# stage
ax[1,0].plot(max_cell_freq1, stage_time_freq1, '.', ls = '', markersize = 6,
        color = 'olive')
ax[1,0].set_xlim(celldist_along_beam_freq1.iloc[0,0],celldist_along_beam_freq1.iloc[-1,0] )
ax[1,0].tick_params(axis='both', which='major', labelsize = 16)
ax[1,0].set_ylabel('Stage (m)', fontsize=18, weight = 'bold')
ax[1,0].set_ylim(0,)
ax[1,0].set_xticks([])
ax[1,0].text(0.1,0.9, '(c)', fontsize = 18, transform = ax[1,0].transAxes)

# stage
ax[1,1].plot(max_cell_freq2, stage_time_freq2, '.', ls = '',  markersize = 6,
        color = 'mediumorchid')
ax[1,1].set_xlim(celldist_along_beam_freq2.iloc[0,0],celldist_along_beam_freq2.iloc[-1,0] )
ax[1,1].tick_params(axis='both', which='major', labelsize = 16)
# ax[1,1].yaxis.set_label_position("right")
# ax[1,1].yaxis.tick_right()
ax[1, 1].set_ylim(0,)
ax[1,1].set_xticks([])
ax[1,1].text(0.1,0.9, '(d)', fontsize = 18, transform = ax[1,1].transAxes)


# spm
ax[2,0].plot(max_cell_freq1, spm_time_freq1, '.', ls = '', markersize = 6,
        color = 'olive')
ax[2,0].set_xlim(celldist_along_beam_freq1.iloc[0,0],celldist_along_beam_freq1.iloc[-1,0] )
ax[2,0].tick_params(axis='both', which='major', labelsize = 16)
ax[2,0].set_ylabel(r'$\mathregular{C_{tot}}$ (g/l)', fontsize=18, weight = 'bold')
ax[2,0].set_ylim(0.01,10)
ax[2,0].set_yscale('log')
ax[2,0].text(0.1,0.9, '(e)', fontsize = 18, transform = ax[2,0].transAxes)

# spm
ax[2,1].plot(max_cell_freq2, spm_time_freq2, '.', ls = '',  markersize = 6,
        color = 'mediumorchid')
ax[2,1].set_xlim(celldist_along_beam_freq2.iloc[0,0],celldist_along_beam_freq2.iloc[-1,0] )
ax[2,1].tick_params(axis='both', which='major', labelsize = 16)
# ax[2,1].yaxis.set_label_position("right")
# ax[2,1].yaxis.tick_right()
ax[2, 1].set_ylim(0.01,10)
ax[2, 1].set_yscale('log')
ax[2,1].text(0.1,0.9, '(f)', fontsize = 18, transform = ax[2,1].transAxes)


# fig.supylabel(r'Frequency', fontsize=18, weight = 'bold')
fig.supxlabel(r'Distance of maximum $\mathregular{B_F}$ from transducer along the beam (m)', fontsize=18, weight = 'bold')

fig.tight_layout()
figname = 'Plots_max_cell'
fig.savefig(outpath_figures+ '\\' + figname +  '.png', dpi = 300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight')
fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 300, bbox_inches='tight')

#%% Plot Fig5
fig, ax = plt.subplots(2, 2, figsize = (12,8), dpi=300)

# Freq1 - good
p2, = ax[0,0].plot(celldist_along_beam_freq1.iloc[:,0], FluidCorrBackscatter_freq1.iloc[5390,:],              
        ls = '-', lw = 2, markersize = 4, color = 'olive', zorder = 24, markeredgecolor = 'black', markeredgewidth = 0.1, 
        label = 'Pump')

p2, = ax[0,0].plot(celldist_along_beam_freq1.iloc[:,0], FluidCorrBackscatter_freq1.iloc[46348,:],              
        ls = '-', lw = 2, markersize = 4, color = 'olive', zorder = 34, markeredgecolor = 'black', markeredgewidth = 0.1, 
        label = 'Pump')

p2, = ax[0,0].plot(celldist_along_beam_freq1.iloc[:,0], FluidCorrBackscatter_freq1.iloc[32792,:],       #9692       
        ls = '-', lw = 2, markersize = 4, color = 'olive', zorder = 24, markeredgecolor = 'black', markeredgewidth = 0.1, 
        label = 'Pump')

p2, = ax[0,0].plot(celldist_along_beam_freq1.iloc[:,0], FluidCorrBackscatter_freq1.iloc[32938,:],               
        ls = '-', lw = 2, markersize = 4, color = 'olive', zorder = 24, markeredgecolor = 'black', markeredgewidth = 0.1, 
        label = 'Pump')

ax[0,0].text(0.02, 0.9, '(a)', transform = ax[0,0].transAxes, fontsize = 20)
ax[0,0].tick_params(axis='both', which='major', labelsize = 20)
ax[0,0].set_xlim(0,celldist_along_beam_freq1.iloc[-1,0])
ax[0,0].set_ylim(60,110)
ax[0,0].set_xticks([])

# Freq2 -good
p2, = ax[0,1].plot(celldist_along_beam_freq2.iloc[:,0], FluidCorrBackscatter_freq2.iloc[5786,:],              
        ls = '-', lw = 2, markersize = 4, color = 'mediumorchid', zorder = 24,markeredgecolor = 'black', markeredgewidth = 0.1, 
        label = 'Pump')

p2, = ax[0,1].plot(celldist_along_beam_freq2.iloc[:,0], FluidCorrBackscatter_freq2.iloc[62187,:],               # max spm
        ls = '-', lw = 2, markersize = 4, color = 'mediumorchid', zorder = 24,markeredgecolor = 'black', markeredgewidth = 0.1, 
        label = 'Pump')

p2, = ax[0,1].plot(celldist_along_beam_freq2.iloc[:,0], FluidCorrBackscatter_freq2.iloc[32722,:],              
        ls = '-', lw = 2, markersize = 4, color = 'mediumorchid', zorder = 24,markeredgecolor = 'black', markeredgewidth = 0.1, 
        label = 'Pump')

p2, = ax[0,1].plot(celldist_along_beam_freq2.iloc[:,0], FluidCorrBackscatter_freq2.iloc[17031,:],              
        ls = '-', lw = 2, markersize = 4, color = 'mediumorchid', zorder = 24,markeredgecolor = 'black', markeredgewidth = 0.1, 
        label = 'Pump')

ax[0,1].text(0.02, 0.9, '(b)', transform = ax[0,1].transAxes, fontsize = 20)
ax[0,1].tick_params(axis='both', which='major', labelsize = 20)
ax[0,1].set_xlim(0,celldist_along_beam_freq1.iloc[-1,0])
ax[0,1].set_ylim(35,70)
ax[0,1].set_xticks([])

# Freq1 - bad
p2, = ax[1,0].plot(celldist_along_beam_freq1.iloc[:,0], FluidCorrBackscatter_freq1.iloc[4762,:],            # min spm   23261
        ls = '-', lw = 2, markersize = 4, color = 'olive', zorder = 24,markeredgecolor = 'black', markeredgewidth = 0.1, 
        label = 'Pump')

p2, = ax[1,0].plot(celldist_along_beam_freq1.iloc[:,0], FluidCorrBackscatter_freq1.iloc[62513,:],             # 62513
        ls = '-', lw = 2, markersize = 4, color = 'olive', zorder = 24,markeredgecolor = 'black', markeredgewidth = 0.1, 
        label = 'Pump')

p2, = ax[1,0].plot(celldist_along_beam_freq1.iloc[:,0], FluidCorrBackscatter_freq1.iloc[64755,:],             #500
        ls = '-', lw = 2, markersize = 4, color = 'olive', zorder = 24,markeredgecolor = 'black', markeredgewidth = 0.1, 
        label = 'Pump')

p2, = ax[1,0].plot(celldist_along_beam_freq1.iloc[:,0], FluidCorrBackscatter_freq1.iloc[65061,:],           #32792   
        ls = '-', lw = 2, markersize = 4, color = 'olive', zorder = 24,markeredgecolor = 'black', markeredgewidth = 0.1, 
        label = 'Pump')

ax[1,0].text(0.02, 0.9, '(c)', transform = ax[1,0].transAxes, fontsize = 20)
ax[1,0].tick_params(axis='both', which='major', labelsize = 20)
ax[1,0].set_xlim(0,celldist_along_beam_freq1.iloc[-1,0])
ax[1,0].set_ylim(85,100)


# Freq2 - bad
p2, = ax[1,1].plot(celldist_along_beam_freq2.iloc[:,0], FluidCorrBackscatter_freq2.iloc[5390,:],              
        ls = '-', lw = 2, markersize = 4, color = 'mediumorchid', zorder = 24,markeredgecolor = 'black', markeredgewidth = 0.1, 
        label = 'Pump')

p2, = ax[1,1].plot(celldist_along_beam_freq2.iloc[:,0], FluidCorrBackscatter_freq2.iloc[11327,:],              
        ls = '-', lw = 2, markersize = 4, color = 'mediumorchid', zorder = 24,markeredgecolor = 'black', markeredgewidth = 0.1, 
        label = 'Pump')

p2, = ax[1,1].plot(celldist_along_beam_freq2.iloc[:,0], FluidCorrBackscatter_freq2.iloc[27662,:],              
        ls = '-', lw = 2, markersize = 4, color = 'mediumorchid', zorder = 24,markeredgecolor = 'black', markeredgewidth = 0.1, 
        label = 'Pump')
 
p2, = ax[1,1].plot(celldist_along_beam_freq2.iloc[:,0], FluidCorrBackscatter_freq2.iloc[57262,:],               
        ls = '-', lw = 2, markersize = 4, color = 'mediumorchid', zorder = 24,markeredgecolor = 'black', markeredgewidth = 0.1, 
        label = 'Pump')

ax[1,1].text(0.02, 0.9, '(d)', transform = ax[1,1].transAxes, fontsize = 20)
ax[1,1].tick_params(axis='both', which='major', labelsize = 20)
ax[1,1].set_xlim(0,celldist_along_beam_freq1.iloc[-1,0])
ax[1,1].set_ylim(55,75)

fig.supxlabel('Distance $\mathregular{r}$ from transducer (m)', fontsize=22, weight = 'bold')
fig.supylabel('Fluid-corrected backscatter $\mathregular{B_F}$ (dB)', fontsize=22, weight = 'bold')

# handles = [p1, p2, p3, p4 , p5, p6]
# #_, labels = ax.get_legend_handles_labels()
# fig.legend(handles = handles, #labels=labels, 
#           handler_map = {tuple: mpl.legend_handler.HandlerTuple(None)}, framealpha = 1, 
#           fontsize = 20, loc = 'lower center', ncol = 3, bbox_to_anchor = (0.5, -0.09))

fig.tight_layout()
figname = 'Fig5'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 300, bbox_inches='tight')
fig.savefig(outpath_figures + '\\' + figname + '.eps', dpi = 300, bbox_inches='tight')


#%% Histograms AlphaSed & BS - MANUSCRIPT

fig, ax = plt.subplots(2, 2, figsize = (16, 8), dpi=300)

# freq1 AlphaSed
ax[0,0].hist(AlphaSed_freq1, bins = 100,density = True, log = True,
        color = 'olive')

ax[0,0].text(0.02, 0.9, '(a)', fontsize = 16, transform = ax[0,0].transAxes) 
ax[0,0].set_xlim(0, )
ax[0,0].set_ylim(1e-5,)
ax[0,0].tick_params(axis='both', which='major', labelsize = 16)
ax[0,0].set_ylabel('Frequency', fontsize=18, weight = 'bold')

# freq1 BS
ax[0,1].hist(BeamAvBS_freq1, bins = 100, density = True,log = True,
       color = 'olive')
ax[0,1].text(0.02, 0.9, '(b)', fontsize = 16, transform = ax[0,1].transAxes) 
ax[0,1].set_xlim(84, )
ax[0,1].set_ylim(1e-5,)
ax[0,1].tick_params(axis='both', which='major', labelsize = 16)
ax[0,1].yaxis.set_label_position("right")
ax[0,1].yaxis.tick_right()
#ax[0,1].set_title('1 MHz', fontsize=14)
ax[0,1].set_ylabel('Frequency', fontsize=18, weight = 'bold')

# freq2 AlphaSed
ax[1,0].hist(AlphaSed_freq2, bins = 100, density = True, log = True,
        color = 'mediumorchid')

ax[1,0].text(0.02, 0.9, '(c)', fontsize = 16, transform = ax[1,0].transAxes) 
ax[1,0].set_xlim(0,)
ax[1,0].set_ylim(1e-5,)
ax[1,0].tick_params(axis='both', which='major', labelsize = 16)
ax[1,0].set_ylabel('Frequency', fontsize=18, weight = 'bold')

# freq2 BS
ax[1,1].hist(BeamAvBS_freq2, bins = 100, density = True, log = True,
       color = 'mediumorchid')

ax[1,1].text(0.02, 0.9, '(d)', fontsize = 16, transform = ax[1,1].transAxes) 
ax[1,1].tick_params(axis='both', which='major', labelsize = 16)
ax[1,1].yaxis.set_label_position("right")
ax[1,1].yaxis.tick_right()
ax[1,1].set_xlim(40,)
ax[1,1].set_ylim(1e-5,)
ax[1,1].set_ylabel('Frequency', fontsize=18, weight = 'bold')

ax[1,0].set_xlabel(r'$\mathregular{\alpha_{Sed}}$ (dB/m)', fontsize=18, weight = 'bold')
ax[1,1].set_xlabel(r'$\mathregular{\overline{B}}$ (dB)', fontsize=18, weight = 'bold')

fig.tight_layout()
figname = 'Histogram_AlphaSed_BS'
fig.savefig(outpath_figures+ '\\' + figname +  '.png', dpi = 250, bbox_inches='tight')
fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight') 
fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 250, bbox_inches='tight')

