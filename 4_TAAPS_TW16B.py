# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 15:56:57 2022

@author: jlaible
"""


# Load packages
import pickle
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
from Functions import user_input_path_freq, user_input_outpath, user_input_outpath_figures, user_input_path_data, user_input_path_data_samples

#%% 1. LOAD DATA
# Acoustic data
# Define input and output path
print('============== SELECT PATH ==============')
print('Select path folder acoustic data')
path_folder = user_input_path_freq()
print('Select path folder of concurrent samplings, missing and deleted data and station data')
path_data = user_input_path_data()
print('Select path folder of sampling data in separate files')
path_folder_samples = user_input_path_data_samples()
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


#%% Define input and output path
# out_path = r'C:\Users\jessica.laible\Documents\Acoustics\5_Zone_xs_method\Far_field'
# path_folder = r'C:\Users\jessica.laible\Documents\Acoustics\3_TW16\Results\9_TW16A_far_field'
# path_folder_samples = r'C:\Users\jessi\OneDrive\Dokumente\INRAE\Grenoble_Campus'

# Choose options
# Time-averaging
# 1h-window around temporal midpoint = True, all during sampling time = False
time_av = False
# B' correction
B_fines_correction = False
# cut at x = 1 higher frequency
cut_x1 = False

# Which plots and computations should be performed?
# Additional plots 
add_plots = False
supp_plots = False
event_plots = False # Plot events defined in events_dates
samples_plots = False # Plot samplings defined in 
discussion_plots = False # Perform discussion steps

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

# Load acoustic data (output from Beam-averaged attenuation and backscatter)
with open(path_folder+'\Beam_averaged_attenuation_backscatter_'+ str(freq1) + '.txt', "rb") as fp:   # Unpickling
    BeamAv_freq1 = pickle.load(fp)
with open(path_folder + '\Beam_averaged_attenuation_backscatter_'+ str(freq2) + '.txt', "rb") as fp:    
    BeamAv_freq2 = pickle.load(fp)
    
with open(path_folder+'\FluidCorrBackscatter_'+ str(freq1) + '.txt', "rb") as fp:   # Unpickling
    FluidCorrBackscatter_freq1 = pickle.load(fp)
with open(path_folder + '\FluidCorrBackscatter_'+ str(freq2) + '.txt', "rb") as fp:    
    FluidCorrBackscatter_freq2 = pickle.load(fp)

with open(path_folder+'\AveCount_db_'+ str(freq1) + '.txt', "rb") as fp:   # Unpickling
    AveCount_db_freq1 = pickle.load(fp)
with open(path_folder + '\AveCount_db_'+ str(freq2) + '.txt', "rb") as fp:    
    AveCount_db_freq2 = pickle.load(fp)
    
with open(path_folder+'\Time_datetime_AveCount_db_'+ str(freq1) + '.txt', "rb") as fp:   # Unpickling
    Time_datetime_AveCount_db_freq1 = pickle.load(fp)
with open(path_folder + '\Time_datetime_AveCount_db_'+ str(freq2) + '.txt', "rb") as fp:    
    Time_datetime_AveCount_db_freq2 = pickle.load(fp)
    
with open(path_folder+'\CelldB_'+ str(freq1) + '.txt', "rb") as fp:   # Unpickling
    CelldB_freq1 = pickle.load(fp)
with open(path_folder + '\CelldB_'+ str(freq2) + '.txt', "rb") as fp:    
    CelldB_freq2 = pickle.load(fp)
    
with open(path_folder + '\Celldist_along_beam_'+ str(freq1) + '.txt', "rb") as fp:    
   celldist_along_beam_freq1 = pickle.load(fp)
with open(path_folder + '\Celldist_along_beam_'+ str(freq2) + '.txt', "rb") as fp:    
    celldist_along_beam_freq2 = pickle.load(fp)

samples = samples.drop(['Unnamed: 0'], axis = 1)
colnames_samples = list(samples.columns.values)
colnames_data_spm_data_raw = list(spm_data_raw.columns.values)
colnames_data_Q_data_raw = list(Q_data_raw.columns.values)
colnames_data_stage_data_raw = list(stage_data_raw.columns.values)

# Parameters plots 
cmap = plt.cm.get_cmap('turbo')
colors = cmap(np.linspace(0,1,len(samples)))

# Fontsizes
fontsize_axis = 18
fontsize_legend = 16
fontsize_legend_title = 18
fontsize_text = 16
fontsize_ticks = 16 


#%% Define reference properties
# Water kinematic viscosity
nu_0 = 0.73 * 1e-6
# rho
rho_sed = 2650

# Concentration
C_sand_ref_g_l = 0.2 # 0.1

# D50 - sand 
D50_sand_ref_mum = 200
D50_sand_ref_phi = -np.log2(D50_sand_ref_mum/1000) 

# D50 - fines
D50_fines_ref_mum = 1
D50_fines_ref_phi = -np.log2(D50_fines_ref_mum/1000)

# Sigma - sand 
sigma_sand_ref_mum = 0.59
sigma_sand_ref_phi = -np.log2(sigma_sand_ref_mum/1000)

# Sigma - fines 
sigma_fines_ref_mum = 1.4
sigma_fines_ref_phi = -np.log2(sigma_fines_ref_mum/1000)

# Reference ranges
# D50 - sand 
D50_sand_ref_range_phi = [D50_sand_ref_phi - 0.4, D50_sand_ref_phi + 0.4] 
D50_sand_ref_range_mum = [2**(-D50_sand_ref_range_phi[i])*1000 for i in range(len(D50_sand_ref_range_phi))]


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

#%% Prepare data
# Delete samples at bridge cross section (made using th BD)
samples = samples.drop(samples[samples['Sampler'] == 'BD'].index, inplace = False)
samples.reset_index(inplace=True)

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

# Csand,turbidity,calc
# Csand_turbidity_calc = [Csand_turbidity_calc_data['Csand_turbidity_calc'][i] for i in range(len(Csand_turbidity_calc_data))]
# Time_Csand_turbidity_calc_datetime = pd.to_datetime(list(Csand_turbidity_calc_data['Time']),format='%Y-%m-%d %H:%M:%S')

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

# Calculate log10 values
samples['log_sand'] = np.log10(samples['Sand_concentration_mg_l'])
samples['log_fine'] = np.log10(samples['Fine_concentration_mg_l'])
ISCO_data['log_sand'] = np.log10(ISCO_data['Sand_concentration_mg_l'])
ISCO_data['log_fine'] = np.log10(ISCO_data['Fine_concentration_mg_l'])

# Get AveCount_db at acoustic times
AveCount_db_freq1_time_freq1 = []
for i in range(len(celldist_along_beam_freq1)):
    AveCount_db_freq1_i = np.round(np.interp(Time_datetime_freq1_all, Time_datetime_AveCount_db_freq1, AveCount_db_freq1.iloc[:,i]),1)
    AveCount_db_freq1_time_freq1.append(AveCount_db_freq1_i)
AveCount_db_freq1_time_freq1 = pd.DataFrame(AveCount_db_freq1_time_freq1).transpose()
AveCount_db_freq2_time_freq2 = []
for i in range(len(celldist_along_beam_freq2)):
    AveCount_db_freq2_i = np.round(np.interp(Time_datetime_freq2_all, Time_datetime_AveCount_db_freq2, AveCount_db_freq2.iloc[:,i]),1)
    AveCount_db_freq2_time_freq2.append(AveCount_db_freq2_i)
AveCount_db_freq2_time_freq2 = pd.DataFrame(AveCount_db_freq2_time_freq2).transpose()


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
AveCount_db_cut_freq1 = AveCount_db_freq1_time_freq1.drop(AveCount_db_freq1_time_freq1.index[[idxx_delete_freq1]]).reset_index(drop = True)
Temperature_freq1 = pd.Series(Temperature_freq1).drop(pd.Series(Temperature_freq1).index[[idxx_delete_freq1]]).reset_index(drop = True).tolist()
FluidCorrBackscatter_freq1 = FluidCorrBackscatter_freq1.drop(FluidCorrBackscatter_freq1.index[idxx_delete_freq1])
CelldB_freq1 = CelldB_freq1.drop(CelldB_freq1.index[idxx_delete_freq1])

Time_datetime_freq2 = Time_datetime_freq2_all.delete([idxx_delete_freq2]) 
BeamAv_freq2 = BeamAv_freq2.drop(BeamAv_freq2.index[[idxx_delete_freq2]]).reset_index(drop = True)
BeamAvBS_freq2 = BeamAvBS_freq2.drop(BeamAvBS_freq2.index[[idxx_delete_freq2]],).reset_index(drop = True)
CelldBAve_freq2 = pd.Series(CelldBAve_freq2).drop(pd.Series(CelldBAve_freq2).index[[idxx_delete_freq2]]).reset_index(drop = True).tolist()
AlphaSed_freq2 = pd.Series(AlphaSed_freq2).drop(pd.Series(AlphaSed_freq2).index[[idxx_delete_freq2]]).reset_index(drop = True).tolist()
AlphaW_freq2 = pd.Series(AlphaW_freq2).drop(pd.Series(AlphaW_freq2).index[[idxx_delete_freq2]]).reset_index(drop = True).tolist()
AveCount_db_cut_freq2 = AveCount_db_freq2_time_freq2.drop(AveCount_db_freq2_time_freq2.index[[idxx_delete_freq2]]).reset_index(drop = True)
Temperature_freq2 = pd.Series(Temperature_freq2).drop(pd.Series(Temperature_freq2).index[[idxx_delete_freq2]]).reset_index(drop = True).tolist()
FluidCorrBackscatter_freq2 = FluidCorrBackscatter_freq2.drop(FluidCorrBackscatter_freq2.index[idxx_delete_freq2])
CelldB_freq2 = CelldB_freq2.drop(CelldB_freq2.index[idxx_delete_freq2])

Time_datetime_freq1_int = [int(Time_datetime_freq1[i].timestamp()) for i in range(len(Time_datetime_freq1))]
Time_datetime_freq2_int = [int(Time_datetime_freq2[i].timestamp()) for i in range(len(Time_datetime_freq2))]

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

Q_time_pump = np.interp(Time_pump_mid_datetime, Time_Q_datetime,Q)
Q_time_ISCO = np.interp(Time_ISCO_mid_datetime, Time_Q_datetime,Q)

#%% Load all HADCP_Export_data from sampling campaigns and interpolate on HADCP cells

# P6 
sampling_dates_P6 = pd.read_csv(path_data + '\Sampling_dates_P6_for_5m_buffer.csv', sep = ';')
datee_P6 = sampling_dates_P6.iloc[:,1]

sand_conc_freq1_P6 = []
sand_flux_freq1_P6 = []
fine_conc_freq1_P6 = []
fine_flux_freq1_P6 = []
fine_sand_ratio_freq1_P6 = []
sand_conc_freq2_P6 = []
sand_flux_freq2_P6 = []
fine_conc_freq2_P6 = []
fine_flux_freq2_P6 = []
fine_sand_ratio_freq2_P6 = []
for i in range(len(datee_P6)): 
    dataa = pd.read_csv(path_folder_samples + '\\' + str(sampling_dates_P6['Gauging'].iloc[i]) + '_Campus\\' + str(sampling_dates_P6['Gauging'].iloc[i]) + '_HADCP_export_P6.csv', sep = ';')
    dataa = dataa.transpose()
    dataa.columns = dataa.iloc[0]
    dataa = dataa.iloc[1:,:]
    dataa.insert(0, 'Sampling_date', datee_P6[i])
    #dataa['Distance'] = dataa.index
    dataa = dataa.reset_index(level = 0)
    
    # correct distance   
    distance1= [np.round(np.float(dataa['index'][i]) - np.float(dataa['index'][0]),3) for i in range(len(dataa))]
    
    # Get values 
    sand_conc_HADCP1 = [np.float(dataa['Sand_conc_HADCP'][i]) for i in range(len(dataa))]
    sand_flux_HADCP1 = [np.float(dataa['Sand_flux_HADCP'][i]) for i in range(len(dataa))]
    fine_conc_HADCP1 = [np.float(dataa['Fine_conc_HADCP'][i]) for i in range(len(dataa))]
    fine_flux_HADCP1 = [np.float(dataa['Fine_flux_HADCP'][i]) for i in range(len(dataa))]
    fine_sand_ratio_HADCP1 = [np.float(dataa['Fine_sand_ratio_HADCP'][i]) for i in range(len(dataa))]
    
    # Get values on HADCP grid
    # Freq 1
    sc_freq1_P6 = np.interp(celldist_along_beam_freq1, distance1, sand_conc_HADCP1)
    sf_freq1_P6 = np.interp(celldist_along_beam_freq1, distance1, sand_flux_HADCP1)
    fc_freq1_P6 = np.interp(celldist_along_beam_freq1, distance1, fine_conc_HADCP1)
    ff_freq1_P6 = np.interp(celldist_along_beam_freq1, distance1, fine_flux_HADCP1)
    fsr_freq1_P6 = np.interp(celldist_along_beam_freq1, distance1, fine_sand_ratio_HADCP1)
    
    # Freq 2
    sc_freq2_P6 = np.interp(celldist_along_beam_freq2, distance1, sand_conc_HADCP1)
    sf_freq2_P6 = np.interp(celldist_along_beam_freq2, distance1, sand_flux_HADCP1)
    fc_freq2_P6 = np.interp(celldist_along_beam_freq2, distance1, fine_conc_HADCP1)
    ff_freq2_P6 = np.interp(celldist_along_beam_freq2, distance1, fine_flux_HADCP1)
    fsr_freq2_P6 = np.interp(celldist_along_beam_freq2, distance1, fine_sand_ratio_HADCP1)
    
    # Append
    sand_conc_freq1_P6.append(sc_freq1_P6)
    sand_flux_freq1_P6.append(sf_freq1_P6)
    fine_conc_freq1_P6.append(fc_freq1_P6)
    fine_flux_freq1_P6.append(ff_freq1_P6)
    fine_sand_ratio_freq1_P6.append(fsr_freq1_P6)
    sand_conc_freq2_P6.append(sc_freq2_P6)
    sand_flux_freq2_P6.append(sf_freq2_P6)
    fine_conc_freq2_P6.append(fc_freq2_P6)
    fine_flux_freq2_P6.append(ff_freq2_P6)
    fine_sand_ratio_freq2_P6.append(fsr_freq2_P6)

sand_conc_freq1_P6 = pd.DataFrame(sand_conc_freq1_P6)
sand_flux_freq1_P6= pd.DataFrame(sand_flux_freq1_P6)
fine_conc_freq1_P6= pd.DataFrame(fine_conc_freq1_P6)
fine_flux_freq1_P6= pd.DataFrame(fine_flux_freq1_P6)
fine_sand_ratio_freq1_P6= pd.DataFrame(fine_sand_ratio_freq1_P6)
sand_conc_freq2_P6= pd.DataFrame(sand_conc_freq2_P6)
sand_flux_freq2_P6= pd.DataFrame(sand_flux_freq2_P6)
fine_conc_freq2_P6= pd.DataFrame(fine_conc_freq2_P6)
fine_flux_freq2_P6= pd.DataFrame(fine_flux_freq2_P6)
fine_sand_ratio_freq2_P6= pd.DataFrame(fine_sand_ratio_freq2_P6)


# Add info 
sand_conc_freq1_P6['Sampler'] = ['P6'] *len(sand_conc_freq1_P6)
sand_conc_freq1_P6['Sampling_date'] = datee_P6
sand_flux_freq1_P6['Sampler'] = ['P6'] *len(sand_flux_freq1_P6)
sand_flux_freq1_P6['Sampling_date'] = datee_P6
fine_conc_freq1_P6['Sampler'] = ['P6'] *len(fine_conc_freq1_P6)
fine_conc_freq1_P6['Sampling_date'] = datee_P6
fine_flux_freq1_P6['Sampler'] = ['P6'] *len(fine_flux_freq1_P6)
fine_flux_freq1_P6['Sampling_date'] = datee_P6
fine_sand_ratio_freq1_P6['Sampler'] = ['P6'] *len(fine_sand_ratio_freq1_P6)
fine_sand_ratio_freq1_P6['Sampling_date'] = datee_P6
sand_conc_freq2_P6['Sampler'] = ['P6'] *len(sand_conc_freq2_P6)
sand_conc_freq2_P6['Sampling_date'] = datee_P6
sand_flux_freq2_P6['Sampler'] = ['P6'] *len(sand_flux_freq2_P6)
sand_flux_freq2_P6['Sampling_date'] = datee_P6
fine_conc_freq2_P6['Sampler'] = ['P6'] *len(fine_conc_freq2_P6)
fine_conc_freq2_P6['Sampling_date'] = datee_P6
fine_flux_freq2_P6['Sampler'] = ['P6'] *len(fine_flux_freq2_P6)
fine_flux_freq2_P6['Sampling_date'] = datee_P6
fine_sand_ratio_freq2_P6['Sampler'] = ['P6'] *len(fine_sand_ratio_freq2_P6)
fine_sand_ratio_freq2_P6['Sampling_date'] = datee_P6

#%% P72
sampling_dates_P72 = pd.read_csv(path_data+ '\Sampling_dates_P72.csv', sep = ';')
datee_P72 = sampling_dates_P72.iloc[:,1]

sand_conc_freq1_P72 = []
sand_flux_freq1_P72 = []
fine_conc_freq1_P72 = []
fine_flux_freq1_P72 = []
fine_sand_ratio_freq1_P72 = []
sand_conc_freq2_P72 = []
sand_flux_freq2_P72 = []
fine_conc_freq2_P72 = []
fine_flux_freq2_P72 = []
fine_sand_ratio_freq2_P72 = []
for i in range(len(datee_P72)): 
    dataa = pd.read_csv(path_folder_samples + '\\' + str(sampling_dates_P72['Gauging'].iloc[i]) + '_Campus\\' + str(sampling_dates_P72['Gauging'].iloc[i]) + '_HADCP_export_P72.csv', sep = ';')
    dataa.insert(1, 'Sampling_date', datee_P72[i])
    
    # correct distance
    distance1 = dataa.columns[2:].values
    distance1 = [float(i) for i in distance1]
    distance1= [distance1[i] - distance1[0] for i in range(len(distance1))]
    
    # Get values 
    sand_conc_HADCP1 = dataa.iloc[0,2:].tolist()
    sand_flux_HADCP1 = dataa.iloc[1,2:].tolist()
    fine_conc_HADCP1 = dataa.iloc[2,2:].tolist()
    fine_flux_HADCP1 = dataa.iloc[3,2:].tolist()
    fine_sand_ratio_HADCP1 = dataa.iloc[4,2:].tolist()
    
    # Get values on HADCP grid
    # Freq 1
    sc_freq1_P72 = np.interp(celldist_along_beam_freq1, distance1, sand_conc_HADCP1)
    sf_freq1_P72 = np.interp(celldist_along_beam_freq1, distance1, sand_flux_HADCP1)
    fc_freq1_P72 = np.interp(celldist_along_beam_freq1, distance1, fine_conc_HADCP1)
    ff_freq1_P72 = np.interp(celldist_along_beam_freq1, distance1, fine_flux_HADCP1)
    fsr_freq1_P72 = np.interp(celldist_along_beam_freq1, distance1, fine_sand_ratio_HADCP1)
    
    # Freq 2
    sc_freq2_P72 = np.interp(celldist_along_beam_freq2, distance1, sand_conc_HADCP1)
    sf_freq2_P72 = np.interp(celldist_along_beam_freq2, distance1, sand_flux_HADCP1)
    fc_freq2_P72 = np.interp(celldist_along_beam_freq2, distance1, fine_conc_HADCP1)
    ff_freq2_P72 = np.interp(celldist_along_beam_freq2, distance1, fine_flux_HADCP1)
    fsr_freq2_P72 = np.interp(celldist_along_beam_freq2, distance1, fine_sand_ratio_HADCP1)
    
    # Append
    sand_conc_freq1_P72.append(sc_freq1_P72)
    sand_flux_freq1_P72.append(sf_freq1_P72)
    fine_conc_freq1_P72.append(fc_freq1_P72)
    fine_flux_freq1_P72.append(ff_freq1_P72)
    fine_sand_ratio_freq1_P72.append(fsr_freq1_P72)
    sand_conc_freq2_P72.append(sc_freq2_P72)
    sand_flux_freq2_P72.append(sf_freq2_P72)
    fine_conc_freq2_P72.append(fc_freq2_P72)
    fine_flux_freq2_P72.append(ff_freq2_P72)
    fine_sand_ratio_freq2_P72.append(fsr_freq2_P72)

sand_conc_freq1_P72 = pd.DataFrame(sand_conc_freq1_P72)
sand_flux_freq1_P72= pd.DataFrame(sand_flux_freq1_P72)
fine_conc_freq1_P72= pd.DataFrame(fine_conc_freq1_P72)
fine_flux_freq1_P72= pd.DataFrame(fine_flux_freq1_P72)
fine_sand_ratio_freq1_P72= pd.DataFrame(fine_sand_ratio_freq1_P72)
sand_conc_freq2_P72= pd.DataFrame(sand_conc_freq2_P72)
sand_flux_freq2_P72= pd.DataFrame(sand_flux_freq2_P72)
fine_conc_freq2_P72= pd.DataFrame(fine_conc_freq2_P72)
fine_flux_freq2_P72= pd.DataFrame(fine_flux_freq2_P72)
fine_sand_ratio_freq2_P72= pd.DataFrame(fine_sand_ratio_freq2_P72)

# Add info 
sand_conc_freq1_P72['Sampler'] = ['P72'] *len(sand_conc_freq1_P72)
sand_conc_freq1_P72['Sampling_date'] = datee_P72
sand_flux_freq1_P72['Sampler'] = ['P72'] *len(sand_flux_freq1_P72)
sand_flux_freq1_P72['Sampling_date'] = datee_P72
fine_conc_freq1_P72['Sampler'] = ['P72'] *len(fine_conc_freq1_P72)
fine_conc_freq1_P72['Sampling_date'] = datee_P72
fine_flux_freq1_P72['Sampler'] = ['P72'] *len(fine_flux_freq1_P72)
fine_flux_freq1_P72['Sampling_date'] = datee_P72
fine_sand_ratio_freq1_P72['Sampler'] = ['P72'] *len(fine_sand_ratio_freq1_P72)
fine_sand_ratio_freq1_P72['Sampling_date'] = datee_P72
sand_conc_freq2_P72['Sampler'] = ['P72'] *len(sand_conc_freq2_P72)
sand_conc_freq2_P72['Sampling_date'] = datee_P72
sand_flux_freq2_P72['Sampler'] = ['P72'] *len(sand_flux_freq2_P72)
sand_flux_freq2_P72['Sampling_date'] = datee_P72
fine_conc_freq2_P72['Sampler'] = ['P72'] *len(fine_conc_freq2_P72)
fine_conc_freq2_P72['Sampling_date'] = datee_P72
fine_flux_freq2_P72['Sampler'] = ['P72'] *len(fine_flux_freq2_P72)
fine_flux_freq2_P72['Sampling_date'] = datee_P72
fine_sand_ratio_freq2_P72['Sampler'] = ['P72'] *len(fine_sand_ratio_freq2_P72)
fine_sand_ratio_freq2_P72['Sampling_date'] = datee_P72

#%% BD
sampling_dates_BD = pd.read_csv(path_data + '\Sampling_dates_BD_SDC.csv', sep = ';')
datee_BD = sampling_dates_BD.iloc[:,1]

sand_conc_freq1_BD = []
sand_flux_freq1_BD = []
fine_conc_freq1_BD = []
fine_flux_freq1_BD = []
fine_sand_ratio_freq1_BD = []
sand_conc_freq2_BD = []
sand_flux_freq2_BD = []
fine_conc_freq2_BD = []
fine_flux_freq2_BD = []
fine_sand_ratio_freq2_BD = []
for i in range(len(datee_BD)): 
    dataa = pd.read_csv(path_folder_samples + '\\' + str(sampling_dates_BD['Sampling_dates_2'].iloc[i]) + '_Campus\\' + str(sampling_dates_BD['Sampling_dates_2'].iloc[i]) + '_HADCP_export_BD.csv', sep = ';')
    dataa.insert(1, 'Sampling_date', datee_BD[i])
    
    # correct distance
    distance1 = dataa.columns[2:].values
    distance1 = [float(i) for i in distance1]
    distance1= [distance1[i] - distance1[0] for i in range(len(distance1))]
    
    # Get values 
    sand_conc_HADCP1 = dataa.iloc[0,2:].tolist()
    sand_conc_HADCP1 = [np.float(sand_conc_HADCP1[i]) for i in range(len(sand_conc_HADCP1))]
    sand_flux_HADCP1 = dataa.iloc[1,2:].tolist()    
    sand_flux_HADCP1 = [np.float(sand_flux_HADCP1[i]) for i in range(len(sand_flux_HADCP1))]
    
    # Get values on HADCP grid
    # Freq 1
    sc_freq1_BD = np.interp(celldist_along_beam_freq1, distance1, sand_conc_HADCP1)
    sf_freq1_BD = np.interp(celldist_along_beam_freq1, distance1, sand_flux_HADCP1)
    # Freq 2
    sc_freq2_BD = np.interp(celldist_along_beam_freq2, distance1, sand_conc_HADCP1)
    sf_freq2_BD = np.interp(celldist_along_beam_freq2, distance1, sand_flux_HADCP1)
    
    # Append
    sand_conc_freq1_BD.append(sc_freq1_BD)
    sand_flux_freq1_BD.append(sf_freq1_BD)    
    sand_conc_freq2_BD.append(sc_freq2_BD)
    sand_flux_freq2_BD.append(sf_freq2_BD)

sand_conc_freq1_BD = pd.DataFrame(sand_conc_freq1_BD)
sand_flux_freq1_BD= pd.DataFrame(sand_flux_freq1_BD)
sand_conc_freq2_BD= pd.DataFrame(sand_conc_freq2_BD)
sand_flux_freq2_BD= pd.DataFrame(sand_flux_freq2_BD)

# Add info 
sand_conc_freq1_BD['Sampler'] = ['BD'] *len(sand_conc_freq1_BD)
sand_conc_freq1_BD['Sampling_date'] = datee_BD
sand_flux_freq1_BD['Sampler'] = ['BD'] *len(sand_flux_freq1_BD)
sand_flux_freq1_BD['Sampling_date'] = datee_BD
sand_conc_freq2_BD['Sampler'] = ['BD'] *len(sand_conc_freq2_BD)
sand_conc_freq2_BD['Sampling_date'] = datee_BD
sand_flux_freq2_BD['Sampler'] = ['BD'] *len(sand_flux_freq2_BD)
sand_flux_freq2_BD['Sampling_date'] = datee_BD

#%% All campaigns

# P6 and P72 combined
sand_conc_freq1 = pd.concat([sand_conc_freq1_P72, sand_conc_freq1_P6])
sand_flux_freq1 = pd.concat([sand_flux_freq1_P72, sand_flux_freq1_P6])
fine_conc_freq1 = pd.concat([fine_conc_freq1_P72, fine_conc_freq1_P6])
fine_flux_freq1 = pd.concat([fine_flux_freq1_P72, fine_flux_freq1_P6])
fine_sand_ratio_freq1 = pd.concat([fine_sand_ratio_freq1_P72, fine_sand_ratio_freq1_P6])
sand_conc_freq2 = pd.concat([sand_conc_freq2_P72, sand_conc_freq2_P6])
sand_flux_freq2 = pd.concat([sand_flux_freq2_P72, sand_flux_freq2_P6])
fine_conc_freq2 = pd.concat([fine_conc_freq2_P72, fine_conc_freq2_P6])
fine_flux_freq2 = pd.concat([fine_flux_freq2_P72, fine_flux_freq2_P6])
fine_sand_ratio_freq2 = pd.concat([fine_sand_ratio_freq2_P72, fine_sand_ratio_freq2_P6])

sampling_dates_P72_P6 = pd.concat([sampling_dates_P72, sampling_dates_P6])
sampling_dates_P72_P6 = sampling_dates_P72_P6.iloc[:,0:3]

# all combined
sand_conc_freq1_all = pd.concat([sand_conc_freq1_P72, sand_conc_freq1_P6, sand_conc_freq1_BD])
sand_flux_freq1_all = pd.concat([sand_flux_freq1_P72, sand_flux_freq1_P6, sand_flux_freq1_BD])
sand_conc_freq2_all = pd.concat([sand_conc_freq2_P72, sand_conc_freq2_P6, sand_conc_freq2_BD])
sand_flux_freq2_all = pd.concat([sand_flux_freq2_P72, sand_flux_freq2_P6, sand_flux_freq2_BD])

sampling_dates_all = pd.concat([sampling_dates_P72, sampling_dates_P6, sampling_dates_BD])
sampling_dates_all = sampling_dates_all.iloc[:,0:3]


#%%###################################################################################

# Along beam 

######################################################################################


#%% Determine mean concentrations and S along the beam for each suspended sediment measurement

# Freq1
mean_sand_conc_freq1 = [np.mean(sand_conc_freq1.iloc[i,:2]) for i in range(len(sand_conc_freq1))]
mean_sand_flux_freq1 = [np.mean(sand_flux_freq1.iloc[i,:-2]) for i in range(len(sand_flux_freq1))]
mean_fine_conc_freq1 = [np.mean(fine_conc_freq1.iloc[i,:-2]) for i in range(len(fine_conc_freq1))]
mean_fine_flux_freq1 = [np.mean(fine_flux_freq1.iloc[i,:-2]) for i in range(len(fine_flux_freq1))]
mean_fine_sand_ratio_freq1 = [np.mean(fine_sand_ratio_freq1.iloc[i,:-2]) for i in range(len(fine_sand_ratio_freq1))]

# Freq2
mean_sand_conc_freq2 = [np.mean(sand_conc_freq2.iloc[i,:-2]) for i in range(len(sand_conc_freq2))]
mean_sand_flux_freq2 = [np.mean(sand_flux_freq2.iloc[i,:-2]) for i in range(len(sand_flux_freq2))]
mean_fine_conc_freq2 = [np.mean(fine_conc_freq2.iloc[i,:-2]) for i in range(len(fine_conc_freq2))]
mean_fine_flux_freq2 = [np.mean(fine_flux_freq2.iloc[i,:-2]) for i in range(len(fine_flux_freq2))]
mean_fine_sand_ratio_freq2 = [np.mean(fine_sand_ratio_freq2.iloc[i,:-2]) for i in range(len(fine_sand_ratio_freq2))]

# Combine P6 and P72 data
samples_P6_P72 = samples[samples['Sampler'] != 'BD']
samples_P6_P72 = samples_P6_P72.sort_values(['Date', 'Start_sampling'])
# #!!!!!!!!!!!!!!!!! as 14.12.2022 not finished
# addrow= [np.nan]*123
# samples_P6_P72.loc[len(samples_P6_P72)] = addrow

# Create one df with means in ensonified volume
enson_volume_values = samples_P6_P72
enson_volume_values['Mean_enson_sand_conc'] = mean_sand_conc_freq1
enson_volume_values['Mean_enson_sand_flux'] = mean_sand_flux_freq1
enson_volume_values['Mean_enson_fine_conc'] = mean_fine_conc_freq1
enson_volume_values['Mean_enson_fine_flux'] = mean_fine_flux_freq1
enson_volume_values['Mean_enson_ratio'] = mean_fine_sand_ratio_freq1
enson_volume_values = enson_volume_values.iloc[:-1,:]

#%%###################################################################################

# Relate C with AlphaSed and B averaged over beam

######################################################################################

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

# Delete samples at bridge cross section (made using th BD)
TAAPS_freq1 = TAAPS_freq1.drop(TAAPS_freq1[TAAPS_freq1['Sampler'] == 'BD'].index, inplace = False)
TAAPS_freq1.reset_index(inplace=True)
TAAPS_freq2 = TAAPS_freq2.drop(TAAPS_freq2[TAAPS_freq2['Sampler'] == 'BD'].index, inplace = False)
TAAPS_freq2.reset_index(inplace=True)

#%% REGRESSION BETWEEN ALPHASED and CFINES ENS
# add spm
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


# Relation with ens
x_range_ens = np.linspace(0,5,100)
x = np.array(TAAPS_freq1['Alpha Sediment (dB/m)'])
y = np.array(mean_fine_conc_freq1)
x = x[:,np.newaxis]
slope_AlphaSed_Cfines_ens_freq1, _, _, _ = np.linalg.lstsq(x,y)
lin_model_AlphaSed_Cfines_ens_freq1 = [TAAPS_freq1['Alpha Sediment (dB/m)'][i]*slope_AlphaSed_Cfines_ens_freq1
                            for i in range(len(TAAPS_freq1['Alpha Sediment (dB/m)']))]
lin_model_AlphaSed_Cfines_ens_freq1_plot = x_range_ens*slope_AlphaSed_Cfines_ens_freq1
R2_AlphaSed_Cfines_ens_freq1 = r2_score(mean_fine_conc_freq1, lin_model_AlphaSed_Cfines_ens_freq1)

x = np.array(TAAPS_freq2['Alpha Sediment (dB/m)'])
y = np.array(mean_fine_conc_freq2)
x = x[:,np.newaxis]
slope_AlphaSed_Cfines_ens_freq2, _, _, _ = np.linalg.lstsq(x,y)
lin_model_AlphaSed_Cfines_ens_freq2 = [TAAPS_freq2['Alpha Sediment (dB/m)'][i]*slope_AlphaSed_Cfines_ens_freq2
                            for i in range(len(TAAPS_freq2['Alpha Sediment (dB/m)']))]
lin_model_AlphaSed_Cfines_ens_freq2_plot = x_range_ens*slope_AlphaSed_Cfines_ens_freq2
R2_AlphaSed_Cfines_ens_freq2 = r2_score(mean_fine_conc_freq2, lin_model_AlphaSed_Cfines_ens_freq2)


#%% Create AlphaSed - Cfines calibration dataset

TAAPS_freq1_fines = TAAPS_freq1.drop(TAAPS_freq1[TAAPS_freq1['Sampler'] == 'BD'].index, inplace = False)
TAAPS_freq1_fines.reset_index(drop = True, inplace = True)
TAAPS_freq2_fines = TAAPS_freq2.drop(TAAPS_freq2[TAAPS_freq2['Sampler'] == 'BD'].index, inplace = False)
TAAPS_freq2_fines.reset_index(drop = True, inplace = True)


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

#%% REGRESSION BETWEEN B and Csand ENS
mean_sand_conc_freq1_mg_l = [mean_sand_conc_freq1[i]*1000 for i in range(len(mean_sand_conc_freq1))]
mean_sand_conc_freq2_mg_l = [mean_sand_conc_freq2[i]*1000 for i in range(len(mean_sand_conc_freq2))]

x_range = np.linspace(0,150,100)
interp_B_Csand_ens_freq1 = np.polyfit(TAAPS_freq1['Beam-Averaged Backscatter (dB)'], np.log10(mean_sand_conc_freq1_mg_l), 1)
lin_model_B_Csand_ens_freq1 = [TAAPS_freq1['Beam-Averaged Backscatter (dB)'][i]*interp_B_Csand_ens_freq1[0]+interp_B_Csand_ens_freq1[1]
                            for i in range(len(TAAPS_freq1['Beam-Averaged Backscatter (dB)']))]
lin_model_B_Csand_ens_freq1_plot = [x_range[i]*interp_B_Csand_ens_freq1[0]+interp_B_Csand_ens_freq1[1]
                            for i in range(len(x_range))]
R2_B_Csand_ens_freq1 = r2_score(np.log10(mean_sand_conc_freq1_mg_l), lin_model_B_Csand_ens_freq1)

interp_B_Csand_ens_freq2 = np.polyfit(TAAPS_freq2['Beam-Averaged Backscatter (dB)'], np.log10(mean_sand_conc_freq2_mg_l), 1)
lin_model_B_Csand_ens_freq2 = [TAAPS_freq2['Beam-Averaged Backscatter (dB)'][i]*interp_B_Csand_ens_freq2[0]+interp_B_Csand_ens_freq2[1]
                            for i in range(len(TAAPS_freq2['Beam-Averaged Backscatter (dB)']))]
lin_model_B_Csand_ens_freq2_plot = [x_range[i]*interp_B_Csand_ens_freq2[0]+interp_B_Csand_ens_freq2[1]
                            for i in range(len(x_range))]
R2_B_Csand_ens_freq2 = r2_score(np.log10(mean_sand_conc_freq2_mg_l), lin_model_B_Csand_ens_freq2)

#%% Perform regression only on measurements in S and D50 range
# D50 - sand 
D50_sand_ref_mum = 200
D50_sand_ref_phi = -np.log2(D50_sand_ref_mum/1000) 

D50_sand_ref_range_phi = [D50_sand_ref_phi - 0.4, D50_sand_ref_phi + 0.4] 
D50_sand_ref_range_mum = [2**(-D50_sand_ref_range_phi[i])*1000 for i in range(len(D50_sand_ref_range_phi))]


# Only perform BBC on samples where S < 2
TAAPS_freq1_S2 = TAAPS_freq1[TAAPS_freq1['S'] < 3]
TAAPS_freq2_S2 = TAAPS_freq2[TAAPS_freq2['S'] < 3]
TAAPS_freq1_S2.reset_index(inplace=False)
TAAPS_freq2_S2.reset_index(inplace=False)
       
#-------------------------------------------------------------------------------------
# Only perform BBC on samples where S < 2 and D50_sand (phi) < 1/4 phi D50_sand_ref
TAAPS_freq1_S2_D50 = TAAPS_freq1_S2[TAAPS_freq1_S2['D50_mum'].le(D50_sand_ref_range_mum[0]) & TAAPS_freq1_S2['D50_mum'].ge(D50_sand_ref_range_mum[1])]
TAAPS_freq2_S2_D50 = TAAPS_freq2_S2[TAAPS_freq2_S2['D50_mum'].le(D50_sand_ref_range_mum[0]) & TAAPS_freq2_S2['D50_mum'].ge(D50_sand_ref_range_mum[1])]
TAAPS_freq1_S2_D50.reset_index(inplace=True)
TAAPS_freq2_S2_D50.reset_index(inplace=True)

# Used rows
idx_used_S2_D50 = TAAPS_freq2_S2_D50['index'].tolist()

# Apply to ens data
mean_sand_conc_freq1_S2_D50_mg_l = [mean_sand_conc_freq1_mg_l[idx_used_S2_D50[i]] 
                                    for i in range(len(idx_used_S2_D50))]
mean_sand_conc_freq2_S2_D50_mg_l = [mean_sand_conc_freq2_mg_l[idx_used_S2_D50[i]] 
                                    for i in range(len(idx_used_S2_D50))]

#%% Regression B with ens data S2_D50
x_range = np.linspace(0,150,100)
interp_B_Csand_ens_freq1_S2_D50 = np.polyfit(TAAPS_freq1_S2_D50['Beam-Averaged Backscatter (dB)'], np.log10(mean_sand_conc_freq1_S2_D50_mg_l), 1)
lin_model_B_Csand_ens_freq1_S2_D50 = [TAAPS_freq1_S2_D50['Beam-Averaged Backscatter (dB)'][i]*interp_B_Csand_ens_freq1_S2_D50[0]+interp_B_Csand_ens_freq1_S2_D50[1]
                            for i in range(len(TAAPS_freq1_S2_D50['Beam-Averaged Backscatter (dB)']))]
lin_model_B_Csand_ens_freq1_S2_D50_plot = [x_range[i]*interp_B_Csand_ens_freq1_S2_D50[0]+interp_B_Csand_ens_freq1_S2_D50[1]
                            for i in range(len(x_range))]
R2_B_Csand_ens_freq1_S2_D50 = r2_score(np.log10(mean_sand_conc_freq1_S2_D50_mg_l), lin_model_B_Csand_ens_freq1_S2_D50)

interp_B_Csand_ens_freq2_S2_D50 = np.polyfit(TAAPS_freq2_S2_D50['Beam-Averaged Backscatter (dB)'], np.log10(mean_sand_conc_freq2_S2_D50_mg_l), 1)
lin_model_B_Csand_ens_freq2_S2_D50 = [TAAPS_freq2_S2_D50['Beam-Averaged Backscatter (dB)'][i]*interp_B_Csand_ens_freq2_S2_D50[0]+interp_B_Csand_ens_freq2_S2_D50[1]
                            for i in range(len(TAAPS_freq2_S2_D50['Beam-Averaged Backscatter (dB)']))]
lin_model_B_Csand_ens_freq2_S2_D50_plot = [x_range[i]*interp_B_Csand_ens_freq2_S2_D50[0]+interp_B_Csand_ens_freq2_S2_D50[1]
                            for i in range(len(x_range))]
R2_B_Csand_ens_freq2_S2_D50 = r2_score(np.log10(mean_sand_conc_freq2_S2_D50_mg_l), lin_model_B_Csand_ens_freq2_S2_D50)


# Regression on samples with S < 2 and D50 range
K1_emp_freq1_S2_D50 = interp_B_Csand_ens_freq1_S2_D50[1]
K1_emp_freq2_S2_D50 = interp_B_Csand_ens_freq2_S2_D50[1]

# Empirical determination of K2
# Regression on samples with S < 2 and D50 range
K2_emp_freq1_S2_D50 = interp_B_Csand_ens_freq1_S2_D50[0]
K2_emp_freq2_S2_D50 = interp_B_Csand_ens_freq2_S2_D50[0]





#%%###################################################################################

# Relate ens - xs

######################################################################################


#%% Determine regressions xs - enson
# R2
corr_enson = enson_volume_values.corr()
r2_enson = corr_enson**2

# Q_xs - sand C ensonified
x_range_Q = np.linspace(0,700, 1000)
interp_Q_sand_C = np.polyfit(enson_volume_values['Q_sampling_m3_s'], enson_volume_values['Mean_enson_sand_conc'], 1)
lin_model_Q_sand_C = [x_range_Q[i]*interp_Q_sand_C[0]+interp_Q_sand_C[1]
                           for i in range(len(x_range_Q))]

# stage_xs - sand C ensonified
x_range_stage = np.linspace(0,5, 1000)
interp_stage_sand_C = np.polyfit(enson_volume_values['Stage_sampling_m'], enson_volume_values['Mean_enson_sand_conc'], 1)
lin_model_stage_sand_C = [x_range_stage[i]*interp_stage_sand_C[0]+interp_stage_sand_C[1]
                           for i in range(len(x_range_stage))]

# sand conc_xs - sand C ensonified
x_range = np.linspace(0,3,100)
x = np.array(enson_volume_values['Sand_concentration_g_l'])
y = np.array(enson_volume_values['Mean_enson_sand_conc'])
x = x[:,np.newaxis]
slope_xs_ens_sand, _, _, _ = np.linalg.lstsq(x, y)
lin_model_xs_ens_sand = [enson_volume_values['Sand_concentration_g_l'][i]*slope_xs_ens_sand
                            for i in range(len(enson_volume_values['Sand_concentration_g_l']))]
lin_model_xs_ens_sand_plot = [x_range[i]*slope_xs_ens_sand
                            for i in range(len(x_range))]
R2_time_xs_ens_sand = r2_score(enson_volume_values['Mean_enson_sand_conc'], lin_model_xs_ens_sand)

# fine conc_xs - fine C ensonified
x = np.array(enson_volume_values['Fine_concentration_g_l'])
y = np.array(enson_volume_values['Mean_enson_fine_conc'])
x = x[:,np.newaxis]
slope_xs_ens_fine, _, _, _ = np.linalg.lstsq(x, y)
lin_model_xs_ens_fine = [enson_volume_values['Fine_concentration_g_l'][i]*slope_xs_ens_fine
                            for i in range(len(enson_volume_values['Fine_concentration_g_l']))]
lin_model_xs_ens_fine_plot = [x_range[i]*slope_xs_ens_fine
                            for i in range(len(x_range))]
R2_time_xs_ens_fine = r2_score(enson_volume_values['Mean_enson_fine_conc'], lin_model_xs_ens_fine)

#%% Determine regressions ens - xs 

# Csand ens - xs
x_range = np.linspace(0,3,100)
x = np.array(enson_volume_values['Mean_enson_sand_conc'])
y = np.array(enson_volume_values['Sand_concentration_g_l'])
x = x[:,np.newaxis]
slope_ens_xs_sand, _, _, _ = np.linalg.lstsq(x, y)
lin_model_ens_xs_sand = [enson_volume_values['Sand_concentration_g_l'][i]*slope_ens_xs_sand
                            for i in range(len(enson_volume_values['Sand_concentration_g_l']))]
lin_model_ens_xs_sand_plot = [x_range[i]*slope_ens_xs_sand
                            for i in range(len(x_range))]
R2_time_ens_xs_sand = r2_score(enson_volume_values['Sand_concentration_g_l'], lin_model_ens_xs_sand)

# Cfines ens - xs
x = np.array(enson_volume_values['Mean_enson_fine_conc'])
y = np.array(enson_volume_values['Fine_concentration_g_l'])
x = x[:,np.newaxis]
slope_ens_xs_fine, _, _, _ = np.linalg.lstsq(x, y)
lin_model_ens_xs_fine = [enson_volume_values['Fine_concentration_g_l'][i]*slope_ens_xs_fine
                            for i in range(len(enson_volume_values['Fine_concentration_g_l']))]
lin_model_ens_xs_fine_plot = [x_range[i]*slope_ens_xs_fine
                            for i in range(len(x_range))]
R2_time_ens_xs_fine = r2_score(enson_volume_values['Fine_concentration_g_l'], lin_model_ens_xs_fine)

#%% ###################################################################################

# Calculate Csand and Cfines xs

#%%#####################################################################################

# Determine Csand_ens
Csand_ens_est_freq1 = [10**(BeamAvBS_freq1[i] * interp_B_Csand_ens_freq1_S2_D50[0]+ interp_B_Csand_ens_freq1_S2_D50[1])/1000
                       for i in range(len(BeamAvBS_freq1))]
Csand_ens_est_freq2 = [10**(BeamAvBS_freq2[i] * interp_B_Csand_ens_freq2_S2_D50[0]+ interp_B_Csand_ens_freq2_S2_D50[1])/1000
                       for i in range(len(BeamAvBS_freq2))]

# Determine Csand_xs
Csand_est_freq1 = [Csand_ens_est_freq1[i] * slope_ens_xs_sand[0]
                       for i in range(len(Csand_ens_est_freq1))]
Csand_est_freq2 = [Csand_ens_est_freq2[i] * slope_ens_xs_sand[0]
                       for i in range(len(Csand_ens_est_freq2))]

# Determine Cfines 
C_fines_est_freq1 = [slope_AlphaSed_Cfines_ens_freq1[0]*AlphaSed_freq1[i]
                     for i in range(len(AlphaSed_freq1))]
C_fines_est_freq2 = [slope_AlphaSed_Cfines_ens_freq2[0]* AlphaSed_freq2[i] 
                     for i in range(len(AlphaSed_freq2))]

C_fines_est_freq1 = [slope_ens_xs_fine[0]*C_fines_est_freq1[i]
                     for i in range(len(AlphaSed_freq1))]
C_fines_est_freq2 = [slope_ens_xs_fine[0]* C_fines_est_freq2[i] 
                     for i in range(len(AlphaSed_freq2))]
C_fines_est_freq1_time_freq2 = np.interp(Time_datetime_freq2, Time_datetime_freq1, C_fines_est_freq1)
C_fines_est_time_freq2 = [(C_fines_est_freq2[i] + C_fines_est_freq1_time_freq2[i])/2
                          for i in range(len(C_fines_est_freq1_time_freq2))]


#%% Determine theoretical slope (alphaunit)

# using Richards viscous attenuation
h = 1/2
D50_fines_ref_Rich_mum = 2
sigma_fines_ref_Rich_mum = 1
ref_dist_fines_Rich_freq1 = compute_model_lognorm_spherical(D50_fines_ref_Rich_mum*1e-6, sigma_fines_ref_Rich_mum, freq1_Hz, h, rho_sed, nu_0)
ref_dist_fines_Rich_freq2 = compute_model_lognorm_spherical(D50_fines_ref_Rich_mum*1e-6, sigma_fines_ref_Rich_mum, freq2_Hz, h, rho_sed, nu_0)
zeta_Rich_fines_freq1 = ref_dist_fines_Rich_freq1.zeta_Rich
zeta_Rich_fines_freq2 = ref_dist_fines_Rich_freq2.zeta_Rich

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
alphaunit_Rich_freq1 = 1/(zeta_Rich_fines_freq1*20/(np.log(10)))
alphaunit_Rich_freq2 = 1/(zeta_Rich_fines_freq2*20/(np.log(10)))

lin_model_alphaSed1_Cfines_theo_Rich_plot = x_range*alphaunit_Rich_freq1
lin_model_alphaSed1_Cfines_theo_Rich = [AlphaSed_freq1_TAAPS_ISCO[i]*alphaunit_Rich_freq1
                        for i in range(len(AlphaSed_freq1_TAAPS_ISCO))]
R2_alphaSed1_Cfines_theo_Rich = r2_score(Cfines_TAAPS_ISCO, lin_model_alphaSed1_Cfines_theo_Rich)

lin_model_alphaSed2_Cfines_theo_Rich_plot = x_range*alphaunit_Rich_freq2
lin_model_alphaSed2_Cfines_theo_Rich = [AlphaSed_freq2_TAAPS_ISCO[i]*alphaunit_Rich_freq2
                        for i in range(len(AlphaSed_freq2_TAAPS_ISCO))]
R2_alphaSed2_Cfines_theo_Rich = r2_score(Cfines_TAAPS_ISCO, lin_model_alphaSed2_Cfines_theo_Rich)

#%%#############################################################################################

# RUTS - BASED DUAL FREQUENCY METHOD

##############################################################################################

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


#%% STEP10: Calculate the reference sand distribution, form factor, target strength TS, UTS and RUTS

# Compute sand reference distribution
# ref_dist_sand_freq1 = compute_model_lognorm_spherical(D50_sand_ref_mum*1e-6, sigma_sand_ref_mum, freq1_Hz, rho_sed, nu_0)
# ref_dist_sand_freq1_a_s = ref_dist_sand_freq1.a_sample
# ref_dist_sand_freq1_proba_vol = ref_dist_sand_freq1.proba_vol
# ref_dist_sand_freq1_cdf_vol = ref_dist_sand_freq1.cdf_vol

# # CALCULATE INTEGRATED FORM FACTOR F (THORNE & MERAL 2008) AND ZETA FOR REFERENCE DISTRIBUTIONS
# # Sand
# f_TM08_freq1_sand_ref = ref_dist_sand_freq1.f_TM08
# ref_dist_sand_freq2 = compute_model_lognorm_spherical(D50_sand_ref_mum*1e-6, sigma_sand_ref_mum, freq2_Hz, rho_sed, nu_0)
# f_TM08_freq2_sand_ref = ref_dist_sand_freq2.f_TM08

# # Compute fines reference distribution
# ref_dist_fines_freq1 = compute_model_lognorm_spherical(D50_fines_ref_mum*1e-6, sigma_fines_ref_mum, freq1_Hz, rho_sed, nu_0)
# ref_dist_fines_freq2 = compute_model_lognorm_spherical(D50_fines_ref_mum*1e-6, sigma_fines_ref_mum, freq2_Hz, rho_sed, nu_0)

# sv
TAAPS_freq1['s_v_sand'] = [TAAPS_freq1['f_int_sand'].iloc[i]**2 *(3*TAAPS_freq1['Sand_concentration_g_l'].iloc[i])/(16*math.pi*TAAPS_freq1['D50_mum'].iloc[i]*1e-6 * rho_sed*1**2)
             for i in range(len(TAAPS_freq1))] # in 1/m³, sizes (a_s) in m
TAAPS_freq2['s_v_sand']  = [TAAPS_freq2['f_int_sand'].iloc[i]**2 *(3*TAAPS_freq2['Sand_concentration_g_l'].iloc[i])/(16*math.pi*TAAPS_freq2['D50_mum'].iloc[i]*1e-6 * rho_sed*1**2)
             for i in range(len(TAAPS_freq2))] # in 1/m³, sizes (a_s) in m

# TS
TS_freq1 = 10*np.log10(TAAPS_freq1['s_v_sand']) + 10*np.log10(2*t_p_freq1*1500*math.pi*(0.96/(k_freq1 * a_T_freq1))**2) + 20*np.log10(celldist_along_beam_freq1[-1])
TS_freq2 = 10*np.log10(TAAPS_freq2['s_v_sand']) + 10*np.log10(2*t_p_freq2*1500*math.pi*(0.96/(k_freq2 * a_T_freq2))**2) + 20*np.log10(celldist_along_beam_freq2[-1])
TS_sand_ref_freq1 = 10*np.log10(C_sand_ref_g_l) + 10*np.log10(2*t_p_freq1*1500*math.pi*(0.96/(k_freq1 * a_T_freq1))**2) + 20*np.log10(celldist_along_beam_freq1[-1])
TS_sand_ref_freq2 = 10*np.log10(C_sand_ref_g_l) + 10*np.log10(2*t_p_freq2*1500*math.pi*(0.96/(k_freq2 * a_T_freq2))**2) + 20*np.log10(celldist_along_beam_freq2[-1])

# UTS SED
TAAPS_freq1['UTS_sand'] = [10*np.log10(TAAPS_freq1['f_int_sand'].iloc[i]**2 *3/(16*math.pi*TAAPS_freq1['D50_mum'].iloc[i]*1e-6 * rho_sed*1**2))
             for i in range(len(TAAPS_freq1))] # in dB, sizes (a_s) in m
TAAPS_freq2['UTS_sand']  = [10*np.log10(TAAPS_freq2['f_int_sand'].iloc[i]**2 *3/(16*math.pi*TAAPS_freq2['D50_mum'].iloc[i]*1e-6 * rho_sed*1**2))
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

#%% STEP 1: Calculate theoretical RUTS relation
# Define D50 range and sigma
D50_sand_range_mum = np.arange(63, 500, 5)
# Calculate <f>
f_TM08_freq1_sand_range = []
for i in range(len(D50_sand_range_mum)):
    ref_dist_sand_freq1_range1 = compute_model_lognorm_spherical(D50_sand_range_mum[i]*1e-6, sigma_sand_ref_mum, freq1_Hz, h, rho_sed, nu_0)
    ff1 = ref_dist_sand_freq1_range1.f_TM08        
    f_TM08_freq1_sand_range.append(ff1)
f_TM08_freq1_sand_range = pd.DataFrame(f_TM08_freq1_sand_range)  
      
f_TM08_freq2_sand_range = []
for i in range(len(D50_sand_range_mum)):
    ref_dist_sand_freq2_range1 = compute_model_lognorm_spherical(D50_sand_range_mum[i]*1e-6, sigma_sand_ref_mum, freq2_Hz, h, rho_sed, nu_0)
    ff1 = ref_dist_sand_freq2_range1.f_TM08        
    f_TM08_freq2_sand_range.append(ff1)
f_TM08_freq2_sand_range = pd.DataFrame(f_TM08_freq2_sand_range)  

# f_TM08_freq2_sed_range use only variation of D50, keeping sigma constant
# calculate UTS theoretical 
UTS_theo_freq1 = [10*np.log10(f_TM08_freq1_sand_range.iloc[i,0]**2 *3/(16*math.pi*D50_sand_range_mum[i]*1e-6 * rho_sed*1**2)) + UTS_beam_freq1
                  for i in range(len(f_TM08_freq1_sand_range))]
UTS_theo_freq2 = [10*np.log10(f_TM08_freq2_sand_range.iloc[i, 0]**2 *3/(16*math.pi*D50_sand_range_mum[i]*1e-6 * rho_sed*1**2)) + UTS_beam_freq2
                  for i in range(len(f_TM08_freq2_sand_range))]
UTS_SED_theo_freq1 = [10*np.log10(f_TM08_freq1_sand_range.iloc[i,0]**2 *3/(16*math.pi*D50_sand_range_mum[i]*1e-6 * rho_sed*1**2))
                  for i in range(len(f_TM08_freq1_sand_range))]
UTS_SED_theo_freq2 = [10*np.log10(f_TM08_freq2_sand_range.iloc[i, 0]**2 *3/(16*math.pi*D50_sand_range_mum[i]*1e-6 * rho_sed*1**2))
                  for i in range(len(f_TM08_freq2_sand_range))]

# Calculate theoretical RUTS
# using D50 sand ref and sigma G
RUTS_theo_freq1 = [UTS_theo_freq1[i] - UTS_ref_freq1 for i in range(len(UTS_theo_freq1))]
RUTS_theo_freq2 = [UTS_theo_freq2[i] - UTS_ref_freq2 for i in range(len(UTS_theo_freq2))]

#%% STEP 2: C_sand_freq2 as initial state
C_sand_S2_D50_freq1_time_freq2_g_l = np.interp(Time_datetime_freq2, Time_datetime_freq1, Csand_est_freq1)

#%% STEP 3A: # Determine meas, where C_sand_est_freq1 < / = / > C_sand_freq2_g_l
idx_C_freq1_smaller_C_freq2_S2_D50 = [i for i in range(len(Csand_est_freq2))
           if np.round(C_sand_S2_D50_freq1_time_freq2_g_l[i],3) < np.round(Csand_est_freq2[i],3)] 
            # D50 sand < D50 sand ref, log_C_sand_S2_D50_freq2_g_l is increased using the RUTS and BBC relations for the two frequencies
            
idx_C_freq1_equal_C_freq2_S2_D50 = [i for i in range(len(Csand_est_freq2))
           if np.round(C_sand_S2_D50_freq1_time_freq2_g_l[i],3) == np.round(Csand_est_freq2[i],3)] 
            # D50 sand = D50 sand ref
            
idx_C_freq1_greater_C_freq2_S2_D50 = [i for i in range(len(Csand_est_freq2))
           if np.round(C_sand_S2_D50_freq1_time_freq2_g_l[i],3) > np.round(Csand_est_freq2[i],3)] 
        # D50 sand > D50 sand ref, log_C_sand_S2_D50_freq2_g_l is reduced using the RUTS and BBC relations for the two frequencies


#%% STEP 3B: Determine effective beam-averaged BS for freq1
BeamAvBS_effective_1_freq1_S2_D50 = [(np.log10(C_sand_S2_D50_freq1_time_freq2_g_l[i]*1000) - K1_emp_freq1_S2_D50)/K2_emp_freq1_S2_D50
                            for i in range(len(C_sand_S2_D50_freq1_time_freq2_g_l))]
BeamAvBS_effective_2_freq1_S2_D50 = [(np.log10(Csand_est_freq2[i]*1000) - K1_emp_freq1_S2_D50)/K2_emp_freq1_S2_D50
                            for i in range(len(Csand_est_freq2))]
B_defect_freq1_S2_D50 = [BeamAvBS_effective_1_freq1_S2_D50[i] - BeamAvBS_effective_2_freq1_S2_D50[i]
                  for i in range(len(Csand_est_freq2))]

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
BeamAvBS_effective_freq2_S2_D50 = [(np.log10(Csand_est_freq2[i]*1000) - K1_emp_freq2_S2_D50)/K2_emp_freq2_S2_D50
                            for i in range(len(Csand_est_freq2))]

# Determine sand concentration
C_sand_S2_D50_g_l = [10**(K1_emp_freq2_S2_D50 + K2_emp_freq2_S2_D50*(BeamAvBS_effective_freq2_S2_D50[i] - B_defect_freq2_S2_D50[i]))/1000
                    for i in range(len(B_defect_freq2_S2_D50))]




# #%% STEP 10: Plot BBC
# print('============== PLOT BBC ==============')
# nb_couleurs = 11
# viridis = cm.get_cmap('RdBu_r', nb_couleurs)
# echelle_colorimetrique = viridis(np.linspace(0, 1, nb_couleurs))
# vecteur_blanc = np.array([1, 1, 1, 1])
# echelle_colorimetrique[5:6,:] = vecteur_blanc
# cmap = mpl_colors_ListedColormap(echelle_colorimetrique)
# cmap.set_under('black')
# cmap.set_over('saddlebrown')
# cbounds = [75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350]
# norm = mpl.colors.BoundaryNorm(cbounds, cmap.N)

# fig, ax = plt.subplots(1, 2, figsize = (14,6), dpi=300)

# # Freq1
# ax[0].plot(TAAPS_freq1['Beam-Averaged Backscatter (dB)'], TAAPS_freq1['log_sand'],
#             color = 'grey', markersize = 5, marker = 'o', ls = '', markeredgewidth = 0.3,
#            markeredgecolor='black', zorder = 0)
# sc = ax[0].scatter(TAAPS_freq1['Beam-Averaged Backscatter (dB)'], TAAPS_freq1['log_sand'],
#             c=TAAPS_freq1['D50_mum'], s= 60, marker = 'o',
#             cmap = cmap,norm=norm, edgecolor='black', linewidth=0.2, zorder = 20, label = 'not used')
# cax = ax[0].scatter(TAAPS_freq1_S2_D50['Beam-Averaged Backscatter (dB)'], TAAPS_freq1_S2_D50['log_sand'],
#             c=TAAPS_freq1_S2_D50['D50_mum'], s= 200, marker = 'D',
#             cmap = cmap,norm=norm, edgecolor='black', linewidth=0.4, zorder = 30, label = 'used')
# ax[0].plot(x_range, lin_model_TAAPS1_logsand_S2_D50, color = 'black')

# # cbar = fig.colorbar(cax, ax=ax, extend='both', ticks=[75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350])
# # cbar.ax.set_yticklabels(['75','100','125','150','175','200','225','250','275','300', '325', '350'],ha='right')
# # cbar.ax.yaxis.set_tick_params(pad=35)
# # cbar.set_label(r'$\mathregular{\overline{D_{50}} \; (\mu m)}$', labelpad= 10, fontsize = 16)

# ax[0].text(0.05, 0.95, 'a)', fontsize = 16, transform = ax[0].transAxes)
# ax[0].text(0.3, 0.95, ('y = ' + str(np.round(interp_TAAPS1_logsand_S2_D50[0],2)) + 'x + (' +
#                   str(np.round(interp_TAAPS1_logsand_S2_D50[1],2)) + ')'),color = 'black', fontsize = 16,
#        transform = ax[0].transAxes)
# ax[0].text(0.3, 0.89, ('R² = ' + str(np.round(R2_TAAPS1_logsand_S2_D50,3)) + ', n = ' +
#                   str(len(TAAPS_freq1_S3_D50))), color = 'black', fontsize = 16,
#        transform = ax[0].transAxes)

# ax[0].set_ylabel(r'$\mathregular{{{log}_{10}}}$ ($\mathregular{\overline{C_{sand}}}$) (mg/l)', fontsize=18, weight = 'bold')
# ax[0].tick_params(axis='both', which='major', labelsize = 16)
# ax[0].set_xlim (94, 110)
# ax[0].set_ylim(0, 3.5)

# # Freq2
# ax[1].plot(TAAPS_freq2['Beam-Averaged Backscatter (dB)'], TAAPS_freq2['log_sand'],
#             color = 'grey', markersize = 5, marker = 'o', ls = '', markeredgewidth = 0.2,
#            markeredgecolor='black', zorder = 0)
# ax[1].scatter(TAAPS_freq2['Beam-Averaged Backscatter (dB)'], TAAPS_freq2['log_sand'],
#             c=TAAPS_freq2['D50_mum'], s= 60, marker = 'o',
#             cmap = cmap,norm=norm, edgecolor='black', linewidth=0.2, zorder = 20)
# ax[1].scatter(TAAPS_freq2_S2_D50['Beam-Averaged Backscatter (dB)'], TAAPS_freq2_S2_D50['log_sand'],
#             c=TAAPS_freq2_S2_D50['D50_mum'], s= 200, marker = 'D',
#             cmap = cmap,norm=norm, edgecolor='black', linewidth=0.2, zorder = 30)
# ax[1].plot(x_range, lin_model_TAAPS3_logsand_S3_D50, color = 'black')

# cbar = fig.colorbar(cax, ax=ax, extend='both', ticks=[75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350])
# cbar.ax.set_yticklabels(['75','100','125','150','175','200','225','250','275','300', '325', '350'],ha='right')
# cbar.ax.yaxis.set_tick_params(pad=35, labelsize = 12)
# cbar.set_label(r'$\mathregular{\overline{D_{50}} \; (\mu m)}$', labelpad= 10, fontsize = 16)

# ax[1].text(0.05, 0.95, 'b)', fontsize = 16, transform = ax[1].transAxes)
# ax[1].text(0.3, 0.95,('y = ' + str(np.round(interp_TAAPS3_logsand_S3_D50[0],2)) + 'x + (' +
#                   str(np.round(interp_TAAPS3_logsand_S3_D50[1],2)) + ')'),color = 'black', fontsize = 16,
#        transform = ax[1].transAxes)
# ax[1].text(0.3, 0.89, ('R² = ' + str(np.round(R2_TAAPS3_logsand_S3_D50,3)) + ', n = ' +
#                   str(len(TAAPS_freq2_S2_D50))), color = 'black', fontsize = 16,
#        transform = ax[1].transAxes)

# ax[1].tick_params(axis='both', which='major', labelsize = 16)
# ax[1].set_xlim (56, 76)
# ax[1].set_ylim(0, 3.5)
# ax[1].yaxis.tick_right()

# ax[0].legend(fontsize = 16, loc = 'lower right')

# fig.supxlabel(r'$\mathregular{\overline{B}}$ (dB)', fontsize=18, weight = 'bold')

# # fig.tight_layout()
# figname = 'Base_backscatter_calibration'
# fig.savefig(outpath_figures +'\\' +  figname + '.png', dpi = 300, bbox_inches='tight')
# # fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight')
# # fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 300, bbox_inches='tight')

#%% ###################################################################################

# Validation

######################################################################################


#%% Get Csand during all samplings
TAAPS_all = pd.read_csv(path_data + '\Time_averaged_acoustics_physical_samples_400.csv', sep = ';')

# Samples
date_sample_list = [str(TAAPS_all['Date'][i]) for i in range(len(TAAPS_all))]
date_sample_datetime = [datetime.strptime(date_sample_list[i],'%Y-%m-%d').date()
               for i in range(len(TAAPS_all))]
start_time_sample_list = [str(TAAPS_all['Start_sampling'][i]) for i in range(len(TAAPS_all))]
start_time_sample_datetime = [datetime.strptime(start_time_sample_list[i],'%H:%M').time()
               for i in range(len(TAAPS_all))]
TAAPS_all_start_datetime = [datetime.combine(date_sample_datetime[i], start_time_sample_datetime[i])
                    for i in range(len(date_sample_datetime))]
end_time_sample_list = [str(TAAPS_all['End_sampling'][i]) for i in range(len(TAAPS_all))]
end_time_sample_datetime = [datetime.strptime(end_time_sample_list[i],'%H:%M').time()
               for i in range(len(TAAPS_all))]
TAAPS_all_end_datetime = [datetime.combine(date_sample_datetime[i], end_time_sample_datetime[i])
                    for i in range(len(date_sample_datetime))]

TAAPS_all_start_midpoint_datetime = TAAPS_all_start_datetime 
TAAPS_all_end_midpoint_datetime = TAAPS_all_end_datetime

# Find index of first and last acoustic data within sampling range
# Frequency 1 
ind_first_beam_av_freq1_all = []
for i in range(len(TAAPS_all_start_datetime)):
    lli = next(x[0] for x in enumerate(Time_datetime_freq1) if x[1] > TAAPS_all_start_midpoint_datetime[i])
    ind_first_beam_av_freq1_all.append(lli)
    
ind_last_beam_av_freq1_all = []
for i in range(len(TAAPS_all_start_datetime)):
    lli = next(x[0] for x in enumerate(Time_datetime_freq1) if TAAPS_all_end_midpoint_datetime[i] < x[1])
    lli = lli # -1 
    ind_last_beam_av_freq1_all.append(lli)

#-------------------------------------------------------------------------------------
# Frequency 2 
ind_first_beam_av_freq2_all = []
for i in range(len(TAAPS_all_start_datetime)):
    lli = next(x[0] for x in enumerate(Time_datetime_freq2) if x[1] > TAAPS_all_start_midpoint_datetime[i])
    ind_first_beam_av_freq2_all.append(lli)
    
ind_last_beam_av_freq2_all = []
for i in range(len(TAAPS_all_start_datetime)):
    lli = next(x[0] for x in enumerate(Time_datetime_freq2) if TAAPS_all_end_midpoint_datetime[i] < x[1])
    lli = lli # -1 
    ind_last_beam_av_freq2_all.append(lli)



#%% Prepare validation data         

# 1) Pumping validation data
# Find nearest acoustic measurement to pump sample
Csand_S2_D50_pump_valid = np.interp(Time_pump_mid_datetime, Time_datetime_freq2, C_sand_S2_D50_g_l)
Cfines_pump_valid = np.interp(Time_pump_mid_datetime, Time_datetime_freq2, C_fines_est_time_freq2)
Q_time_pump = np.interp(Time_pump_mid_datetime, Time_Q_datetime,Q)

# Find nearest 1freq acoustic measurement to pump sample
Csand_S2_D50_freq1_pump_valid = np.interp(Time_pump_mid_datetime, Time_datetime_freq1, Csand_est_freq1)
Csand_S2_D50_freq2_pump_valid = np.interp(Time_pump_mid_datetime, Time_datetime_freq2, Csand_est_freq2)


# 2) ISCO validation data
# Find nearest acoustic measurement to ISCO sample
Csand_S2_D50_ISCO_valid = np.interp(Time_ISCO_mid_datetime, Time_datetime_freq2, C_sand_S2_D50_g_l)
D50_S2_D50_ISCO_valid = np.interp(Time_ISCO_GSD_mid_datetime, Time_datetime_freq2, D50_est_S2_D50)
Cfines_ISCO_valid = np.interp(Time_ISCO_mid_datetime, Time_datetime_freq2, C_fines_est_time_freq2)
Q_time_ISCO = np.interp(Time_ISCO_mid_datetime, Time_Q_datetime,Q)                    
                                   
# Correct ISCO sand xx
# Csand_S2_ISCO_corr = [43.987*Csand_S2_ISCO_valid[i]**(3.7037) for i in range(len(Csand_S2_ISCO_valid))]

# Find nearest 1freq acoustic measurement to ISCO sample
Csand_S2_D50_freq1_ISCO_valid = np.interp(Time_ISCO_mid_datetime, Time_datetime_freq1, Csand_est_freq1)
Csand_S2_D50_freq2_ISCO_valid = np.interp(Time_ISCO_mid_datetime, Time_datetime_freq2, Csand_est_freq2)

# 3) Calculated Csand from turbidity 
# Csand_S2_D50_Csand_T_valid = np.interp(Time_Csand_turbidity_calc_datetime, Time_datetime_freq2, C_sand_S2_D50_g_l)
# Csand_S2_D50_freq1_Csand_T_valid = np.interp(Time_Csand_turbidity_calc_datetime,Time_datetime_freq1, Csand_est_freq1)
# Csand_S2_D50_freq2_Csand_T_valid = np.interp(Time_Csand_turbidity_calc_datetime,Time_datetime_freq2, Csand_est_freq2)

# other
# Get C sand est 1freq during sampling time
C_sand_S2_D50_freq1_samples_g_l = [np.nanmean(Csand_est_freq1[ind_first_beam_av_freq1_all[i]:ind_last_beam_av_freq1_all[i]+1])
                     for i in range(len(ind_first_beam_av_freq1_all))]
C_sand_S2_D50_freq2_samples_g_l = [np.nanmean(Csand_est_freq2[ind_first_beam_av_freq2_all[i]:ind_last_beam_av_freq2_all[i]+1])
                     for i in range(len(ind_first_beam_av_freq2_all))]      

# # Get D50_xs_sand during sampling time
D50_S2_D50_samples_g_l = [np.nanmean(D50_est_S2_D50[ind_first_beam_av_freq2_all[i]:ind_last_beam_av_freq2_all[i]+1])
                     for i in range(len(ind_first_beam_av_freq2_all))]

# Get C sand est during sampling time
C_sand_S2_D50_samples_g_l = [np.nanmean(C_sand_S2_D50_g_l[ind_first_beam_av_freq2_all[i]:ind_last_beam_av_freq2_all[i]+1])
                     for i in range(len(ind_first_beam_av_freq2_all))]

# Get Cfines est during sampling time
C_fines_samples_g_l = [np.nanmean(C_fines_est_time_freq2[ind_first_beam_av_freq2_all[i]:ind_last_beam_av_freq2_all[i]+1])
                     for i in range(len(ind_first_beam_av_freq2_all))]

#%% Validation Fines ##############


#%% Regress Cfines,ISCO and Cfines,meas with Cfines,HADCP
x_range_fines = np.linspace(0,20,10000)
# force through origin (and loglog plot adapted)
# meas
from scipy.optimize import curve_fit
x_range = np.logspace(-3, 1, base=10)
def myExpFunc(x, a, b):
    return a * np.power(x, b)

# x = [(TAAPS_freq1['Fine_concentration_g_l'][i]) for i in range(len(TAAPS_freq1['Fine_concentration_g_l'])) if idx[i] == True]
# y = [(C_fines_samples_g_l[i]) for i in range(len(C_fines_samples_g_l)) if idx[i] == True]
# popt_Cmeas_CHADCP_fines, pcov = curve_fit(myExpFunc, x, y)
# R2_Cmeas_CHADCP_fines = r2_score(y, myExpFunc(x, *popt_Cmeas_CHADCP_fines))

# x = [(ISCO_data['Fine_concentration_g_l'][i]) for i in range(len(ISCO_data['Fine_concentration_g_l'])) if idx[i] == True]
# y = [(Cfines_ISCO_valid[i]) for i in range(len(Cfines_ISCO_valid)) if idx[i] == True]
# popt_Cmeas_CISCO_fines, pcov = curve_fit(myExpFunc, x, y)
# R2_Cmeas_CISCO_fines = r2_score(y, myExpFunc(x, *popt_Cmeas_CISCO_fines))

# samples 
interp_Cmeas_CHADCP_fines = np.polyfit(np.log10(TAAPS_all['Fine_concentration_g_l']),np.log10(C_fines_samples_g_l), 1)
lin_model_Cmeas_CHADCP_fines = [10**(np.log10(TAAPS_all['Fine_concentration_g_l'][i])*interp_Cmeas_CHADCP_fines[0]+interp_Cmeas_CHADCP_fines[1])
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

#%% Validation Single-frequency ##############

#%% Regressions Single-frequency 

# Freq1 
interp_Cmeas_CHADCP_sand_freq1 = np.polyfit(np.log10(TAAPS_all['Sand_concentration_g_l']),np.log10(C_sand_S2_D50_freq1_samples_g_l), 1)
lin_model_Cmeas_CHADCP_sand_freq1 = [10**(np.log10(TAAPS_all['Sand_concentration_g_l'][i])*interp_Cmeas_CHADCP_sand_freq1[0]+interp_Cmeas_CHADCP_sand_freq1[1])
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

# interp_CT_CHADCP_sand_freq1 = np.polyfit(np.log10(Csand_turbidity_calc),np.log10(Csand_S2_D50_freq1_Csand_T_valid), 1)
# lin_model_CT_CHADCP_sand_freq1 = [10**(np.log10(Csand_turbidity_calc[i])*interp_CT_CHADCP_sand_freq1[0]+interp_CT_CHADCP_sand_freq1[1])
#                            for i in range(len(Csand_S2_D50_freq1_Csand_T_valid))]
# lin_model_CT_CHADCP_sand_freq1_plot = [10**(np.log10(x_range[i])*interp_CT_CHADCP_sand_freq1[0]+interp_CT_CHADCP_sand_freq1[1])
#                            for i in range(len(x_range))]
# R2_CT_CHADCP_sand_freq1 = r2_score(Csand_S2_D50_freq1_Csand_T_valid, lin_model_CT_CHADCP_sand_freq1)

# Freq2
interp_Cmeas_CHADCP_sand_freq2 = np.polyfit(np.log10(TAAPS_all['Sand_concentration_g_l']),np.log10(C_sand_S2_D50_freq2_samples_g_l), 1)
lin_model_Cmeas_CHADCP_sand_freq2 = [10**(np.log10(TAAPS_all['Sand_concentration_g_l'][i])*interp_Cmeas_CHADCP_sand_freq2[0]+interp_Cmeas_CHADCP_sand_freq2[1])
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

# interp_CT_CHADCP_sand_freq2 = np.polyfit(np.log10(Csand_turbidity_calc),np.log10(Csand_S2_D50_freq2_Csand_T_valid), 1)
# lin_model_CT_CHADCP_sand_freq2 = [10**(np.log10(Csand_turbidity_calc[i])*interp_CT_CHADCP_sand_freq2[0]+interp_CT_CHADCP_sand_freq2[1])
#                            for i in range(len(Csand_S2_D50_freq2_Csand_T_valid))]
# lin_model_CT_CHADCP_sand_freq2_plot = [10**(np.log10(x_range[i])*interp_CT_CHADCP_sand_freq2[0]+interp_CT_CHADCP_sand_freq2[1])
#                            for i in range(len(x_range))]
# R2_CT_CHADCP_sand_freq2 = r2_score(Csand_S2_D50_freq2_Csand_T_valid, lin_model_CT_CHADCP_sand_freq2)

#%% Validation Dual-frequency ##############

#%% Regressions Dual-frequency 
 
x_range_sand = np.linspace(0,10,10000)

interp_Cmeas_CHADCP_sand = np.polyfit(np.log10(TAAPS_all['Sand_concentration_g_l']),np.log10(C_sand_S2_D50_samples_g_l), 1)
lin_model_Cmeas_CHADCP_sand = [10**(np.log10(TAAPS_all['Sand_concentration_g_l'][i])*interp_Cmeas_CHADCP_sand[0]+interp_Cmeas_CHADCP_sand[1])
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

# interp_CT_CHADCP_sand = np.polyfit(np.log10(Csand_turbidity_calc),np.log10(Csand_S2_D50_freq1_Csand_T_valid), 1)
# lin_model_CT_CHADCP_sand = [10**(np.log10(Csand_turbidity_calc[i])*interp_CT_CHADCP_sand[0]+interp_CT_CHADCP_sand[1])
#                            for i in range(len(Csand_S2_D50_freq1_Csand_T_valid))]
# lin_model_CT_CHADCP_sand_plot = [10**(np.log10(x_range_sand[i])*interp_CT_CHADCP_sand[0]+interp_CT_CHADCP_sand[1])
#                            for i in range(len(x_range_sand))]
# R2_CT_CHADCP_sand = r2_score(Csand_S2_D50_freq1_Csand_T_valid, lin_model_CT_CHADCP_sand)


#%% Plot Csand - Csand, HADCP - Dual-frequency data - MANUSCRIPT
fig, ax = plt.subplots(1, 2, figsize = (12,6), dpi=300)

#Csand 
p1 = ax[0].errorbar(TAAPS_all['Sand_concentration_g_l'], C_sand_S2_D50_samples_g_l,  marker = 'D', 
            xerr = TAAPS_all['U_C']*TAAPS_all['Sand_concentration_g_l']/100, elinewidth = 1, capsize = 1.5, zorder = 40,
        ls = '', markersize = 7, color = 'teal', markeredgecolor = 'black', markeredgewidth = 0.2, label = '$\mathregular{\overline{C_{sand}}}$')
p2, = ax[0].plot(pump_data['Sand_concentration_g_l'], Csand_S2_D50_pump_valid, marker = 's',       zorder = 30,       
        ls = '', markersize = 7, color = 'mediumvioletred', markeredgecolor = 'black', markeredgewidth = 0.1,
        label = '$\mathregular{\overline{C_{sand, pump}}}$')
p3, = ax[0].plot(ISCO_data['ISCO_sand_concentration_corr_g_l'], Csand_S2_D50_ISCO_valid, marker = 'o', zorder = 20,             
        ls = '', markersize = 5, color = 'yellowgreen', markeredgecolor = 'black', markeredgewidth = 0.1,
        label = '$\mathregular{C_{sand, ISCO, corr}}$') 
# p4, = ax[0].plot(Csand_turbidity_calc, Csand_S2_D50_Csand_T_valid, marker = '.',              
#         ls = '', markersize = 1, color = 'lightgrey',
#         label = '$\mathregular{C_{sand, T, calc}}$', zorder = 0)

ax[0].plot(2*x_range, x_range,  zorder = 40,
        ls = ':', lw = 1, color = 'black')
ax[0].plot(x_range, x_range,  zorder = 40,
        ls = '-', lw = 1, color = 'black')
ax[0].plot(x_range, 2*x_range,  zorder = 40,
        ls = ':', lw = 1, color = 'black') 

ax[0].plot(x_range_sand, lin_model_Cmeas_CHADCP_sand_plot, lw = 2, color = 'teal', zorder = 40)
ax[0].plot(x_range_sand, lin_model_CISCO_CHADCP_sand_plot, lw = 2, color = 'mediumvioletred', ls = '-.',zorder = 40)
ax[0].plot(x_range_sand, lin_model_Cpump_CHADCP_sand_plot, lw = 2, color = 'yellowgreen', ls = '--',zorder = 40)
# ax[0].plot(x_range_sand, lin_model_CT_CHADCP_sand_plot, lw = 2, color = 'grey', ls = ':',zorder = 40)

ax[0].text(0.02, 0.95, 'a)', transform = ax[0].transAxes, fontsize = 16)
ax[0].text(0.2, 0.92, 'R² = ' + str(abs(np.round(R2_Cmeas_CHADCP_sand,2))), fontsize = 16, transform = ax[0].transAxes, color = 'teal')  
ax[0].text(0.2, 0.87, 'R² = ' + str(abs(np.round(R2_Cpump_CHADCP_sand,2))), fontsize = 16, transform = ax[0].transAxes, color = 'mediumvioletred') 
ax[0].text(0.2, 0.82, 'R² = ' + str(abs(np.round(R2_CISCO_CHADCP_sand,2))), fontsize = 16, transform = ax[0].transAxes, color = 'yellowgreen') 
# ax[0].text(0.2, 0.77, 'R² = ' + str(abs(np.round(R2_CT_CHADCP_sand,2))), fontsize = 16, transform = ax[0].transAxes, color = 'grey') 


ax[0].set_xlabel('$\mathregular{C_{sand}}$ (g/l)', fontsize=20, weight = 'bold')
ax[0].set_ylabel('$\mathregular{\overline{C_{sand, HADCP}}}$ (g/l)', fontsize=20, weight = 'bold')
ax[0].tick_params(axis='both', which='major', labelsize = 16)
ax[0].set_xlim(0.001,5)
ax[0].set_ylim(0.001,5)
ax[0].set_xscale('log')
ax[0].set_yscale('log')

# D50 
x_range_d50 = np.arange(63,1000,10)
ax[1].plot(ISCO_GSD_data['D50'], D50_S2_D50_ISCO_valid, marker = 'o',              
        ls = '', markersize = 5, color = 'yellowgreen', markeredgecolor = 'black', markeredgewidth = 0.1,
        label = 'ISCO')
ax[1].plot(TAAPS_all['D50_mum'], D50_S2_D50_samples_g_l,  marker = 'D',             
        ls = '', markersize = 7, color = 'teal', markeredgecolor = 'black', markeredgewidth = 0.2, label = 'XS')
ax[1].plot(2*x_range_d50, x_range_d50,  
        ls = ':', lw = 1, color = 'black')
ax[1].plot(np.arange(63,1000,10), np.arange(63,1000,10),  
        ls = '-', lw = 1, color = 'black')
ax[1].plot(x_range_d50, 2*x_range_d50,    
        ls = ':', lw = 1, color = 'black')
ax[1].text(0.02, 0.95, 'b)', transform = ax[1].transAxes, fontsize = 16)

# ax[0].legend(fontsize = 16, loc = 'lower right', framealpha = 1)
ax[1].set_xlabel('$\mathregular{D_{50}\; (\mu m)}$', fontsize=20, weight = 'bold')
ax[1].set_ylabel('$\mathregular{\overline{D_{50, HADCP}}\; (\mu m)}$', fontsize=20, weight = 'bold')
ax[1].tick_params(axis='both', which='major', labelsize = 16)
ax[1].set_xlim(63,1000)
ax[1].set_ylim(63,1500)
ax[1].set_xscale('log')
ax[1].set_yscale('log')
# ax[1].yaxis.set_label_position("right")
# ax[1].yaxis.tick_right()

handles = [p1, p2, p3]
#_, labels = ax.get_legend_handles_labels()
fig.legend(handles = handles, #labels=labels, 
          handler_map = {tuple: mpl.legend_handler.HandlerTuple(None)},
          fontsize = 14, loc = 'lower center', ncol = 1, bbox_to_anchor = (0.89, 0.13))
#ax.legend(fontsize = 14, loc = 'lower center', ncol = 4, bbox_to_anchor = (0.5, -0.23))

fig.tight_layout()
figname = 'C_sand_D50_meas_HADCP_S2_D50'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 300, bbox_inches='tight')
fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight') 
fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 300, bbox_inches='tight')

#%% Calculate Flux per time step (C * Q)

Phi_sand_S2_D50_kg_s = C_sand_S2_D50_g_l *Q_time_freq2

# Determine time intervall between HADCP meas
Time_interval_freq2 = [(Time_datetime_freq2[i+1] -Time_datetime_freq2[i]).total_seconds() for i in range(len(Time_datetime_freq2)-1)]

total_Phi_sand_S2_D50_kg = np.nansum(Phi_sand_S2_D50_kg_s[1:]*Time_interval_freq2)
total_Phi_sand_S2_D50_t = total_Phi_sand_S2_D50_kg/1000
Phi_sand_S2_D50_kg_s_cumsum = np.nancumsum(Phi_sand_S2_D50_kg_s[1:]*Time_interval_freq2)
Q_time_freq2_cumsum = np.nancumsum(Q_time_freq2[1:]*Time_interval_freq2)

# for SFCP 400 kHz
Time_interval_freq1 = [(Time_datetime_freq1[i+1] -Time_datetime_freq1[i]).total_seconds() for i in range(len(Time_datetime_freq1)-1)]
Phi_sand_freq1_kg_s = Csand_est_freq1 *Q_time_freq1

total_Phi_sand_freq1_kg = np.nansum(Phi_sand_freq1_kg_s[1:]*Time_interval_freq1)
total_Phi_sand_freq1_t = total_Phi_sand_freq1_kg/1000
Phi_sand_freq1_kg_s_cumsum = np.nancumsum(Phi_sand_freq1_kg_s[1:]*Time_interval_freq1)
Q_time_freq1_cumsum = np.nancumsum(Q_time_freq1[1:]*Time_interval_freq1)

# for SFCP 1 MHz
Phi_sand_freq2_kg_s = Csand_est_freq2*Q_time_freq2

total_Phi_sand_freq2_kg = np.nansum(Phi_sand_freq2_kg_s[1:]*Time_interval_freq2)
total_Phi_sand_freq2_t = total_Phi_sand_freq2_kg/1000
Phi_sand_freq2_kg_s_cumsum = np.nancumsum(Phi_sand_freq2_kg_s[1:]*Time_interval_freq2)
Q_time_freq2_cumsum = np.nancumsum(Q_time_freq2[1:]*Time_interval_freq2)



#%% Sediment rating curve       
# Use power law with critical Q

Q_range = np.linspace(0,1000, 500)  

a_power_cr = 0.0003
b_power_cr = 2.3
Q_cr = 50
Qss_range_power_cr = [a_power_cr *(Q_range[i]-Q_cr)**b_power_cr for i in range(len(Q_range))]
Q_range_Cr = [Q_range[i] for i in range(len(Q_range))]

# Determine R2
Qss_power_cr_samples = [a_power_cr *(TAAPS_all['Q_sampling_m3_s'][i]-Q_cr)**b_power_cr for i in range(len(TAAPS_all))]
R2_power_cr = r2_score(TAAPS_all['Sand_flux_kg_s'], Qss_power_cr_samples)

# Plot 
err_Q = TAAPS_all['U_Q']/100*TAAPS_all['Q_sampling_m3_s']
err_Phi = TAAPS_all['U_F']/100*TAAPS_all['Sand_flux_kg_s']

TAAPS_all['err_Q'] = err_Q
TAAPS_all['err_Phi'] = err_Phi

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
    
# Calculate C
Csand_power_cr_time_Q = Phi_power_cr_time_Q/Q
 
 
#%% Plot Fig3
fig, ax = plt.subplots(1, 1, figsize = (8, 6), dpi=300)

p1, = ax.plot(Q_range_Cr, Qss_range_power_cr,
        lw = 2, color = 'black', label = r'$\mathregular{\Phi_{cr}}$', zorder = 40) #$\mathregular{\Phi_{sand,cr} = a_{cr}(Q-Q_{cr})^{b_{cr}}}$
 
p2 = ax.errorbar(TAAPS_all['Q_sampling_m3_s'], TAAPS_all['Sand_flux_kg_s'],
           ls=' ', marker= 'D', markersize = '7', color='darkorange', markeredgecolor = 'black', markeredgewidth=0.5,                 
           xerr = TAAPS_all['err_Q'], 
           yerr = TAAPS_all['err_Phi'], elinewidth = 0.7, capsize = 1.5,
           label = r'$\mathregular{\Phi_{meas}}$', zorder = 30)        

p3, = ax.plot(pump_data['Q_sampling_m3_s'], pump_data['Sand_flux_kg_s'],              
        ls=' ', marker= 's', markersize = '7', color='mediumblue', markeredgecolor = 'black', markeredgewidth=0.5,
        label = r'$\mathregular{\Phi_{ pump}}$', zorder = 30)

p4, = ax.plot(ISCO_data['Q_sampling_m3_s'][10], ISCO_data['Phi_sand_kg_s'][10], marker = 'o',              
           ls = '', markersize = 6, color = 'yellowgreen',markeredgecolor = 'black', markeredgewidth=0.5,
           label = '$\mathregular{\Phi_{ISCO}}$', zorder = 0)

ax.plot(ISCO_data['Q_sampling_m3_s'], ISCO_data['Phi_sand_kg_s'], marker = 'o',             
           ls = '', markersize = 4, color = 'yellowgreen', markeredgecolor = 'black', markeredgewidth=0.1, 
           zorder = 0)

#ax.legend(fontsize = 16,loc = 'lower right', framealpha = 1)
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
figname = 'Fig3'
fig.savefig(outpath_figures+ '\\' + figname +  '.png', dpi = 300, bbox_inches='tight')
fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight')

#%% Plot Fig 5
fig, ax = plt.subplots(1, 3, figsize = (12,4), dpi=300)

# Cal 
p1 = ax[0].errorbar(TAAPS_all['Sand_concentration_g_l'], C_sand_S2_D50_samples_g_l,  marker = 'D', 
            xerr = TAAPS_all['U_C']*TAAPS_all['Sand_concentration_g_l']/100, elinewidth = 1, capsize = 1.5, zorder = 40,
        ls = '', markersize = 7, color = 'darkorange', markeredgecolor = 'black', markeredgewidth = 0.2, label = '$\mathregular{\overline{C_{sand}}}$')

ax[0].plot(2*x_range, x_range,  zorder = 40,
        ls = ':', lw = 1, color = 'black')
ax[0].plot(x_range, x_range,  zorder = 40,
        ls = '-', lw = 1, color = 'black')
ax[0].plot(x_range, 2*x_range,  zorder = 40,
        ls = ':', lw = 1, color = 'black')
ax[0].plot(5*x_range, x_range,    
        ls = (0, (1, 10)), lw = 1, color = 'black')
ax[0].plot(x_range, 5*x_range,    
        ls = (0, (1, 10)), lw = 1, color = 'black')

ax[0].text(0.02, 0.95, '(g)', transform = ax[0].transAxes, fontsize = 12)

ax[0].set_xlabel('$\mathregular{C_{sand, cal}}$ (g/l)', fontsize=14, weight = 'bold')
ax[0].set_ylabel('$\mathregular{\overline{C_{sand, HADCP}}}$ (g/l)', fontsize=14, weight = 'bold')
ax[0].tick_params(axis='both', which='major', labelsize = 12)
ax[0].set_xlim(0.01,2)
ax[0].set_ylim(0.01,2)
ax[0].set_xscale('log')
ax[0].set_yscale('log')

# Val 
p2, = ax[1].plot(pump_data['Sand_concentration_g_l'], Csand_S2_D50_pump_valid, marker = 's',       zorder = 31,       
        ls = '', markersize = 7, color = 'mediumblue', markeredgecolor = 'black', markeredgewidth = 0.1,
        label = '$\mathregular{\overline{C_{sand, pump}}}$')
p3, = ax[1].plot(ISCO_data['ISCO_sand_concentration_corr_g_l'], Csand_S2_D50_ISCO_valid, marker = 'o', zorder = 21,             
        ls = '', markersize = 4, color = 'yellowgreen', markeredgecolor = 'black', markeredgewidth = 0.1,
        label = '$\mathregular{C_{sand, ISCO, corr}}$') 

ax[1].plot(2*x_range, x_range,  zorder = 41,
        ls = ':', lw = 1, color = 'black')
ax[1].plot(x_range, x_range,  zorder = 41,
        ls = '-', lw = 1, color = 'black')
ax[1].plot(x_range, 2*x_range,  zorder = 41,
        ls = ':', lw = 1, color = 'black')
ax[1].plot(5*x_range, x_range,    
        ls = (0, (1, 10)), lw = 1, color = 'black')
ax[1].plot(x_range, 5*x_range,    
        ls = (0, (1, 10)), lw = 1, color = 'black') 

ax[1].text(0.02, 0.95, '(h)', transform = ax[1].transAxes, fontsize = 12)

ax[1].set_xlabel('$\mathregular{C_{sand, val}}$ (g/l)', fontsize=14, weight = 'bold')
ax[1].set_ylabel('$\mathregular{\overline{C_{sand, HADCP}}}$ (g/l)', fontsize=14, weight = 'bold')
ax[1].tick_params(axis='both', which='major', labelsize = 12)
ax[1].set_xlim(0.01,2)
ax[1].set_ylim(0.01,2)
ax[1].set_xscale('log')
ax[1].set_yscale('log')

# D50 
x_range_d50 = np.arange(63,1000,10)
ax[2].plot(TAAPS_all['D50_mum'], D50_S2_D50_samples_g_l,  marker = 'D',             
        ls = '', markersize = 7, color = 'maroon', markeredgecolor = 'black', markeredgewidth = 0.1, zorder = 40, label = 'XS')
ax[2].plot(2*x_range_d50, x_range_d50,  
        ls = ':', lw = 1, color = 'black')
ax[2].plot(np.arange(63,1000,10), np.arange(63,1000,10),  
        ls = '-', lw = 1, color = 'black')
ax[2].plot(x_range_d50, 2*x_range_d50,    
        ls = ':', lw = 1, color = 'black')
ax[2].plot(5*x_range_d50, x_range_d50,    
        ls = (0, (1, 10)), lw = 1, color = 'black')
ax[2].plot(x_range_d50, 5*x_range_d50,    
        ls = (0, (1, 10)), lw = 1, color = 'black')
ax[2].text(0.02, 0.95, '(i)', transform = ax[2].transAxes, fontsize = 12)

ax[2].set_xlabel('$\mathregular{D_{50, cal}\; (\mu m)}$', fontsize=14, weight = 'bold')
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

# handles = [p1, p2, p3]
# #_, labels = ax.get_legend_handles_labels()
# fig.legend(handles = handles, #labels=labels, 
#           handler_map = {tuple: mpl.legend_handler.HandlerTuple(None)},
#           fontsize = 14, loc = 'lower center', ncol = 1, bbox_to_anchor = (0.89, 0.13))

fig.tight_layout()
figname = 'Fig5'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 250, bbox_inches='tight')
#fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight') 
#fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 300, bbox_inches='tight')



#%% Plot Fig6
fig, ax = plt.subplots(2, 2, figsize = (12,12), dpi=300)

# Freq1 
p1 = ax[0,0].errorbar(TAAPS_all['Sand_concentration_g_l'], C_sand_S2_D50_freq1_samples_g_l,  marker = 'D', 
            xerr = TAAPS_all['U_C']*TAAPS_all['Sand_concentration_g_l']/100, elinewidth = 1, capsize = 1.5, zorder = 10,
        ls = '', markersize = 8, color = 'darkorange', markeredgecolor = 'black', markeredgewidth = 0.2, label = '$\mathregular{\overline{C_{sand}}}$')

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

ax[0,0].text(0.02, 0.95, '(a)', transform = ax[0,0].transAxes, fontsize = 16)
ax[0,0].set_ylabel('$\mathregular{\overline{C_{sand, 400 kHz}}}$ (g/l)', fontsize=20, weight = 'bold')
# ax[0,0].set_xlabel('$\mathregular{\overline{C_{sand, cal}}}$ (g/l)', fontsize=20, weight = 'bold')
ax[0,0].tick_params(axis='both', which='major', labelsize = 16)
ax[0,0].set_xlim(0.01,1.5)
ax[0,0].set_ylim(0.01,1.5)
ax[0,0].set_xscale('log')
ax[0,0].set_yscale('log')
ax[0,0].xaxis.set_ticklabels([])

# Freq1 
p2, = ax[0,1].plot(pump_data['Sand_concentration_g_l'], Csand_S2_D50_freq1_pump_valid, marker = 's',              
        ls = '', markersize = 8, color = 'mediumblue', markeredgecolor = 'black', markeredgewidth = 0.1, zorder = 20,
        label = 'Pump')
p3, = ax[0,1].plot(ISCO_data['ISCO_sand_concentration_corr_g_l'], Csand_S2_D50_freq1_ISCO_valid, marker = 'o',              
        ls = '', markersize = 5, color = 'yellowgreen', markeredgecolor = 'black', markeredgewidth = 0.1,
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

ax[0,1].text(0.02, 0.95, '(b)', transform = ax[0,1].transAxes, fontsize = 16)
# ax[0,1].set_xlabel('$\mathregular{\overline{C_{sand, val}}}$ (g/l)', fontsize=20, weight = 'bold')
# ax[0,1].set_ylabel('$\mathregular{\overline{C_{sand, HADCP, 400 kHz}}}$ (g/l)', fontsize=20, weight = 'bold')
ax[0,1].tick_params(axis='both', which='major', labelsize = 16)
ax[0,1].set_xlim(0.01,1.5)
ax[0,1].set_ylim(0.01,1.5)
ax[0,1].set_xscale('log')
ax[0,1].set_yscale('log')

# Freq2 
p1 = ax[1,0].errorbar(TAAPS_all['Sand_concentration_g_l'], C_sand_S2_D50_freq2_samples_g_l,  marker = 'D', 
            xerr = TAAPS_all['U_C']*TAAPS_all['Sand_concentration_g_l']/100, elinewidth = 1, capsize = 1.5, zorder = 10,
        ls = '', markersize = 8, color = 'darkorange', markeredgecolor = 'black', markeredgewidth = 0.2, label = 'Samples')

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

ax[1,0].text(0.02, 0.95, '(c)', transform = ax[1,0].transAxes, fontsize = 16)
# ax[0].legend(fontsize = 16, loc = 'lower right', framealpha = 1)
ax[1,0].set_xlabel('$\mathregular{\overline{C_{sand, meas}}}$ (g/l)', fontsize=20, weight = 'bold')
ax[1,0].set_ylabel('$\mathregular{\overline{C_{sand, 1MHz}}}$ (g/l)', fontsize=20, weight = 'bold')
ax[1,0].tick_params(axis='both', which='major', labelsize = 16)
ax[1,0].set_xlim(0.01,1.5)
ax[1,0].set_ylim(0.01,1.5)
ax[1,0].set_xscale('log')
ax[1,0].set_yscale('log')
ax[1,0].xaxis.set_ticklabels([])

# Freq2 
p2, = ax[1,1].plot(pump_data['Sand_concentration_g_l'], Csand_S2_D50_freq2_pump_valid, marker = 's',              
        ls = '', markersize = 8, color = 'mediumblue', markeredgecolor = 'black', markeredgewidth = 0.1, zorder = 20,
        label = 'Pump')
ax[1,1].plot(ISCO_data['ISCO_sand_concentration_corr_g_l'], Csand_S2_D50_freq2_ISCO_valid, marker = 'o',              
        ls = '', markersize = 5, color = 'yellowgreen', markeredgecolor = 'black', markeredgewidth = 0.1,
        label = 'ISCO')
p3, = ax[1,1].plot(ISCO_data['ISCO_sand_concentration_corr_g_l'][0], Csand_S2_D50_freq2_ISCO_valid[0], marker = 'o',              
        ls = '', markersize = 8, color = 'yellowgreen', markeredgecolor = 'black', markeredgewidth = 0.1,
        label = 'ISCO')

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
 
 
ax[1,1].text(0.02, 0.95, '(d)', transform = ax[1,1].transAxes, fontsize = 16)
# ax[0].legend(fontsize = 16, loc = 'lower right', framealpha = 1)
ax[1,1].set_xlabel('$\mathregular{\overline{C_{sand, meas}}}$ (g/l)', fontsize=20, weight = 'bold')
# ax[1,1].set_ylabel('$\mathregular{\overline{C_{sand, HADCP, 1MHz}}}$ (g/l)', fontsize=20, weight = 'bold')
ax[1,1].tick_params(axis='both', which='major', labelsize = 16)
ax[1,1].set_xlim(0.01,1.5)
ax[1,1].set_ylim(0.01,1.5)
ax[1,1].set_xscale('log')
ax[1,1].set_yscale('log')

handles = [p1, p2, p3, p4 , p5, p6]
#_, labels = ax.get_legend_handles_labels()
fig.legend(handles = handles, #labels=labels, 
          handler_map = {tuple: mpl.legend_handler.HandlerTuple(None)}, framealpha = 1, 
          fontsize = 16, loc = 'lower center', ncol = 3, bbox_to_anchor = (0.5, -0.08))

fig.tight_layout()
figname = 'Fig6'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight') 
# fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 300, bbox_inches='tight')




#%% Plot C est fines with C meas - HADCP
fig, ax = plt.subplots(1, 1, figsize = (8,6), dpi=300)

ax.plot(TAAPS_all['Fine_concentration_g_l'], C_fines_samples_g_l,  marker = 'D', 
        ls = '', markersize = 8, color = 'darkorange', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 40,
        label = 'Samples')
ax.plot(ISCO_data['Fine_concentration_g_l'], Cfines_ISCO_valid, marker = 'o',              
        ls = '', markersize = 5, color = 'yellowgreen', markeredgecolor = 'black', markeredgewidth = 0.1, zorder = 30) 
ax.plot(ISCO_data['Fine_concentration_g_l'][10], Cfines_ISCO_valid[10], marker = 'o',              
        ls = '', markersize = 7, color = 'yellowgreen', markeredgecolor = 'black', markeredgewidth = 0.1, zorder = 30,
        label = 'ISCO') 

ax.plot(5*x_range, x_range,  zorder = 41,
         ls = (0, (1, 10)), lw = 1, color = 'black') 
ax.plot(2*x_range, x_range,  zorder = 41,
        ls = ':', lw = 1, color = 'black')
p4, = ax.plot(x_range, x_range,  zorder = 41,
        ls = '-', lw = 1, color = 'black', label = 'Perfect agreement')
p5, = ax.plot(x_range, 2*x_range,  zorder = 41,
        ls = ':', lw = 1, color = 'black', label = 'Error of a factor of 2') 
p6, = ax.plot(x_range, 5*x_range,  zorder = 41,
         ls = (0, (1, 10)), lw = 1, color = 'black', label = 'Error of a factor of 5') 


ax.text(0.05, 0.95, 'c)', fontsize = 16, transform = ax.transAxes)
ax.legend(fontsize = 16, loc = 'lower right', framealpha = 1)
ax.set_xlabel('$\mathregular{C_{fines, meas}}$ (g/l)', fontsize=20, weight = 'bold')
ax.set_ylabel('$\mathregular{\overline{C_{fines, HADCP}}}$ (g/l)', fontsize=20, weight = 'bold')
ax.tick_params(axis='both', which='major', labelsize = 16)
ax.set_xlim(0.005,10)
ax.set_ylim(0.005,10)
ax.set_xscale('log')
ax.set_yscale('log')

fig.tight_layout()
figname = 'Fig_A2'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 150, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight') 
# fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 300, bbox_inches='tight')



   

#%% Plot Q - C est S2_D50 sand
C_power_cr = [Qss_range_power_cr[i]/Q_range_Cr[i] for i in range(len(Qss_range_power_cr))]

fig, ax = plt.subplots(1, 1, figsize = (8,6), dpi=300)

ax.plot(Q_time_freq2[0], C_sand_S2_D50_g_l[0], '.', markersize = 8, alpha = 0.8,
        color = 'tan', label = 'TW16-B')
ax.plot(Q_time_freq2, C_sand_S2_D50_g_l, '.', markersize = 3, alpha = 0.8,
        color = 'tan')
ax.plot(TAAPS_all['Q_sampling_m3_s'], TAAPS_all['Sand_concentration_g_l'], 'D', 
        color = 'darkorange', markeredgecolor = 'black', markeredgewidth = 0.2, label = 'Sampler', zorder = 30)
ax.plot(Q_time_pump, pump_data['Sand_concentration_g_l'], marker = 's',              
        ls = '', markersize = 6, color = 'mediumblue', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 20,
        label = 'Pump')
ax.plot(Q_time_ISCO, ISCO_data['ISCO_sand_concentration_corr_g_l'], marker = 'o',              
        ls = '', markersize = 5, color = 'yellowgreen', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 10,
        label = 'ISCO')

p1, = ax.plot(Q_range_Cr, C_power_cr,
        lw = 2, color = 'darkorange', label = r'Rating curve', zorder = 40) #$\mathregular{\Phi_{sand,cr} = a_{cr}(Q-Q_{cr})^{b_{cr}}}$
 
ax.text(0.05, 0.95, 'c)', fontsize = 16, transform = ax.transAxes)
ax.set_xlabel('Q (m³/s)', fontsize=20, weight = 'bold')
ax.set_ylabel('$\mathregular{\overline{C_{sand}}}$ (g/l)', fontsize=20, weight = 'bold')
ax.tick_params(axis='both', which='major', labelsize = 16)
ax.set_ylim(0.001,7)
ax.set_xlim(0,700)
ax.set_yscale('log')
# ax.legend(fontsize = 14, loc = 'lower right')

fig.tight_layout()
figname = 'Fig_A3_TW16B'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight') 
# fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 300, bbox_inches='tight')


#%% Plot Q - C est S2_D50 sand - legend
C_power_cr = [Qss_range_power_cr[i]/Q_range_Cr[i] for i in range(len(Qss_range_power_cr))]

fig, ax = plt.subplots(1, 1, figsize = (8,6), dpi=300)

ax.plot(Q_time_freq2[0], C_sand_S2_D50_g_l[0], '.', markersize = 8, alpha = 0.8,
        color = 'tan', label = 'TW16-B')
ax.plot(Q_time_freq2, C_sand_S2_D50_g_l, '.', markersize = 3, alpha = 0.8,
        color = 'tan')
ax.plot(TAAPS_all['Q_sampling_m3_s'], TAAPS_all['Sand_concentration_g_l'], 'D', 
        color = 'darkorange', markeredgecolor = 'black', markeredgewidth = 0.2, label = 'Sampler', zorder = 30)
ax.plot(Q_time_pump, pump_data['Sand_concentration_g_l'], marker = 's',              
        ls = '', markersize = 6, color = 'mediumblue', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 20,
        label = 'Pump')
ax.plot(Q_time_ISCO, ISCO_data['ISCO_sand_concentration_corr_g_l'], marker = 'o',              
        ls = '', markersize = 5, color = 'yellowgreen', markeredgecolor = 'black', markeredgewidth = 0.2, zorder = 10,
        label = 'ISCO')

p1, = ax.plot(Q_range_Cr, C_power_cr,
        lw = 2, color = 'darkorange', label = r'Rating curve', zorder = 40) #$\mathregular{\Phi_{sand,cr} = a_{cr}(Q-Q_{cr})^{b_{cr}}}$
 
ax.text(0.05, 0.95, 'c)', fontsize = 16, transform = ax.transAxes)
ax.set_xlabel('Q (m³/s)', fontsize=20, weight = 'bold')
ax.set_ylabel('$\mathregular{\overline{C_{sand}}}$ (g/l)', fontsize=20, weight = 'bold')
ax.tick_params(axis='both', which='major', labelsize = 16)
ax.set_ylim(0.001,7)
ax.set_xlim(0,700)
ax.set_yscale('log')
ax.legend(fontsize = 16, loc = 'lower center', ncol = 3, bbox_to_anchor = (0.5, -0.1))

fig.tight_layout()
figname = 'Fig_A3_TW16B_legend'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 300, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight') 
# fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 300, bbox_inches='tight')

#%% Plot Fig 5 samples
fig, ax = plt.subplots(1, 1, figsize = (7,6), dpi=300)

# Cal 
p1 = ax.errorbar(TAAPS_all['Sand_concentration_g_l'], C_sand_S2_D50_samples_g_l,  marker = 'D', 
            xerr = TAAPS_all['U_C']*TAAPS_all['Sand_concentration_g_l']/100, elinewidth = 1, capsize = 1.5, zorder = 40,
        ls = '', markersize = 9, color = 'darkorange', markeredgecolor = 'black', markeredgewidth = 0.2, label = 'Sampler')

p2, = ax.plot(pump_data['Sand_concentration_g_l'], Csand_S2_D50_pump_valid, marker = 's',       zorder = 31,       
        ls = '', markersize = 9, color = 'mediumblue', markeredgecolor = 'black', markeredgewidth = 0.1,
        label = 'Pump')

ax.plot(5*x_range, x_range,  zorder = 41,
         ls = (0, (1, 10)), lw = 1, color = 'black') 
ax.plot(2*x_range, x_range,  zorder = 40,
        ls = ':', lw = 1, color = 'black')
p3, = ax.plot(x_range, x_range,  zorder = 40,
        ls = '-', lw = 1, color = 'black', label = 'Perfect agreement')
p4, = ax.plot(x_range, 2*x_range,  zorder = 40,
        ls = ':', lw = 1, color = 'black', label = 'Error of a factor of 2') 
p5, = ax.plot(x_range, 5*x_range,  zorder = 41,
         ls = (0, (1, 10)), lw = 1, color = 'black', label = 'Error of a factor of 5') 

ax.set_xlabel('$\mathregular{\overline{C_{sand, meas}}}$ (g/l)', fontsize=18, weight = 'bold')
ax.set_ylabel('$\mathregular{\overline{C_{sand, TW16-B}}}$ (g/l)', fontsize=18, weight = 'bold')
ax.tick_params(axis='both', which='major', labelsize = 16)
ax.set_xlim(0.01,2)
ax.set_ylim(0.01,2)
ax.set_xscale('log')
ax.set_yscale('log')

# handles = [p1, p2, p3, p4, p5]
# #_, labels = ax.get_legend_handles_labels()
# fig.legend(handles = handles, #labels=labels, 
#           handler_map = {tuple: mpl.legend_handler.HandlerTuple(None)},
#           fontsize = 16, loc = 'upper left', framealpha= 1)
fig.tight_layout()
figname = 'Fig5_samples_pump_TW16B'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 300, bbox_inches='tight')


#%% Plot Fig 5 samples pump grain size
fig, ax = plt.subplots(1, 1, figsize = (7,6), dpi=300)

x_range_d50 = np.arange(63,1000,10)
ax.plot(TAAPS_all['D50_mum'], D50_S2_D50_samples_g_l,  marker = 'D',             
        ls = '', markersize = 9, color = 'maroon', markeredgecolor = 'black', markeredgewidth = 0.1, zorder = 40, label = 'Sampler')

ax.plot(5*x_range_d50, x_range_d50,    
         ls = (0, (1, 10)), lw = 1, color = 'black')
ax.plot(2*x_range_d50, x_range_d50,  
        ls = ':', lw = 1, color = 'black')
ax.plot(np.arange(63,1000,10), np.arange(63,1000,10),  
        ls = '-', lw = 1, color = 'black', label = 'Perfect agreeement')
ax.plot(x_range_d50, 2*x_range_d50,    
        ls = ':', lw = 1, color = 'black', label = 'Error of a factor of 2')
ax.plot(x_range_d50, 5*x_range_d50,    
        ls = (0, (1, 10)), lw = 1, color = 'black', label = 'Error of a factor of 5')

ax.set_xlabel('$\mathregular{\overline{D_{50, meas}}\; (\mu m)}$', fontsize=18, weight = 'bold')
ax.set_ylabel('$\mathregular{\overline{D_{50, TW16-B}}\; (\mu m)}$', fontsize=18, weight = 'bold')
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
ax.legend(loc = 'upper left', fontsize = 16, framealpha = 1)

fig.tight_layout()
figname = 'Fig5_samples_pump_grain_size_TW16B'
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
                    'Instrument background (dB)',  'Effective background (dB)',  'Q_m3_s',  'SPM',  'Stage',  #'SNR', 'SNR_cell5', 'SNR_cell10',
                    'Beam-Averaged Backscatter (dB) Freq1', 
                    'Alpha Sediment (dB/m) Freq1', 'BeamAvBS_effective_1_freq1_S2_D50', 'BeamAvBS_effective_2_freq1_S2_D50', 
                    'B_defect_freq1_S2_D50', 'B_defect_freq2_S2_D50', 'BeamAvBS_effective_freq2_S2_D50', 'Beff_Bdefect_freq2_S2_D50',
                    'Csand_S2_D50_g_l', 'D50_est_S2_D50', 'Cfines', 'S_S2_D50']
results.columns = colnames_results


#%% STEP 9: Plot B and Csand ENS S2_D50
x_range = np.linspace(0,150,100)
fig, ax = plt.subplots(1, 2, figsize = (12,6), dpi=300)

# freq1_S2_D50 
ax[0].plot(x_range, lin_model_B_Csand_ens_freq1_plot, color = 'grey')
ax[0].plot(x_range, lin_model_B_Csand_ens_freq1_S2_D50_plot, color = 'navy', lw = 2)
ax[0].plot(TAAPS_freq1['Beam-Averaged Backscatter (dB)'], np.log10(mean_sand_conc_freq1_mg_l),
        'o', color = 'grey', markersize = 10, markeredgewidth = 0.2, 
        alpha = 0.7, markeredgecolor = 'black', lw = 0, label = 'all')
m2, = ax[0].plot(TAAPS_freq1_S2_D50['Beam-Averaged Backscatter (dB)'], np.log10(mean_sand_conc_freq1_S2_D50_mg_l),
        'D', color = 'navy', markersize = 10, markeredgewidth = 0.2, 
        markeredgecolor = 'black', lw = 0, label = 'S, $\mathregular{D_{50, sand}}$')

ax[0].text(0.05, 0.95, 'a)', fontsize = 14, transform = ax[0].transAxes)
ax[0].text(0.2, 0.95, ('y = ' + str(np.round(interp_B_Csand_ens_freq1_S2_D50[0],2)) + 'x + (' + str(np.round(interp_B_Csand_ens_freq1_S2_D50[1],2)) + ')'), 
        color = 'black', fontsize = 16, transform = ax[0].transAxes)
ax[0].text(0.2, 0.89, ('R² = ' + str(float(np.round(R2_B_Csand_ens_freq1_S2_D50,2)))), 
        color = 'black', fontsize = 16, transform = ax[0].transAxes)
ax[0].text(0.5, 0.2, ('y = ' + str(np.round(interp_B_Csand_ens_freq1[0],2)) + 'x + (' + str(np.round(interp_B_Csand_ens_freq1[1],2)) + ')'), 
        color = 'grey', fontsize = 16, transform = ax[0].transAxes)
ax[0].text(0.5, 0.14, ('R² = ' + str(float(np.round(R2_B_Csand_ens_freq1,2)))), 
        color = 'grey', fontsize = 16, transform = ax[0].transAxes)

ax[0].set_ylabel('$\mathregular{\overline{\log_{10}(C_{sand, ens}})}$ (mg/l)', fontsize=18, weight = 'bold')
ax[0].tick_params(axis='both', which='major', labelsize = 16)
ax[0].set_xlim (94, 110)
ax[0].set_ylim(0, 3.5)

# freq2_S2_D50
ax[1].plot(x_range, lin_model_B_Csand_ens_freq2_plot, color = 'grey')
ax[1].plot(x_range, lin_model_B_Csand_ens_freq2_S2_D50_plot, color = 'darkred', lw = 2)
m1, = ax[1].plot(TAAPS_freq2['Beam-Averaged Backscatter (dB)'],np.log10(mean_sand_conc_freq2_mg_l),
        'o', color = 'grey', markersize = 10, markeredgewidth = 0.2, 
        alpha = 0.7, markeredgecolor = 'black', lw = 0, label = 'all')
m4, = ax[1].plot(TAAPS_freq2_S2_D50['Beam-Averaged Backscatter (dB)'],np.log10(mean_sand_conc_freq2_S2_D50_mg_l),
        'D', color = 'darkred', markersize = 10, markeredgewidth = 0.2, 
        markeredgecolor = 'black', lw = 0, label = 'S, $\mathregular{D_{50, sand}}$')

ax[1].text(0.05, 0.95, 'b)', fontsize = 14, transform = ax[1].transAxes)
ax[1].text(0.2, 0.95, ('y = ' + str(np.round(interp_B_Csand_ens_freq2_S2_D50[0],2)) + 'x + (' + str(np.round(interp_B_Csand_ens_freq2_S2_D50[1],2)) + ')'), 
        color = 'black', fontsize = 16, transform = ax[1].transAxes)
ax[1].text(0.2, 0.89, ('R² = ' + str(float(np.round(R2_B_Csand_ens_freq2_S2_D50,2)))), 
        color = 'black', fontsize = 16, transform = ax[1].transAxes)

ax[1].text(0.5, 0.3, ('y = ' + str(np.round(interp_B_Csand_ens_freq2[0],2)) + 'x + (' + str(np.round(interp_B_Csand_ens_freq2[1],2)) + ')'), 
        color = 'grey', fontsize = 16, transform = ax[1].transAxes)
ax[1].text(0.5, 0.24, ('R² = ' + str(float(np.round(R2_B_Csand_ens_freq2,2)))), 
        color = 'grey', fontsize = 16, transform = ax[1].transAxes)

ax[1].tick_params(axis='both', which='major', labelsize = 16)
ax[1].set_xlim (56, 76)
ax[1].set_ylim(0, 3.5)
ax[1].yaxis.tick_right()

fig.supxlabel(r'$\mathregular{\overline{B}}$ (dB/m)', fontsize=18, weight = 'bold')

handles = [(m1), (m2, m4)]
_, labels = ax[0].get_legend_handles_labels()
fig.legend(handles = handles, labels=labels, 
          handler_map = {tuple: mpl.legend_handler.HandlerTuple(None)},
          loc="lower right", ncol=1, bbox_to_anchor=(0.95, 0.15), 
          fontsize = 16)

fig.tight_layout()
figname = 'B_Csand_ens_S2_D50'
fig.savefig(outpath_figures +'\\' +  figname + '.png', dpi = 300, bbox_inches='tight')



#%% STEP 9: Plot B and Csand ENS S2_D50
x_range = np.linspace(0,150,100)
fig, ax = plt.subplots(1, 1, figsize = (7,6), dpi=300)

# freq1_S2_D50 
ax.plot(x_range, lin_model_B_Csand_ens_freq1_S2_D50_plot, color = 'black', lw = 2)
ax.plot(TAAPS_freq1['Beam-Averaged Backscatter (dB)'], np.log10(mean_sand_conc_freq1_mg_l),
        'D', color = 'grey', markersize = 12, markeredgewidth = 0.2, 
        alpha = 0.7, markeredgecolor = 'black', lw = 0, label = 'all')
m2, = ax.plot(TAAPS_freq1_S2_D50['Beam-Averaged Backscatter (dB)'], np.log10(mean_sand_conc_freq1_S2_D50_mg_l),
        'D', color = 'darkorange', markersize = 12, markeredgewidth = 0.2, 
        markeredgecolor = 'black', lw = 0, label = 'Sampler')

ax.text(0.05, 0.95, '400 kHz', fontsize = 16, transform = ax.transAxes, weight = 'bold') 
ax.text(0.5, 0.15, ('y = ' + str(np.round(interp_B_Csand_ens_freq1_S2_D50[0],2)) + 'x + (' + str(np.round(interp_B_Csand_ens_freq1_S2_D50[1],2)) + ')'), 
        color = 'black', fontsize = 16, transform = ax.transAxes)
ax.text(0.5, 0.08, 'R² = ' + str(float(np.round(R2_B_Csand_ens_freq1_S2_D50,2))) + ', n = ' + str(len(mean_sand_conc_freq1_S2_D50_mg_l)), 
        color = 'black', fontsize = 16, transform = ax.transAxes)

ax.set_ylabel('$\mathregular{log_{10}(\overline{C_{sand, ens}})}$ (mg/l)', fontsize=18, weight = 'bold')
ax.tick_params(axis='both', which='major', labelsize = 16)
ax.set_xlim (94, 115)
ax.set_ylim(0, 3.5)
ax.set_xlabel(r'$\mathregular{\overline{B_{400 kHz}}}$ (dB/m)', fontsize=18, weight = 'bold')
# ax.legend(loc = 'lower right', fontsize = 16)
# handles = [(m1), (m2, m4)]
# _, labels = ax.get_legend_handles_labels()
# fig.legend(handles = handles, labels=labels, 
#           handler_map = {tuple: mpl.legend_handler.HandlerTuple(None)},
#           loc="lower right", ncol=1, bbox_to_anchor=(0.95, 0.15), 
#           fontsize = 16)

fig.tight_layout()
figname = 'B_Csand_ens_400'
fig.savefig(outpath_figures +'\\' +  figname + '.png', dpi = 300, bbox_inches='tight')


#%% Determine regressions ens - xs 

# Csand ens - xs
x_range = np.linspace(0,3,100)
x = np.array(enson_volume_values['Mean_enson_sand_conc'])
y = np.array(enson_volume_values['Sand_concentration_g_l'])
x = x[:,np.newaxis]
slope_ens_xs_sand, _, _, _ = np.linalg.lstsq(x, y)
lin_model_ens_xs_sand = [enson_volume_values['Sand_concentration_g_l'][i]*slope_ens_xs_sand
                            for i in range(len(enson_volume_values['Sand_concentration_g_l']))]
lin_model_ens_xs_sand_plot = [x_range[i]*slope_ens_xs_sand
                            for i in range(len(x_range))]
R2_time_ens_xs_sand = r2_score(enson_volume_values['Sand_concentration_g_l'], lin_model_ens_xs_sand)

# Cfines ens - xs
x = np.array(enson_volume_values['Mean_enson_fine_conc'])
y = np.array(enson_volume_values['Fine_concentration_g_l'])
x = x[:,np.newaxis]
slope_ens_xs_fine, _, _, _ = np.linalg.lstsq(x, y)
lin_model_ens_xs_fine = [enson_volume_values['Fine_concentration_g_l'][i]*slope_ens_xs_fine
                            for i in range(len(enson_volume_values['Fine_concentration_g_l']))]
lin_model_ens_xs_fine_plot = [x_range[i]*slope_ens_xs_fine
                            for i in range(len(x_range))]
R2_time_ens_xs_fine = r2_score(enson_volume_values['Fine_concentration_g_l'], lin_model_ens_xs_fine)


#%% Sand HADCP - cross-section

fig, ax = plt.subplots(1, 1, figsize=(7,6), dpi=100)

# Sand 
ax.plot(enson_volume_values['Mean_enson_sand_conc'], enson_volume_values['Sand_concentration_g_l'],
        'D', markersize = 12, color = 'yellowgreen', markeredgecolor = 'black', markeredgewidth = 0.5)
ax.plot(x_range, lin_model_ens_xs_sand_plot,
        ls = '-', lw = 1.5, color = 'black', zorder = 0)

ax.text(0.2, 0.95, 'y = ' + str(np.round(slope_ens_xs_sand[0],2)) + 'x', 
        fontsize = fontsize_axis, transform=ax.transAxes)
ax.text(0.2, 0.89, 'R² = ' + str(np.round(R2_time_ens_xs_sand,2)) + ', n = ' + str(len(enson_volume_values)), fontsize = fontsize_axis, transform=ax.transAxes )

ax.set_xlabel('$\mathregular{\overline{C_{sand, ens}}}$ (g/l)', fontsize=fontsize_axis, weight = 'bold')
ax.set_ylabel('$\mathregular{\overline{C_{sand}}}$ (g/l)', fontsize=fontsize_axis, weight = 'bold')
ax.tick_params(axis='both', which='major', labelsize = fontsize_ticks)
ax.set_xlim(0,1.5)
ax.set_ylim(0,1.5)

fig.tight_layout()
figname = 'Concentration_sand_ens_xs'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 300, bbox_inches='tight')
fig.savefig(outpath_figures + '\\' + figname + '.eps', dpi = 300, bbox_inches='tight')



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


#%% Q - S

fig, ax = plt.subplots(1, 1, figsize=(7,6), dpi=100)

# Sand 
ax.scatter(TAAPS_freq1['Q_sampling_m3_s'], TAAPS_freq1['S'],
        marker = 'D', s = 60, c = TAAPS_freq1['S'], ec = 'black', lw = 0.5)

# ax.hlines(D50_sand_ref_range_mum[0], 0, 500, lw = 1,ls = '--', color = 'black')
# ax.hlines(D50_sand_ref_range_mum[1], 0, 500, lw = 1, ls = '--', color = 'black')
# ax.hlines(200, 0, 500, lw = 1, color = 'black')

ax.set_xlabel('Q (m³/s)', fontsize=fontsize_axis, weight = 'bold')
ax.set_ylabel('S', fontsize=fontsize_axis, weight = 'bold')
ax.tick_params(axis='both', which='major', labelsize = fontsize_ticks)
ax.set_xlim(0,500)
ax.set_ylim(0,)

fig.tight_layout()
figname = 'Q_S'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 300, bbox_inches='tight')
fig.savefig(outpath_figures + '\\' + figname + '.eps', dpi = 300, bbox_inches='tight')



#%% STEP 9: Plot ALPHASED and CFINES ENS
fig, ax = plt.subplots(1, 2, figsize = (12,6), dpi=300)

# Freq1 
ax[0].plot(x_range_ens, lin_model_AlphaSed_Cfines_ens_freq1_plot, color = 'black')
# ax[0].plot(x_range, lin_model_alphaSed1_spm_origin_plot, color = 'grey', ls = '--')

m1, = ax[0].plot(TAAPS_freq1['Alpha Sediment (dB/m)'],mean_fine_conc_freq1,
        'D', color = 'darkorange', markersize = 12, markeredgewidth = 0.2, 
        markeredgecolor = 'black', lw = 0, label = '$\mathregular{C_{sand, ens}}$')

# m1, = ax[0].plot(AlphaSed_freq12, spm_time_freq12, 
#         'o', color = 'cornflowerblue', markersize = 4, zorder = 0, alpha = 0.5, markeredgewidth = 0, lw = 0,
#         label = r'$\mathregular{C_{turbidity}}$')

ax[0].text(0.05, 0.95, 'a)', fontsize = 16, transform = ax[0].transAxes)
ax[0].text(0.15, 0.95, ('y = ' + str(np.round(slope_AlphaSed_Cfines_ens_freq1[0],2)) + 'x'), 
        color = 'black', fontsize = 18, transform = ax[0].transAxes)
ax[0].text(0.15, 0.89, ('R² = ' + str(float(np.round(R2_AlphaSed_Cfines_ens_freq1,2)))) + ', n = ' + str(len(mean_fine_conc_freq1)), 
        color = 'black', fontsize = 18, transform = ax[0].transAxes)
# ax[0].text(0.15, 0.8, ('y = ' + str(np.round(slope_AlphaSed_freq1_spm[0],2)) + 'x'), 
#         color = 'grey', fontsize = 16, transform = ax[0].transAxes)
# ax[0].text(0.15, 0.74, ('R² = ' + str(float(np.round(R2_AlphaSed_freq1_spm_origin,2))) + ', n = ' + str(len(spm_time_freq1))), 
#         color = 'grey', fontsize = 16, transform = ax[0].transAxes)

ax[0].set_ylabel('$\mathregular{C_{fines}}$ (g/l)', fontsize=18, weight = 'bold')
ax[0].tick_params(axis='both', which='major', labelsize = 16)
ax[0].set_xlim (0, 2)
ax[0].set_ylim(0,5)

# Freq2
ax[1].plot(x_range_ens, lin_model_AlphaSed_Cfines_ens_freq2_plot, color = 'black')
# ax[1].plot(x_range, lin_model_alphaSed2_spm_origin_plot, color = 'grey', ls = '--')
m2, = ax[1].plot(TAAPS_freq2['Alpha Sediment (dB/m)'],mean_fine_conc_freq2,
        'D', color = 'darkorange', markersize = 12, markeredgewidth = 0.2, 
        markeredgecolor = 'black', lw = 0, label = '$\mathregular{C_{sand, ens}}$')
# m3, = ax[1].plot(AlphaSed_freq22, spm_time_freq22, markeredgewidth = 0, lw = 0, 
#         marker = 'o', color = 'chocolate', markersize = 4, zorder = 0, alpha = 0.5)

ax[1].text(0.05, 0.95, 'b)', fontsize = 16, transform = ax[1].transAxes)
ax[1].text(0.15, 0.95, ('y = ' + str(np.round(slope_AlphaSed_Cfines_ens_freq2[0],2)) + 'x'),
        color = 'black', fontsize = 18, transform = ax[1].transAxes)
ax[1].text(0.15, 0.89, ('R² = ' + str(float(np.round(R2_AlphaSed_Cfines_ens_freq2,2))))+ ', n = ' + str(len(mean_fine_conc_freq1)),  
        color = 'black', fontsize = 18, transform = ax[1].transAxes)
# ax[1].text(0.15, 0.8, ('y = ' + str(np.round(slope_AlphaSed_freq2_spm[0],2)) + 'x'),
#         color = 'grey', fontsize = 16, transform = ax[1].transAxes)
# ax[1].text(0.15, 0.74, ('R² = ' + str(float(np.round(R2_AlphaSed_freq2_spm_origin,2))) + ', n = ' + str(len(spm_time_freq2))), 
#         color = 'grey', fontsize = 16, transform = ax[1].transAxes)

ax[1].tick_params(axis='both', which='major', labelsize = 16)
ax[1].set_xlim (0, 4)
ax[1].set_ylim(0,5)
ax[1].yaxis.tick_right()

fig.supxlabel(r'$\mathregular{α_{sed}}$ (dB/m)', fontsize=18, weight = 'bold')

# handles = [(m1, m2)]
# _, labels = ax[0].get_legend_handles_labels()
# fig.legend(handles = handles, labels=labels, 
#           handler_map = {tuple: mpl.legend_handler.HandlerTuple(None)},
#           loc="lower right", ncol=1, bbox_to_anchor=(0.97, 0.15), 
#           markerscale = 1, fontsize = 16)

fig.tight_layout()
figname = 'AlphaSed_Cfines_ens'
fig.savefig(outpath_figures +'\\' +  figname + '.png', dpi = 200, bbox_inches='tight')
fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight') 
fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 300, bbox_inches='tight')



#%% Plot C est S2_D50 sands
fig, ax = plt.subplots(2, 1, figsize = (10,6), dpi=300)

# Freq1
p6, = ax[0].plot(Time_datetime_freq2[0], C_sand_S2_D50_g_l[0], '.',markersize = 10,          
        color = 'sienna', label = 'TW16-A')
ax[0].plot(Time_datetime_freq2, C_sand_S2_D50_g_l, '.', markersize = 1,
        color = 'sienna', label = 'TW16-A')
p1, = ax[0].plot(TAAPS_freq1.iloc[:,1], TAAPS_freq1['Sand_concentration_g_l'], 'D', 
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

# Freq2
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
p3 = ax[0].hlines(0.2, Time_datetime_freq1[0], Time_datetime_freq1[-1], lw = 1, ls = '--', color = 'black',
             label = 'Reference sand') #label = r'$\mathregular{\overline{C_{sand,ref}}}$')
p4 = ax[1].hlines(200, Time_datetime_freq1[0], Time_datetime_freq1[-1], lw = 1, ls = '--', color = 'black',
             label = 'Reference sand') #label = r'$\mathregular{\overline{D_{50,sand,ref}}}$')

ax[0].text(0.02, 0.9, '(a)', fontsize = 14, transform = ax[0].transAxes)
ax[1].text(0.02, 0.9, '(b)', fontsize = 14, transform = ax[1].transAxes)
ax[0].set_ylabel('$\mathregular{\overline{C_{sand}}}$ (g/l)', fontsize=16, weight = 'bold')
ax[1].set_ylabel('$\mathregular{\overline{D_{50}} \; (\mu m)}$', fontsize=16, weight = 'bold')
ax[1].set_xlabel('Time', fontsize=16, weight = 'bold')

# handles = [(p6, p5), p1, p2, (p3, p4)]
# # _, labels = ax[0].get_legend_handles_labels()
# fig.legend(handles = handles, ['TW16-A', 'Sampler', 'Pump PP36', 'Reference sand'], #labels, 
#           handler_map = {tuple: mpl.legend_handler.HandlerTuple(None)},
#           fontsize = 16, loc = 'lower center', ncol = 3, bbox_to_anchor = (0.5, -0.13))

l = fig.legend([(p6, p5), p1, p2, (p3, p4)], ['TW16-A', 'Sampler', 'Pump', 'Reference sand'],
               handler_map={tuple: HandlerTuple(ndivide=None)},
               fontsize = 16, loc = 'lower center', ncol = 4, bbox_to_anchor = (0.5, -0.07))

fig.tight_layout()
figname = 'Time_Csand_D50'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 400, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight') 
# fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 300, bbox_inches='tight')


#%% Plot Time -  C sand
fig, ax = plt.subplots(1, 1, figsize = (8,4), dpi=300)

ax.plot(Time_datetime_freq2, C_sand_S2_D50_g_l, '.', markersize = 3,
        color = 'peru', label = 'fines')
ax.plot(TAAPS_freq2.iloc[:,1], TAAPS_freq2['Sand_concentration_g_l'], 'D', 
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
ax.plot(TAAPS_freq2.iloc[:,1], TAAPS_freq2['Sand_flux_kg_s'], 'D', 
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
ax.plot(TAAPS_freq2.iloc[:,1], TAAPS_freq2['D50_mum'], 'D', 
        color = 'black', label = 'sand')

ax.set_xlabel('Time', fontsize=16, weight = 'bold')
ax.set_ylabel('$\mathregular{\overline{D_{50}} \; (\mu m)}$', fontsize=16, weight = 'bold')
ax.xaxis.set_major_locator(md.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(md.DateFormatter('%d.%m.%Y'))
ax.set_xlim(Time_datetime_freq2[0], Time_datetime_freq2[-1])
ax.tick_params(axis='both', which='major', labelsize = 14)
ax.set_ylim(63,500)

fig.tight_layout()
figname = 'Time_D50_est_S2_D50_sand_dual'
fig.savefig(outpath_figures + '\\' + figname + '.png', dpi = 400, bbox_inches='tight')
# fig.savefig(outpath_figures+ '\\' + figname +  '.eps', dpi = 300, bbox_inches='tight') 
fig.savefig(outpath_figures+ '\\' + figname +  '.pdf', dpi = 300, bbox_inches='tight')