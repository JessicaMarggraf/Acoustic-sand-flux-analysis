# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 11:33:59 2021

@author: jessica.laible

Beam-averaged attenuation and backscatter

based on Ron Griffiths USGS 2021, Topping & Wright (2016) 

Input data: output data from AQD_append script (Amplitude_velocity_freq.txt and WHD_freq.txt)
Output data: Beam_averaged backscatter, relative backscatter, FCB, attenuation due to sediment 
             (AlphaSed) and water (AlphaW)

Necessary definitions: 
    - in and outpath
    - used frequencies (1.)
    - Scalefactor (1.)
    - Conditions for outliers (3.)
    - Guess noise floor offset (4.)

"""
print('****************************************************************\n'
      '**********                                            **********\n'
      '**********         RUN 3 BEAM AVERAGING TW16A         **********\n'
      '**********                                            **********\n'
      '****************************************************************\n')
print('General import')

# Load packages
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
import tkinter.filedialog
import sys
import matplotlib.dates as md
from datetime import date, timedelta, datetime, time
import pickle
import math
from classes.Choice_2A_monthly_files import user_input_path, user_input_outpath, user_input_outpath_figures, ask_float

#%% 1. LOAD AND PREPARE DATA
print('Define paths and choose files...')
# path_folder = user_input_path()
outpath = user_input_outpath()
outpath_figures = user_input_outpath_figures()

# Load data
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
filepath_amp_vel = tk.filedialog.askopenfilename(title='Choose Amplitude_velocity-file (.csv)')
Amp_vel_raw = pd.read_csv(filepath_amp_vel, sep=',')
colnames_Amp_vel_raw = list(Amp_vel_raw.columns.values)
root.destroy()

root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
filepath_Whd_wad = tk.filedialog.askopenfilename(title='Choose Whd-file (.csv)')
Whd_wad = pd.read_csv(filepath_Whd_wad, sep=',')
colnames_Whd_wad = list(Whd_wad.columns.values)
root.destroy()

# root = tk.Tk()
# root.withdraw()
# root.attributes("-topmost", True)
# filepath_stage = tk.filedialog.askopenfilename(title='Choose water stage data (.csv)')
# stage_data_raw = pd.read_csv(filepath_stage, sep=';')
# colnames_stage = list(stage_data_raw.columns.values)
# root.destroy()

#%% Prepare basic information
scalefactor = 0.43 # for Nortek instruments

# Find instrument frequency
#freq1_text = '1MHz'
freq1 = 400
#freq2_text = '400kHz'
freq2 = 1000

if str(freq1) in filepath_amp_vel:
    freq = 400
if str(freq2) in filepath_amp_vel:
    freq = 1000

# Verify if files with same frequency choosen
if str(freq1) in filepath_Whd_wad:
    freq_Whd_wad = 400
if str(freq2) in filepath_Whd_wad:
    freq_Whd_wad = 1000

if freq is not freq_Whd_wad:
    sys.exit("Error message: Frequencies do not match. Choose files with same frequency.")

#%% 2. PLOT DATA TO DETERMINE OUTLIERS
print('Plot instrument orientation...')

# Heading
fig, ax = plt.subplots(1, 1, figsize = (8,4), dpi=300)

ax.plot(Amp_vel_raw['Heading'],'.', label = 'Heading', color = 'grey')

ax.grid(linewidth = 0.2)
ax.set_xlabel('Time', fontsize=16, weight = 'bold')
ax.set_ylabel('Heading (°)', fontsize=16, weight = 'bold')
ax.tick_params(axis='both', which='major', labelsize = 14)

fig.tight_layout()
figname = 'Heading_non_filtered_'
fig.savefig(outpath_figures + '\\' + figname + str(freq) +'.png', dpi = 400, bbox_inches='tight')

# -------------------------------------------------------------------------------
# Pitch
fig, ax = plt.subplots(1, 1, figsize = (8,4), dpi=300)

ax.plot(Amp_vel_raw['Pitch'],'.', label = 'Pitch', color = 'grey')

ax.grid(linewidth = 0.2)
ax.set_xlabel('Time', fontsize=16, weight = 'bold')
ax.set_ylabel('Pitch (°)', fontsize=16, weight = 'bold')
ax.tick_params(axis='both', which='major', labelsize = 14)

fig.tight_layout()
figname = 'Pitch_non_filtered_'
fig.savefig(outpath_figures + '\\' + figname + str(freq) +'.png', dpi = 400, bbox_inches='tight')

# -------------------------------------------------------------------------------
# Roll
fig, ax = plt.subplots(1, 1, figsize = (8,4), dpi=300)

ax.plot(Amp_vel_raw['Roll'],'.', label = 'Roll', color = 'grey')

ax.set_xlabel('Time', fontsize=16, weight = 'bold')
ax.set_ylabel('Roll (°)', fontsize=16, weight = 'bold')
ax.tick_params(axis='both', which='major', labelsize = 14)

fig.tight_layout()
figname = 'Roll_non_filtered_'
fig.savefig(outpath_figures+ '\\' + figname + str(freq) + '.png', dpi = 400, bbox_inches='tight')

#%% 3. DELETE OUTLIERS
print('Define thresholds...')
limit_low_heading = ask_float(text='Lower threshold heading', title='Define instrument angle thresholds', default_value='328')
limit_upper_heading = ask_float(text='Upper threshold heading', title='Define instrument angle thresholds', default_value='350')
limit_low_pitch = ask_float(text='Lower threshold pitch', title='Define instrument angle thresholds', default_value='-1.5')
limit_upper_pitch = ask_float(text='Upper threshold pitch', title='Define instrument angle thresholds', default_value='2.5')
limit_low_roll = ask_float(text='Lower threshold roll', title='Define instrument angle thresholds', default_value='-3.3')
limit_upper_roll = ask_float(text='Upper threshold roll', title='Define instrument angle thresholds', default_value='3.3')

print('Delete outliers...')
Amp_vel = Amp_vel_raw.drop(Amp_vel_raw[Amp_vel_raw['Heading'] <= limit_low_heading.value].index, inplace = False)
Amp_vel = Amp_vel.drop(Amp_vel[Amp_vel['Heading'] >= limit_upper_heading.value].index, inplace = False)
Amp_vel = Amp_vel.drop(Amp_vel[Amp_vel['Pitch'] <= limit_low_pitch.value].index, inplace = False)
Amp_vel = Amp_vel.drop(Amp_vel[Amp_vel['Pitch'] >= limit_upper_pitch.value].index, inplace = False)
Amp_vel = Amp_vel.drop(Amp_vel[Amp_vel['Roll'] <= limit_low_roll.value].index, inplace = False)
Amp_vel = Amp_vel.drop(Amp_vel[Amp_vel['Roll'] >= limit_upper_roll.value].index, inplace = False)

no_deleted_data = int(len(Amp_vel_raw))-int(len(Amp_vel))
perc_deleted_data = np.round(no_deleted_data/int(len(Amp_vel_raw))*100, 2)
Amp_vel.reset_index(inplace=True)

#%% 4. PREPARE BASIC DATA (TIMESTAMP, AMPLITUDE, VELOCITY AND NOISE DATA)

print('Plot instrument orientation...')
# Define timestamps
Time = Amp_vel['Time']
Time_list = list(Time)
Time_datetime_PC = pd.to_datetime(Time_list)

Time_noise = Whd_wad['Time_whd']
Time_noise_list = list(Time_noise)
Time_noise_datetime_PC = pd.to_datetime(Time_noise_list)

# Correct time to UTC time (wrong time on computer on site)
time_diff_min = timedelta(minutes = 11)
time_diff_sec = timedelta(seconds = 14)
Time_datetime = Time_datetime_PC - time_diff_min - time_diff_sec
Time_noise_datetime = Time_noise_datetime_PC - time_diff_min - time_diff_sec

# Amplitude and velocity per beam
amplitude1 = Amp_vel.loc[:, Amp_vel.columns.str.startswith('Amplitude1')]
amplitude2 = Amp_vel.loc[:, Amp_vel.columns.str.startswith('Amplitude2')]

velocity1 = Amp_vel.loc[:, Amp_vel.columns.str.startswith('Velocity1')]
velocity2 = Amp_vel.loc[:, Amp_vel.columns.str.startswith('Velocity2')]

# Noise data per beam
noise_beam1 = list(Whd_wad['noise1'])
noise_beam2 = list(Whd_wad['noise2'])

# Choose noise floor offset
noise_floor_offset = 25 # Guess

# # Load stage data
# stage_data = stage_data_raw.drop(stage_data_raw[stage_data_raw['Value'] == '-9999'].index, inplace = False)
# stage = stage_data['Value']
# stage = stage.astype(float)

# Time_stage = stage_data['DateHeure']
# Time_stage_list = list(Time_stage)
# Time_stage_datetime = pd.to_datetime(Time_stage_list,format='%d.%m.%Y %H:%M')

# # Get stage at Acoustic time
# stage_time_freq = np.interp(Time_datetime, Time_stage_datetime,stage)

#%% 5. CONTROL PLOTS ON OUTLIERS

# Heading
fig, ax = plt.subplots(1, 1, figsize = (8,4), dpi=300)

ax.plot(Time_datetime, Amp_vel['Heading'],'.', label = 'Heading', color = 'grey')

ax.xaxis.set_major_locator(md.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(md.DateFormatter('%d-%m-%Y'))
ax.set_xlabel('Time', fontsize=16, weight = 'bold')
ax.set_ylabel('Heading (°)', fontsize=16, weight = 'bold')
ax.tick_params(axis='both', which='major', labelsize = 14)

fig.tight_layout()
figname = 'Heading_filtered_'
fig.savefig(outpath_figures + '\\' + figname + str(freq) + '.png', dpi = 400, bbox_inches='tight')

# -------------------------------------------------------------------------------
# Pitch
fig, ax = plt.subplots(1, 1, figsize = (8,4), dpi=300)

ax.plot(Time_datetime, Amp_vel['Pitch'],'.', label = 'Pitch', color = 'grey')

ax.xaxis.set_major_locator(md.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(md.DateFormatter('%d-%m-%Y'))
ax.set_xlabel('Time', fontsize=16, weight = 'bold')
ax.set_ylabel('Pitch (°)', fontsize=16, weight = 'bold')
ax.tick_params(axis='both', which='major', labelsize = 14)

fig.tight_layout()
figname = 'Pitch_filtered_'
fig.savefig(outpath_figures + '\\' + figname + str(freq) + '.png', dpi = 400, bbox_inches='tight')

# -------------------------------------------------------------------------------
# Roll
fig, ax = plt.subplots(1, 1, figsize = (8,4), dpi=300)

ax.plot(Time_datetime, Amp_vel['Roll'],'.', label = 'Roll', color = 'grey')

ax.xaxis.set_major_locator(md.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(md.DateFormatter('%d-%m-%Y'))
ax.set_xlabel('Time', fontsize=16, weight = 'bold')
ax.set_ylabel('Roll (°)', fontsize=16, weight = 'bold')
ax.tick_params(axis='both', which='major', labelsize = 14)

fig.tight_layout()
figname = 'Roll_filtered_'
fig.savefig(outpath_figures+ '\\' + figname + str(freq) + '.png', dpi = 400, bbox_inches='tight')


#%% 6. DETERMINE CELLSIZE, NOISE DATA AT AMPLITUDE MEASUREMENTS AND AVERAGE BEAMS
print('Average beams...')
# Calculate cell size along beam
# Define cell number and size for both instruments
cellsize_perpendicular = Amp_vel['Cellsize'][0]
cellsize_along_beam = cellsize_perpendicular * 1.1033779
numcell = int(colnames_Amp_vel_raw[-1][-2:])

# Determine blanking
blanking_perpendicular = Amp_vel['Blanking'][0]
blanking_along_beam = blanking_perpendicular * 1.1033779;

# Calculate cell distance
# distance along beam in meters to each cell
celldist_along_beam = [cellsize_along_beam*i + blanking_along_beam for i in range(0,numcell)]

# Average the two beams
# Average the beam count data, if only 1 beam is used AveCount is equal to that beam
AveCount = pd.DataFrame((np.array(amplitude1)+np.array(amplitude2))/2)
AveCount_db =  pd.DataFrame(((np.array(amplitude1)+np.array(amplitude2))/2)*scalefactor)

#----------------------------------------------------------------------------
# Noise data

# Using whd-wad-data
# a) Get noise data at beam data timestamps (if whd_wad-data used)
noise_beam1_time = np.interp(Time_datetime, Time_noise_datetime,noise_beam1)
noise_beam2_time = np.interp(Time_datetime, Time_noise_datetime,noise_beam2)

# b) Average the two beams
average_noise  = [noise_floor_offset + (noise_beam1_time[i] + noise_beam2_time[i])/2
                  for i in range(len(noise_beam1_time))]
average_noise_db  = [average_noise[i]*scalefactor
                  for i in range(len(average_noise))]
instrument_noise_db = [((noise_beam1_time[i] + noise_beam2_time[i])/2)*scalefactor
                  for i in range(len(noise_beam1_time))]


#%% 7. CALCULATE SNR AND INTERFERENCES FROM THE SURFACE
# print('Calculate signal-to-noise ratio SNR...')
# snr = [[AveCount_db.iloc[j,i]/average_noise_db[j]
#              for i in range(len(celldist_along_beam))]
#              for j in range(len(AveCount_db))]
# snr = pd.DataFrame(snr)

# # Determine SNR
# SNR_lar = snr[snr > 1.5]
# SNR_lar_trans = SNR_lar.transpose()

# # Determine last valid index (to be used to cutoff)
# last_valid_idx_SNR = SNR_lar_trans.apply(pd.DataFrame.last_valid_index).tolist()

# print('Calculate interferences with surface...')
# if freq == 400:
#     beam_angle_freq = 1.7
# if freq == 1000:
#     beam_angle_freq = 3.4
# beam_angle_freq_rad = beam_angle_freq*(math.pi/180)
# max_height_range_freq = math.sin(beam_angle_freq_rad)*celldist_along_beam[-1]

# # Calculate A, B and C depending on water height (Comprehensive Manual p. 27)
# A_freq = stage_time_freq-0.5
# B_freq = [(stage_time_freq[i]-0.5)/math.cos(45*(math.pi/180)) for i in range(len(stage_time_freq))]
# C_freq = [(stage_time_freq[i]-0.5)/math.cos(75*(math.pi/180)) for i in range(len(stage_time_freq))]

# # Calculate distance along beam where surface and acoustic beam meet and add buffer
# distance_along_beam_surface_freq = (stage_time_freq-0.5) / math.sin(beam_angle_freq_rad)
# buffer = distance_along_beam_surface_freq - 5

# # Create export data
# interfer_data = pd.DataFrame([list(A_freq),B_freq,C_freq, list(distance_along_beam_surface_freq), list(buffer)])
# interfer_data = interfer_data.transpose()
# interfer_data.columns = ['90°', '45°', '15°', 'crossing', 'buffer_5m']

# # Calculate end id
# end_idx_cross = []
# for j in range(len(interfer_data)):
#     hh = interfer_data['buffer_5m'][j]
#     hf = celldist_along_beam <= hh # buffer distance of 5 m before intersection of surface and beam
#     hi = [i for i, x in enumerate(hf) if x]
#     if hi == []:
#         hg = 0
#     else:
#         hg = hi[-1]
#     end_idx_cross.append(hg)
# plt.hist(end_idx_cross)

# # Determine lowest last valid index between last_valid_idx_SNR and end_idx_cross (interference)
# end_idxes = pd.DataFrame([end_idx_cross,last_valid_idx_SNR]).transpose()
# last_valid_idx = end_idxes.min(axis =1).tolist()

# end_idxes.columns = ['Last_valid_index_5m_buffer', 'Last_valid_index_SNR']
# end_idxes['Last_valid_distance_5m_buffer'] = [celldist_along_beam[int(end_idx_cross[i])] for i in range(len(end_idx_cross))]
# end_idxes['Last_valid_distance_SNR'] = [celldist_along_beam[int(last_valid_idx_SNR[i])] for i in range(len(last_valid_idx_SNR))]
# end_idxes['Date'] = Time_list
# end_idxes.to_csv(outpath + '\\' + 'Last_valid_index_uncorrected_' + str(freq) + '.csv', sep = ';')

#%% 8. CALCULATE ALPHA_WATER, FLUID-CORRECTED BACKSCATTER
print('Calculate AlphaW...')
# Calculate Alpha water from Schulkin and Marsh 1962, Topping and Wright appendix 4
AlphaW  = [((0.5*2.34e-6*(21.9*10**(6-1520/(Amp_vel['Temperature'][i]+273)))*freq**2)/
           ((21.9*10**(6-1520/(Amp_vel['Temperature'][i]+273)))**2+freq**2)+((3.38e-6)*(freq**2)/
           (21.9*10**(6-1520/(Amp_vel['Temperature'][i]+273))))*(1-6.54*2e-4))/0.115129255
           for i in range(len(Amp_vel['Temperature']))]

print('Calculate fluid-corrected backscatter...')
# Calculate acoustic backscatter
FluidCorrBackscatter_list = []
# znear = []
near = []
for i in range(0,len(AveCount)):
    inner = []
    for j in range(0,numcell):
        near = 1 # no correction of near field
        if AveCount.iloc[i,j] >= average_noise[i]:
            ll = scalefactor* AveCount.iloc[i,j] + 20*np.log10(near*
            celldist_along_beam[j]) + 2 * AlphaW[i] * celldist_along_beam[j] # equation 24
            inner.append(ll)
        else:
            li = math.nan
            inner.append(li)       
    FluidCorrBackscatter_list.append(inner)

FluidCorrBackscatter = pd.DataFrame(FluidCorrBackscatter_list, columns = range(1,numcell+1))

print('Use far field only')
# Exclude near field (using Downing near field distance)
nan_replace = [np.nan]*len(FluidCorrBackscatter)
if freq == 400: # 8.4 m
    FluidCorrBackscatter[1] = nan_replace
    FluidCorrBackscatter[2] = nan_replace
    FluidCorrBackscatter[3] = nan_replace
    FluidCorrBackscatter[4] = nan_replace
    FluidCorrBackscatter[5] = nan_replace
    FluidCorrBackscatter[6] = nan_replace    
    FluidCorrBackscatter[7] = nan_replace
    FluidCorrBackscatter[8] = nan_replace
    FluidCorrBackscatter[9] = nan_replace
    FluidCorrBackscatter[10] = nan_replace
    FluidCorrBackscatter[11] = nan_replace
    FluidCorrBackscatter[12] = nan_replace
    FluidCorrBackscatter[13] = nan_replace
if freq == 1000: # 1.31 m
    FluidCorrBackscatter[1] = nan_replace
    FluidCorrBackscatter[2] = nan_replace
    FluidCorrBackscatter[3] = nan_replace


#%% 9. CORRECT TAIL in FCB (where value - nan - value - nan)
print('Correct fluid-corrected backscatter...')
# Determine first and last valid index
FluidCorrBackscatter_trans = FluidCorrBackscatter.transpose()
last_valid_idx = FluidCorrBackscatter_trans.apply(pd.DataFrame.last_valid_index)
first_valid_idx = FluidCorrBackscatter_trans.apply(pd.DataFrame.first_valid_index)

# Create dataframe without rows containing nans only
only_nan_FCB = FluidCorrBackscatter[FluidCorrBackscatter.isna().all(axis=1)]
only_nan_FCB_idx = only_nan_FCB.index.values.tolist()
FluidCorrBackscatter_no_nans = FluidCorrBackscatter.drop(only_nan_FCB_idx)
FluidCorrBackscatter_no_nans.reset_index(inplace=True)
FluidCorrBackscatter_no_nans_index = list(FluidCorrBackscatter_no_nans['index'])
FluidCorrBackscatter_no_nans = FluidCorrBackscatter_no_nans.drop(['index'], axis = 1)

FluidCorrBackscatter_no_nans_trans = FluidCorrBackscatter_no_nans.transpose()
last_valid_idx_no_nans = FluidCorrBackscatter_no_nans_trans.apply(pd.DataFrame.last_valid_index)
first_valid_idx_no_nans = FluidCorrBackscatter_no_nans_trans.apply(pd.DataFrame.first_valid_index)

# Find rows needing correction (where value - nan - value - nan)
nan_in_valids = [] # returns index in FluidCorrBackscatter_no_nans
for i in range(len(FluidCorrBackscatter_no_nans)):
    nn = FluidCorrBackscatter_no_nans.iloc[i,int(first_valid_idx_no_nans[i]):int(last_valid_idx_no_nans[i])]
    nnan = nn.isna().any()
    nan_in_valids.append(nnan)

# Determine adjusted last valid index (so that values appearing after the first nan are excluded)
index_nan_in_valids = [i for i, x in enumerate(nan_in_valids) if x]

last_valid_index_adjust = []
for i in range(len(index_nan_in_valids)):
    ll = FluidCorrBackscatter_no_nans.iloc[index_nan_in_valids[i],int(first_valid_idx_no_nans[index_nan_in_valids[i]]):int(last_valid_idx_no_nans[index_nan_in_valids[i]])]
    ll.reset_index(inplace=True, drop = True)
    lll = np.isnan(ll).argmax(axis=0)
    lll = lll + int(first_valid_idx_no_nans[index_nan_in_valids[i]])
    last_valid_index_adjust.append(lll)

# Replace initial last valid index by adjusted last valid index
index_change_last_valid_index = [FluidCorrBackscatter_no_nans_index[index_nan_in_valids[i]]
                                 for i in range(len(index_nan_in_valids))]
last_valid_idx = pd.DataFrame(last_valid_idx)
for i in range(len(last_valid_index_adjust)):
    last_valid_idx.loc[index_change_last_valid_index[i]] = last_valid_index_adjust[i]
last_valid_idx = last_valid_idx.squeeze()

# Calculate number of valid values per row (measurement)
number_valid_values = last_valid_idx-first_valid_idx


#%% 10. CORRECT FCB
print('Correct fluid-corrected backscatter...')
ff = []
indexes_used_corr = []
for j in range(len(FluidCorrBackscatter)):
    fff = []
    if number_valid_values[j] >= 2:
        for i in np.arange(0,numcell):
            if i > int(last_valid_idx[j])-1:
                fe = np.nan
            else:
                fe = FluidCorrBackscatter.iloc[j,i]
            fff.append(fe)
        ff.append(fff)
        indexes_used_corr.append(j)

FluidCorrBackscatter_corr = pd.DataFrame(ff, columns = range(1,numcell+1))

# Keep only measurements and data where AlphaSed is calculated
Time_list_corr = [Time_list[indexes_used_corr[i]]
              for i in range(len(indexes_used_corr))]
AlphaW_corr = [AlphaW[indexes_used_corr[i]]
              for i in range(len(indexes_used_corr))]
Temperature_corr = [Amp_vel['Temperature'][indexes_used_corr[i]]
              for i in range(len(indexes_used_corr))]
instrument_noise_db_corr = [instrument_noise_db[indexes_used_corr[i]]
              for i in range(len(indexes_used_corr))]
average_noise_db_corr = [average_noise_db[indexes_used_corr[i]]
              for i in range(len(indexes_used_corr))]
# interfer_data_corr = [interfer_data.iloc[indexes_used_corr[i],:]
#               for i in range(len(indexes_used_corr))]
# interfer_data_corr = pd.DataFrame(interfer_data_corr)


#%% 11. DETERMINE ALPHASED

print('Determine AlphaSed...')
# Determine new first and last valid indexes
FluidCorrBackscatter_trans_corr = FluidCorrBackscatter_corr.transpose()
last_valid_idx_corr = FluidCorrBackscatter_trans_corr.apply(pd.DataFrame.last_valid_index)
first_valid_idx_corr = FluidCorrBackscatter_trans_corr.apply(pd.DataFrame.first_valid_index)
last_valid_idx_corr = [last_valid_idx_corr[i]-1 for i in range(len(last_valid_idx_corr))]
first_valid_idx_corr = [first_valid_idx_corr[i]-1 for i in range(len(first_valid_idx_corr))]

from sklearn.linear_model import LinearRegression
# Fit linear regression
slope = []
intercept = []
r2 = []
for i in range(len(FluidCorrBackscatter_corr)):
    x= np.array([celldist_along_beam[int(first_valid_idx_corr[i]):int(last_valid_idx_corr[i])]]).reshape(-1,1)
    fcb_corr = np.array([FluidCorrBackscatter_corr.iloc[i,int(first_valid_idx_corr[i]):int(last_valid_idx_corr[i])]]).reshape(-1,1)
    model_sk = LinearRegression().fit(x,fcb_corr)
    r_sq = model_sk.score(x, fcb_corr)
    r2.append(r_sq)
    slope_j = model_sk.coef_
    slope_j = float(slope_j)
    slope.append(slope_j)
    intercept_j = model_sk.intercept_
    intercept.append(intercept_j)


AlphaSed = [abs(slope[i])/2 for i in range(len(slope))]

#%% 12. DETERMINE BACKSCATTER
print('Determine backscatter B in each valid cell...')
# Calculate Backscatter in dB in each cell
CelldB_list = []
for i in range(0,len(FluidCorrBackscatter_corr)):
    inner = []
    for j in range(0,numcell):
        if FluidCorrBackscatter.iloc[i,j] == np.nan or AlphaSed[i] == np.nan:
            ll = np.nan
            inner.append(ll)
        else:
            li = FluidCorrBackscatter_corr.iloc[i,j]+ AlphaSed[i]* 2 * celldist_along_beam[j]
            inner.append(li)
    CelldB_list.append(inner)

CelldB = pd.DataFrame(CelldB_list, columns = range(1,numcell+1))

print('Determine beam-averaged backscatter')
# Calculate beam averaged backscatter
CelldBAve = list(CelldB.mean(axis = 1))

#%% 13. EXPORT DATA
print('Export data...')
export_data = pd.concat([pd.DataFrame(Time_list_corr), pd.DataFrame(AlphaSed), pd.DataFrame(CelldBAve), pd.DataFrame(AlphaW_corr),
                         pd.DataFrame(Temperature_corr), pd.DataFrame(instrument_noise_db_corr), pd.DataFrame(average_noise_db_corr)], axis = 1)
export_data.columns = (['Date', 'Alpha Sediment (dB/m)', 'Beam-Averaged Backscatter (dB)', 'AlphaW', 'Temperature',
                        'Instrument background (dB)', 'Effective background (dB)'])
export_data.to_csv(outpath + '\Beam_averaged_attenuation_backscatter_'+ str(freq) + '.csv', sep= ';', index=False)

with open(outpath + '\Beam_averaged_attenuation_backscatter_'+ str(freq) + '.txt', "wb") as fp:
    pickle.dump(export_data, fp)

with open(outpath + '\FluidCorrBackscatter_'+ str(freq) + '.txt', "wb") as fp:
    pickle.dump(FluidCorrBackscatter_corr, fp)
FluidCorrBackscatter_corr.to_csv(outpath + '\FluidCorrBackscatter_'+ str(freq) + '.csv', sep= ';', index=False)

with open(outpath + '\CelldB_'+ str(freq) + '.txt', "wb") as fp:
    pickle.dump(CelldB, fp)
CelldB.to_csv(outpath + '\CelldB_'+ str(freq) + '.csv', sep= ';', index=False)

with open(outpath + '\AveCount_db_'+ str(freq) + '.txt', "wb") as fp:
    pickle.dump(AveCount_db, fp)
AveCount_db.to_csv(outpath + '\AveCount_db_'+ str(freq) + '.csv', sep= ';', index=False)

with open(outpath + '\Celldist_along_beam_'+ str(freq) + '.txt', "wb") as fp:
    pickle.dump(celldist_along_beam, fp)
pd.DataFrame(celldist_along_beam).to_csv(outpath + '\Celldist_along_beam_'+ str(freq) + '.csv', sep= ';', index=False)

with open(outpath + '\Time_datetime_AveCount_db_'+ str(freq) + '.txt', "wb") as fp:
    pickle.dump(Time_datetime, fp)
pd.DataFrame(Time_datetime).to_csv(outpath + '\Time_datetime_AveCount_db_'+ str(freq) + '.csv', sep= ';', index=False)

# interfer_data_corr.to_csv(outpath + '\\' +'Usable_part_surface_' + str(freq) + '.csv', sep = ';')

last_valid_distance = [celldist_along_beam[int(last_valid_idx_corr[i])] for i in range(len(last_valid_idx_corr))]
last_valid_idx_corr_export = pd.DataFrame([Time_list_corr, last_valid_idx_corr, last_valid_distance]).transpose()
last_valid_idx_corr_export.columns = ['Date', 'Last_valid_index', 'Last_valid_celldistance_along_beam']
last_valid_idx_corr_export.to_csv(outpath + '\\' + 'Last_valid_index_corrected_' + str(freq) + '.csv', sep = ';')

# indexes = np.arange(0,len(interfer_data),1)
# indexes_not_used = list(set(indexes) - set(indexes_used_corr))
# Time_list_not_used = [Time_list[indexes_not_used[i]]
#               for i in range(len(indexes_not_used))]
# pd.DataFrame(Time_list_not_used).to_csv(outpath + '\\' + 'Time_list_not_used_beam_averaging_' + str(freq) + '.csv', sep = ';')

print('============== end ==============')

