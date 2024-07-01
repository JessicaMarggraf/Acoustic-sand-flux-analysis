# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 16:30:55 2021

Copyright (C) 2024  INRAE

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

@author: jessica.laible

Prepares necessary input files for further analysis (Beam_averaged_attenuation_backscatter)

Input data: ASCII-converted raw data (.a1, .a2, .v1, .v2, .sen, .whd, .wad, .ssl)
            in one file per data type
"""
print('****************************************************************\n'
      '**********                                            **********\n'
      '**********             RUN 2C AQD APPEND              **********\n'
      '**********                                            **********\n'
      '****************************************************************\n')
print('General import')
# Load packages
import numpy as np
import pandas as pd
import tkinter as tk
import tkinter.filedialog as fd
import sys
import pickle 
from Functions import choice_freq, user_input_path, user_input_outpath
try:
    #%% 1. LOAD DATA
    print('Define paths and frequency...')
    freq_choice = choice_freq()
    freq_text = freq_choice.freq

    # Define input and output path
    path = user_input_path()
    outpath = user_input_outpath()

    #%%
    a1 = pd.read_csv(path + '\A1_' + freq_text + '.a1', sep='\s+',  header=None, error_bad_lines=False, engine='python')
    a2 = pd.read_csv(path + '\A2_' + freq_text + '.a2', sep='\s+',  header=None, error_bad_lines=False, engine='python')
    v1 = pd.read_csv(path + '\V1_' + freq_text + '.v1', sep='\s+',  header=None, error_bad_lines=False, engine='python')
    v2 = pd.read_csv(path + '\V2_' + freq_text + '.v2', sep='\s+',  header=None, error_bad_lines=False, engine='python')
    sen = pd.read_csv(path + '\SEN_' + freq_text + '.sen', sep='\s+',  header=None, error_bad_lines=False, engine='python')
    whd = pd.read_csv(path + '\WHD_' + freq_text + '.whd', sep='\s+',  header=None, error_bad_lines=False, engine='python')
    wad = pd.read_csv(path + '\WAD_' + freq_text + '.wad', sep='\s+',  header=None, error_bad_lines=False, engine='python')
    ssl = pd.read_csv(path + '\SSL_' + freq_text + '.ssl', sep="\\s{1,}", header=None, index_col=False)

    whd = whd.iloc[:,0:25]

    #%% 2. PREPARE BASIC INFORMATION
    print('Prepare data...')
    # Find instrument frequency
    # freq1_text = '400kHz'
    # freq2_text = '1MHz'

    if freq_text == '400kHz':
        freq = 400
    if freq_text == '1MHz':
        freq = 1000

    # burst
    burst_wad = 15

    # Control dataframe lengths
    if len(a1.iloc[:,1]) == len(a2.iloc[:,1]) == len(v1.iloc[:,1]) == len(v2.iloc[:,1]) == len(sen.iloc[:,1]) == False:
        sys.exit("Error message: Amplitude, Velocity, or Sen file lengths do not match. Check data.")

    #----------------------------------------------------------------------------
    # Create time stamps
    # for Amplitude and velocity using .sen - file
    time = []
    for i in range((len(sen))):
        timex = pd.Timestamp(sen[2][i], sen[0][i], sen[1][i], sen[3][i], sen[4][i], sen[5][i])
        time.append(timex)

    # for .whd using .sen - file
    time_whd = []
    for i in range((len(whd))):
        timex_whd = pd.Timestamp(whd[2][i], whd[0][i], whd[1][i], whd[3][i], whd[4][i], whd[5][i])
        time_whd.append(timex_whd)


    #%% Rearrange and organize data

    # Determine blanking and cell-size --> Supposing they are always the same!!!
    if freq == 1000:
        blanking = 0.2
        cellsize = 0.4
        number_cells = 64
    if freq == 400:
        blanking = 1.02
        cellsize = 0.5
        number_cells = 50

    blanking_list = [blanking]*len(a1)

    cellsize_list = [cellsize]*len(a1)

    # Create dataframe with all necessary data
    Amplitude_velocity = pd.concat([pd.DataFrame(time),pd.DataFrame(blanking_list), pd.DataFrame(cellsize_list),
                                    sen, a1, a2, v1, v2], axis = 1)

    # Add column names
    cell_number = np.linspace(1,number_cells,number_cells)
    cell_number = [int(cell_number[i]) for i in range(len(cell_number))]
    cell_number_str = [str(cell_number[i]) for i in range(len(cell_number))]

    amplitude_cols = ['Amplitude']*number_cells
    velocity_cols = ['Velocity']*number_cells
    beam1_cols = ['1_']*number_cells
    beam2_cols = ['2_']*number_cells

    cols_general = ['Time', 'Blanking', 'Cellsize', 'month', 'day', 'year', 'hour', 'minute', 'second', 'Error_code',
          'Status_code', 'Voltage', 'Sound_speed', 'Heading', 'Pitch', 'Roll', 'Pressure', 'Temperature',
          'Analog_in_1', 'Analog_in_2']
    colnamex = [cols_general +
          [amplitude_cols[i] + beam1_cols[i] + cell_number_str[i] for i in range(len(amplitude_cols))] +
          [amplitude_cols[i] + beam2_cols[i] + cell_number_str[i] for i in range(len(amplitude_cols))] +
          [velocity_cols[i] + beam1_cols[i] + cell_number_str[i] for i in range(len(amplitude_cols))]+
          [velocity_cols[i] + beam2_cols[i] + cell_number_str[i] for i in range(len(amplitude_cols))]]
    colnames = colnamex[0]

    Amplitude_velocity.columns = [colnames]


    # Create whd dataframe
    whd_export = pd.concat([pd.DataFrame(time_whd), whd], axis = 1)
    whd_export.columns = ['Time_whd', 'month', 'day', 'year', 'hour', 'minute', 'second', 'Burst',
                          'non', 'Cell_position', 'Voltage', 'Sound_speed', 'Heading', 'Pitch', 'Roll',
                          'min_Pressure', 'max_Pressure', 'Temperature', 'Cell_size', 'noise1', 'noise2',
                          'noise3', 'noise4', 'AST_start', 'AST_end', 'AST_clipped']


    #%% Export files
    print('Export Amp_vel and whd files...')
    Amplitude_velocity.to_csv(outpath + '\Amplitude_velocity_'+ str(freq) + '.csv', index=False)
    whd_export.to_csv(outpath + '\WHD_'+ str(freq) + '.csv', index=False)

    with open(outpath + '\Amplitude_velocity_'+ str(freq) + '.txt', "wb") as fp:
        pickle.dump(Amplitude_velocity, fp)
    with open(outpath + '\WHD_'+ str(freq) + '.txt', "wb") as fp:
        pickle.dump(whd_export, fp)

    print('============== end ==============')
    input('Press any key to close this window')

except Exception as e:
    print('Error : ' + e)
    input('Press any key to close this window')
