# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 08:51:54 2021
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

Create monthly files out of ASCII-converted hourly data


"""
print('General imports')
import glob
import shutil
from Functions import user_input_2A, choice_freq, user_input_path, user_input_outpath, choice_unit

#%% Define paths and month
print('Import paths and frequency...')
tk_2A = user_input_2A()
tk_2A.open()
input_getted = tk_2A.input_map
month = input_getted[0]

freq_choice = choice_freq()
freq = freq_choice.freq

path = user_input_path() 
outpath = user_input_outpath()



#%% Create monthly files
print('Create monthly files...')
#A1 (Amplitude beam 1)
with open(outpath + '\\' + '\A1_' + freq + '_' + month + '.a1', 'wb') as wfd: # Specify folder and filename
    for f in glob.glob(path + '\\' +'*.a1'): # Specify folder, where all the (hourly) data are located
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd)

#A2 (Amplitude beam 2)
with open(outpath + '\\' + '\A2_' + freq + '_' + month + '.a2', 'wb') as wfd: # Specify folder and filename
    for f in glob.glob(path + '\\' + '*.a2'): # Specify folder, where all the (hourly) data are located
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd)    
            
#V1 (Velocity beam 1)
with open(outpath + '\\' + '\V1_' + freq + '_' + month + '.v1', 'wb') as wfd: # Specify folder and filename
    for f in glob.glob(path + '\\' +'*.v1'): # Specify folder, where all the (hourly) data are located
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd)

#V2 (Velocity beam 2)
with open(outpath + '\\' + '\V2_' + freq + '_' + month + '.v2', 'wb') as wfd: # Specify folder and filename
    for f in glob.glob(path + '\\' + '*.v2'): # Specify folder, where all the (hourly) data are located
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd)    
            
#hdr (Header -  en-tête)
with open(outpath + '\\' + '\HDR_' + freq + '_' + month + '.hdr', 'wb') as wfd: # Specify folder and filename
    for f in glob.glob(path + '\\' + '*.hdr'): # Specify folder, where all the (hourly) data are located
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd)
            
#sen (Sensor)
with open(outpath + '\\' + '\SEN_' + freq + '_' + month + '.sen', 'wb') as wfd: # Specify folder and filename
    for f in glob.glob(path + '\\'  +'*.sen'): # Specify folder, where all the (hourly) data are located
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd)
                        
#ssl
with open(outpath + '\\' +  '\SSL_' + freq + '_' + month + '.ssl', 'wb') as wfd: # Specify folder and filename
    for f in glob.glob(path + '\\' + '*.ssl'): # Specify folder, where all the (hourly) data are located
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd)
            
#wad
with open(outpath + '\\' +'\WAD_' + freq + '_' + month + '.wad', 'wb') as wfd: # Specify folder and filename
    for f in glob.glob(path + '\\' + '*.wad'): # Specify folder, where all the (hourly) data are located
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd)
            
#whd
with open(outpath + '\\' + '\WHD_' + freq + '_' + month + '.whd', 'wb') as wfd: # Specify folder and filename
    for f in glob.glob(path + '\\' +'*.whd'): # Specify folder, where all the (hourly) data are located
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd)
   
            
            
                    
            
