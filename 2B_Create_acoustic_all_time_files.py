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

Create all data files (concatenate monthly files)

"""

import glob
import shutil
from Functions import choice_freq, user_input_path, user_input_outpath

#%% Define paths and frequency
print('Define paths and frequency...')
freq_choice = choice_freq()
freq = freq_choice.freq

path = user_input_path()
outpath = user_input_outpath()

#%%  Create all data files
print(' Create all data files...')
#A1  utpath + '\\' +'*' + '\\' + *.a1'
with open(outpath + '\A1_' + freq + '.a1','wb') as wfd: # Folder and filename of output 
    for f in glob.glob(path + '\\' +'*' + '\\' + '*.a1'): # Input files (all files ending with ".a1" in \Concatenated)
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd)

#A2
with open(outpath + '\A2_' + freq + '.a2','wb') as wfd:
    for f in glob.glob(path + '\\' +'*' + '\\' + '*.a2'):
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd)

#V1
with open(outpath + '\V1_' + freq + '.v1','wb') as wfd:
    for f in glob.glob(path + '\\' +'*' + '\\' + '*.v1'):
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd)
            
#V2
with open(outpath + '\V2_' + freq + '.v2','wb') as wfd:
    for f in glob.glob(path + '\\' +'*' + '\\' + '*.v2'):
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd)
 
            shutil.copyfileobj(fd, wfd)
            
#hdr
with open(outpath + '\HDR_' + freq + '.hdr','wb') as wfd:
    for f in glob.glob(path + '\\' +'*' + '\\' + '*.hdr'):
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd)
            
#sen
with open(outpath + '\SEN_' + freq + '.sen','wb') as wfd:
    for f in glob.glob(path + '\\' +'*' + '\\' + '*.sen'):
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd)
            
#ssl
with open(outpath + '\SSL_' + freq + '.ssl','wb') as wfd:
    for f in glob.glob(path + '\\' +'*' + '\\' + '*.ssl'):
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd)
            
#wad
with open(outpath + '\WAD_' + freq + '.wad','wb') as wfd:
    for f in glob.glob(path + '\\' +'*' + '\\' + '*.wad'):
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd)
            
#whd
with open(outpath + '\WHD_' + freq + '.whd','wb') as wfd:
    for f in glob.glob(path + '\\' +'*' + '\\' + '*.whd'):
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd)
            
