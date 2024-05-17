# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 15:56:57 2022

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

Creates the "samples" file using the outputs of the sand flux analysis program 
developed by INRAE: https://gitlab.irstea.fr/jessica.laible/analysis-solid-gauging


@author: jlaible
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import pickle
from matplotlib import cm
from matplotlib.colors import ListedColormap as mpl_colors_ListedColormap 
from scipy.interpolate import Akima1DInterpolator
from matplotlib.ticker import FixedLocator, FixedFormatter

#%% Change the following pathes

sampler = ['BD', 'P72', 'P6']
path_folder = r'C:\Users\jessica.laible\Documents\Mesures\Manips&Sites\Grenoble_Campus'
outpath = r'C:\Users\jessica.laible\Documents\Mesures\Manips&Sites\Grenoble_Campus\Autres'
outpath_figures = r'C:\Users\jessica.laible\Documents\Mesures\Manips&Sites\Grenoble_Campus\Autres\Figures'

# Path où se trouvent les fichiers avec les dates de campagne de mesure séparés en préleveurs
sampling_dates_BD = pd.read_csv(r'C:\Users\jessica.laible\Documents\Mesures\Manips&Sites\Grenoble_Campus\Autres\Sampling_dates_' + sampler[0] +'.csv', sep = ';')
sampling_dates_P72 = pd.read_csv(r'C:\Users\jessica.laible\Documents\Mesures\Manips&Sites\Grenoble_Campus\Autres\Sampling_dates_' + sampler[1] +'.csv', sep = ';')
sampling_dates_P6 = pd.read_csv(r'C:\Users\jessica.laible\Documents\Mesures\Manips&Sites\Grenoble_Campus\Autres\Sampling_dates_' + sampler[2] +'.csv', sep = ';')


#%% Load all Export data from sampling campaigns

sampler = ['BD', 'P72', 'P6']

ISO_size_classes = pd.read_csv(outpath + '\\ISO_size_classes.csv', sep = ';')
ISO_size_classes = ISO_size_classes.iloc[:,1]
ISO_size_classes = [i for i in ISO_size_classes]

# For classified distribution
size_classes_inf = np.array(ISO_size_classes[0:-1])* 1e-6 / 2
size_classes_sup = np.array(ISO_size_classes[1:])* 1e-6 / 2
size_classes_center = [10**((np.log10(size_classes_inf[i]) + np.log10(size_classes_sup[i]))/2) 
                     for i in range(len(size_classes_inf))]
size_classes_center= np.array(size_classes_center)

colnamess1 = ['Date', 'Start_sampling', 'End_sampling', 'Sampler', 'Q_sampling_m3_s', 'Stage_sampling_m', 'spm_sampling_g_l', 
                                   'Sand_concentration_g_l', 'Sand_flux_kg_s', 'No_samples_used_for_ISO_gsd', 'D10_mum', 'D50_mum', 'D90_mum',
                                   'Fine_concentration_g_l', 'Fine_flux_kg_s','U_C', 'U_Q', 'U_F', 'Method']
colnamess = colnamess1 + ISO_size_classes

#%% BD
datee_BD = sampling_dates_BD['Sampling_dates_2']
samples_BD = pd.DataFrame(columns = colnamess1, index = datee_BD)
samples_BD_ISO = pd.DataFrame(columns = colnamess, index = datee_BD)
for i in range(len(datee_BD)): # sampling_dates   
    with open(path_folder + '\\' + str(datee_BD.iloc[i]) + '_Campus\\' + str(datee_BD.iloc[i]) + '_Analysis_solid_gauging_' + sampler[0] +'.txt', "rb") as fp:   
        export_data = pickle.load(fp)
        #ISO_gsd_all_BD.append(export_data.ISO_mean_gsd)
       
        start_time = np.min(export_data.analysis_data['Time']) 
        end_time = np.max(export_data.analysis_data['Time'])        
        Q = (np.nanmean(export_data.Q_sampling['Value']))
        stage = (np.nanmean(export_data.stage_sampling['Value']))
        spm = (np.nanmean(export_data.spm_sampling['Value']))
        if export_data.summary_SDC is not None:
            sand_C = export_data.summary_SDC['Conc_mean_SDC_g_l'].iloc[0]        
            sand_flux = export_data.summary_SDC['total_sand_flux_SDC_kg_s'].iloc[0]
            methodi = 'SDC'
        else:
            sand_C = export_data.summary_NN['NN_mean_sand_conc_g_l'].iloc[0]
            sand_flux = export_data.summary_NN['NN_total_sand_flux_kg_s'].iloc[0]
            methodi = 'NN'
        if export_data.uncertainty is not None:
            U_C = export_data.uncertainty['U\'_C'].iloc[0]
            U_Q = export_data.uncertainty['U\'_Q'].iloc[0]
            U_F = export_data.uncertainty['U\'F'].iloc[0]
        ISO_gsd = export_data.ISO_mean_gsd
        if ISO_gsd is not None:
            D10 = np.interp(10, ISO_gsd.iloc[0,:], ISO_size_classes)
            D50 = np.interp(50, ISO_gsd.iloc[0,:], ISO_size_classes)
            D90 = np.interp(90, ISO_gsd.iloc[0,:], ISO_size_classes)
            ISO_gsdd = export_data.ISO_mean_gsd.iloc[0]
        else: 
            D10 = None
            D50 = None
            D90 = None            
            ISO_gsdd = [None]*100
        number_samples_for_ISO = export_data.ISO_mean_gsd_number
        fine_C = np.nan
        fine_flux = np.nan                          
   
    samples_a = pd.DataFrame([datee_BD.iloc[i], start_time, end_time, 'BD', Q, stage, spm, sand_C, 
                                  sand_flux, number_samples_for_ISO, D10, D50, D90, fine_C, fine_flux, 
                                  U_C, U_Q, U_F, methodi]).transpose()   
    samples_b = pd.concat([samples_a, pd.DataFrame(ISO_gsdd).transpose()], axis = 1)
    samples_BD_ISO.iloc[i,:] = samples_b
    samples_BD.iloc[i,:] = samples_a
    
samples_BD.to_csv(outpath + '\\Samples_' + sampler[0] +'.csv', sep = ';')
samples_BD_ISO.to_csv(outpath + '\\Samples_' + sampler[0] +'_ISO_GSD.csv', sep = ';')

#%% P72
datee_P72 = sampling_dates_P72['Sampling_dates_2']
samples_P72 = pd.DataFrame(columns = colnamess1, index = datee_P72)
samples_P72_ISO = pd.DataFrame(columns = colnamess, index = datee_P72)
for i in range(len(datee_P72)):  
    with open(path_folder + '\\' + str(datee_P72.iloc[i]) + '_Campus\\' + str(datee_P72.iloc[i]) + '_Analysis_solid_gauging_' + sampler[1] +'.txt', "rb") as fp:   
        export_data = pickle.load(fp)       
        start_time = np.min(export_data.analysis_data['Time']) 
        end_time = np.max(export_data.analysis_data['Time']) 
        #sampler = max(set(list(analysis_data['Sampler'])), key=list(analysis_data['Sampler']).count).strip()
        Q = (np.nanmean(export_data.Q_sampling['Value']))
        stage = (np.nanmean(export_data.stage_sampling['Value']))
        spm = (np.nanmean(export_data.spm_sampling['Value']))
        if export_data.summary_SDC is not None:
            sand_C = export_data.summary_SDC['Conc_mean_SDC_g_l'].iloc[0]        
            sand_flux = export_data.summary_SDC['total_sand_flux_SDC_kg_s'].iloc[0]
            methodi = 'SDC'
        else:
            sand_C = export_data.summary_segment['Conc_mean_segment'].iloc[0]
            sand_flux = export_data.summary_segment['total_sand_flux_segment_kg_s'].iloc[0]
            methodi = 'Segment'
        ISO_gsd = export_data.ISO_mean_gsd
        if ISO_gsd is not None:
            D10 = np.interp(10, ISO_gsd.iloc[0,:], ISO_size_classes)
            D50 = np.interp(50, ISO_gsd.iloc[0,:], ISO_size_classes)
            D90 = np.interp(90, ISO_gsd.iloc[0,:], ISO_size_classes)            
            ISO_gsdd = export_data.ISO_mean_gsd.iloc[0]
        else: 
            D10 = None
            D50 = None
            D90 = None
            ISO_gsdd = [None]*100
        number_samples_for_ISO = export_data.ISO_mean_gsd_number 
        if export_data.summary_fine is not None:
            fine_C = pd.DataFrame(export_data.summary_fine['Conc_mean_SDC_fines_g_l']).iloc[0,0]
            fine_flux = pd.DataFrame(export_data.summary_fine['total_fine_flux_SDC_kg_s']).iloc[0,0]
        if export_data.summary_fine is None:
            fine_C =  export_data.summary_NN['NN_mean_fine_conc_g_l'].iloc[0]
            fine_flux = export_data.summary_NN['NN_total_fine_flux_kg_s'].iloc[0]        
        if export_data.uncertainty is not None:
            U_C = export_data.uncertainty['U\'_C'].iloc[0]
            U_Q = export_data.uncertainty['U\'_Q'].iloc[0]
            U_F = export_data.uncertainty['U\'F'].iloc[0]
   
    samples_a = pd.DataFrame([datee_P72.iloc[i], start_time, end_time, 'P72', Q, stage, spm, sand_C, 
                                  sand_flux, number_samples_for_ISO, D10, D50, D90, fine_C, fine_flux, 
                                  U_C, U_Q, U_F, methodi]).transpose()
    samples_b = pd.concat([samples_a, pd.DataFrame(ISO_gsdd).transpose()], axis = 1)
    samples_P72.iloc[i,:] = samples_a
    samples_P72_ISO.iloc[i,:] = samples_b
  
samples_P72.to_csv(outpath + '\\Samples_' + sampler[1] +'.csv', sep = ';')
samples_P72_ISO.to_csv(outpath + '\\Samples_' + sampler[1] +'_ISO_GSD.csv', sep = ';')

#%% P6
datee_P6 = sampling_dates_P6['Sampling_dates_2']
gauging_P6 = sampling_dates_P6['Gauging']
samples_P6_ISO = pd.DataFrame(columns = colnamess, index = datee_P6)
samples_P6 = pd.DataFrame(columns = colnamess1, index = datee_P6)
for i in range(len(datee_P6)):  
    with open(path_folder + '\\' + str(datee_P6.iloc[i]) + '_Campus\\' + str(gauging_P6.iloc[i]) + '_Analysis_solid_gauging_' + sampler[2] +'.txt', "rb") as fp:   
        export_data = pickle.load(fp)        
        start_time = np.min(export_data.analysis_data['Time']) 
        end_time = np.max(export_data.analysis_data['Time']) 
        if export_data.Q_sampling is not None:
            Q = (np.nanmean(export_data.Q_sampling['Value']))
        if export_data.stage_sampling is not None:
            stage = (np.nanmean(export_data.stage_sampling['Value']))
        if export_data.spm_sampling is not None:
            spm = (np.nanmean(export_data.spm_sampling['Value']))
        if export_data.summary_SDC is not None:
            sand_C = export_data.summary_SDC['Conc_mean_SDC_g_l'].iloc[0]       
            sand_flux = export_data.summary_SDC['total_sand_flux_SDC_kg_s'].iloc[0]
            methodi = 'SDC'
        else:
            sand_C = export_data.summary_segment['Conc_mean_segment'].iloc[0]
            sand_flux = export_data.summary_segment['total_sand_flux_segment_kg_s'].iloc[0]
            methodi = 'Segment'
        ISO_gsd = export_data.ISO_mean_gsd
        if ISO_gsd is not None:
            D10 = np.interp(10, ISO_gsd.iloc[0,:], ISO_size_classes)
            D50 = np.interp(50, ISO_gsd.iloc[0,:], ISO_size_classes)
            D90 = np.interp(90, ISO_gsd.iloc[0,:], ISO_size_classes)
            ISO_gsdd = export_data.ISO_mean_gsd.iloc[0]
        else: 
            D10 = None
            D50 = None
            D90 = None
            ISO_gsdd = [None]*100
        number_samples_for_ISO = export_data.ISO_mean_gsd_number    
        if export_data.summary_fine is not None:
            fine_C = pd.DataFrame(export_data.summary_fine['Conc_mean_SDC_fines_g_l']).iloc[0,0]
            fine_flux = pd.DataFrame(export_data.summary_fine['total_fine_flux_SDC_kg_s']).iloc[0,0]
        if export_data.summary_fine is None:
            fine_C =  export_data.summary_NN['NN_mean_fine_conc_g_l'].iloc[0]
            fine_flux = export_data.summary_NN['NN_total_fine_flux_kg_s'].iloc[0]  
        if export_data.uncertainty is not None:
            U_C = export_data.uncertainty['U\'_C'].iloc[0]
            U_Q = export_data.uncertainty['U\'_Q'].iloc[0]
            U_F = export_data.uncertainty['U\'F'].iloc[0]                
   
    samples_a = pd.DataFrame([datee_P6.iloc[i], start_time, end_time, 'P6', Q, stage, spm, sand_C, 
                                  sand_flux, number_samples_for_ISO, D10, D50, D90, fine_C, fine_flux, 
                                  U_C, U_Q, U_F, methodi]).transpose()   
    samples_b = pd.concat([samples_a, pd.DataFrame(ISO_gsdd).transpose()], axis = 1)
    samples_P6.iloc[i,:] = samples_a
    samples_P6_ISO.iloc[i,:] = samples_b
    
samples_P6['Q_sampling_m3_s'].iloc[-1]= 450 ########!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
samples_P6_ISO['Q_sampling_m3_s'].iloc[-1]= 450 ########!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

samples_P6.to_csv(outpath + '\\Samples_' + sampler[2] +'.csv', sep = ';')
samples_P6_ISO.to_csv(outpath + '\\Samples_' + sampler[2] +'_ISO_GSD.csv', sep = ';')

#%% Concat P72 and P6
# without ISO GSD
samples_P72_P6 = pd.concat([samples_P72, samples_P6])
samples_P72_P6.reset_index(drop=True, inplace=True)
samples_BD.reset_index(drop=True, inplace=True)
samples_P72.reset_index(drop=True, inplace=True)
samples_P6.reset_index(drop=True, inplace=True)

# Concat all samples
samples_all = pd.concat([samples_P72_P6, samples_BD])
samples_all.reset_index(drop=True, inplace=True)

# calculate absolute uncertainties
err_F_BD = samples_BD['U_F']/100*samples_BD['Sand_flux_kg_s']
err_C_BD = samples_BD['U_C']/100*samples_BD['Sand_concentration_g_l']
err_Q_BD = samples_BD['U_Q']/100*samples_BD['Q_sampling_m3_s']
err_F_P6 = samples_P6['U_F']/100*samples_P6['Sand_flux_kg_s']
err_C_P6 = samples_P6['U_C']/100*samples_P6['Sand_concentration_g_l']
err_Q_P6 = samples_P6['U_Q']/100*samples_P6['Q_sampling_m3_s']
err_F_P72 = samples_P72['U_F']/100*samples_P72['Sand_flux_kg_s']
err_C_P72 = samples_P72['U_C']/100*samples_P72['Sand_concentration_g_l']
err_Q_P72 = samples_P72['U_Q']/100*samples_P72['Q_sampling_m3_s']
err_F_P72_P6 = pd.concat([err_F_P72, err_F_P6])
err_F_P72_P6.reset_index(drop = True, inplace = True)
err_Q_P72_P6 = pd.concat([err_Q_P72, err_Q_P6])
err_Q_P72_P6.reset_index(drop = True, inplace = True)

# with ISO GSD
samples_P72_P6_ISO = pd.concat([samples_P72_ISO, samples_P6_ISO])
samples_P72_P6_ISO.reset_index(drop=True, inplace=True)
samples_BD_ISO.reset_index(drop=True, inplace=True)
samples_P72_ISO.reset_index(drop=True, inplace=True)
samples_P6_ISO.reset_index(drop=True, inplace=True)

# Concat all samples
samples_all_ISO = pd.concat([samples_P72_P6_ISO, samples_BD_ISO])
samples_all_ISO.reset_index(drop=True, inplace=True)

# calculate absolute uncertainties
err_F_BD_ISO = samples_BD_ISO['U_F']/100*samples_BD_ISO['Sand_flux_kg_s']
err_C_BD_ISO = samples_BD_ISO['U_C']/100*samples_BD_ISO['Sand_concentration_g_l']
err_Q_BD_ISO = samples_BD_ISO['U_Q']/100*samples_BD_ISO['Q_sampling_m3_s']
err_F_P6_ISO = samples_P6_ISO['U_F']/100*samples_P6_ISO['Sand_flux_kg_s']
err_C_P6_ISO = samples_P6_ISO['U_C']/100*samples_P6_ISO['Sand_concentration_g_l']
err_Q_P6_ISO = samples_P6_ISO['U_Q']/100*samples_P6_ISO['Q_sampling_m3_s']
err_F_P72_ISO = samples_P72_ISO['U_F']/100*samples_P72_ISO['Sand_flux_kg_s']
err_C_P72_ISO = samples_P72_ISO['U_C']/100*samples_P72_ISO['Sand_concentration_g_l']
err_Q_P72_ISO = samples_P72_ISO['U_Q']/100*samples_P72_ISO['Q_sampling_m3_s']
err_F_P72_P6_ISO = pd.concat([err_F_P72, err_F_P6_ISO])
err_F_P72_P6_ISO.reset_index(drop = True, inplace = True)
err_Q_P72_P6_ISO = pd.concat([err_Q_P72, err_Q_P6_ISO])
err_Q_P72_P6_ISO.reset_index(drop = True, inplace = True)

#%% Calculate sigma 
def compare_malvern(sizes_malvern, proba_malvern):
       # Transforming sizes to log
    log_sizes_malvern = np.log(sizes_malvern)
    
    #  Cumulative fraction
    malvern_cumul = np.cumsum(proba_malvern)

    # Interpolation model
    f_logsize_malvern = Akima1DInterpolator(log_sizes_malvern, malvern_cumul)

    ####  Interpolation ###

    # Re-sampling size classes
    # Phi steps
    log_scale = 0.01
    # Computing maximum phi (minimum diameter)
    log_min = np.log(0.01* 1e-6)
    # Computing minimum phi (maximum diameter)
    log_max = np.log(3500* 1e-6)
    # Creating an array of equally spaced phi
    log_steps_array = np.arange(log_min, log_max, log_scale)

    # Interpolating
    cumul_log_resampled_malvern = f_logsize_malvern(log_steps_array)

    # Computing the volume fraction resampled of each class
    proba_vol_resampled_malvern = np.diff(cumul_log_resampled_malvern)
    
    return(proba_vol_resampled_malvern)


# calculate pdf from cdf
samples_all_gsd = samples_all_ISO.dropna(subset = ['D10_mum'], how = 'any')
ISO_pdf = []
for i in range(len(samples_all_gsd)):
    ISO_pdf1 =  samples_all_gsd.iloc[i,19:]
    ISO_pdf2 = [ISO_pdf1.iloc[i+1] - ISO_pdf1.iloc[i] for i in range(len(ISO_pdf1)-1)]
    ISO_pdf2.append(0)
    ISO_pdf.append(ISO_pdf2)
ISO_pdf = pd.DataFrame(ISO_pdf)

# Resample 
resampled = [compare_malvern(size_classes_center, ISO_pdf.iloc[i,1:])
              for i in range(len(samples_all_gsd))]
resampled = pd.DataFrame(resampled)

# Calculate sigma 
sigma = [np.exp(np.nanstd(resampled.iloc[i,:]))/2 #VERIFY!!!! PROBABLY WRONG!
          for i in range(len(resampled))]

indexx = samples_all_gsd.index
ll = [0]*len(samples_all)
for i in range(len(indexx)):
    ll[indexx[i]] = np.round(sigma[i],3)
ll = [i if i != 0 else None for i in ll]

samples_all_ISO['sigma_mum'] = ll
samples_all['sigma_mum'] = ll

#%% Export data
samples_all = samples_all.applymap(lambda x: round(x, 3) if isinstance(x, (float, int)) else x)
samples_all_ISO = samples_all_ISO.applymap(lambda x: round(x, 3) if isinstance(x, (float, int)) else x)
samples_all['S'] = samples_all['Fine_concentration_g_l']/samples_all['Sand_concentration_g_l']
samples_all_ISO['S'] = samples_all_ISO['Fine_concentration_g_l']/samples_all_ISO['Sand_concentration_g_l']

samples_all.to_csv(outpath + '\\Samples_HADCP.csv', sep = ';')
samples_all_ISO.to_csv(outpath + '\\Samples_HADCP_ISO_GSD.csv', sep = ';')

# %% PLOT Flux - Q 
fig, ax = plt.subplots(figsize=(8, 6), dpi=400)
sc = ax.errorbar(samples_BD['Q_sampling_m3_s'], samples_BD['Sand_flux_kg_s'],
                 ls=' ', marker= 'o', markersize = '8', color='seagreen', markeredgecolor = 'black', markeredgewidth=0.5,                 
                xerr = err_Q_BD, yerr = err_F_BD, elinewidth = 0.7, capsize = 1.5, label = 'BD')
sc = ax.errorbar(samples_P6['Q_sampling_m3_s'], samples_P6['Sand_flux_kg_s'],
                 ls=' ', marker= 'D', markersize = '7', color='mediumblue', markeredgecolor = 'black', markeredgewidth=0.5,                 
                xerr = err_Q_P6, yerr = err_F_P6, elinewidth = 0.7, capsize = 1.5, label = 'P6')
sc = ax.errorbar(samples_P72['Q_sampling_m3_s'], samples_P72['Sand_flux_kg_s'],
                 ls=' ', marker= 's', markersize = '7', color='darkorange', markeredgecolor = 'black', markeredgewidth=0.5,                 
                 xerr = err_Q_P72, yerr = err_F_P72, elinewidth = 0.7, capsize = 1.5, label = 'P72')

ax.legend(fontsize = 14)
ax.set_xlabel('Q (m³/s)', fontsize=16, weight = 'bold')
ax.set_ylabel('$\mathregular{\Phi_{sand}}$ (kg/s)', fontsize=16, weight = 'bold')
ax.set_yscale('log')
ax.grid(True, which = 'both', linewidth = 0.2)
ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax.set_xlim(0,)
ax.set_ylim(0.1, 400)
  
y_formatter = FixedFormatter(['0.1', '1','10','100'])
y_locator = FixedLocator([0.1, 1, 10,100])
ax.yaxis.set_major_formatter(y_formatter)
ax.yaxis.set_major_locator(y_locator)

fig.tight_layout()
figname = '\\Discharge_sand_flux'
fig.savefig(outpath_figures + figname + '.png', dpi=400, bbox_inches='tight')

#%% Plot sand flux - Q and S
nb_couleurs = 11  # en combien de parties je découpe mon échelle colorimétrique
viridis = cm.get_cmap('RdBu_r', nb_couleurs)  # _r signifie que je prends l'échelle dans l'ordre inverse, ici clair vers sombre
echelle_colorimetrique = viridis(np.linspace(0, 1, nb_couleurs))  # j'extrais des couleurs discrètes
vecteur_blanc = np.array([1, 1, 1, 1])
echelle_colorimetrique[5:6,:] = vecteur_blanc  # je remplace la première couleur par du blanc (défini en RGB) : 0 ou 1 ?...
cmap = mpl_colors_ListedColormap(echelle_colorimetrique)  # je crée la colormap à partir de l'échelle colorimétrique
cmap.set_under('black')  # en dessous de la limite min : gris
cmap.set_over('saddlebrown')  # 
cbounds = [0.1, 0.2, 0.25, 0.33, 0.5, 0.8, 1.2, 2, 3, 4, 5, 10]
norm = mpl.colors.BoundaryNorm(cbounds, cmap.N)

ratio_P72_P6 = samples_P72_P6['Fine_concentration_g_l']/samples_P72_P6['Sand_concentration_g_l']
samples_P72_P6['Ratio'] = ratio_P72_P6

fig, ax = plt.subplots(figsize=(8, 6), dpi=400)

cax = ax.scatter(samples_P72_P6['Q_sampling_m3_s'], samples_P72_P6['Sand_flux_kg_s'],
             c=samples_P72_P6['Ratio'], s = 100, edgecolor = 'black', linewidth = 0.3, cmap=cmap,norm=norm, 
             zorder = 100)
caxx = ax.errorbar(samples_P72_P6['Q_sampling_m3_s'], samples_P72_P6['Sand_flux_kg_s'],  
                 ls = '', ecolor = 'black',
                 xerr = err_Q_P72_P6, yerr = err_F_P72_P6, elinewidth = 0.5, capsize = 1,
                 zorder = 0)


cbar = fig.colorbar(cax, ax=ax, extend='both', ticks=[0.1, 0.2, 0.25, 0.33, 0.5, 0.8, 1.2, 2, 3, 4, 5, 10])
cbar.ax.set_yticklabels(['0.1','0.2','0.25','0.33','0.5','0.8','1.2','2','3', '4', '5','10'],ha='right')
cbar.ax.yaxis.set_tick_params(pad=35) 
cbar.set_label(r'$C_{fines}/C_{sand}$', labelpad= 10, fontsize = 16)
 
ax.set_xlim(0,)
ax.set_ylim(0.1, 400)
ax.set_xlabel('Q (m³/s)', fontsize=16, weight = 'bold')
ax.set_ylabel('$\mathregular{\Phi_{sand}}$ (kg/s)', fontsize=16, weight = 'bold')
ax.set_yscale('log')
ax.grid(True, which = 'both', linewidth = 0.2)
ax.tick_params(axis = 'both', which = 'major', labelsize = 12)

y_formatter = FixedFormatter(['0.1', '1','10','100'])
y_locator = FixedLocator([0.1, 1, 10,100])
ax.yaxis.set_major_formatter(y_formatter)
ax.yaxis.set_major_locator(y_locator)
    
fig.tight_layout()
figname = '\\Discharge_sand_flux_S'
fig.savefig(outpath_figures + figname + '.png', dpi=400, bbox_inches='tight')

# %% PLOT sand C - Q 
fig, ax = plt.subplots(figsize=(8, 6), dpi=400)
sc = ax.errorbar(samples_BD['Q_sampling_m3_s'], samples_BD['Sand_concentration_g_l'],
                 ls=' ', marker= 'o', markersize = '8', color='seagreen', markeredgecolor = 'black', markeredgewidth=0.5,                 
                xerr = err_Q_BD, yerr = err_C_BD, elinewidth = 0.7, capsize = 1.5, label = 'BD')
sc = ax.errorbar(samples_P6['Q_sampling_m3_s'], samples_P6['Sand_concentration_g_l'],
                 ls=' ', marker= 'D', markersize = '7', color='mediumblue', markeredgecolor = 'black', markeredgewidth=0.5,                 
                xerr = err_Q_P6, yerr = err_C_P6, elinewidth = 0.7, capsize = 1.5, label = 'P6')
sc = ax.errorbar(samples_P72['Q_sampling_m3_s'], samples_P72['Sand_concentration_g_l'],
                 ls=' ', marker= 's', markersize = '7', color='darkorange', markeredgecolor = 'black', markeredgewidth=0.5,                 
                 xerr = err_Q_P72, yerr = err_C_P72, elinewidth = 0.7, capsize = 1.5, label = 'P72')

ax.legend(fontsize = 14)
ax.set_xlabel('Q (m³/s)', fontsize=16, weight = 'bold')
ax.set_ylabel('$\mathregular{\overline{C_{sand}}}$ (g/l)', fontsize=16, weight = 'bold')
ax.set_yscale('log')
ax.grid(True, which = 'both', linewidth = 0.2)
ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax.set_xlim(0,)
ax.set_ylim(0.001, 2)
  
y_formatter = FixedFormatter(['0.001', '0.01', '0.1', '1'])
y_locator = FixedLocator([0.001, 0.01, 0.1, 1, 10,100])
ax.yaxis.set_major_formatter(y_formatter)
ax.yaxis.set_major_locator(y_locator)

fig.tight_layout()
figname = '\\Discharge_sand_concentration'
fig.savefig(outpath_figures + figname + '.png', dpi=400, bbox_inches='tight')

# %% PLOT Q - sand D50
fig, ax = plt.subplots(figsize=(8, 6), dpi=400)
sc = ax.errorbar(samples_BD['Q_sampling_m3_s'], samples_BD['D50_mum'],
                 ls=' ', marker= 'o', markersize = '8', color='seagreen', markeredgecolor = 'black', markeredgewidth=0.5,                 
                label = 'BD')
sc = ax.errorbar(samples_P6['Q_sampling_m3_s'], samples_P6['D50_mum'],
                 ls=' ', marker= 'D', markersize = '7', color='mediumblue', markeredgecolor = 'black', markeredgewidth=0.5,                 
                label = 'P6')

ax.legend(fontsize = 14)
ax.set_xlabel('Q (m³/s)', fontsize=16, weight = 'bold')
ax.set_ylabel('$\mathregular{\overline{D_{50}} \; (\mu m)}$', fontsize=16, weight = 'bold')
ax.set_yscale('log')
ax.grid(True, which = 'both', linewidth = 0.2)
ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax.set_xlim(0,)
ax.set_ylim(80,400)
  
y_formatter = FixedFormatter(['100', '200', '300', '400'])
y_locator = FixedLocator([100, 200, 300, 400])
ax.yaxis.set_major_formatter(y_formatter)
ax.yaxis.set_major_locator(y_locator)

fig.tight_layout()
figname = '\\Discharge_D50'
fig.savefig(outpath_figures + figname + '.png', dpi=400, bbox_inches='tight')


#%% D50 - concentration sand 
fig, ax = plt.subplots(figsize=(8, 6), dpi=400)
sc = ax.plot(samples_BD['Sand_concentration_g_l'],samples_BD['D50_mum'], 
                ls=' ', marker= 'o', markersize = '8', color='seagreen', markeredgecolor = 'black', markeredgewidth=0.5,                 
                label = 'BD') #xerr = err_C_BD, elinewidth = 0.7, capsize = 1.5, 
sc = ax.plot(samples_P6['Sand_concentration_g_l'], samples_P6['D50_mum'],
                  ls=' ', marker= 'D', markersize = '7', color='mediumblue', markeredgecolor = 'black', markeredgewidth=0.5,                 
                label = 'P6') #xerr = err_C_P6, elinewidth = 0.7, capsize = 1.5, 

ax.legend(fontsize = 14)
ax.set_ylabel('$\mathregular{\overline{D_{50}} \; (\mu m)}$', fontsize=16, weight = 'bold')
ax.set_xlabel('$\mathregular{\overline{C_{sand}}}$ (g/l)', fontsize=16, weight = 'bold')
ax.set_xscale('log')
ax.grid(True, which = 'both', linewidth = 0.2)
ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax.set_ylim(0,)
#ax.set_xlim(0.005, 0.5)
  
x_formatter = FixedFormatter(['0.001', '0.01', '0.1', '1'])
x_locator = FixedLocator([0.001, 0.01, 0.1, 1, 10,100])
ax.xaxis.set_major_formatter(x_formatter)
ax.xaxis.set_major_locator(x_locator)

fig.tight_layout()
figname = '\\Sand_concentration_D50'
fig.savefig(outpath_figures + figname + '.png', dpi=400, bbox_inches='tight')

#%% sigma - concentration sand 
fig, ax = plt.subplots(figsize=(8, 6), dpi=400)
sc = ax.plot(samples_all[samples_all['Sampler'] == 'BD']['Sand_concentration_g_l'],
             samples_all[samples_all['Sampler'] == 'BD']['sigma_mum'], 
                ls=' ', marker= 'o', markersize = '8', color='seagreen', markeredgecolor = 'black', markeredgewidth=0.5,                 
                label = 'BD') #xerr = err_C_BD, elinewidth = 0.7, capsize = 1.5, 
sc = ax.plot(samples_all[samples_all['Sampler'] == 'P6']['Sand_concentration_g_l'], 
             samples_all[samples_all['Sampler'] == 'P6']['sigma_mum'],
                  ls=' ', marker= 'D', markersize = '7', color='mediumblue', markeredgecolor = 'black', markeredgewidth=0.5,                 
                label = 'P6') #xerr = err_C_P6, elinewidth = 0.7, capsize = 1.5, 

ax.legend(fontsize = 14)
ax.set_ylabel('$\mathregular{\sigma}$', fontsize=16, weight = 'bold')
ax.set_xlabel('$\mathregular{\overline{C_{sand}}}$ (g/l)', fontsize=16, weight = 'bold')
ax.set_xscale('log')
ax.grid(True, which = 'both', linewidth = 0.2)
ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
#ax.set_ylim(50,400)
#ax.set_xlim(0.005, 0.5)
  
x_formatter = FixedFormatter(['0.001', '0.01', '0.1', '1'])
x_locator = FixedLocator([0.001, 0.01, 0.1, 1, 10,100])
ax.xaxis.set_major_formatter(x_formatter)
ax.xaxis.set_major_locator(x_locator)

fig.tight_layout()
figname = '\\Sand_concentration_sigma'
fig.savefig(outpath_figures + figname + '.png', dpi=400, bbox_inches='tight')

#%% D50 - sigma
fig, ax = plt.subplots(figsize=(8, 6), dpi=400)
sc = ax.plot(samples_all[samples_all['Sampler'] == 'BD']['D50_mum'],
             samples_all[samples_all['Sampler'] == 'BD']['sigma_mum'], 
                ls=' ', marker= 'o', markersize = '8', color='seagreen', markeredgecolor = 'black', markeredgewidth=0.5,                 
                label = 'BD') #xerr = err_C_BD, elinewidth = 0.7, capsize = 1.5, 
sc = ax.plot(samples_all[samples_all['Sampler'] == 'P6']['D50_mum'], 
             samples_all[samples_all['Sampler'] == 'P6']['sigma_mum'],
                  ls=' ', marker= 'D', markersize = '7', color='mediumblue', markeredgecolor = 'black', markeredgewidth=0.5,                 
                label = 'P6') #xerr = err_C_P6, elinewidth = 0.7, capsize = 1.5, 

ax.legend(fontsize = 14)
ax.set_ylabel('$\mathregular{\sigma \; (\mu m)}$', fontsize=16, weight = 'bold')
ax.set_xlabel('$\mathregular{\overline{D_{50}} \; (\mu m)}$', fontsize=16, weight = 'bold')
ax.set_xscale('log')
ax.grid(True, which = 'both', linewidth = 0.2)
ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
#ax.set_ylim(50,400)
#ax.set_xlim(0.005, 0.5)
  
x_formatter = FixedFormatter(['100', '200', '300', '1'])
x_locator = FixedLocator([100, 200, 300, 1, 10,100])
ax.xaxis.set_major_formatter(x_formatter)
ax.xaxis.set_major_locator(x_locator)

fig.tight_layout()
figname = '\\Sand_D50_sigma'
fig.savefig(outpath_figures + figname + '.png', dpi=400, bbox_inches='tight')






