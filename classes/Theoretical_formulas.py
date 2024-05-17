# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 09:42:22 2023

@author: jlaible
"""

import math as math
import numpy as np

def form_factor_function_ThorneMeral2008(a_s, freq, C):
    """This function computes the form factor based on the equation of
    Thorne and Meral (2008) rigid sphere"""
    # computing the wave number
    k = 2 * math.pi * freq / C
    
    x = k*a_s
    
    f = (x**2 * (1 - 0.35 * np.exp(-((x - 1.5) / 0.7)**2)) * 
       (1 + 0.5 * np.exp(-((x - 1.8) / 2.2)**2))) / (1 + 0.9 * x**2)
    
    return(f)

def form_factor_function_MoateThorne2012(a_s, freq, C=1500, mica = 'false'):
    """This function computes the form factor based on the equation of
    Moate and Thorne (2012)"""
        # computing the wave number
    k = 2*math.pi*freq / C
    
    x = k*a_s
    
    if mica == 'false':
        f=(x**2 * (1 - 0.25 * np.exp(-((x - 1.5) / 0.35)**2)) * (1 + 0.6 * np.exp(-((x - 2.9) / 1.15)**2))) / (42 + 28 * x**2)
    elif mica == 'true':
        d1=0.2
        d2=1.7
        d3=0.15
        d4=0.2
        d5=3.5
        d6=0.9
        d7=0
        d8=0
        d9=1
        d10=1.4
        d11=0.3
        f=(x**2 * (1 - d1 * np.exp(-((x - d2) / d3)**2)) * (1 + d4 * np.exp(-((x - d5) / d6)**2)) * (1 + d7 * np.exp(-((x - d8) / d9)**2))) / (d10 + d11 * x**2)

    return(f)


def xi_v_function_urick(a_s, freq, rho_sed, nu_0, rho_0=1000, C=1500):
    """This function computes the total normalized viscous absorption cross
    section, following the equation of Urick (1948)"""
    # angular frequencies
    omega = 2*math.pi*freq
    
    # wave number
    k = omega / C
    
    # coefficient b (inverse of the visquous boundary layer thickness)
    b = np.sqrt(omega / (2*nu_0))
    
    # coefficient delta
    delta = 0.5*(1 + 9/(2*b*a_s))
    
    # coefficient g (density ratio)
    g = rho_sed / rho_0
    
    # coefficient s
    s = (9 / (4*b*a_s))*(1 + 1/(b*a_s))
    
    # Computing the cross section
    Xi_v = (2/3)*k*a_s*(g-1)**2*(s/(s**2 + (g + delta)**2))
    
    return(Xi_v)

def xi_s_function_ThorneMeral2008(a_s, rho_sed, freq, C=1500):
    """This function computes the total normalized scattering cross
    section, following the equation of Moate and Thorne (2012)"""
    # computing the wave number
    k = 2*math.pi*freq / C    
    x = k*a_s
    
    Xi_s = rho_sed * 0.29 * x**4 / (0.95 + 1.28 * x**2 + 0.25 * x**4)
    
    return(Xi_s)

def xi_s_function_MoateThorne2012(a_s, freq, C=1500):
    """This function computes the total normalized scattering cross
    section, following the equation of Moate and Thorne (2012)"""
    # computing the wave number
    k = 2*math.pi*freq / C
    
    x = k*a_s
    
    Xi_s = 0.09 * x**4 / (1380 + 560 * x**2 + 150 * x**4)
    
    return(Xi_s)


class compute_model_lognorm_spherical:
    def __init__(self, a50, sigma, freq, rho_sed, nu_0):
        # Position parameter = median diameter of the Volume PSD
        self.a50 = a50
        # Scale parameter
        self.sigma = sigma
        # Frequencies
        self.freq = freq
        # Density
        self.rho_sed = rho_sed
        # kinematic viscosity
        self.nu_0 = nu_0

        # Computing acoustic variables
        self.compute()
    
    def simple_gaussian(self, x, m, s):
        """ This function returns a gaussian distribution """
        res = (1 / np.sqrt(2 * math.pi * s**2) * np.exp(-(x - m)**2 / (2 * s**2)))
        return res

    def compute(self) :
        """ This function computes zeta and ks2 by integration over a log-normal distribution. The
        integral is computed as a sum"""
        # center of the sampling array
        F = round(np.log10(self.a50))
        # creating a sampling array of size classes
        a_sample = np.logspace(F - 6, F + 6, 3000)
        # computing the probability over the sampling array
        gaussian = self.simple_gaussian(np.log(a_sample), np.log(self.a50), self.sigma)
        proba_vol = gaussian / sum(gaussian)
        # computing the cdf
        cdf_vol = np.nancumsum(proba_vol)*100
        
        # Computing number probability
        ss = np.sum(proba_vol / a_sample**3)
        proba_num = proba_vol/a_sample**3 / ss
        
        # Integrating zeta and ks2 over the distribution
        # Initializing 
        temp_a1 = 0
        temp_a2f2_MT12 = 0
        temp_a2f2_TM08 = 0
        temp_a3 = 0   
        temp_a2Xis = 0
        temp_a2Xiv = 0
        # Summing the integrals
        for i in range(len(a_sample)):
            a = a_sample[i]      
            temp_a1 += a * proba_num[i]
            temp_a2f2_TM08 += a**2 * form_factor_function_ThorneMeral2008(a, self.freq, 1500)**2 * proba_num[i]
            temp_a2f2_MT12 += a**2 * form_factor_function_MoateThorne2012(a, self.freq,mica ='false')**2 * proba_num[i]
            temp_a3 += a**3 * proba_num[i]
            temp_a2Xis += a**2 * xi_s_function_MoateThorne2012(a, self.freq) * proba_num[i]
            temp_a2Xiv += a**2 * xi_v_function_urick(a, self.freq, self.rho_sed, self.nu_0) * proba_num[i]
        
        cc = [a_sample[i] *proba_num[i] for i in range(len(proba_num))]
        ccc = [a_sample[i]**2 *proba_num[i]*
               form_factor_function_ThorneMeral2008(a_sample[i], self.freq, 1500)**2 
               for i in range(len(proba_num))]
        cccc = [a_sample[i]**3 *proba_num[i]
               for i in range(len(proba_num))]
        cff = ((np.nansum(cc)*np.nansum(ccc))/np.nansum(cccc))**0.5
        
        # computing output values   
        self.temp_a2f2_MT12 = temp_a2f2_MT12
        self.temp_a1 = temp_a1
        self.temp_a2f2_TM08 = temp_a2f2_TM08
        self.temp_a3 = temp_a3
        self.f_TM08 = ((temp_a1*temp_a2f2_TM08)/temp_a3)**0.5
        self.ks2 = temp_a2f2_MT12 / temp_a3
        self.proba_vol = proba_vol
        self.cdf_vol = cdf_vol
        self.proba_num = proba_num
        self.a_sample = a_sample
        self.f_TM08_2 = cff
        self.zeta = 3 * (temp_a2Xiv / self.rho_sed + temp_a2Xis) / (4 * temp_a3)
        self.zetav = 3 * (temp_a2Xiv / self.rho_sed) / (4 * temp_a3)
        self.zetas = 3 * (temp_a2Xis) / (4 * temp_a3)
        

# Using proba_vol for f
class compute_model_lognorm_spherical_probavol:
    def __init__(self, a50, sigma, freq, rho_sed, nu_0):
        # Position parameter = median diameter of the Volume PSD
        self.a50 = a50
        # Scale parameter
        self.sigma = sigma
        # Frequencies
        self.freq = freq
        # Density
        self.rho_sed = rho_sed
        # kinematic viscosity
        self.nu_0 = nu_0

        # Computing acoustic variables
        self.compute()
    
    def simple_gaussian(self, x, m, s):
        """ This function returns a gaussian distribution """
        res = (1 / np.sqrt(2 * math.pi * s**2) * np.exp(-(x - m)**2 / (2 * s**2)))
        return res

    def compute(self) :
        """ This function computes zeta and ks2 by integration over a log-normal distribution. The
        integral is computed as a sum"""
        # center of the sampling array
        F = round(np.log10(self.a50))
        # creating a sampling array of size classes
        a_sample = np.logspace(F - 6, F + 6, 3000)
        # computing the probability over the sampling array
        gaussian = self.simple_gaussian(np.log(a_sample), np.log(self.a50), self.sigma)
        proba_vol = gaussian / sum(gaussian)
        # computing the cdf
        cdf_vol = np.nancumsum(proba_vol)*100   
        
        gs = [2**(-35.25 + 0.0625*j) for j in range(0,672)]
        gsbound = [2**(-35.2188 + 0.0625*j) for j in range(0,672)]
        
        # Computing number probability
        ss = np.sum(proba_vol / a_sample**3)
        proba_num = proba_vol/a_sample**3 / ss
        
        d500 = 0.5*cdf_vol[-1]
        
        for j in range(len(cdf_vol)):
            if d500 >= cdf_vol[j-1] and d500 <= cdf_vol[j]:
                d50Bimodal=((d500-cdf_vol[j-1])/(cdf_vol[j]-cdf_vol[j-1]))*(np.log(gsbound[j])-np.log(gsbound[j-1]))+np.log(gsbound[j-1])
        d50_sed_i = np.exp(d50Bimodal)/1000    
               
            
        # Calculate form function TM08
        # Integrating over the distribution        
        temp_a1 = 0
        temp_a2f2_TM08_freq = 0
        # temp_a2f2_TM08_freq2 = 0
        temp_a3 = 0  
        # Summing the integrals
        for i in range(len(proba_num)):
            a = np.array(gs)[i]/1000
            temp_a2f2_TM08_freq += (a/2)**2 * form_factor_function_ThorneMeral2008((a/2), self.freq, 1450)**2 * proba_num[i]   
            # temp_a2f2_TM08_freq2 += (a/2)**2 * form_factor_function_ThorneMeral2008((a/2), freq2_Hz, 1450)**2 * proba_num[i]    
            temp_a3 += (a/2)**3 * proba_num[i]
    
        # computing output values   
        f_TM08_freq = (((d50_sed_i/2)*temp_a2f2_TM08_freq)/temp_a3)**0.5
        # f_TM08_freq2_i = (((d50_sed_i/2)*temp_a2f2_TM08_freq2)/temp_a3)**0.5
            
        self.f_TM08 = f_TM08_freq
        
        
        # # Integrating zeta and ks2 over the distribution
        # # Initializing 
        # temp_a1 = 0
        # temp_a2f2_MT12 = 0
        # temp_a2f2_TM08 = 0
        # temp_a3 = 0   
        # temp_a2Xis = 0
        # temp_a2Xiv = 0
        # # Summing the integrals
        # for i in range(len(a_sample)):
        #     a = a_sample[i]      
        #     temp_a1 += (a/2) * proba_num[i]
        #     temp_a2f2_TM08 += (a/2)**2 * form_factor_function_ThorneMeral2008(a, self.freq, 1500)**2 * proba_num[i]
        #     temp_a2f2_MT12 += (a/2)**2 * form_factor_function_MoateThorne2012(a, self.freq,mica ='false')**2 * proba_num[i]
        #     temp_a3 += (a/2)**3 * proba_vol[i]
        #     temp_a2Xis += (a/2)**2 * xi_s_function_MoateThorne2012(a, self.freq) * proba_num[i]
        #     temp_a2Xiv += (a/2)**2 * xi_v_function_urick(a, self.freq, self.rho_sed, self.nu_0) * proba_num[i]
        
        # cc = [a_sample[i] *proba_num[i] for i in range(len(proba_vol))]
        # ccc = [a_sample[i]**2 *proba_num[i]*
        #        form_factor_function_ThorneMeral2008(a_sample[i], self.freq, 1500)**2 
        #        for i in range(len(proba_num))]
        # cccc = [a_sample[i]**3 *proba_num[i]
        #        for i in range(len(proba_num))]
        # cff = ((np.nansum(cc)*np.nansum(ccc))/np.nansum(cccc))**0.5
        
        # # computing output values   
        # self.temp_a2f2_MT12 = temp_a2f2_MT12
        # self.temp_a1 = temp_a1
        # self.temp_a2f2_TM08 = temp_a2f2_TM08
        # self.temp_a3 = temp_a3
        # self.f_TM08 = ((temp_a1*temp_a2f2_TM08)/temp_a3)**0.5
        # self.ks2 = temp_a2f2_MT12 / temp_a3
        # self.proba_vol = proba_vol
        # self.cdf_vol = cdf_vol
        # self.a_sample = a_sample
        # self.f_TM08_2 = cff
        # self.zeta = 3 * (temp_a2Xiv / self.rho_sed + temp_a2Xis) / (4 * temp_a3)
        # self.zetav = 3 * (temp_a2Xiv / self.rho_sed) / (4 * temp_a3)
        # self.zetas = 3 * (temp_a2Xis) / (4 * temp_a3)