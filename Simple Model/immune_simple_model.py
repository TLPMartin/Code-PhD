# -*- coding: utf-8 -*-
"""
Created on 01 July 2023

Modules for the simple immune model

@author: LIM Roktaek
"""

####-- Modules

import os
import sys
import numpy
import pandas
import shutil
from scipy.integrate import solve_ivp
from scipy.interpolate import splev
from scipy.interpolate import splrep

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

####-- END Modules

###
def error_exit():
    checker = input('Enter "x" to finish this program: ')
    if checker == 'x':
        sys.exit()
    #
    #-- return
    return 0
###

###
def func_gamma(gamma,k,c,m):
    f_eval = gamma*numpy.power(c,m)/(numpy.power(k,m) + numpy.power(c,m))
    #-- return
    return f_eval
###

###
def func_IL22_IL17(gamma_IL_22,k_IL22,c_IL22,gamma_IL_17,k_IL17,c_IL17,m,a_Epi):
    f_eval = (func_gamma(gamma_IL_22,k_IL22,c_IL22,m) + func_gamma(gamma_IL_17,k_IL17,c_IL17,m))*a_Epi
    #-- return
    return f_eval
###

###
def func_initialize_antigen_load(df):
    #-- values on antigen load at t = t_start
    beta_eAg = 0.0
    beta_tAg = 0.0
    Ag_load_sum = 0.0
    for idx in range(0,int(df['N_Ag'])):
        df[f'eAg_load_{idx:d}'] = df[f'eAg_load_0_{idx:d}']
        beta_eAg += df[f'beta_{idx:d}']*df[f'eAg_load_{idx:d}']
        df[f'tAg_load_{idx:d}'] = df[f'tAg_load_0_{idx:d}']
        beta_tAg += df[f'beta_{idx:d}']*df[f'tAg_load_{idx:d}']
        Ag_load_sum += df[f'eAg_load_{idx:d}'] + df[f'tAg_load_{idx:d}']
    #
    df['beta_eAg'] = beta_eAg
    df['beta_tAg'] = beta_tAg
    df['Ag_load_sum'] = Ag_load_sum
    #-- values on antigen load at t = t_start
    #-- return
    return df
###

###
def func_d_C_SAA(c_SAA,delta_SAA,m,rho_SAA_Epi,gamma_IL22_SAA_Epi,k_IL22_Epi,c_IL22,K_Epi_SAA,
                 beta_eAg,Ag_load_sum,A_Epi):
    tmp = (1.0 + func_gamma(gamma_IL22_SAA_Epi,k_IL22_Epi,c_IL22,m))*(beta_eAg)/(K_Epi_SAA + Ag_load_sum)*A_Epi
    dy = rho_SAA_Epi*tmp - delta_SAA*c_SAA
    #-- return
    return dy
###

###
def func_d_C_SAA_rate(c_SAA,delta_SAA,m,rho_SAA_Epi,gamma_IL22_SAA_Epi,k_IL22_Epi,c_IL22,K_Epi_SAA,
                      beta_eAg,Ag_load_sum,A_Epi):
    tmp = (1.0 + func_gamma(gamma_IL22_SAA_Epi,k_IL22_Epi,c_IL22,m))*(beta_eAg)/(K_Epi_SAA + Ag_load_sum)*A_Epi
    r_dict = { 'g_SAA_Epi' : rho_SAA_Epi*tmp, 'd_SAA' : delta_SAA*c_SAA,
               'x_SAA_IL22' : func_gamma(gamma_IL22_SAA_Epi,k_IL22_Epi,c_IL22,m) }
    #-- return
    return r_dict
###

###
def func_d_C_IL1beta23(c_IL1beta23,delta_IL1beta23,m,
                       rho_IL1beta23_Mac,gamma_SAA_IL1beta23_Mac,k_SAA_Mac,c_SAA,
                       rho_IL1beta23_eMac,gamma_IL10_IL1beta23_Mac,k_IL10_Mac,c_IL10,
                       K_Mac_IL1beta23,beta_eAg,Ag_load_sum,A_Mac):
    tmp_n = (1.0 + func_gamma(gamma_SAA_IL1beta23_Mac,k_SAA_Mac,c_SAA,m))
    tmp_d = 1.0/(1.0 + func_gamma(gamma_IL10_IL1beta23_Mac,k_IL10_Mac,c_IL10,m))
    tmp1 = tmp_n*tmp_d*A_Mac
    tmp2 = tmp_d*(beta_eAg)/(K_Mac_IL1beta23 + Ag_load_sum)*A_Mac
    dy = rho_IL1beta23_Mac*tmp1 + rho_IL1beta23_eMac*tmp2 - delta_IL1beta23*c_IL1beta23
    #-- return
    return dy
###

###
def func_d_C_IL1beta23_rate(c_IL1beta23,delta_IL1beta23,m,
                            rho_IL1beta23_Mac,gamma_SAA_IL1beta23_Mac,k_SAA_Mac,c_SAA,
                            rho_IL1beta23_eMac,gamma_IL10_IL1beta23_Mac,k_IL10_Mac,c_IL10,
                            K_Mac_IL1beta23,beta_eAg,Ag_load_sum,A_Mac):
    tmp_n = (1.0 + func_gamma(gamma_SAA_IL1beta23_Mac,k_SAA_Mac,c_SAA,m))
    tmp_d = 1.0/(1.0 + func_gamma(gamma_IL10_IL1beta23_Mac,k_IL10_Mac,c_IL10,m))
    tmp1 = tmp_n*tmp_d*A_Mac
    tmp2 = tmp_d*(beta_eAg)/(K_Mac_IL1beta23 + Ag_load_sum)*A_Mac
    r_dict = { 'g_IL1beta23_Mac' : rho_IL1beta23_Mac*tmp1, 'g_IL1beta23_eMac' : rho_IL1beta23_eMac*tmp2,
               'd_IL1beta23' : delta_IL1beta23*c_IL1beta23,
               'x_IL1beta23_SAA' : func_gamma(gamma_SAA_IL1beta23_Mac,k_SAA_Mac,c_SAA,m),
               'x_IL1beta23_IL10' : 1.0 + func_gamma(gamma_IL10_IL1beta23_Mac,k_IL10_Mac,c_IL10,m) }
    #-- return
    return r_dict
###

###
def func_d_C_IL22(c_IL22,delta_IL22,m,
                  rho_IL22_ILC3,gamma_IL1beta23_ILC3,k_IL1beta23_ILC3,c_IL1beta23,A_ILC3,
                  rho_IL22_TH17,TH17_sum):
    tmp = func_gamma(gamma_IL1beta23_ILC3,k_IL1beta23_ILC3,c_IL1beta23,m)*A_ILC3
    dy = rho_IL22_ILC3*tmp + rho_IL22_TH17*TH17_sum - delta_IL22*c_IL22
    #-- return
    return dy
###

###
def func_d_C_IL22_rate(c_IL22,delta_IL22,m,
                       rho_IL22_ILC3,gamma_IL1beta23_ILC3,k_IL1beta23_ILC3,c_IL1beta23,A_ILC3,
                       rho_IL22_TH17,TH17_sum):
    tmp = func_gamma(gamma_IL1beta23_ILC3,k_IL1beta23_ILC3,c_IL1beta23,m)*A_ILC3
    r_dict = { 'g_IL22_ILC3' : rho_IL22_ILC3*tmp, 'g_IL22_TH17' : rho_IL22_TH17*TH17_sum,
               'd_IL22' : delta_IL22*c_IL22 }
    #-- return
    return r_dict
###

###
def func_d_C_IL2(c_IL2,delta_IL2,m,
                 rho_IL2_ILC3,gamma_IL1beta23_ILC3,k_IL1beta23_ILC3,c_IL1beta23,A_ILC3,
                 rho_IL2_eDC,rho_IL2_tDC,K_Mac_IL2,beta_eAg,beta_tAg,Ag_load_sum,A_DC):
    tmp1 = func_gamma(gamma_IL1beta23_ILC3,k_IL1beta23_ILC3,c_IL1beta23,m)*A_ILC3
    tmp_d = 1.0/(K_Mac_IL2 + Ag_load_sum)
    tmp2 = beta_eAg*tmp_d*A_DC
    tmp3 = beta_tAg*tmp_d*A_DC
    dy = rho_IL2_ILC3*tmp1 + rho_IL2_eDC*tmp2 + rho_IL2_tDC*tmp3 - delta_IL2*c_IL2
    #-- return
    return dy
###

###
def func_d_C_IL2_rate(c_IL2,delta_IL2,m,
                      rho_IL2_ILC3,gamma_IL1beta23_ILC3,k_IL1beta23_ILC3,c_IL1beta23,A_ILC3,
                      rho_IL2_eDC,rho_IL2_tDC,K_Mac_IL2,beta_eAg,beta_tAg,Ag_load_sum,A_DC):
    tmp1 = func_gamma(gamma_IL1beta23_ILC3,k_IL1beta23_ILC3,c_IL1beta23,m)*A_ILC3
    tmp_d = 1.0/(K_Mac_IL2 + Ag_load_sum)
    tmp2 = beta_eAg*tmp_d*A_DC
    tmp3 = beta_tAg*tmp_d*A_DC
    r_dict = { 'g_IL2_ILC3' : rho_IL2_ILC3*tmp1, 'g_IL2_eDC' : rho_IL2_eDC*tmp2, 'g_IL2_tDC' : rho_IL2_tDC*tmp3,
               'd_IL2' : delta_IL2*c_IL2 }
    #-- return
    return r_dict
###

###
def func_d_C_IL17(c_IL17,delta_IL17,m,
                  rho_IL17_TH17,gamma_SAA_IL17_TH17,k_SAA_TH17,c_SAA,
                  gamma_IL10_IL17_TH17,k_IL10_TH17,c_IL10,TH17_sum):
    tmp_n = (1.0 + func_gamma(gamma_SAA_IL17_TH17,k_SAA_TH17,c_SAA,m))
    tmp_d = 1.0/(1.0 + func_gamma(gamma_IL10_IL17_TH17,k_IL10_TH17,c_IL10,m))
    dy = rho_IL17_TH17*tmp_n*tmp_d*TH17_sum - delta_IL17*c_IL17
    #-- return
    return dy
###

###
def func_d_C_IL17_rate(c_IL17,delta_IL17,m,
                       rho_IL17_TH17,gamma_SAA_IL17_TH17,k_SAA_TH17,c_SAA,
                       gamma_IL10_IL17_TH17,k_IL10_TH17,c_IL10,TH17_sum):
    tmp_n = (1.0 + func_gamma(gamma_SAA_IL17_TH17,k_SAA_TH17,c_SAA,m))
    tmp_d = 1.0/(1.0 + func_gamma(gamma_IL10_IL17_TH17,k_IL10_TH17,c_IL10,m))
    r_dict = { 'g_IL17_TH17' : rho_IL17_TH17*tmp_n*tmp_d*TH17_sum, 'd_IL17' : delta_IL17*c_IL17,
               'x_IL17_SAA' : func_gamma(gamma_SAA_IL17_TH17,k_SAA_TH17,c_SAA,m),
               'x_IL17_IL10' : 1.0 + func_gamma(gamma_IL10_IL17_TH17,k_IL10_TH17,c_IL10,m) }
    #-- return
    return r_dict
###

###
def func_d_C_IL10(c_IL10,delta_IL10,m,
                  rho_IL10_Treg,gamma_IL10_Treg,k_IL10_Treg,Treg_sum,
                  rho_IL10_tTreg,gamma_IL10_tTreg,k_IL10_tTreg,A_tTreg,
                  rho_IL10_Mac,A_Mac,rho_IL10_tMac,K_Mac_IL10,beta_tAg,Ag_load_sum,
                  rho_IL10_Epi,A_Epi):
    tmp1 = (1.0 + func_gamma(gamma_IL10_Treg,k_IL10_Treg,c_IL10,m))*Treg_sum
    tmp2 = (1.0 + func_gamma(gamma_IL10_tTreg,k_IL10_tTreg,c_IL10,m))*A_tTreg
    tmp3 = beta_tAg/(K_Mac_IL10 + Ag_load_sum)*A_Mac
    dy = rho_IL10_Treg*tmp1 + rho_IL10_tTreg*tmp2 + rho_IL10_Mac*A_Mac + rho_IL10_tMac*tmp3 + rho_IL10_Epi*A_Epi - delta_IL10*c_IL10
    #-- return
    return dy
###

###
def func_d_C_IL10_rate(c_IL10,delta_IL10,m,
                       rho_IL10_Treg,gamma_IL10_Treg,k_IL10_Treg,Treg_sum,
                       rho_IL10_tTreg,gamma_IL10_tTreg,k_IL10_tTreg,A_tTreg,
                       rho_IL10_Mac,A_Mac,rho_IL10_tMac,K_Mac_IL10,beta_tAg,Ag_load_sum,
                       rho_IL10_Epi,A_Epi):
    tmp1 = (1.0 + func_gamma(gamma_IL10_Treg,k_IL10_Treg,c_IL10,m))*Treg_sum
    tmp2 = (1.0 + func_gamma(gamma_IL10_tTreg,k_IL10_tTreg,c_IL10,m))*A_tTreg
    tmp3 = beta_tAg/(K_Mac_IL10 + Ag_load_sum)*A_Mac
    r_dict = { 'g_IL10_Treg' : rho_IL10_Treg*tmp1, 'g_IL10_tTreg' : rho_IL10_tTreg*tmp2,
               'g_IL10_Mac' : rho_IL10_Mac*A_Mac, 'g_IL10_tMAc' : rho_IL10_tMac*tmp3, 'g_IL10_Epi' : rho_IL10_Epi*A_Epi,
               'd_IL10' : delta_IL10*c_IL10,
               'x_IL10_Treg' : func_gamma(gamma_IL10_Treg,k_IL10_Treg,c_IL10,m),
               'x_IL10_tTreg' : func_gamma(gamma_IL10_tTreg,k_IL10_tTreg,c_IL10,m) }
    #-- return
    return r_dict
###

###
def func_d_A_TH17(A_TH17,mu_TH17,m,A_max_LP,A_LP_sum,alpha,phi_TH17,psi_TH17,Ag_load_del,
                  gamma_IL1beta23_Treg,k_IL1beta23_Treg,c_IL1beta23,A_Treg,gamma_TH17,
                  lambda_TH17,xi_TH17_Epi,K_Epi,Ag_load_sum,Ag_load,A_Epi,
                  xi_TH17_Mac,gamma_IL10_Mac,k_IL10_Mac,c_IL10,K_Mac,A_Mac,A_DC):
    tmp_Epi = xi_TH17_Epi*Ag_load/(K_Epi + Ag_load_sum)*A_Epi
    tmp_d_IL10 = 1.0/(1.0 + func_gamma(gamma_IL10_Mac,k_IL10_Mac,c_IL10,m))
    tmp_Mac = xi_TH17_Mac*tmp_d_IL10*Ag_load/(K_Mac + Ag_load_sum)*(A_Mac + A_DC)
    l_TH17 = lambda_TH17*(tmp_Epi + tmp_Mac)
    tmp1 = (A_max_LP - A_LP_sum)/A_max_LP*(alpha*phi_TH17*psi_TH17*Ag_load_del + psi_TH17*l_TH17*A_TH17)
    tmp2 = func_gamma(gamma_IL1beta23_Treg,k_IL1beta23_Treg,c_IL1beta23,m)*A_Treg
    dy = tmp1 + tmp2 - gamma_TH17*A_TH17 - mu_TH17*A_TH17
    #-- return
    return dy
###

###
def func_d_A_TH17_rate(A_TH17,mu_TH17,m,A_max_LP,A_LP_sum,alpha,phi_TH17,psi_TH17,Ag_load_del,
                       gamma_IL1beta23_Treg,k_IL1beta23_Treg,c_IL1beta23,A_Treg,gamma_TH17,
                       lambda_TH17,xi_TH17_Epi,K_Epi,Ag_load_sum,Ag_load,A_Epi,
                       xi_TH17_Mac,gamma_IL10_Mac,k_IL10_Mac,c_IL10,K_Mac,A_Mac,A_DC):
    tmp_Epi = xi_TH17_Epi*Ag_load/(K_Epi + Ag_load_sum)*A_Epi
    tmp_d_IL10 = 1.0/(1.0 + func_gamma(gamma_IL10_Mac,k_IL10_Mac,c_IL10,m))
    tmp_Mac = xi_TH17_Mac*tmp_d_IL10*Ag_load/(K_Mac + Ag_load_sum)*(A_Mac + A_DC)
    l_TH17 = lambda_TH17*(tmp_Epi + tmp_Mac)
    r_dict = { 'g_TH17_Ag' : (A_max_LP - A_LP_sum)/A_max_LP*alpha*phi_TH17*psi_TH17*Ag_load_del,
               'g_TH17' : (A_max_LP - A_LP_sum)/A_max_LP*psi_TH17*l_TH17*A_TH17,
               'g_TH17_Treg' : func_gamma(gamma_IL1beta23_Treg,k_IL1beta23_Treg,c_IL1beta23,m)*A_Treg,
               'd_TH17_Treg' : gamma_TH17*A_TH17,
               'd_TH17' : mu_TH17*A_TH17 }
    #-- return
    return r_dict
###

###
def func_d_A_Treg(A_Treg,mu_Treg,m,
                  gamma_IL2_Treg,k_IL2_Treg,c_IL2,
                  A_max_LP,A_LP_sum,alpha,phi_Treg,psi_Treg,Ag_load_del,
                  gamma_IL1beta23_Treg,k_IL1beta23_Treg,c_IL1beta23,gamma_TH17,A_TH17,
                  lambda_Treg,xi_Treg_Epi,K_Epi,Ag_load_sum,Ag_load,A_Epi,
                  xi_Treg_Mac,gamma_IL10_Mac,k_IL10_Mac,c_IL10,K_Mac,A_Mac,A_DC):
    tmp_Epi = xi_Treg_Epi*Ag_load/(K_Epi + Ag_load_sum)*A_Epi
    tmp_d_IL10 = 1.0/(1.0 + func_gamma(gamma_IL10_Mac,k_IL10_Mac,c_IL10,m))
    tmp_Mac = xi_Treg_Mac*tmp_d_IL10*Ag_load/(K_Mac + Ag_load_sum)*(A_Mac + A_DC)
    l_Treg = lambda_Treg*(tmp_Epi + tmp_Mac)
    tmp1 = (A_max_LP - A_LP_sum)/A_max_LP*((1.0 - alpha)*phi_Treg*psi_Treg*Ag_load_del + psi_Treg*l_Treg*A_Treg)
    tmp2 = func_gamma(gamma_IL1beta23_Treg,k_IL1beta23_Treg,c_IL1beta23,m)*A_Treg
    tmp_mu_d = 1.0/(1.0 + func_gamma(gamma_IL2_Treg,k_IL2_Treg,c_IL2,m))
    dy = tmp1 - tmp2 + gamma_TH17*A_TH17 - mu_Treg*tmp_mu_d*A_Treg
    #-- return
    return dy
###

###
def func_d_A_Treg_rate(A_Treg,mu_Treg,m,
                       gamma_IL2_Treg,k_IL2_Treg,c_IL2,
                       A_max_LP,A_LP_sum,alpha,phi_Treg,psi_Treg,Ag_load_del,
                       gamma_IL1beta23_Treg,k_IL1beta23_Treg,c_IL1beta23,gamma_TH17,A_TH17,
                       lambda_Treg,xi_Treg_Epi,K_Epi,Ag_load_sum,Ag_load,A_Epi,
                       xi_Treg_Mac,gamma_IL10_Mac,k_IL10_Mac,c_IL10,K_Mac,A_Mac,A_DC):
    tmp_Epi = xi_Treg_Epi*Ag_load/(K_Epi + Ag_load_sum)*A_Epi
    tmp_d_IL10 = 1.0/(1.0 + func_gamma(gamma_IL10_Mac,k_IL10_Mac,c_IL10,m))
    tmp_Mac = xi_Treg_Mac*tmp_d_IL10*Ag_load/(K_Mac + Ag_load_sum)*(A_Mac + A_DC)
    l_Treg = lambda_Treg*(tmp_Epi + tmp_Mac)
    tmp_mu_d = 1.0/(1.0 + func_gamma(gamma_IL2_Treg,k_IL2_Treg,c_IL2,m))
    r_dict = { 'g_Treg_Ag' : (A_max_LP - A_LP_sum)/A_max_LP*(1.0 - alpha)*phi_Treg*psi_Treg*Ag_load_del,
               'g_Treg' : (A_max_LP - A_LP_sum)/A_max_LP*psi_Treg*l_Treg*A_Treg,
               'g_Treg_TH17' : gamma_TH17*A_TH17,
               'd_Treg_TH17' : func_gamma(gamma_IL1beta23_Treg,k_IL1beta23_Treg,c_IL1beta23,m)*A_Treg,
               'd_Treg' : mu_Treg*tmp_mu_d*A_Treg,
               'x_Treg_mu_IL2' : 1.0 + func_gamma(gamma_IL2_Treg,k_IL2_Treg,c_IL2,m) }
    #-- return
    return r_dict
###

###
def ODE_simple_immune_model(t,y,pv):
    #-- y[0] = SAA
    #-- y[1] = IL-1beta/23
    #-- y[2] = IL-22
    #-- y[3] = IL-2
    #-- y[4] = IL-17
    #-- y[5] = IL-10
    #-- y[6] = TH17_1
    #-- y[7] = TH17_2
    #-- y[5 + N_Ag] = TH17_N_Ag
    #-- y[5 + N_Ag + 1] = Treg_1
    #-- y[5 + 2*N_Ag] = Treg_N_Ag
    #-- dy_dt
    dy_dt = numpy.zeros((len(y),))
    #-- dy_dt
    #-- TH17_sum and Treg_sum
    TH17_sum = numpy.sum(y[6:6 + int(pv['N_Ag'])])
    Treg_sum = numpy.sum(y[6 + int(pv['N_Ag']):])
    A_LP_sum = pv['A_LP'] + TH17_sum + Treg_sum
    #-- TH17_sum and Treg_sum
    #-- y[0] = c_SAA
    dy_dt[0] = func_d_C_SAA(y[0],pv['delta_SAA'],pv['m'],
                            pv['rho_SAA_Epi'],pv['gamma_IL22_SAA_Epi'],pv['k_IL22_Epi'],y[2],
                            pv['K_Epi_SAA'],pv['beta_eAg'],pv['Ag_load_sum'],pv['A_Epi'])
    #-- y[0] = c_SAA
    #-- y[1] = IL-1beta/23
    dy_dt[1] = func_d_C_IL1beta23(y[1],pv['delta_IL1beta23'],pv['m'],
                                  pv['rho_IL1beta23_Mac'],pv['gamma_SAA_IL1beta23_Mac'],pv['k_SAA_Mac'],y[0],
                                  pv['rho_IL1beta23_eMac'],pv['gamma_IL10_IL1beta23_Mac'],pv['k_IL10_Mac'],y[5],
                                  pv['K_Mac_IL1beta23'],pv['beta_eAg'],pv['Ag_load_sum'],pv['A_Mac'])
    #-- y[1] = IL-1beta/23
    #-- y[2] = IL-22
    dy_dt[2] = func_d_C_IL22(y[2],pv['delta_IL22'],pv['m'],
                             pv['rho_IL22_ILC3'],pv['gamma_IL1beta23_ILC3'],pv['k_IL1beta23_ILC3'],y[1],pv['A_ILC3'],
                             pv['rho_IL22_TH17'],TH17_sum)
    #-- y[2] = IL-22
    #-- y[3] = IL-2
    dy_dt[3] = func_d_C_IL2(y[3],pv['delta_IL2'],pv['m'],
                            pv['rho_IL2_ILC3'],pv['gamma_IL1beta23_ILC3'],pv['k_IL1beta23_ILC3'],y[1],pv['A_ILC3'],
                            pv['rho_IL2_eDC'],pv['rho_IL2_tDC'],pv['K_Mac_IL2'],
                            pv['beta_eAg'],pv['beta_tAg'],pv['Ag_load_sum'],pv['A_DC'])
    #-- y[3] = IL-2
    #-- y[4] = IL-17
    dy_dt[4] = func_d_C_IL17(y[4],pv['delta_IL17'],pv['m'],
                             pv['rho_IL17_TH17'],pv['gamma_SAA_IL17_TH17'],pv['k_SAA_TH17'],y[0],
                             pv['gamma_IL10_IL17_TH17'],pv['k_IL10_TH17'],y[5],TH17_sum)
    #-- y[4] = IL-17
    #-- y[5] = IL-10
    dy_dt[5] = func_d_C_IL10(y[5],pv['delta_IL10'],pv['m'],
                             pv['rho_IL10_Treg'],pv['gamma_IL10_Treg'],pv['k_IL10_Treg'],Treg_sum,
                             pv['rho_IL10_tTreg'],pv['gamma_IL10_tTreg'],pv['k_IL10_tTreg'],pv['A_tTreg'],
                             pv['rho_IL10_Mac'],pv['A_Mac'],pv['rho_IL10_tMac'],pv['K_Mac_IL10'],
                             pv['beta_tAg'],pv['Ag_load_sum'],pv['rho_IL10_Epi'],pv['A_Epi'])
    #-- y[5] = IL-10
    #-- TH17_i, 6:6 + int(pv['N_Ag']
    for idx in range(0,int(pv['N_Ag'])):
        dy_dt[6 + idx] = func_d_A_TH17(y[6 + idx],pv['mu_TH17'],pv['m'],pv['A_max_LP'],A_LP_sum,
                                       pv[f'alpha_{idx:d}'],pv['phi_TH17'],pv[f'psi_TH17_{idx:d}'],
                                       pv[f'eAg_load_del_{idx:d}'] + pv[f'tAg_load_del_{idx:d}'],
                                       pv['gamma_IL1beta23_Treg'],pv['k_IL1beta23_Treg'],y[1],
                                       y[6 + int(pv['N_Ag']) + idx],pv['gamma_TH17'],
                                       pv['lambda_TH17'],pv['xi_TH17_Epi'],pv['K_Epi'],pv['Ag_load_sum'],
                                       pv[f'eAg_load_{idx:d}'] + pv[f'tAg_load_{idx:d}'],
                                       pv['A_Epi'],pv['xi_TH17_Mac'],pv['gamma_IL10_Mac'],pv['k_IL10_Mac'],y[5],
                                       pv['K_Mac'],pv['A_Mac'],pv['A_DC'])
    #
    #-- TH17_i, 6:6 + int(pv['N_Ag']
    #-- Treg_i, 6 + int(pv['N_Ag']
    for idx in range(0,int(pv['N_Ag'])):
        dy_dt[6 + int(pv['N_Ag']) + idx] = func_d_A_Treg(y[6 + int(pv['N_Ag']) + idx],pv['mu_Treg'],pv['m'],
                                                         pv['gamma_IL2_Treg'],pv['k_IL2_Treg'],y[3],
                                                         pv['A_max_LP'],A_LP_sum,pv[f'alpha_{idx:d}'],
                                                         pv['phi_Treg'],pv[f'psi_Treg_{idx:d}'],
                                                         pv[f'eAg_load_del_{idx:d}'] + pv[f'tAg_load_del_{idx:d}'],
                                                         pv['gamma_IL1beta23_Treg'],pv['k_IL1beta23_Treg'],y[1],
                                                         pv['gamma_TH17'],y[6 + idx],pv['lambda_Treg'],
                                                         pv['xi_Treg_Epi'],pv['K_Epi'],pv['Ag_load_sum'],
                                                         pv[f'eAg_load_{idx:d}'] + pv[f'tAg_load_{idx:d}'],
                                                         pv['A_Epi'],pv['xi_Treg_Mac'],
                                                         pv['gamma_IL10_Mac'],pv['k_IL10_Mac'],y[5],
                                                         pv['K_Mac'],pv['A_Mac'],pv['A_DC'])
    #
    #-- Treg_i, 6 + int(pv['N_Ag']
    #-- return
    return dy_dt
###

###
def compute_dy_eval(n_eval,r_dict,tag):
    dy_eval = numpy.zeros((n_eval,))
    for item in r_dict.keys():
        if item.startswith('g_'):
            dy_eval += r_dict[item]
        elif item.startswith('d_'):
            dy_eval -= r_dict[item]
        elif item.startswith('x_'):
            continue
        else:
            print('\n' + tag)
            print(item)
            print(item + ' dose not start with g_, d_, or x_')
            error_exit()
        #
    #
    #-- return
    return dy_eval
###

###
def ODE_simple_immune_model_eval(y,pv,df_Ag):
    #-- y[0] = SAA
    #-- y[1] = IL-1beta/23
    #-- y[2] = IL-22
    #-- y[3] = IL-2
    #-- y[4] = IL-17
    #-- y[5] = IL-10
    #-- y[6] = TH17_1
    #-- y[7] = TH17_2
    #-- y[5 + N_Ag] = TH17_N_Ag
    #-- y[5 + N_Ag + 1] = Treg_1
    #-- y[5 + 2*N_Ag] = Treg_N_Ag
    #-- dy_dt
    n_sol = len(y.index.values)
    dy_dt = pandas.DataFrame({})
    dy_dt['t'] = y.index.values
    dy_dt = dy_dt.set_index('t')
    #-- dy_dt
    #-- TH17_sum and Treg_sum
    TH17_sum = numpy.zeros((len(dy_dt.index.values),))
    Treg_sum = numpy.zeros((len(dy_dt.index.values),))
    for idx in range(0,int(pv['N_Ag'])):
        TH17_sum += y.loc[:,f'TH17_{idx:d}'].values
        Treg_sum += y.loc[:,f'Treg_{idx:d}'].values
    #
    A_LP_sum = pv['A_LP'] + TH17_sum + Treg_sum
    #-- TH17_sum and Treg_sum
    #-- y[0] = c_SAA
    tmp_tag = 'd_SAA'
    dy_dt[tmp_tag] = func_d_C_SAA(y.loc[:,'SAA'].values,pv['delta_SAA'],pv['m'],
                                  pv['rho_SAA_Epi'],pv['gamma_IL22_SAA_Epi'],pv['k_IL22_Epi'],
                                  y.loc[:,'IL-22'].values,pv['K_Epi_SAA'],
                                  df_Ag.loc[:,'beta_eAg'].values,df_Ag.loc[:,'Ag_load_sum'].values,
                                  pv['A_Epi'])
    tmp_dict = func_d_C_SAA_rate(y.loc[:,'SAA'].values,pv['delta_SAA'],pv['m'],
                                 pv['rho_SAA_Epi'],pv['gamma_IL22_SAA_Epi'],pv['k_IL22_Epi'],
                                 y.loc[:,'IL-22'].values,pv['K_Epi_SAA'],
                                 df_Ag.loc[:,'beta_eAg'].values,df_Ag.loc[:,'Ag_load_sum'].values,
                                 pv['A_Epi'])
    tmp_eval = compute_dy_eval(n_sol,tmp_dict,tmp_tag)
    print('\n' + tmp_tag + ' check')
    print(numpy.amax(numpy.absolute(dy_dt[tmp_tag].values - tmp_eval)))
    for item in tmp_dict.keys():
        dy_dt[tmp_tag + '_' + item] = tmp_dict[item]
    #
    #-- y[0] = c_SAA
    #-- y[1] = IL-1beta/23
    tmp_tag = 'd_IL-1beta/23'
    dy_dt[tmp_tag] = func_d_C_IL1beta23(y.loc[:,'IL-1beta/23'].values,pv['delta_IL1beta23'],pv['m'],
                                        pv['rho_IL1beta23_Mac'],pv['gamma_SAA_IL1beta23_Mac'],pv['k_SAA_Mac'],
                                        y.loc[:,'SAA'].values,pv['rho_IL1beta23_eMac'],pv['gamma_IL10_IL1beta23_Mac'],
                                        pv['k_IL10_Mac'],y.loc[:,'IL-10'].values,pv['K_Mac_IL1beta23'],
                                        df_Ag.loc[:,'beta_eAg'].values,df_Ag.loc[:,'Ag_load_sum'].values,
                                        pv['A_Mac'])
    tmp_dict = func_d_C_IL1beta23_rate(y.loc[:,'IL-1beta/23'].values,pv['delta_IL1beta23'],pv['m'],
                                       pv['rho_IL1beta23_Mac'],pv['gamma_SAA_IL1beta23_Mac'],pv['k_SAA_Mac'],
                                       y.loc[:,'SAA'].values,pv['rho_IL1beta23_eMac'],pv['gamma_IL10_IL1beta23_Mac'],
                                       pv['k_IL10_Mac'],y.loc[:,'IL-10'].values,pv['K_Mac_IL1beta23'],
                                       df_Ag.loc[:,'beta_eAg'].values,df_Ag.loc[:,'Ag_load_sum'].values,
                                       pv['A_Mac'])
    tmp_eval = compute_dy_eval(n_sol,tmp_dict,tmp_tag)
    print('\n' + tmp_tag + ' check')
    print(numpy.amax(numpy.absolute(dy_dt[tmp_tag].values - tmp_eval)))
    for item in tmp_dict.keys():
        dy_dt[tmp_tag + '_' + item] = tmp_dict[item]
    #
    #-- y[1] = IL-1beta/23
    #-- y[2] = IL-22
    tmp_tag = 'd_IL-22'
    dy_dt[tmp_tag] = func_d_C_IL22(y.loc[:,'IL-22'].values,pv['delta_IL22'],pv['m'],
                                   pv['rho_IL22_ILC3'],pv['gamma_IL1beta23_ILC3'],pv['k_IL1beta23_ILC3'],
                                   y.loc[:,'IL-1beta/23'].values,pv['A_ILC3'],
                                   pv['rho_IL22_TH17'],TH17_sum)
    tmp_dict = func_d_C_IL22_rate(y.loc[:,'IL-22'].values,pv['delta_IL22'],pv['m'],
                                  pv['rho_IL22_ILC3'],pv['gamma_IL1beta23_ILC3'],pv['k_IL1beta23_ILC3'],
                                  y.loc[:,'IL-1beta/23'].values,pv['A_ILC3'],
                                  pv['rho_IL22_TH17'],TH17_sum)
    tmp_eval = compute_dy_eval(n_sol,tmp_dict,tmp_tag)
    print('\n' + tmp_tag + ' check')
    print(numpy.amax(numpy.absolute(dy_dt[tmp_tag].values - tmp_eval)))
    for item in tmp_dict.keys():
        dy_dt[tmp_tag + '_' + item] = tmp_dict[item]
    #
    #-- y[2] = IL-22
    #-- y[3] = IL-2
    tmp_tag = 'd_IL-2'
    dy_dt[tmp_tag] = func_d_C_IL2(y.loc[:,'IL-2'].values,pv['delta_IL2'],pv['m'],
                                  pv['rho_IL2_ILC3'],pv['gamma_IL1beta23_ILC3'],pv['k_IL1beta23_ILC3'],
                                  y.loc[:,'IL-1beta/23'].values,pv['A_ILC3'],
                                  pv['rho_IL2_eDC'],pv['rho_IL2_tDC'],pv['K_Mac_IL2'],
                                  df_Ag.loc[:,'beta_eAg'].values,df_Ag.loc[:,'beta_tAg'].values,
                                  df_Ag.loc[:,'Ag_load_sum'].values,pv['A_DC'])
    tmp_dict = func_d_C_IL2_rate(y.loc[:,'IL-2'].values,pv['delta_IL2'],pv['m'],
                                 pv['rho_IL2_ILC3'],pv['gamma_IL1beta23_ILC3'],pv['k_IL1beta23_ILC3'],
                                 y.loc[:,'IL-1beta/23'].values,pv['A_ILC3'],
                                 pv['rho_IL2_eDC'],pv['rho_IL2_tDC'],pv['K_Mac_IL2'],
                                 df_Ag.loc[:,'beta_eAg'].values,df_Ag.loc[:,'beta_tAg'].values,
                                 df_Ag.loc[:,'Ag_load_sum'].values,pv['A_DC'])
    tmp_eval = compute_dy_eval(n_sol,tmp_dict,tmp_tag)
    print('\n' + tmp_tag + ' check')
    print(numpy.amax(numpy.absolute(dy_dt[tmp_tag].values - tmp_eval)))
    for item in tmp_dict.keys():
        dy_dt[tmp_tag + '_' + item] = tmp_dict[item]
    #
    #-- y[3] = IL-2
    #-- y[4] = IL-17
    tmp_tag = 'd_IL-17'
    dy_dt[tmp_tag] = func_d_C_IL17(y.loc[:,'IL-17'].values,pv['delta_IL17'],pv['m'],
                                   pv['rho_IL17_TH17'],pv['gamma_SAA_IL17_TH17'],pv['k_SAA_TH17'],
                                   y.loc[:,'SAA'].values,pv['gamma_IL10_IL17_TH17'],pv['k_IL10_TH17'],
                                   y.loc[:,'IL-10'].values,TH17_sum)
    tmp_dict = func_d_C_IL17_rate(y.loc[:,'IL-17'].values,pv['delta_IL17'],pv['m'],
                                  pv['rho_IL17_TH17'],pv['gamma_SAA_IL17_TH17'],pv['k_SAA_TH17'],
                                  y.loc[:,'SAA'].values,pv['gamma_IL10_IL17_TH17'],pv['k_IL10_TH17'],
                                  y.loc[:,'IL-10'].values,TH17_sum)
    tmp_eval = compute_dy_eval(n_sol,tmp_dict,tmp_tag)
    print('\n' + tmp_tag + ' check')
    print(numpy.amax(numpy.absolute(dy_dt[tmp_tag].values - tmp_eval)))
    for item in tmp_dict.keys():
        dy_dt[tmp_tag + '_' + item] = tmp_dict[item]
    #
    #-- y[4] = IL-17
    #-- y[5] = IL-10
    tmp_tag = 'd_IL-10'
    dy_dt[tmp_tag] = func_d_C_IL10(y.loc[:,'IL-10'].values,pv['delta_IL10'],pv['m'],
                                   pv['rho_IL10_Treg'],pv['gamma_IL10_Treg'],pv['k_IL10_Treg'],Treg_sum,
                                   pv['rho_IL10_tTreg'],pv['gamma_IL10_tTreg'],pv['k_IL10_tTreg'],pv['A_tTreg'],
                                   pv['rho_IL10_Mac'],pv['A_Mac'],pv['rho_IL10_tMac'],pv['K_Mac_IL10'],
                                   df_Ag.loc[:,'beta_tAg'].values,df_Ag.loc[:,'Ag_load_sum'].values,
                                   pv['rho_IL10_Epi'],pv['A_Epi'])
    tmp_dict = func_d_C_IL10_rate(y.loc[:,'IL-10'].values,pv['delta_IL10'],pv['m'],
                                  pv['rho_IL10_Treg'],pv['gamma_IL10_Treg'],pv['k_IL10_Treg'],Treg_sum,
                                  pv['rho_IL10_tTreg'],pv['gamma_IL10_tTreg'],pv['k_IL10_tTreg'],pv['A_tTreg'],
                                  pv['rho_IL10_Mac'],pv['A_Mac'],pv['rho_IL10_tMac'],pv['K_Mac_IL10'],
                                  df_Ag.loc[:,'beta_tAg'].values,df_Ag.loc[:,'Ag_load_sum'].values,
                                  pv['rho_IL10_Epi'],pv['A_Epi'])
    tmp_eval = compute_dy_eval(n_sol,tmp_dict,tmp_tag)
    print('\n' + tmp_tag + ' check')
    print(numpy.amax(numpy.absolute(dy_dt[tmp_tag].values - tmp_eval)))
    for item in tmp_dict.keys():
        dy_dt[tmp_tag + '_' + item] = tmp_dict[item]
    #
    #-- y[5] = IL-10
    #-- TH17_i, 6:6 + int(pv['N_Ag']
    tmp_tag = 'd_TH17_'
    for idx in range(0,int(pv['N_Ag'])):
        tmp_eval_1 = df_Ag.loc[:,f'eAg_del_{idx:d}'].values + df_Ag.loc[:,f'tAg_del_{idx:d}'].values
        tmp_eval_2 = df_Ag.loc[:,f'eAg_{idx:d}'].values + df_Ag.loc[:,f'tAg_{idx:d}'].values
        dy_dt[tmp_tag + f'{idx:d}'] = func_d_A_TH17(y.loc[:,f'TH17_{idx:d}'].values,
                                                    pv['mu_TH17'],pv['m'],pv['A_max_LP'],A_LP_sum,
                                                    pv[f'alpha_{idx:d}'],pv['phi_TH17'],pv[f'psi_TH17_{idx:d}'],
                                                    tmp_eval_1,pv['gamma_IL1beta23_Treg'],pv['k_IL1beta23_Treg'],
                                                    y.loc[:,'IL-1beta/23'].values,
                                                    y.loc[:,f'Treg_{idx:d}'].values,pv['gamma_TH17'],
                                                    pv['lambda_TH17'],pv['xi_TH17_Epi'],pv['K_Epi'],
                                                    df_Ag.loc[:,'Ag_load_sum'].values,tmp_eval_2,
                                                    pv['A_Epi'],pv['xi_TH17_Mac'],pv['gamma_IL10_Mac'],pv['k_IL10_Mac'],
                                                    y.loc[:,'IL-10'].values,pv['K_Mac'],pv['A_Mac'],pv['A_DC'])
        tmp_dict = func_d_A_TH17_rate(y.loc[:,f'TH17_{idx:d}'].values,
                                     pv['mu_TH17'],pv['m'],pv['A_max_LP'],A_LP_sum,
                                     pv[f'alpha_{idx:d}'],pv['phi_TH17'],pv[f'psi_TH17_{idx:d}'],
                                     tmp_eval_1,pv['gamma_IL1beta23_Treg'],pv['k_IL1beta23_Treg'],
                                     y.loc[:,'IL-1beta/23'].values,
                                     y.loc[:,f'Treg_{idx:d}'].values,pv['gamma_TH17'],
                                     pv['lambda_TH17'],pv['xi_TH17_Epi'],pv['K_Epi'],
                                     df_Ag.loc[:,'Ag_load_sum'].values,tmp_eval_2,
                                     pv['A_Epi'],pv['xi_TH17_Mac'],pv['gamma_IL10_Mac'],pv['k_IL10_Mac'],
                                     y.loc[:,'IL-10'].values,pv['K_Mac'],pv['A_Mac'],pv['A_DC'])
        tmp_eval = compute_dy_eval(n_sol,tmp_dict,tmp_tag + f'{idx:d}')
        print('\n' + tmp_tag + f'{idx:d}' + ' check')
        print(numpy.amax(numpy.absolute(dy_dt[tmp_tag + f'{idx:d}'].values - tmp_eval)))
        for item in tmp_dict.keys():
            dy_dt[tmp_tag + f'{idx:d}' + '_' + item] = tmp_dict[item]
        #
    #
    #-- TH17_i, 6:6 + int(pv['N_Ag']
    #-- Treg_i, 6 + int(pv['N_Ag']
    tmp_tag = 'd_Treg_'
    for idx in range(0,int(pv['N_Ag'])):
        tmp_eval_1 = df_Ag.loc[:,f'eAg_del_{idx:d}'].values + df_Ag.loc[:,f'tAg_del_{idx:d}'].values
        tmp_eval_2 = df_Ag.loc[:,f'eAg_{idx:d}'].values + df_Ag.loc[:,f'tAg_{idx:d}'].values
        dy_dt[tmp_tag + f'{idx:d}'] = func_d_A_Treg(y.loc[:,f'Treg_{idx:d}'].values,pv['mu_Treg'],pv['m'],
                                                    pv['gamma_IL2_Treg'],pv['k_IL2_Treg'],
                                                    y.loc[:,'IL-2'].values,pv['A_max_LP'],A_LP_sum,
                                                    pv[f'alpha_{idx:d}'],pv['phi_Treg'],pv[f'psi_Treg_{idx:d}'],
                                                    tmp_eval_1,pv['gamma_IL1beta23_Treg'],pv['k_IL1beta23_Treg'],
                                                    y.loc[:,'IL-1beta/23'].values,pv['gamma_TH17'],
                                                    y.loc[:,f'TH17_{idx:d}'].values,pv['lambda_Treg'],
                                                    pv['xi_Treg_Epi'],pv['K_Epi'],
                                                    df_Ag.loc[:,'Ag_load_sum'].values,tmp_eval_2,
                                                    pv['A_Epi'],pv['xi_Treg_Mac'],pv['gamma_IL10_Mac'],pv['k_IL10_Mac'],
                                                    y.loc[:,'IL-10'].values,pv['K_Mac'],pv['A_Mac'],pv['A_DC'])
        tmp_dict = func_d_A_Treg_rate(y.loc[:,f'Treg_{idx:d}'].values,pv['mu_Treg'],pv['m'],
                                      pv['gamma_IL2_Treg'],pv['k_IL2_Treg'],
                                      y.loc[:,'IL-2'].values,pv['A_max_LP'],A_LP_sum,
                                      pv[f'alpha_{idx:d}'],pv['phi_Treg'],pv[f'psi_Treg_{idx:d}'],
                                      tmp_eval_1,pv['gamma_IL1beta23_Treg'],pv['k_IL1beta23_Treg'],
                                      y.loc[:,'IL-1beta/23'].values,pv['gamma_TH17'],
                                      y.loc[:,f'TH17_{idx:d}'].values,pv['lambda_Treg'],
                                      pv['xi_Treg_Epi'],pv['K_Epi'],
                                      df_Ag.loc[:,'Ag_load_sum'].values,tmp_eval_2,
                                      pv['A_Epi'],pv['xi_Treg_Mac'],pv['gamma_IL10_Mac'],pv['k_IL10_Mac'],
                                      y.loc[:,'IL-10'].values,pv['K_Mac'],pv['A_Mac'],pv['A_DC'])
        tmp_eval = compute_dy_eval(n_sol,tmp_dict,tmp_tag + f'{idx:d}')
        print('\n' + tmp_tag + f'{idx:d}' + ' check')
        print(numpy.amax(numpy.absolute(dy_dt[tmp_tag + f'{idx:d}'].values - tmp_eval)))
        for item in tmp_dict.keys():
            dy_dt[tmp_tag + f'{idx:d}' + '_' + item] = tmp_dict[item]
        #
    #
    #-- Treg_i, 6 + int(pv['N_Ag']
    #-- return
    return dy_dt
###

###
def solve_ODE_simple_immune_model(t_start,t_end,dt,y_init,pv):
    sol = solve_ivp(fun=lambda t,y : ODE_simple_immune_model(t,y,pv),\
                    t_span=[t_start,t_end],y0=y_init,method='LSODA',\
                    t_eval=[t_start,t_end],max_step=dt)
    #-- return
    return sol
###

###
def solve_ODE_simple_immune_model_i(t_start,t_end,dt,y_init,pv):
    sol = solve_ivp(fun=lambda t,y : ODE_simple_immune_model(t,y,pv),\
                    t_span=[t_start,t_end],y0=y_init,method='LSODA',\
                    t_eval=numpy.round(numpy.linspace(t_start,t_end,int((t_end-t_start)/dt) + 1),decimals=12),
                    max_step=dt)
    sol.y[:,0] = y_init
    #-- return
    return sol
###

###
def initialization_simple_immune_simulation(df):
    #-- setup initial condition
    #-- y[0] = SAA
    #-- y[1] = IL-1beta/23
    #-- y[2] = IL-22
    #-- y[3] = IL-2
    #-- y[4] = IL-17
    #-- y[5] = IL-10
    #-- y[6] = TH17_1
    #-- y[7] = TH17_2
    #-- y[5 + N_Ag] = TH17_N_Ag
    #-- y[5 + N_Ag + 1] = Treg_1
    #-- y[5 + 2*N_Ag] = Treg_N_Ag
    #-- setup parameter values corresponding to specific antigens
    for item in ['Ag_load_0','alpha','beta','h','nu','del','psi_TH17','psi_Treg','TH17_0','Treg_0']:
        df_tmp = pandas.read_csv('antigen_specific_p/' + item + '.csv').set_index('antigen')
        for idx in range(0,int(df.loc['N_Ag','value'])):
            df.loc[item + f'_{idx:d}','value'] = df_tmp.loc[idx,'value']
            df.loc[item + f'_{idx:d}','unit'] = 'unit'
            df.loc[item + f'_{idx:d}','type'] = 'code'
        #
    #
    for idx in range(0,int(df.loc['N_Ag','value'])):
        df.loc[f'eAg_load_0_{idx:d}','value'] = 0.0
        df.loc[f'eAg_load_0_{idx:d}','unit'] = 'unit'
        df.loc[f'eAg_load_0_{idx:d}','type'] = 'code'
        df.loc[f'tAg_load_0_{idx:d}','value'] = 0.0
        df.loc[f'tAg_load_0_{idx:d}','unit'] = 'unit'
        df.loc[f'tAg_load_0_{idx:d}','type'] = 'code'
        if df.loc[f'alpha_{idx:d}','value'] >= 0.5:
            df.loc[f'eAg_load_0_{idx:d}','value'] = df.loc[f'Ag_load_0_{idx:d}','value']
        else:
            df.loc[f'tAg_load_0_{idx:d}','value'] = df.loc[f'Ag_load_0_{idx:d}','value']
        #
    #
    #-- setup parameter values corresponding to specific antigens
    #-- setup initial condition
    n_Eq = 6 + 2*int(df.loc['N_Ag','value'])
    y_init = numpy.zeros((n_Eq,))
    y_init[0] = df.loc['C_SAA_0','value']
    y_init[1] = df.loc['C_IL1beta23_0','value']
    y_init[2] = df.loc['C_IL22_0','value']
    y_init[3] = df.loc['C_IL2_0','value']
    y_init[4] = df.loc['C_IL17_0','value']
    y_init[5] = df.loc['C_IL10_0','value']
    for idx in range(0,int(df.loc['N_Ag','value'])):
        y_init[6 + idx] = df.loc[f'TH17_0_{idx:d}','value']
        y_init[6 + int(df.loc['N_Ag','value']) + idx] = df.loc[f'Treg_0_{idx:d}','value']
    #
    #-- setup initial condition
    #-- dt setup, dt_max = 0.01 (rel_error < 5%), dt_max = 0.001 (rel_error < 1%)
    dt_sets = numpy.array([0.01,0.005,0.002,0.001])
    time_scale = list()
    for item in ['delta_SAA','delta_IL1beta23','delta_IL22','delta_IL2','delta_IL17','delta_IL10','mu_TH17','mu_Treg']:
        time_scale.append(1.0/df.loc[item,'value'])
    #
    tmp_idx = numpy.where(dt_sets < numpy.amin(time_scale))
    df.loc['dt','value'] = numpy.amax(dt_sets[tmp_idx[0]])
    #-- dt setup, dt_max = 0.01 (rel_error < 5%), dt_max = 0.001 (rel_error < 1%)
    #-- initialization, int_t_start^t_start * ds = 0
    #-- f_IL22_IL17 = (func_gamma(gamma_IL22,k_IL22,c_IL22,m) + func_gamma(gamma_IL17,k_IL17,c_IL17,m))*A_Epi
    dt = round(df.loc['dt','value'],6)
    t_start = round(df.loc['t_start','value'],6)
    t_end = round(df.loc['t_end','value'],6)
    c_IL22 = numpy.zeros((int((t_end-t_start)/dt) + 1,))
    c_IL22[0] = df.loc['C_IL22_0','value']
    c_IL17 = numpy.zeros((int((t_end-t_start)/dt) + 1,))
    c_IL17[0] = df.loc['C_IL17_0','value']
    f_IL22_IL17 = numpy.zeros((int((t_end-t_start)/dt) + 1,))
    f_IL22 = numpy.zeros((int((t_end-t_start)/dt) + 1,))
    f_IL17 = numpy.zeros((int((t_end-t_start)/dt) + 1,))
    tmp1 = func_gamma(df.loc['gamma_IL22','value'],df.loc['k_IL22','value'],y_init[2],df.loc['m','value'])*df.loc['A_Epi','value']
    tmp2 = func_gamma(df.loc['gamma_IL17','value'],df.loc['k_IL17','value'],y_init[4],df.loc['m','value'])*df.loc['A_Epi','value']
    f_IL22_IL17[0] = tmp1 + tmp2
    f_IL22[0] = tmp1
    f_IL17[0] = tmp2
    df.loc['t_c','value'] = t_start
    df.loc['t_c','unit'] = 'unit'
    df.loc['t_c','type'] = 'type'
    df.loc['tau_inv','value'] = 1.0/df.loc['tau','value']
    df.loc['tau_inv','unit'] = 'unit'
    df.loc['tau_inv','type'] = 'type'
    df.loc['f_IL22_IL17','value'] = f_IL22_IL17[0]
    df.loc['f_IL22_IL17','unit'] = 'unit'
    df.loc['f_IL22_IL17','type'] = 'code'
    #-- f_IL22_IL17 = (func_gamma(gamma_IL22,k_IL22,c_IL22,m) + func_gamma(gamma_IL17,k_IL17,c_IL17,m))*A_Epi
    #-- initialization, int_t_start^t_start * ds = 0 
    #-- antigen load, int_t_start^t_start * ds = 0.0 -> Ag_load_i = Ag_load_0_i
    eAg_load = numpy.zeros((int(df.loc['N_Ag','value']),int((t_end-t_start)/dt) + 1))
    tAg_load = numpy.zeros((int(df.loc['N_Ag','value']),int((t_end-t_start)/dt) + 1))
    eAg_load_del = numpy.zeros((int(df.loc['N_Ag','value']),int((t_end-t_start)/dt) + 1))
    tAg_load_del = numpy.zeros((int(df.loc['N_Ag','value']),int((t_end-t_start)/dt) + 1))
    for idx in range(0,int(df.loc['N_Ag','value'])):
        eAg_load[idx,0] = df.loc[f'eAg_load_0_{idx:d}','value']
        df.loc[f'eAg_load_{idx:d}','value'] = eAg_load[idx,0]
        df.loc[f'eAg_load_{idx:d}','unit'] = 'unit'
        df.loc[f'eAg_load_{idx:d}','type'] = 'code'
        df.loc[f'eAg_load_t_del_{idx:d}','value'] = df.loc['t_c','value'] - df.loc[f'del_{idx:d}','value']
        df.loc[f'eAg_load_t_del_{idx:d}','unit'] = 'unit'
        df.loc[f'eAg_load_t_del_{idx:d}','type'] = 'code'
        if df.loc[f'eAg_load_t_del_{idx:d}','value'] == df.loc['t_start','value']:
            df.loc[f'eAg_load_del_{idx:d}','value'] = eAg_load[idx,0]
            eAg_load_del[idx,0] = df.loc[f'eAg_load_del_{idx:d}','value']
        else:
            df.loc[f'eAg_load_del_{idx:d}','value'] = 0.0
            eAg_load_del[idx,0] = 0.0
        #
        df.loc[f'eAg_load_del_{idx:d}','unit'] = 'unit'
        df.loc[f'eAg_load_del_{idx:d}','type'] = 'code'
        tAg_load[idx,0] = df.loc[f'tAg_load_0_{idx:d}','value']
        df.loc[f'tAg_load_{idx:d}','value'] = tAg_load[idx,0]
        df.loc[f'tAg_load_{idx:d}','unit'] = 'unit'
        df.loc[f'tAg_load_{idx:d}','type'] = 'code'
        df.loc[f'tAg_load_t_del_{idx:d}','value'] = df.loc['t_c','value'] - df.loc[f'del_{idx:d}','value']
        df.loc[f'tAg_load_t_del_{idx:d}','unit'] = 'unit'
        df.loc[f'tAg_load_t_del_{idx:d}','type'] = 'code'
        if df.loc[f'tAg_load_t_del_{idx:d}','value'] == df.loc['t_start','value']:
            df.loc[f'tAg_load_del_{idx:d}','value'] = tAg_load[idx,0]
            tAg_load_del[idx,0] = df.loc[f'tAg_load_del_{idx:d}','value']
        else:
            df.loc[f'tAg_load_del_{idx:d}','value'] = 0.0
            tAg_load_del[idx,0] = 0.0
        #
        df.loc[f'tAg_load_del_{idx:d}','unit'] = 'unit'
        df.loc[f'tAg_load_del_{idx:d}','type'] = 'code'
    #
    df.loc['Ag_load_sum','value'] = 0.0
    df.loc['Ag_load_sum','unit'] = 'unit'
    df.loc['Ag_load_sum','type'] = 'code'
    df.loc['beta_eAg','value'] = 0.0
    df.loc['beta_eAg','unit'] = 'unit'
    df.loc['beta_eAg','type'] = 'code'
    df.loc['beta_tAg','value'] = 0.0
    df.loc['beta_tAg','unit'] = 'unit'
    df.loc['beta_tAg','type'] = 'code'
    #-- antigen load, int_t_start^t_start * ds = 0.0 -> Ag_load_i = Ag_load_0_i
    #-- # of cells in LP
    df.loc['A_LP','value'] = 0.0
    df.loc['A_LP','unit'] = 'unit'
    df.loc['A_LP','type'] = 'code'
    for itme in ['A_Mac','A_ILC3','A_DC','A_Stro','A_tTreg']:
        df.loc['A_LP','value'] += df.loc[itme,'value']
    #
    #-- # of cells in LP
    #-- parameters
    pv = df['value'].copy(deep=True)
    pv = func_initialize_antigen_load(pv)
    t_eval = numpy.round(numpy.linspace(t_start,t_end,int((t_end-t_start)/dt) + 1),decimals=6)
    sol_y = numpy.zeros((n_Eq,len(t_eval)))
    sol_y[:,0] = y_init
    #-- parameters
    #-- return
    return t_eval,sol_y,pv,f_IL22_IL17,eAg_load,tAg_load,f_IL22,f_IL17,eAg_load_del,tAg_load_del
###

###
def update_antigen_load(df,c_idx,c_IL22,c_IL17,t_eval,f_IL22_IL17,eAg_load,tAg_load,
                        f_IL22,f_IL17,eAg_load_del,tAg_load_del):
    #-- time update
    df['t_c'] = t_eval[c_idx]
    #-- time update
    #-- update values, f_IL22_IL17
    tmp1 = func_gamma(df['gamma_IL22'],df['k_IL22'],c_IL22,df['m'])*df['A_Epi']
    tmp2 = func_gamma(df['gamma_IL17'],df['k_IL17'],c_IL17,df['m'])*df['A_Epi']
    f_IL22_IL17[c_idx] = tmp1 + tmp2
    f_IL22[c_idx] = tmp1
    f_IL17[c_idx] = tmp2
    df['f_IL22_IL17'] = f_IL22_IL17[c_idx]
    #-- update values, f_IL22_IL17
    #-- compute int_t0^t f_IL22_IL17*exp(-1/tau*(t - s)) ds
    exp_eval = numpy.exp(df['tau_inv']*(t_eval[0:c_idx + 1] - df['t_c']))
    tmp_eval = f_IL22_IL17[0:c_idx + 1]*exp_eval
    int_f_IL22_IL17 = numpy.trapz(tmp_eval,x=t_eval[0:c_idx + 1])
    #-- compute int_t0^t f_IL22_IL17*exp(-1/tau*(t - s)) ds
    #-- eAg_load and eAg_load
    beta_eAg = 0.0
    beta_tAg = 0.0
    Ag_load_sum = 0.0
    for idx in range(0,int(df['N_Ag'])):
        #-- factor
        fac = 1.0/(1.0 + numpy.power(df['tau_inv']*df[f'nu_{idx:d}']*int_f_IL22_IL17,df[f'h_{idx:d}']))
        #-- factor
        #-- eAg
        eAg_load[idx,c_idx] = df[f'eAg_load_0_{idx:d}']*fac
        df[f'eAg_load_{idx:d}'] = eAg_load[idx,c_idx]
        df[f'eAg_load_t_del_{idx:d}'] = df['t_c'] - df[f'del_{idx:d}']
        if df[f'eAg_load_t_del_{idx:d}'] >= df['t_start']:
            f_eAg = splrep(t_eval[0:c_idx + 1],eAg_load[idx,0:c_idx + 1],k=1)
            df[f'eAg_load_del_{idx:d}'] = splev(df[f'eAg_load_t_del_{idx:d}'],f_eAg)
        else:
            df[f'eAg_load_del_{idx:d}'] = 0.0
        #
        beta_eAg += df[f'beta_{idx:d}']*df[f'eAg_load_{idx:d}']
        eAg_load_del[idx,c_idx] = df[f'eAg_load_del_{idx:d}']
        #-- eAg
        #-- tAg
        tAg_load[idx,c_idx] = df[f'tAg_load_0_{idx:d}']*fac
        df[f'tAg_load_{idx:d}'] = tAg_load[idx,c_idx]
        df[f'tAg_load_t_del_{idx:d}'] = df['t_c'] - df[f'del_{idx:d}']
        if df[f'tAg_load_t_del_{idx:d}'] >= df['t_start']:
            f_tAg = splrep(t_eval[0:c_idx + 1],tAg_load[idx,0:c_idx + 1],k=1)
            df[f'tAg_load_del_{idx:d}'] = splev(df[f'tAg_load_t_del_{idx:d}'],f_tAg)
        else:
            df[f'tAg_load_del_{idx:d}'] = 0.0
        #
        beta_tAg += df[f'beta_{idx:d}']*df[f'tAg_load_{idx:d}']
        tAg_load_del[idx,c_idx] = df[f'tAg_load_del_{idx:d}']
        #-- tAg
        Ag_load_sum += df[f'eAg_load_{idx:d}'] + df[f'tAg_load_{idx:d}']
    #
    df['beta_eAg'] = beta_eAg
    df['beta_tAg'] = beta_tAg
    df['Ag_load_sum'] = Ag_load_sum
    #-- eAg_load and eAg_load
    #-- return
    return df,f_IL22_IL17,eAg_load,tAg_load,f_IL22,f_IL17,eAg_load_del,tAg_load_del
###

###
def handle_negative_solution(t_0,t_1,sol_0,dt,df):
    #-- tmp_idx = numpy.where(sol.y[:,1] < 0.0), len(tmp_idx[0]) != 0
    check_neg = True
    #-- tmp_idx = numpy.where(sol.y[:,1] < 0.0), len(tmp_idx[0]) != 0
    #-- solve IVP with small dt
    dt_new = dt*0.1
    while (check_neg == True):
        sol = solve_ODE_simple_immune_model_i(t_0,t_1,dt_new,sol_0,df)
        if numpy.amin(sol.y) >= 0.0:
            check_neg = False
            status = True
        #
        dt_new = dt_new*0.1
        if dt_new < df['dt_min']:
            check_neg = False
            status = False
        #
    #
    #-- solve IVP with small dt
    #-- return
    return status,sol.t,sol.y
###

###
def save_IVP_solution(df,t_eval,sol_y,df_file_name):
    c_dict = { 0 : 'SAA', 1 : 'IL-1beta/23', 2 : 'IL-22', 3 : 'IL-2', 4 : 'IL-17', 5 : 'IL-10' }
    df_out = pandas.DataFrame({})
    df_out['t'] = t_eval
    for c_key in c_dict.keys():
        df_out[c_dict[c_key]] = sol_y[c_key,:]
    #
    for idx in range(0,int(df['N_Ag'])):
        df_out[f'TH17_{idx:d}'] = sol_y[6 + idx,:]
    #
    for idx in range(0,int(df['N_Ag'])):
        df_out[f'Treg_{idx:d}'] = sol_y[6 + int(df['N_Ag']) + idx,:]
    #
    df_out.to_csv(df_file_name,index=False)
    #-- return
    return 0
###

###
def save_solution(df,t_eval,sol_y,eAg_load,tAg_load,f_IL22_IL17,
                  f_IL22,f_IL17,eAg_load_del,tAg_load_del,df_file_name):
    c_dict = { 0 : 'SAA', 1 : 'IL-1beta/23', 2 : 'IL-22', 3 : 'IL-2', 4 : 'IL-17', 5 : 'IL-10' }
    df_out = pandas.DataFrame({})
    df_out['t'] = t_eval
    for c_key in c_dict.keys():
        df_out[c_dict[c_key]] = sol_y[c_key,:]
    #
    for idx in range(0,int(df['N_Ag'])):
        df_out[f'TH17_{idx:d}'] = sol_y[6 + idx,:]
    #
    for idx in range(0,int(df['N_Ag'])):
        df_out[f'Treg_{idx:d}'] = sol_y[6 + int(df['N_Ag']) + idx,:]
    #
    df_out['f_IL22_IL17'] = f_IL22_IL17
    df_out['f_IL22'] = f_IL22
    df_out['f_IL17'] = f_IL17
    df_out['beta_eAg'] = 0.0
    df_out['beta_tAg'] = 0.0
    df_out['Ag_load_sum'] = 0.0
    for idx in range(0,int(df['N_Ag'])):
        df_out[f'eAg_{idx:d}'] = eAg_load[idx,:]
        df_out['beta_eAg'] += df[f'beta_{idx:d}']*eAg_load[idx,:]
        df_out[f'tAg_{idx:d}'] = tAg_load[idx,:]
        df_out['beta_tAg'] += df[f'beta_{idx:d}']*tAg_load[idx,:]
        df_out[f'Ag_{idx:d}'] = df_out[f'eAg_{idx:d}'].values + df_out[f'tAg_{idx:d}'].values
        df_out[f'eAg_del_{idx:d}'] = eAg_load_del[idx,:]
        df_out[f'tAg_del_{idx:d}'] = tAg_load_del[idx,:]
        df_out['Ag_load_sum'] += eAg_load[idx,:] + tAg_load[idx,:]
    #
    df_out.to_csv(df_file_name,index=False)
    #-- return
    return 0
###

###
def solve_ODEs(out_dir,df):
    #-- simple immune model, variables
    #-- y[0] = SAA
    #-- y[1] = IL-1beta-23
    #-- y[2] = IL-22
    #-- y[3] = IL-2
    #-- y[4] = IL-17
    #-- y[5] = IL-10
    #-- y[6] = TH17_1
    #-- y[7] = TH17_2
    #-- y[5 + N_Ag] = TH17_N_Ag
    #-- y[5 + N_Ag + 1] = Treg_1
    #-- y[5 + 2*N_Ag] = Treg_N_Ag
    #-- simple immune model, variables
    #-- initialization of the simple immune model simulation
    t_eval,sol_y,pv,f_IL22_IL17,eAg_load,tAg_load,f_IL22,f_IL17,eAg_load_del,tAg_load_del = initialization_simple_immune_simulation(df)
    #-- initialization of the simple immune model simulation
    #-- solve IVP, method of steps
    warning_message = list()
    for idx in range(1,len(t_eval)):
        sol = solve_ODE_simple_immune_model(t_eval[idx - 1],t_eval[idx],pv['dt'],sol_y[:,idx - 1],pv)
        sol_y[:,idx] = sol.y[:,1]
        #-- check negative solutions
        if numpy.amin(sol.y[:,1]) < 0:
            warning_message.append(f't = {t_eval[idx]:f}, numpy.amin(sol) < 0\n')
            h_status,sol_t,sol_f = handle_negative_solution(t_eval[idx - 1],t_eval[idx],sol_y[:,idx - 1],pv['dt'],pv)
            sol_y[:,idx] = sol_f[:,-1]
            if h_status == False:
                save_IVP_solution(pv,sol_t,sol_f,out_dir + 'simulation_result_small_dt_fail.csv')
                save_solution(pv,t_eval[:idx + 1],sol_y[:,:idx + 1],eAg_load[:,:idx + 1],tAg_load[:,:idx + 1],
                              f_IL22_IL17[:idx + 1],f_IL22[:idx + 1],f_IL17[:idx + 1],
                              eAg_load_del[:,:idx + 1],tAg_load_del[:,:idx + 1],out_dir + 'simulation_result_fail.csv')
                warning_message.append(f't = {t_eval[idx]:f}, fail to resolve the negative solution problem\n')
                if len(warning_message) != 0:
                    w_file = open(out_dir + 'warning_messages.txt','w')
                    for w_idx in range(0,len(warning_message)):
                        w_file.write(warning_message[w_idx])
                    #
                    w_file.close()
                #
                return 0
            #
            warning_message.append(f't = {t_eval[idx]:f}, success to resolve the negative solution problem\n')
        #
        #-- check negative solutions
        #-- update f_IL22_IL17, eAg_load, tAg_load, eAg_load_del, tAg_load_del
        pv,f_IL22_IL17,eAg_load,tAg_load,f_IL22,f_IL17,eAg_load_del,tAg_load_del = update_antigen_load(pv,idx,sol_y[2,idx],sol_y[4,idx],t_eval,
                                                                                                       f_IL22_IL17,eAg_load,tAg_load,
                                                                                                       f_IL22,f_IL17,eAg_load_del,tAg_load_del)
        #-- update f_IL22_IL17, eAg_load, tAg_load, eAg_load_del, tAg_load_del
    #
    #-- solve IVP, method of steps
    #-- save solutions
    save_solution(pv,t_eval,sol_y,eAg_load,tAg_load,f_IL22_IL17,f_IL22,f_IL17,eAg_load_del,tAg_load_del,
                  out_dir + 'simulation_result.csv')
    if len(warning_message) != 0:
        w_file = open(out_dir + 'warning_messages.txt','w')
        for w_idx in range(0,len(warning_message)):
            w_file.write(warning_message[w_idx])
        #
        w_file.close()
    #
    #-- save solutions
    #-- return
    return 0
###

###
def draw_figure(x_len,y_len,x_eval,y_eval,x_max,y_max,x_label,y_label,tag,fig_name):
    #-- default settings
    mpl.use('Agg')
    mpl.rcParams['font.sans-serif'] = 'Arial'
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['figure.figsize'] = [3.5,3.5] # 3.5 inches == 89 mm, 4.7 inches == 120 mm, 5.354
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['font.size'] = 10.0
    mpl.rcParams['legend.fontsize'] = 9.0
    mpl.rcParams['figure.titlesize'] = 10.0
    mpl.rcParams['scatter.marker'] = 's'
    mpl.rcParams['axes.linewidth'] = 0.5
    line_color = '#069AF3'
    #-- default settings
    #-- figure
    fig = plt.figure(figsize=(x_len,y_len),dpi=300)
    gs = GridSpec(1,1,figure=fig)
    ax_fig = fig.add_subplot(gs[0])
    ax_fig.tick_params(axis='x',which='both',top=True,bottom=True,labelbottom=True,labeltop=False,direction='in',width=0.5,length=1.5)
    ax_fig.tick_params(axis='y',which='both',left=True,right=True,labelleft=True,labelright=False,direction='in',width=0.5,length=1.5)
    ax_fig.plot(x_eval,y_eval,color=line_color,linestyle='solid',linewidth=1.0,label=tag)
    ax_fig.set_xlim([0.0,x_max])
    ax_fig.set_ylim([0.0,y_max])
    ax_fig.set_xlabel(x_label)
    ax_fig.set_ylabel(y_label)
    ax_fig.legend(loc='upper right')
    #-- figure
    #-- draw
    # plt.savefig('figure_PPI.tiff',format='tiff',dpi=300, bbox_inches='tight',pad_inches=save_pad)
    # plt.savefig('figure_PPI.eps',format='eps',dpi=300, bbox_inches='tight',pad_inches=save_pad)
    # plt.savefig('figure_PPI.pdf',format='pdf',dpi=300, bbox_inches='tight',pad_inches=save_pad)
    plt.savefig(fig_name + '.tiff',format='tiff',dpi=300,bbox_inches='tight')
    plt.close()
    # img = PIL.Image.open(fig_name + '.tiff')
    # print('\ntiff size')
    # print(img.size)
    #-- draw
    #-- return
    return 0
###

###
def draw_figure_Ag(x_len,y_len,x_eval,y_Ag,y_eAg,y_tAg,x_max,y_max,x_label,y_label,tag,fig_name):
    #-- default settings
    mpl.use('Agg')
    mpl.rcParams['font.sans-serif'] = 'Arial'
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['figure.figsize'] = [3.5,3.5] # 3.5 inches == 89 mm, 4.7 inches == 120 mm, 5.354
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['font.size'] = 10.0
    mpl.rcParams['legend.fontsize'] = 9.0
    mpl.rcParams['figure.titlesize'] = 10.0
    mpl.rcParams['scatter.marker'] = 's'
    mpl.rcParams['axes.linewidth'] = 0.5
    line_color_a = '#000000'
    line_color_e = '#069AF3'
    line_color_t = '#8C000F'
    #-- default settings
    #-- figure
    fig = plt.figure(figsize=(x_len,y_len),dpi=300)
    gs = GridSpec(1,1,figure=fig)
    ax_fig = fig.add_subplot(gs[0])
    ax_fig.tick_params(axis='x',which='both',top=True,bottom=True,labelbottom=True,labeltop=False,direction='in',width=0.5,length=1.5)
    ax_fig.tick_params(axis='y',which='both',left=True,right=True,labelleft=True,labelright=False,direction='in',width=0.5,length=1.5)
    ax_fig.plot(x_eval,y_Ag,color=line_color_a,linestyle='solid',linewidth=1.0,label='Ag_' + tag)
    ax_fig.plot(x_eval,y_eAg,color=line_color_e,linestyle='solid',linewidth=1.0,label='eAg_' + tag)
    ax_fig.plot(x_eval,y_tAg,color=line_color_t,linestyle='solid',linewidth=1.0,label='tAg_' + tag)
    ax_fig.set_xlim([0.0,x_max])
    ax_fig.set_ylim([0.0,y_max])
    ax_fig.set_xlabel(x_label)
    ax_fig.set_ylabel(y_label)
    ax_fig.legend(loc='upper right')
    #-- figure
    #-- draw
    # plt.savefig('figure_PPI.tiff',format='tiff',dpi=300, bbox_inches='tight',pad_inches=save_pad)
    # plt.savefig('figure_PPI.eps',format='eps',dpi=300, bbox_inches='tight',pad_inches=save_pad)
    # plt.savefig('figure_PPI.pdf',format='pdf',dpi=300, bbox_inches='tight',pad_inches=save_pad)
    plt.savefig(fig_name + '.tiff',format='tiff',dpi=300,bbox_inches='tight')
    plt.close()
    # img = PIL.Image.open(fig_name + '.tiff')
    # print('\ntiff size')
    # print(img.size)
    #-- draw
    #-- return
    return 0
###

###
def draw_figure_f_gamma(x_len,y_len,x_eval,y_IL22,y_IL17,x_max,x_label,y_label,pv,fig_name):
    #-- default settings
    mpl.use('Agg')
    mpl.rcParams['font.sans-serif'] = 'Arial'
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['figure.figsize'] = [3.5,3.5] # 3.5 inches == 89 mm, 4.7 inches == 120 mm, 5.354
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['font.size'] = 10.0
    mpl.rcParams['legend.fontsize'] = 9.0
    mpl.rcParams['figure.titlesize'] = 10.0
    mpl.rcParams['scatter.marker'] = 's'
    mpl.rcParams['axes.linewidth'] = 0.5
    line_color_e = '#069AF3'
    line_color_t = '#8C000F'
    #-- default settings
    #-- figure
    fig = plt.figure(figsize=(x_len,y_len),dpi=300)
    gs = GridSpec(1,1,figure=fig)
    ax_fig = fig.add_subplot(gs[0])
    ax_fig.tick_params(axis='x',which='both',top=True,bottom=True,labelbottom=True,labeltop=False,direction='in',width=0.5,length=1.5)
    ax_fig.tick_params(axis='y',which='both',left=True,right=True,labelleft=True,labelright=False,direction='in',width=0.5,length=1.5)
    y_IL22_eval = func_gamma(pv['gamma_IL22'],pv['k_IL22'],y_IL22,pv['m'])*pv['A_Epi']
    y_IL17_eval = func_gamma(pv['gamma_IL17'],pv['k_IL17'],y_IL17,pv['m'])*pv['A_Epi']
    y_max = numpy.amax([numpy.amax(y_IL22_eval),numpy.amax(y_IL17_eval)])*1.1
    ax_fig.plot(x_eval,y_IL22_eval,color=line_color_e,linestyle='solid',linewidth=1.0,label=r'$\Gamma_{\mathrm{IL-22}}$' + '*A_Epi')
    ax_fig.plot(x_eval,y_IL17_eval,color=line_color_t,linestyle='solid',linewidth=1.0,label=r'$\Gamma_{\mathrm{IL-17}}$' + '*A_Epi')
    ax_fig.set_xlim([0.0,x_max])
    ax_fig.set_ylim([0.0,y_max])
    ax_fig.set_xlabel(x_label)
    ax_fig.set_ylabel(y_label)
    ax_fig.legend(loc='best')
    #-- figure
    #-- draw
    # plt.savefig('figure_PPI.tiff',format='tiff',dpi=300, bbox_inches='tight',pad_inches=save_pad)
    # plt.savefig('figure_PPI.eps',format='eps',dpi=300, bbox_inches='tight',pad_inches=save_pad)
    # plt.savefig('figure_PPI.pdf',format='pdf',dpi=300, bbox_inches='tight',pad_inches=save_pad)
    plt.savefig(fig_name + '.tiff',format='tiff',dpi=300,bbox_inches='tight')
    plt.close()
    # img = PIL.Image.open(fig_name + '.tiff')
    # print('\ntiff size')
    # print(img.size)
    #-- draw
    #-- return
    return 0
###

###
def draw_figure_df(x_len,y_len,df_xy,x_max,y_max,x_label,y_label,tag,fig_name):
    #-- default settings
    mpl.use('Agg')
    mpl.rcParams['font.sans-serif'] = 'Arial'
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['figure.figsize'] = [3.5,3.5] # 3.5 inches == 89 mm, 4.7 inches == 120 mm, 5.354
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['font.size'] = 10.0
    mpl.rcParams['legend.fontsize'] = 7.0
    mpl.rcParams['figure.titlesize'] = 10.0
    mpl.rcParams['scatter.marker'] = 's'
    mpl.rcParams['axes.linewidth'] = 0.5
    #-- default settings
    #-- figure
    fig = plt.figure(figsize=(x_len,y_len),dpi=300)
    gs = GridSpec(1,1,figure=fig)
    ax_fig = fig.add_subplot(gs[0])
    ax_fig.tick_params(axis='x',which='both',top=True,bottom=True,labelbottom=True,labeltop=False,direction='in',width=0.5,length=1.5)
    ax_fig.tick_params(axis='y',which='both',left=True,right=True,labelleft=True,labelright=False,direction='in',width=0.5,length=1.5)
    for col in df_xy.columns.values:
        if 'g_' in col.replace(tag,''):
            ax_fig.plot(df_xy.index.values,df_xy.loc[:,col].values,linestyle='solid',linewidth=1.0,label=col.replace(tag,''))
        elif 'd_' in col.replace(tag,''):
            ax_fig.plot(df_xy.index.values,df_xy.loc[:,col].values,linestyle='dotted',linewidth=1.0,label=col.replace(tag,''))
        #
    #
    ax_fig.set_xlim([0.0,x_max])
    ax_fig.set_ylim([0.0,y_max])
    ax_fig.set_xlabel(x_label)
    ax_fig.set_ylabel(y_label)
    ax_fig.legend(loc='center left',bbox_to_anchor=(1.04,0.5))
    #-- figure
    #-- draw
    # plt.savefig('figure_PPI.tiff',format='tiff',dpi=300, bbox_inches='tight',pad_inches=save_pad)
    # plt.savefig('figure_PPI.eps',format='eps',dpi=300, bbox_inches='tight',pad_inches=save_pad)
    # plt.savefig('figure_PPI.pdf',format='pdf',dpi=300, bbox_inches='tight',pad_inches=save_pad)
    plt.savefig(fig_name + '.tiff',format='tiff',dpi=300,bbox_inches='tight')
    plt.close()
    # img = PIL.Image.open(fig_name + '.tiff')
    # print('\ntiff size')
    # print(img.size)
    #-- draw
    #-- return
    return 0
###

###
def draw_figures_ODE_solutions(df_p,out_dir):
    #-- read dataframe
    _,_,pv,_,_,_,_,_,_,_ = initialization_simple_immune_simulation(df_p)
    df_sol = pandas.read_csv(out_dir + 'simulation_result.csv')
    #-- read dataframe
    #-- ODE terms, cytokines
    #-- y[0] = SAA, y[1] = IL-1beta/23, y[2] = IL-22
    #-- y[3] = IL-2, y[4] = IL-17, y[5] = IL-10
    #-- y[6] = TH17_1, y[7] = TH17_2, y[5 + N_Ag] = TH17_N_Ag
    #-- y[5 + N_Ag + 1] = Treg_1, y[5 + 2*N_Ag] = Treg_N_Ag
    ODE_keys = ['t','SAA','IL-1beta/23','IL-22','IL-2','IL-17','IL-10']
    for idx in range(0,int(pv['N_Ag'])):
        ODE_keys.append(f'TH17_{idx:d}')
    #
    for idx in range(0,int(pv['N_Ag'])):
        ODE_keys.append(f'Treg_{idx:d}')
    #
    Ag_keys = ['t','beta_eAg','beta_tAg','Ag_load_sum']
    for idx in range(0,int(pv['N_Ag'])):
        Ag_keys.append(f'eAg_{idx:d}')
        Ag_keys.append(f'eAg_del_{idx:d}')
        Ag_keys.append(f'tAg_{idx:d}')
        Ag_keys.append(f'tAg_del_{idx:d}')
    #
    df_d_sol = ODE_simple_immune_model_eval(df_sol.loc[:,ODE_keys].set_index('t'),pv,
                                            df_sol.loc[:,Ag_keys].set_index('t'))
    df_d_sol.to_csv(out_dir + 'dy_dt.csv')
    #-- ODE terms, cytokines
    #-- figure size
    x_len = 3.5
    y_len = 3.5
    x_max = df_sol['t'].max()
    #-- figure size
    #-- cytokines
    c_dict = { 0 : 'SAA', 1 : 'IL-1beta/23', 2 : 'IL-22', 3 : 'IL-2', 4 : 'IL-17', 5 : 'IL-10' }
    f_dict = { 0 : 'SAA', 1 : 'IL1beta23', 2 : 'IL22', 3 : 'IL2', 4 : 'IL17', 5 : 'IL10' }
    for item in c_dict.keys():
        # if item == 0:
        print('\n' + c_dict[item])
        y_max = df_sol[c_dict[item]].max()*1.1
        draw_figure(x_len,y_len,df_sol['t'].values,df_sol[c_dict[item]].values,x_max,y_max,
                    'Time (day)','Concentration (pg/mm^3)',f_dict[item],
                    out_dir + 'time_series_' + f_dict[item])
        col_list = list()
        col_list_x = list()
        for col in df_d_sol.columns.values:
            if col.startswith('d_' + c_dict[item] + '_'):
                if col != 'd_' + c_dict[item]:
                    if 'x_' in col:
                        col_list_x.append(col)
                    else:
                        col_list.append(col)
                    #
                #
            #
        #
        y_max = df_d_sol.loc[:,col_list].max().max()*1.1
        draw_figure_df(x_len,y_len,df_d_sol.loc[:,col_list],x_max,y_max,'Time (day)','Rate (concentration/day)',
                       'd_' + c_dict[item] + '_',out_dir + 'time_series_' + f_dict[item] + '_dydt')
        if len(col_list_x) != 0:
            for col in col_list_x:
                y_max = df_d_sol.loc[:,col].max()*1.1
                tag = col.replace('d_' + c_dict[item] + '_','')
                print(tag)
                draw_figure(x_len,y_len,df_d_sol.index.values,df_d_sol[col].values,
                            x_max,y_max,'Time (day)','Gamma',tag,
                            out_dir + 'time_series_' + f_dict[item] + '_' + tag)
            #
        #
    #
    #-- cytokines
    #-- T cells
    for idx in range(0,int(df_p.loc['N_Ag','value'])):
        # if idx == 0:
        tag = f'TH17_{idx:d}'
        print('\n' + tag)
        y_max = df_sol[tag].max()*1.1
        draw_figure(x_len,y_len,df_sol['t'].values,df_sol[tag].values,x_max,y_max,
                    'Time (day)','Density (1000 cells/mm^3)',tag,
                    out_dir + 'time_series_' + tag)
        col_list = list()
        for col in df_d_sol.columns.values:
            if col.startswith('d_' + tag):
                if col != 'd_' + tag:
                    col_list.append(col)
                #
            #
        #
        y_max = df_d_sol.loc[:,col_list].max().max()*1.1
        draw_figure_df(x_len,y_len,df_d_sol.loc[:,col_list],x_max,y_max,'Time (day)','Rate (density/day)',
                       'd_' + tag + '_',out_dir + 'time_series_' + tag + '_dydt')
        #
    #
    for idx in range(0,int(df_p.loc['N_Ag','value'])):
        # if idx == 0:
        tag = f'Treg_{idx:d}'
        print('\n' + tag)
        y_max = df_sol[tag].max()*1.1
        draw_figure(x_len,y_len,df_sol['t'].values,df_sol[tag].values,x_max,y_max,
                    'Time (day)','Density (1000 cells/mm^3)',tag,
                    out_dir + 'time_series_' + tag)
        col_list = list()
        col_list_x = list()
        for col in df_d_sol.columns.values:
            if col.startswith('d_' + tag):
                if col != 'd_' + tag:
                    if 'x_' in col:
                        col_list_x.append(col)
                    else:
                        col_list.append(col)
                    #
                #
            #
        #
        y_max = df_d_sol.loc[:,col_list].max().max()*1.1
        draw_figure_df(x_len,y_len,df_d_sol.loc[:,col_list],x_max,y_max,'Time (day)','Rate (density/day)',
                       'd_' + tag + '_',out_dir + 'time_series_' + tag + '_dydt')
        #
        if len(col_list_x) != 0:
            for col in col_list_x:
                y_max = df_d_sol.loc[:,col].max()*1.1
                x_tag = col.replace('d_' + tag + '_','')
                print(x_tag)
                draw_figure(x_len,y_len,df_d_sol.index.values,df_d_sol[col].values,
                            x_max,y_max,'Time (day)','Gamma',x_tag,
                            out_dir + 'time_series_' + tag + '_' + x_tag)
            #
        #
    #
    #-- T cells
    #-- Antigen loads
    for idx in range(0,int(df_p.loc['N_Ag','value'])):
        tag = f'Ag_{idx:d}'
        tag_e = f'eAg_{idx:d}'
        tag_t = f'tAg_{idx:d}'
        print('\n' + tag)
        y_max = df_sol[tag].max()*1.1
        draw_figure_Ag(x_len,y_len,df_sol['t'].values,df_sol[tag].values,
                       df_sol[tag_e].values,df_sol[tag_t].values,x_max,y_max,
                       'Time (day)','Antigen loads (pg/mm^3)',f'{idx:d}',
                       out_dir + f'time_series_Ag_load_{idx:d}')
    #
    #-- Antigen loads
    #-- Antigen load gamma
    draw_figure_f_gamma(x_len,y_len,df_sol['t'].values,df_sol['IL-22'].values,df_sol['IL-17'].values,
                        x_max,'Time (day)',r'$\Gamma$' + '*A_Epi',pv,out_dir + 'time_series_f_Ag_gamma')
    #-- Antigen load gamma
    #-- return
    return 0
###

###
def draw_figures_selected_ODE_solutions(df_p,df_sol,out_dir,var_list):
    #-- read dataframe
    _,_,pv,_,_,_,_,_,_,_ = initialization_simple_immune_simulation(df_p)
    #-- read dataframe
    #-- ODE terms, cytokines
    #-- y[0] = SAA, y[1] = IL-1beta/23, y[2] = IL-22
    #-- y[3] = IL-2, y[4] = IL-17, y[5] = IL-10
    #-- y[6] = TH17_1, y[7] = TH17_2, y[5 + N_Ag] = TH17_N_Ag
    #-- y[5 + N_Ag + 1] = Treg_1, y[5 + 2*N_Ag] = Treg_N_Ag
    ODE_keys = ['t','SAA','IL-1beta/23','IL-22','IL-2','IL-17','IL-10']
    for idx in range(0,int(pv['N_Ag'])):
        ODE_keys.append(f'TH17_{idx:d}')
    #
    for idx in range(0,int(pv['N_Ag'])):
        ODE_keys.append(f'Treg_{idx:d}')
    #
    Ag_keys = ['t','beta_eAg','beta_tAg','Ag_load_sum']
    for idx in range(0,int(pv['N_Ag'])):
        Ag_keys.append(f'eAg_{idx:d}')
        Ag_keys.append(f'eAg_del_{idx:d}')
        Ag_keys.append(f'tAg_{idx:d}')
        Ag_keys.append(f'tAg_del_{idx:d}')
    #
    df_d_sol = ODE_simple_immune_model_eval(df_sol.loc[:,ODE_keys].set_index('t'),pv,
                                            df_sol.loc[:,Ag_keys].set_index('t'))
    df_d_sol.to_csv(out_dir + 'dy_dt.csv')
    #-- ODE terms, cytokines
    #-- figure size
    x_len = 3.5
    y_len = 3.5
    x_max = df_sol['t'].max()
    #-- figure size
    #-- cytokines
    c_dict = { 0 : 'SAA', 1 : 'IL-1beta/23', 2 : 'IL-22', 3 : 'IL-2', 4 : 'IL-17', 5 : 'IL-10' }
    f_dict = { 0 : 'SAA', 1 : 'IL1beta23', 2 : 'IL22', 3 : 'IL2', 4 : 'IL17', 5 : 'IL10' }
    for item in c_dict.keys():
        if c_dict[item] in var_list:
            print('\n' + c_dict[item])
            y_max = df_sol[c_dict[item]].max()*1.1
            draw_figure(x_len,y_len,df_sol['t'].values,df_sol[c_dict[item]].values,x_max,y_max,
                        'Time (day)','Concentration (pg/mm^3)',f_dict[item],
                        out_dir + 'time_series_' + f_dict[item])
            col_list = list()
            col_list_x = list()
            for col in df_d_sol.columns.values:
                if col.startswith('d_' + c_dict[item] + '_'):
                    if col != 'd_' + c_dict[item]:
                        if 'x_' in col:
                            col_list_x.append(col)
                        else:
                            col_list.append(col)
                        #
                    #
                #
            #
            y_max = df_d_sol.loc[:,col_list].max().max()*1.1
            draw_figure_df(x_len,y_len,df_d_sol.loc[:,col_list],x_max,y_max,'Time (day)','Rate (concentration/day)',
                        'd_' + c_dict[item] + '_',out_dir + 'time_series_' + f_dict[item] + '_dydt')
            if len(col_list_x) != 0:
                for col in col_list_x:
                    y_max = df_d_sol.loc[:,col].max()*1.1
                    tag = col.replace('d_' + c_dict[item] + '_','')
                    print(tag)
                    draw_figure(x_len,y_len,df_d_sol.index.values,df_d_sol[col].values,
                                x_max,y_max,'Time (day)','Gamma',tag,
                                out_dir + 'time_series_' + f_dict[item] + '_' + tag)
                #
            #
        #
    #
    #-- cytokines
    #-- T cells
    for idx in range(0,int(df_p.loc['N_Ag','value'])):
        tag = f'TH17_{idx:d}'
        if tag in var_list:
            print('\n' + tag)
            y_max = df_sol[tag].max()*1.1
            draw_figure(x_len,y_len,df_sol['t'].values,df_sol[tag].values,x_max,y_max,
                        'Time (day)','Density (1000 cells/mm^3)',tag,
                        out_dir + 'time_series_' + tag)
            col_list = list()
            for col in df_d_sol.columns.values:
                if col.startswith('d_' + tag):
                    if col != 'd_' + tag:
                        col_list.append(col)
                    #
                #
            #
            y_max = df_d_sol.loc[:,col_list].max().max()*1.1
            draw_figure_df(x_len,y_len,df_d_sol.loc[:,col_list],x_max,y_max,'Time (day)','Rate (density/day)',
                        'd_' + tag + '_',out_dir + 'time_series_' + tag + '_dydt')
            #
        #
    #
    for idx in range(0,int(df_p.loc['N_Ag','value'])):
        tag = f'Treg_{idx:d}'
        if tag in var_list:
            print('\n' + tag)
            y_max = df_sol[tag].max()*1.1
            draw_figure(x_len,y_len,df_sol['t'].values,df_sol[tag].values,x_max,y_max,
                        'Time (day)','Density (1000 cells/mm^3)',tag,
                        out_dir + 'time_series_' + tag)
            col_list = list()
            col_list_x = list()
            for col in df_d_sol.columns.values:
                if col.startswith('d_' + tag):
                    if col != 'd_' + tag:
                        if 'x_' in col:
                            col_list_x.append(col)
                        else:
                            col_list.append(col)
                        #
                    #
                #
            #
            y_max = df_d_sol.loc[:,col_list].max().max()*1.1
            draw_figure_df(x_len,y_len,df_d_sol.loc[:,col_list],x_max,y_max,'Time (day)','Rate (density/day)',
                        'd_' + tag + '_',out_dir + 'time_series_' + tag + '_dydt')
            #
            if len(col_list_x) != 0:
                for col in col_list_x:
                    y_max = df_d_sol.loc[:,col].max()*1.1
                    x_tag = col.replace('d_' + tag + '_','')
                    print(x_tag)
                    draw_figure(x_len,y_len,df_d_sol.index.values,df_d_sol[col].values,
                                x_max,y_max,'Time (day)','Gamma',x_tag,
                                out_dir + 'time_series_' + tag + '_' + x_tag)
                #
            #
        #
    #
    #-- T cells
    #-- Antigen loads
    for idx in range(0,int(df_p.loc['N_Ag','value'])):
        tag = f'Ag_{idx:d}'
        if tag in var_list:
            tag_e = f'eAg_{idx:d}'
            tag_t = f'tAg_{idx:d}'
            print('\n' + tag)
            y_max = df_sol[tag].max()*1.1
            draw_figure_Ag(x_len,y_len,df_sol['t'].values,df_sol[tag].values,
                        df_sol[tag_e].values,df_sol[tag_t].values,x_max,y_max,
                        'Time (day)','Antigen loads (pg/mm^3)',f'{idx:d}',
                        out_dir + f'time_series_Ag_load_{idx:d}')
        #
    #
    #-- Antigen loads
    #-- Antigen load gamma
    draw_figure_f_gamma(x_len,y_len,df_sol['t'].values,df_sol['IL-22'].values,df_sol['IL-17'].values,
                        x_max,'Time (day)',r'$\Gamma$' + '*A_Epi',pv,out_dir + 'time_series_f_Ag_gamma')
    #-- Antigen load gamma
    #-- return
    return 0
###

###
def make_antigen_specific_tables(src_dir,dst_dir,table_dict):
    #-- make dir for antigen specific tables
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    else:
        shutil.rmtree(dst_dir,ignore_errors=True)
        os.makedirs(dst_dir)
    #
    #-- make dir for antigen specific tables
    #-- make antigen specific tables
    for item in os.listdir(src_dir):
        if item.replace('.csv','') in table_dict.keys():
            df_tmp = pandas.read_csv(src_dir + item)
            df_tmp.loc[0,'value'] = table_dict[item.replace('.csv','')]
            df_tmp.to_csv(dst_dir + item,index=False)
        else:
            src = src_dir + item
            dst = dst_dir + item
            shutil.copy(src,dst)
        #
    #
    #-- make antigen specific tables
    #-- return
    return 0
###

###
def main_Treg_proliferation_rate_test():
    #-- Ag and Treg_lambda coeffs
    Ag_eval = ['1.00']
    Treg_eval = ['1.']
    #-- Ag and Treg_lambda coeffs
    #-- save default Treg_lambda
    df_file_name = 'parameter_table.xlsx'
    df = pandas.read_excel(df_file_name).set_index('parameter')
    Treg_lambda_default = df.loc['lambda_Treg','value']
    #-- save default Treg_lambda
    #-- read default Ag_load
    df_file_name = 'antigen_specific_p_default/Ag_load_0.csv'
    df = pandas.read_csv(df_file_name).set_index('antigen')
    Ag_default = float(df.loc[0,'value'])
    #-- read default Ag_load
    #-- default dir for antigen specific tables
    Ag_specific_table_default = 'antigen_specific_p_default/'
    #-- default dir for antigen specific tables
    #-- dir of antigen specific tables for test
    Ag_specific_table_test = 'antigen_specific_p/'
    #-- dir of antigen specific tables for test
    #-- test simulations
    for alpha in ['0.6']:
        #-- Treg proliferation rate test dir
        out_dir_r = 'Treg_p_rate_test_a' + alpha.replace('.','') + '/'
        if not os.path.exists(out_dir_r):
            os.mkdir(out_dir_r)
        else:
            shutil.rmtree(out_dir_r,ignore_errors=True)
            os.makedirs(out_dir_r)
        #
        out_dir_sol = 'Treg_p_rate_test_a' + alpha.replace('.','') + '/' + 'sols/'
        if not os.path.exists(out_dir_sol):
            os.mkdir(out_dir_sol)
        else:
            shutil.rmtree(out_dir_sol,ignore_errors=True)
            os.makedirs(out_dir_sol)
        #
        out_dir_fig = 'Treg_p_rate_test_a' + alpha.replace('.','') + '/' + 'figs/'
        if not os.path.exists(out_dir_fig):
            os.mkdir(out_dir_fig)
        else:
            shutil.rmtree(out_dir_fig,ignore_errors=True)
            os.makedirs(out_dir_fig)
        #
        #-- Treg proliferation rate test dir
        for idx_Treg in range(0,len(Treg_eval)):
            for idx_Ag in range(0,len(Ag_eval)):
                #-- change antigen specific parameters
                t_dict = { 'alpha' : float(alpha), 'Ag_load_0' : float(Ag_eval[idx_Ag])*Ag_default }
                make_antigen_specific_tables(Ag_specific_table_default,Ag_specific_table_test,t_dict)
                #-- change antigen specific parameters
                #-- change Treg_lambda value
                new_Treg = float(Treg_eval[idx_Treg])*Treg_lambda_default
                df_file_name = 'parameter_table.xlsx'
                df = pandas.read_excel(df_file_name).set_index('parameter')
                df.loc['lambda_Treg','value'] = new_Treg
                #-- change Treg_lambda value
                #-- solve ODEs
                solve_ODEs(out_dir_r,df)
                src = out_dir_r + 'simulation_result.csv'
                dst = out_dir_sol + 'simulation_result_' + 'Treg_' + Treg_eval[idx_Treg].replace('.','') + '_Ag_' + Ag_eval[idx_Ag].replace('.','p') + '.csv'
                shutil.move(src,dst)
                df_sol = pandas.read_csv(dst)
                tmp_out_name = out_dir_fig + 'Treg_' + Treg_eval[idx_Treg].replace('.','') + '_Ag_' + Ag_eval[idx_Ag].replace('.','p') + '_'
                draw_figures_selected_ODE_solutions(df,df_sol,tmp_out_name,['SAA','IL-1beta/23','IL-22','IL-2','IL-17','IL-10','TH17_0','Treg_0','Ag_0'])
                #-- solve ODEs
                #-- read solution values
                print(f'\nalpha = {float(alpha):.2f}')
                print('Treg_coeff = ' + Treg_eval[idx_Treg] + ', ' + 'Ag_coeff = ' + Ag_eval[idx_Ag])
                print(f'Treg_default = {Treg_lambda_default:.2f}, Treg_new = {new_Treg:.2f}')
                print(f'Ag_default = {Ag_default:.2f}, New_Ag = {float(Ag_eval[idx_Ag])*Ag_default:.2f}')
                #-- read solution values
                #-- delete antigen specific tables for the test
                shutil.rmtree(Ag_specific_table_test,ignore_errors=True)
                #-- delete antigen specific tables for the test
            #
        #
    #
    #-- test simulations
    #-- return
    return 0
###

###
def make_Treg_proliferation_rate_table(out_dir,Ag_th,c_tag,s_tag):
    #-- Ag and Treg_lambda coeffs
    Ag_eval = ['1.00']
    Treg_eval = ['1.']
    #-- Ag and Treg_lambda coeffs
    #-- data frame
    df = pandas.DataFrame({})
    df['Ag_load/Treg_lambda'] = Ag_eval
    df = df.set_index('Ag_load/Treg_lambda')
    for idx_Treg in range(0,len(Treg_eval)):
        df[Treg_eval[idx_Treg]] = '1000.0'
    #
    #-- data frame
    #-- read result
    for idx_Treg in range(0,len(Treg_eval)):
        for idx_Ag in range(0,len(Ag_eval)):
            tmp_name = out_dir + 'sols/simulation_result_' + 'Treg_' + Treg_eval[idx_Treg].replace('.','') + '_Ag_' + Ag_eval[idx_Ag].replace('.','p') + '.csv'
            df_tmp = pandas.read_csv(tmp_name)
            if c_tag.startswith('Ag_'):
                tmp_idx = df_tmp[ df_tmp[c_tag] < Ag_th ].index.values
                # if float(Ag_eval[idx_Ag]) > 0.2:
                #     Ag_Fin.append(numpy.min(df_tmp[c_tag].values))
                #     Ag_Sat.append(numpy.min(df_tmp[ df_tmp[c_tag] < 1.05*numpy.min(df_tmp[c_tag].values)].index.values))
                if len(tmp_idx) != 0:
                    df.loc[Ag_eval[idx_Ag],Treg_eval[idx_Treg]] = str(df_tmp.loc[tmp_idx,'t'].min())
                else:
                    df.loc[Ag_eval[idx_Ag],Treg_eval[idx_Treg]] = '1000.0'
                #
            elif '_max' in c_tag:
                df.loc[Ag_eval[idx_Ag],Treg_eval[idx_Treg]] = str(df_tmp[c_tag.replace('_max','')].max())
            elif '_avg' in c_tag:
                t_eval = df_tmp['t'].values
                f_eval = df_tmp[c_tag.replace('_avg','')].values
                df.loc[Ag_eval[idx_Ag],Treg_eval[idx_Treg]] = numpy.trapz(f_eval,t_eval)/numpy.amax(t_eval)
            #
        #
    #
    #-- read result
    #-- save result
    df.to_csv(out_dir + s_tag + c_tag + '.csv')
    #-- save result
    #-- return
    return 0
###

###
def main_Treg_proliferation_rate_table(out_dir,s_tag):
    tag_list = ['Ag_0','IL-10_max','IL-10_avg','IL-22_max','IL-22_avg','IL-17_max','IL-17_avg']
    for tag in tag_list:
        make_Treg_proliferation_rate_table(out_dir,200.0,tag,s_tag)
    #
    #-- return
    return 0
###

####-- END Functions

####-- Main

# main_test()

main_Treg_proliferation_rate_test()
main_Treg_proliferation_rate_table('Results/','a06_')


###END
