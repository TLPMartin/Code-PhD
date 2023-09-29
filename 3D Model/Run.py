# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 15:19:24 2021

@author: User
"""

import math as math
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy import signal
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas 
import collections
from random import choice
from mpl_toolkits.mplot3d import Axes3D
import time
import multiprocessing 
import concurrent.futures

from Function import *
from SetUp import *

###--- modified by RLim ---###
import os
import copy
import sys
import json
import shutil
import matplotlib as mpl
mpl.use('Agg')

def error_exit():
    checker = input('Enter "x" to finish this program: ')
    if checker == 'x':
        sys.exit()
    #
    #-- return
    return 0

###--- modified by RLim ---###

#Geographical global settings#########################################################

J = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]




###############################################################################
#Time Parameters
###############################################################################

dt = 0.002
T = 14

Int_Sav = 100

SubT = 10
L_T = [0]

###############################################################################
#Data values 
###############################################################################
size_cube = 0.3
Villus_Cells = 8
Crypts_Cells = 0.6*0.09

TcellsLP = 40
Macro = 4
DC = 0.4

IL10MLN_Obj = 0.1

IL10LP_Obj = 1
IL17LP_Obj = 2.5
IL1Beta23_Obj = 0.025

Number_Tcells_MLN = 4500


Model = {}

def Main(dt, T, SubT, N_chemical, Folder):
    start_time = time.time()
    
    #Model = {}
    L_T = [0]
    print('Hello word')

    df = pd.read_csv('Parameter.csv', delimiter=',')
    Topological_Parameter(Model)

    Biological_Parameter(Model, df, N_chemical)

    ###--- modified by RLim ---###
    Model['A_tAg'] = np.array([[[[0]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2)]*(Model['N_chemical']+1))
    Model['A_eAg'] = np.array([[[[0]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2)]*(Model['N_chemical']+1))
    for i in range(Model['N_chemical'] + 1):
        for (x, y, z) in Model['Volume_Lamina_Propria_Epi_Complementary']:
            # if Model['a'][i] < 0.5:
            #     Model['A_tAg'][i][x, y, z] = Model['Ag_load'][i] / ((Model['N_chemical'] + 990))
            # #
            # if Model['a'][i] >= 0.5:
            #     Model['A_eAg'][i][x, y, z] = Model['Ag_load'][i] / ((Model['N_chemical'] + 990))
            # #
            if Model['a'][i] < 0.5:
                Model['A_tAg'][i][x, y, z] = 0.0
            #
            if Model['a'][i] >= 0.5:
                Model['A_eAg'][i][x, y, z] = 0.0
            #
        #
        for index in range(len(Model['Volume_Crypt'])):
            (x, y, z) = Model['Volume_Crypt'][index]
            if Model['a'][i] < 0.5:
                Model['A_tAg'][i][x, y, z+1] = Model['Ag_load'][i] / ((Model['N_chemical'] + 1))
            #
            if Model['a'][i] >= 0.5:
                Model['A_eAg'][i][x, y, z+1] = Model['Ag_load'][i] / ((Model['N_chemical'] + 1))
            #
        #
    #
    tmp_eAg = list(np.zeros((N_chemical + 1,)))
    tmp_tAg = list(np.zeros((N_chemical + 1,)))
    for i in range(Model['N_chemical'] + 1):
        for index in range(len(Model['Volume_Crypt'])):
            (x, y, z) = Model['Volume_Crypt'][index]
            tmp_eAg[i] += Model['A_eAg'][0][x,y,z+1]
            tmp_tAg[i] += Model['A_tAg'][0][x,y,z+1]
        #
    #
    Ag_threshold = list(np.zeros((N_chemical + 1,)))
    T_threshold = list(np.zeros((N_chemical + 1,)))
    T_threshold_check = list(np.zeros((N_chemical + 1,)))
    for i in range(Model['N_chemical'] + 1):
        Ag_threshold[i] = tmp_eAg[i]*size_cube**3 + tmp_tAg[i]*size_cube**3
        T_threshold[i] = '{:.4f}'.format(14.995)
        T_threshold_check[i] = False
    #
    tmp_list = copy.deepcopy(Model['Ag_load'])
    tmp_list = list(np.array(tmp_list)*Ag_coeff)
    Model['Ag_load'] = tmp_list
    ###--- modified by RLim ---###

    Diffusion_Maps(Model)

    Overpopulation_Test(Model, dt)

    VS_TH17_LP, VS_Treg_LP, L_out_TH17, L_out_Treg, VS_eDC_MLN, VS_tDC_MLN, L_out_eDC, L_out_tDC = Initial_Population(Model, dt)
    
    ###############################################################################
    #Full Lists of Abundances and Concentrations
    ###############################################################################
    
    
    #Full Time Lists of Agents in the Lamina Propria
    
    
    L_A_Mon_LP = [np.sum(Model['A_Mon_LP'])]
    L_A_Mac_LP = [np.sum(Model['A_Mac_LP'])]
    L_A_ImDC_LP = [np.sum(Model['A_ImDC_LP'])]
    L_A_eDC_LP = [np.sum(Model['A_eDC_LP'])]
    L_A_tDC_LP = [np.sum(Model['A_tDC_LP'])]
    L_A_TH17_LP = [np.copy(Model['A_TH17_LP'])]
    L_A_Treg_LP = [np.copy(Model['A_Treg_LP'])]
    L_A_sTH17_LP = [np.copy(Model['A_sTH17_LP'])]
    L_A_sTreg_LP = [np.copy(Model['A_sTreg_LP'])]
    L_A_tTreg_LP = [np.sum(Model['A_tTreg_LP'])]
    L_A_eAg_Mac_LP = [np.sum(Model['A_eAg_Mac_LP'])]
    L_A_tAg_Mac_LP = [np.sum(Model['A_tAg_Mac_LP'])]
    L_A_eAg = [np.copy(Model['A_eAg'])]
    L_A_tAg = [np.copy(Model['A_tAg'])]
    L_S_TH17_LP = [np.sum(Model['S_TH17_LP_ini'][-1])]
    L_S_Treg_LP = [np.sum(Model['S_Treg_LP_ini'][-1])]
    
    Model['L_out_TH17'] = []
    
    Model['L_out_Treg'] = []
    
    
    #Full Time Lists of Agents in the Epithelium
    
    
    L_A_sEpi = [np.copy(Model['A_sEpi'])]
    L_A_Epi = [np.copy(Model['A_Epi'])]
    L_Crypt = [np.copy(C(Model))]
    L_A_eAg_Epi = [np.sum(Model['A_eAg_Epi'])]
    L_A_tAg_Epi = [np.sum(Model['A_tAg_Epi'])]
    L_A_eAg_sEpi = [np.sum(Model['A_eAg_sEpi'])]
    L_A_tAg_sEpi = [np.sum(Model['A_tAg_sEpi'])]
    L_Storage_IL = [np.copy(Model['Storage_IL'])]
    
    
    #Full Time Lists of Agents in the Mesenteric Lymph Node
    
    
    L_A_eDC_MLN = [np.sum(Model['A_eDC_MLN'])]
    
    L_A_tDC_MLN = [np.sum(Model['A_tDC_MLN'])]
    
    L_A_nTc_MLN = [np.sum(Model['A_nTc_MLN'])]
    
    L_A_nTe_MLN = [np.sum(Model['A_nTe_MLN'])]
    
    L_A_nTt_MLN = [np.sum(Model['A_nTt_MLN'])]
    
    L_A_TH17_MLN = [np.sum(Model['A_TH17_MLN'], axis=(1, 2, 3))]
    
    L_A_Treg_MLN = [np.sum(Model['A_Treg_MLN'], axis=(1, 2, 3))]
    
    L_A_fTH17_MLN = [np.sum(Model['A_TH17_MLN'], axis=(1, 2, 3))]
    
    L_A_fTreg_MLN = [np.sum(Model['A_Treg_MLN'], axis=(1, 2, 3))]
    
    L_A_mTH17_MLN = [np.sum(Model['A_mTH17_MLN'], axis=(1, 2, 3))]
    
    L_A_mTreg_MLN = [np.sum(Model['A_mTreg_MLN'], axis=(1, 2, 3))]
    
    L_S_eDC_MLN = [np.sum(Model['S_eDC_MLN_ini'][-1])]
    
    L_S_tDC_MLN = [np.sum(Model['S_tDC_MLN_ini'][-1])]
    
    
    #Full Time Lists of Soluble Mediators in the Lamina Propria
    
    
    L_C_IL10_LP = [np.sum(Model['C_IL10_LP'])]
    
    L_C_TGFBeta_LP = [np.copy(Model['C_TGFBeta_LP'])]
    
    L_C_IL22_LP = [np.sum(Model['C_IL22_LP'])]
    
    L_C_IL1Beta_23_LP = [np.sum(Model['C_IL1Beta_23_LP'])]
    
    L_C_IL2_LP = [np.sum(Model['C_IL2_LP'])]
    
    L_C_IL17_LP = [np.copy(Model['C_IL17_LP'])]
    
    L_C_Csf2_LP = [np.sum(Model['C_Csf2_LP'])]
    
    L_C_CCL2_LP = [np.sum(Model['C_CCL2_LP'])]
    
    L_C_CCL25_LP = [np.sum(Model['C_CCL25_LP'])]
    
    L_C_SAA_LP = [np.copy(Model['C_SAA_LP'])]
    
    L_C_eRA_LP = [np.sum(Model['C_eRA_LP'])]
    
    L_C_tRA_LP = [np.sum(Model['C_tRA_LP'])]
    
    
    #Full Time Lists of Soluble Mediators in the Mesenteric Lymph Node
    
    
    L_C_IL21_MLN = [np.sum(Model['C_IL21_MLN'])]
    
    L_C_IL2_MLN = [np.sum(Model['C_IL2_MLN'])]
    
    L_C_IL6_MLN = [np.sum(Model['C_IL6_MLN'])]
    
    L_C_IL10_MLN = [np.sum(Model['C_IL10_MLN'])]
    
    L_C_TGFBeta_MLN = [np.sum(Model['C_TGFBeta_MLN'])]
    
    L_C_eRA_MLN = [np.sum(Model['C_eRA_MLN'])]
    
    L_C_tRA_MLN = [np.sum(Model['C_tRA_MLN'])]
    

    
    
    #Overpopulation_Test(Model, dt)

###############################################################################
    #Simulation
###############################################################################   
    Model['Impact_Paneth'] = [0] * len(Model['Volume_Crypt'])

    for t in range(1, int(T/dt)):
        # print(t)
        # if t % 1000 == 1:
        #     print(t)

        
        Cor = (math.exp(-t*dt/Model['tau_Mic']) - math.exp(dt/Model['tau_Mic'])) / (1 - math.exp(dt/Model['tau_Mic']))
        if Cor == 0:
            Cor = 1




        Model['A_tAg'] = np.array([[[[0]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2)]*(Model['N_chemical']+1))
        Model['A_eAg'] = np.array([[[[0]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2)]*(Model['N_chemical']+1))




        for i in range(Model['N_chemical'] + 1):
            for (x, y, z) in Model['Volume_Lamina_Propria_Epi_Complementary']:
                # if Model['a'][i] < 0.5:
                #     Model['A_tAg'][i][x, y, z] = Model['Ag_load'][i] / ((Model['N_chemical'] + 990))
                # if Model['a'][i] >= 0.5:
                #     Model['A_eAg'][i][x, y, z] = Model['Ag_load'][i] / ((Model['N_chemical'] + 990))
                if Model['a'][i] < 0.5:
                    Model['A_tAg'][i][x, y, z] = 0.0
                if Model['a'][i] >= 0.5:
                    Model['A_eAg'][i][x, y, z] = 0.0

            for index in range(len(Model['Volume_Crypt'])):
                (x, y, z) = Model['Volume_Crypt'][index]
                Model['Impact_Paneth'][index] = (math.exp(-dt) * Model['Impact_Paneth'][index] + dt * Model['Storage_IL'][index] * math.exp(-(t*dt + dt - (t + 1)*dt)) )/Cor
                # if Model['a'][i] < 0.5:
                #     Model['A_tAg'][i][x, y, z+1] = (Model['Ag_load'][i] / ((Model['N_chemical'] + 990))) / (1 + (Model['k_IL17_22_Mic'] * Model['Impact_Paneth'][index] / Model['tau_Mic'])**Model['h'])
                # if Model['a'][i] >= 0.5:
                #     Model['A_eAg'][i][x, y, z+1] = (Model['Ag_load'][i] / ((Model['N_chemical'] + 990))) / (1 + (Model['k_IL17_22_Mic'] * Model['Impact_Paneth'][index] / Model['tau_Mic'])**Model['h'])
                if Model['a'][i] < 0.5:
                    Model['A_tAg'][i][x, y, z+1] = (Model['Ag_load'][i] / ((Model['N_chemical'] + 1))) / (1 + (Model['k_IL17_22_Mic'] * nu_Ag * Model['Impact_Paneth'][index] / Model['tau_Mic'])**Model['h'])
                if Model['a'][i] >= 0.5:
                    Model['A_eAg'][i][x, y, z+1] = (Model['Ag_load'][i] / ((Model['N_chemical'] + 1))) / (1 + (Model['k_IL17_22_Mic'] * nu_Ag * Model['Impact_Paneth'][index] / Model['tau_Mic'])**Model['h'])





        VA_tAg = np.copy(Model['A_tAg'])
        VA_eAg = np.copy(Model['A_eAg'])

        ###--- modified by RLim ---###
        print('{:.4f}'.format(t*dt))
        tmp_eAg = list(np.zeros((N_chemical + 1,)))
        tmp_tAg = list(np.zeros((N_chemical + 1,)))
        for i in range(Model['N_chemical'] + 1):
            for index in range(len(Model['Volume_Crypt'])):
                (x, y, z) = Model['Volume_Crypt'][index]
                tmp_eAg[i] += Model['A_eAg'][0][x,y,z+1]
                tmp_tAg[i] += Model['A_tAg'][0][x,y,z+1]
            #
        #
        for i in range(Model['N_chemical'] + 1):
            print('eAg={0:.4e}, tAg={1:.4e}'.format(tmp_eAg[i]*size_cube**3,tmp_tAg[i]*size_cube**3))
            tmp_threshold = tmp_eAg[i]*size_cube**3 + tmp_tAg[i]*size_cube**3
            if T_threshold_check[i] == False:
                if tmp_threshold < Ag_threshold[i]:
                    T_threshold[i] = '{:.4f}'.format(float(t)*float(dt))
                    T_threshold_check[i] = True
                #
            #
        #
        # error_exit()
        ###--- modified by RLim ---###



        ###############################################################################
        #Agents Time Dynamics
        ###############################################################################
        with concurrent.futures.ThreadPoolExecutor() as executor:
            
        
            x_process = executor.submit(Update_LP, Model, dt, t, L_out_TH17, L_out_Treg, VS_TH17_LP, VS_Treg_LP)
            y_process = executor.submit(Update_MLN, Model, dt, t, L_out_eDC, L_out_tDC, VS_eDC_MLN, VS_tDC_MLN)
            z_process = executor.submit(Update_LP_Cyt, Model, dt, t)
            u_process = executor.submit(Update_MLN_Cyt, Model, dt, t)
            v_process = executor.submit(Update_Epi, Model, dt, t)
            
            
            L_out_TH17, L_out_Treg, VA_Mon_LP, VA_Mac_LP, VA_ImDC_LP, VA_TH17_LP, VA_sTH17_LP, VA_Treg_LP, VA_sTreg_LP, VA_eDC_LP, VA_tDC_LP, VC_eRA_LP, VC_tRA_LP, VS_TH17_LP, VS_Treg_LP, VA_eAg_Mac_LP, VA_tAg_Mac_LP = x_process.result()
            
            
            L_out_eDC, L_out_tDC, VS_eDC_MLN, VS_tDC_MLN, VA_eDC_MLN, VA_tDC_MLN, VA_nTc_MLN, VA_nTe_MLN, VA_nTt_MLN, VA_TH17_MLN, VA_Treg_MLN, VA_mTH17_MLN, VA_mTreg_MLN, VA_fTH17_MLN, VA_fTreg_MLN, VC_eRA_MLN, VC_tRA_MLN = y_process.result()
            
            
            VC_IL10_LP, VC_TGFBeta_LP, VC_IL17_LP, VC_IL22_LP, VC_IL2_LP, VC_CCL2_LP, VC_CCL25_LP, VC_IL1Beta_23_LP, VC_Csf2_LP, VC_SAA_LP = z_process.result()
            VC_IL21_MLN, VC_IL2_MLN, VC_IL6_MLN, VC_TGFBeta_MLN, VC_IL10_MLN = u_process.result()
            VA_Epi, VA_sEpi, VA_eAg_Epi, VA_tAg_Epi, VA_eAg_sEpi, VA_tAg_sEpi = v_process.result()
        
        ###############################################################################
        #Update Populations Lists Time Dynamics
        ###############################################################################
        
        Reverse_VA_A(Model, dt, t, VS_TH17_LP, VS_Treg_LP, VS_eDC_MLN, VS_tDC_MLN, VA_Mac_LP, VA_eAg_Mac_LP, VA_tAg_Mac_LP, VA_Mon_LP, VA_ImDC_LP, VA_eDC_LP, VA_tDC_LP, VA_TH17_LP, VA_Treg_LP, VA_sTH17_LP, VA_sTreg_LP, VC_eRA_LP, VC_tRA_LP, VA_eAg, VA_tAg, VA_eDC_MLN, VA_tDC_MLN, VC_eRA_MLN, VC_tRA_MLN, VA_nTc_MLN, VA_nTe_MLN, VA_nTt_MLN, VA_fTH17_MLN, VA_fTreg_MLN, VA_TH17_MLN, VA_Treg_MLN, VA_mTH17_MLN, VA_mTreg_MLN, VC_IL10_LP, VC_TGFBeta_LP, VC_IL17_LP, VC_IL22_LP, VC_IL2_LP, VC_CCL2_LP, VC_CCL25_LP, VC_IL1Beta_23_LP, VC_Csf2_LP, VC_IL21_MLN, VC_IL2_MLN, VC_IL6_MLN, VC_TGFBeta_MLN, VC_IL10_MLN, VA_sEpi, VA_Epi, VA_eAg_Epi, VA_tAg_Epi, VA_eAg_sEpi, VA_tAg_sEpi, VC_SAA_LP)
        
        
        Model['Storage_IL'] = []
        for (x, y, z) in Model['Volume_Crypt']:
            Model['Storage_IL'].append(Model['Paneth_Portion'] * C(Model)[x, y, z] * (Gamma(Model['Gamma_IL22_Epi'], Model['K_IL22_Epi'], Model['m'], Model['C_IL22_LP'][x, y, z]) * Model['A_Epi'][x, y, z] + Gamma(Model['Gamma_IL17_Epi'], Model['K_IL17_Epi'], Model['m'], Model['C_IL17_LP'][x, y, z]) * Model['A_Epi'][x, y, z]))

        if t % Int_Sav == 1:
            
            L_T.append(t*dt)
        
            L_A_Mac_LP.append(np.sum(Model['A_Mac_LP']))
            L_A_Mon_LP.append(np.sum(Model['A_Mon_LP']))
            L_A_ImDC_LP.append(np.sum(Model['A_ImDC_LP']))
            L_A_eDC_LP.append(np.sum(Model['A_eDC_LP']))
            L_A_tDC_LP.append(np.sum(Model['A_tDC_LP']))
            L_C_eRA_LP.append(np.sum(Model['C_eRA_LP']))
            L_C_tRA_LP.append(np.sum(Model['C_tRA_LP']))
            L_A_eAg_Mac_LP.append(np.sum(Model['A_eAg_Mac_LP']))
            L_A_tAg_Mac_LP.append(np.sum(Model['A_tAg_Mac_LP']))
            L_A_TH17_LP.append(np.copy(Model['A_TH17_LP']))
            L_A_Treg_LP.append(np.copy(Model['A_Treg_LP']))
            L_A_sTH17_LP.append(np.copy(Model['A_sTH17_LP']))
            L_A_sTreg_LP.append(np.copy(Model['A_sTreg_LP']))
            
            L_A_tTreg_LP.append(np.sum(Model['A_tTreg_LP']))
        
            L_A_eAg.append(np.copy(Model['A_eAg']))
            L_A_tAg.append(np.copy(Model['A_tAg']))
            
            L_A_eDC_MLN.append(np.sum(Model['A_eDC_MLN']))
            L_A_tDC_MLN.append(np.sum(Model['A_tDC_MLN']))
            L_C_eRA_MLN.append(np.sum(Model['C_eRA_MLN']))
            L_C_tRA_MLN.append(np.sum(Model['C_tRA_MLN']))
            L_A_nTe_MLN.append(np.sum(Model['A_nTe_MLN']))
            L_A_nTt_MLN.append(np.sum(Model['A_nTt_MLN']))
            L_A_nTc_MLN.append(np.sum(Model['A_nTc_MLN']))
            L_A_fTH17_MLN.append(np.sum(Model['A_fTH17_MLN'], axis=(1, 2, 3)))
            L_A_fTreg_MLN.append(np.sum(Model['A_fTreg_MLN'], axis=(1, 2, 3)))
            L_A_TH17_MLN.append(np.sum(Model['A_TH17_MLN'], axis=(1, 2, 3)))
            L_A_Treg_MLN.append(np.sum(Model['A_Treg_MLN'], axis=(1, 2, 3)))
            L_A_mTH17_MLN.append(np.sum(Model['A_mTH17_MLN'], axis=(1, 2, 3)))
            L_A_mTreg_MLN.append(np.sum(Model['A_mTreg_MLN'], axis=(1, 2, 3)))
            
            L_C_IL10_LP.append(np.sum(Model['C_IL10_LP']))
            L_C_TGFBeta_LP.append(np.copy(Model['C_TGFBeta_LP']))
            L_C_IL17_LP.append(np.copy(Model['C_IL17_LP']))
            L_C_IL22_LP.append(np.sum(Model['C_IL22_LP']))
            L_C_IL2_LP.append(np.sum(Model['C_IL2_LP']))
            L_C_CCL2_LP.append(np.sum(Model['C_CCL2_LP']))
            L_C_CCL25_LP.append(np.sum(Model['C_CCL25_LP']))
            L_C_IL1Beta_23_LP.append(np.sum(Model['C_IL1Beta_23_LP']))
            L_C_Csf2_LP.append(np.sum(Model['C_Csf2_LP']))
            L_C_SAA_LP.append(np.copy(Model['C_SAA_LP']))
            
            L_C_IL21_MLN.append(np.sum(Model['C_IL21_MLN']))
            L_C_IL2_MLN.append(np.sum(Model['C_IL2_MLN']))
            L_C_IL6_MLN.append(np.sum(Model['C_IL6_MLN']))
            L_C_TGFBeta_MLN.append(np.sum(Model['C_TGFBeta_MLN']))
            L_C_IL10_MLN.append(np.sum(Model['C_IL10_MLN']))
            
            L_A_Epi.append(np.copy(Model['A_Epi']))
            L_A_sEpi.append(np.copy(Model['A_sEpi']))
            L_Crypt.append(np.copy(C(Model)))
            L_A_eAg_Epi.append(np.sum(Model['A_eAg_Epi']))
            L_A_tAg_Epi.append(np.sum(Model['A_tAg_Epi']))
            L_A_eAg_sEpi.append(np.sum(Model['A_eAg_sEpi']))
            L_A_tAg_sEpi.append(np.sum(Model['A_tAg_sEpi']))
    
            
            L_S_TH17_LP.append(np.sum(Model['S_TH17_LP']))
            L_S_Treg_LP.append(np.sum(Model['S_Treg_LP']))
            
            L_S_eDC_MLN.append(np.sum(Model['S_eDC_MLN']))
            L_S_tDC_MLN.append(np.sum(Model['S_tDC_MLN']))

        ###--- modified by RLim ---###
        if t % Int_Sav == 1:
            eAg_eval.append(tmp_eAg)
            tAg_eval.append(tmp_tAg)
        ###--- modified by RLim ---###

        ###############################################################################
        #Agents Space Dynamics
        ###############################################################################
        with concurrent.futures.ThreadPoolExecutor() as executor:
        
            
            x_process = executor.submit(Mov_Agent_LP, Model, dt, t)
            y_process = executor.submit(Mov_Agent_MLN, Model, dt, t)
            z_process = executor.submit(Mov_Cyt_LP, Model, dt, t, SubT)
            u_process = executor.submit(Mov_Cyt_MLN, Model, dt, t, SubT)
            v_process = executor.submit(Mov_Epi, Model, dt, t)
            
            VA_Mon_LP, VA_Mac_LP, VA_ImDC_LP, VA_TH17_LP, VA_sTH17_LP, VA_Treg_LP, VA_sTreg_LP, VA_eDC_LP, VA_tDC_LP, VC_eRA_LP, VC_tRA_LP, VA_eAg_Mac_LP, VA_tAg_Mac_LP = x_process.result()
            VA_eDC_MLN, VA_tDC_MLN, VA_nTc_MLN, VA_nTe_MLN, VA_nTt_MLN, VA_TH17_MLN, VA_Treg_MLN, VA_mTH17_MLN, VA_mTreg_MLN, VA_fTH17_MLN, VA_fTreg_MLN, VC_eRA_MLN, VC_tRA_MLN = y_process.result()
            VC_IL10_LP, VC_TGFBeta_LP, VC_IL17_LP, VC_IL22_LP, VC_IL2_LP, VC_CCL2_LP, VC_CCL25_LP, VC_IL1Beta_23_LP, VC_Csf2_LP, VC_SAA_LP = z_process.result()
            VC_IL21_MLN, VC_IL2_MLN, VC_IL6_MLN, VC_TGFBeta_MLN, VC_IL10_MLN = u_process.result()
            VA_Epi, VA_sEpi, VA_eAg_Epi, VA_tAg_Epi, VA_eAg_sEpi, VA_tAg_sEpi = v_process.result()
        

        
        ###############################################################################
        #Update Populations Lists Space Dynamics
        ###############################################################################
        Reverse_VA_A(Model, dt, t, VS_TH17_LP, VS_Treg_LP, VS_eDC_MLN, VS_tDC_MLN, VA_Mac_LP, VA_eAg_Mac_LP, VA_tAg_Mac_LP, VA_Mon_LP, VA_ImDC_LP, VA_eDC_LP, VA_tDC_LP, VA_TH17_LP, VA_Treg_LP, VA_sTH17_LP, VA_sTreg_LP, VC_eRA_LP, VC_tRA_LP, VA_eAg, VA_tAg, VA_eDC_MLN, VA_tDC_MLN, VC_eRA_MLN, VC_tRA_MLN, VA_nTc_MLN, VA_nTe_MLN, VA_nTt_MLN, VA_fTH17_MLN, VA_fTreg_MLN, VA_TH17_MLN, VA_Treg_MLN, VA_mTH17_MLN, VA_mTreg_MLN, VC_IL10_LP, VC_TGFBeta_LP, VC_IL17_LP, VC_IL22_LP, VC_IL2_LP, VC_CCL2_LP, VC_CCL25_LP, VC_IL1Beta_23_LP, VC_Csf2_LP, VC_IL21_MLN, VC_IL2_MLN, VC_IL6_MLN, VC_TGFBeta_MLN, VC_IL10_MLN, VA_sEpi, VA_Epi, VA_eAg_Epi, VA_tAg_Epi, VA_eAg_sEpi, VA_tAg_sEpi, VC_SAA_LP)


    T = L_T

    TH17LP = L_A_TH17_LP
    TregLP = L_A_Treg_LP

    sTH17LP = L_A_sTH17_LP
    sTregLP = L_A_sTreg_LP
    tTregLP = L_A_tTreg_LP

    TH17MLN = L_A_TH17_MLN
    TregMLN = L_A_Treg_MLN

    mTH17MLN = L_A_mTH17_MLN
    mTregMLN = L_A_mTreg_MLN

    fTH17MLN = L_A_fTH17_MLN
    fTregMLN = L_A_fTreg_MLN

    eDCLP = L_A_eDC_LP
    tDCLP = L_A_tDC_LP

    eDCMLN = L_A_eDC_MLN
    tDCMLN = L_A_tDC_MLN

    Mac = L_A_Mac_LP
    Mon = L_A_Mon_LP
    ImDC = L_A_ImDC_LP

    eAg = L_A_eAg
    tAg = L_A_tAg

    eAgMac = L_A_eAg_Mac_LP
    tAgMac = L_A_tAg_Mac_LP

    nTc = L_A_nTc_MLN
    nTt = L_A_nTt_MLN
    nTe = L_A_nTe_MLN

    S_TH17 = L_S_TH17_LP
    S_Treg = L_S_Treg_LP

    S_eDC = L_S_eDC_MLN
    S_tDC = L_S_tDC_MLN

    Epi = L_A_Epi
    sEpi = L_A_sEpi

    eAgEpi = L_A_eAg_Epi
    tAgEpi = L_A_tAg_Epi
    eAgsEpi = L_A_eAg_sEpi
    tAgsEpi = L_A_tAg_sEpi
    
    IL10LP = L_C_IL10_LP
    TGFBetaLP = L_C_TGFBeta_LP
    IL17LP = L_C_IL17_LP
    IL22LP = L_C_IL22_LP
    IL2LP = L_C_IL2_LP
    CCL2LP = L_C_CCL2_LP
    CCL25LP = L_C_CCL25_LP
    IL1Beta23LP = L_C_IL1Beta_23_LP
    Csf2LP = L_C_Csf2_LP
    SAALP = L_C_SAA_LP
    
    IL21MLN = L_C_IL21_MLN
    IL2MLN = L_C_IL2_MLN
    IL6MLN = L_C_IL6_MLN
    TGFBetaMLN = L_C_TGFBeta_MLN
    IL10MLN = L_C_IL10_MLN

    z_Epi = Model['z_Epi']
    z_max_LP = Model['z_max_LP']
    Crypt_portion = Model['Crypt_portion']

    # plt.plot(T, [np.sum(TH17LP[i])*size_cube**3 for i in range(0, len(TH17LP))], label = 'TH17LP')
    # plt.plot(T, [np.sum(TregLP[i])*size_cube**3 for i in range(0, len(TregLP))], label = 'TregLP')
    # plt.plot(T, [np.sum(sTH17LP[i])*size_cube**3 for i in range(0, len(sTH17LP))], label = 'sTH17LP')
    # plt.plot(T, [np.sum(sTregLP[i])*size_cube**3 for i in range(0, len(sTregLP))], label = 'sTregLP')
    # plt.title('T cells LP')
    # plt.xlabel('Time in days')
    # plt.ylabel('Cell number (x1000)')
    # plt.legend()
    # plt.savefig(Folder + '/TcellsLP.png')
    # plt.close()

    # plt.plot(T, [np.sum(eAgMac[i])*size_cube**3 for i in range(0, len(TH17LP))], label = 'eMac')
    # plt.plot(T, [np.sum(tAgMac[i])*size_cube**3 for i in range(0, len(TregLP))], label = 'tMac')
    # plt.plot(T, [np.sum(eAgEpi[i])*size_cube**3 for i in range(0, len(sTH17LP))], label = 'eEpi')
    # plt.plot(T, [np.sum(tAgEpi[i])*size_cube**3 for i in range(0, len(sTregLP))], label = 'tEpi')
    # #plt.plot(T, [np.sum(eAg[i])*size_cube**3 for i in range(0, len(sTH17LP))], label = 'eAg')
    # #plt.plot(T, [np.sum(tAg[i])*size_cube**3 for i in range(0, len(sTregLP))], label = 'tAg')
    # plt.title('Ag LP')
    # plt.xlabel('Time in days')
    # plt.ylabel('Cell number (x1000)')
    # plt.legend()
    # plt.savefig(Folder + '/AgLP.png')
    # plt.close()

    # plt.plot(T, [np.sum(TH17MLN[i])*size_cube**3 for i in range(0, len(TH17MLN))], label = 'TH17MLN')
    # plt.plot(T, [np.sum(TregMLN[i])*size_cube**3 for i in range(0, len(TregMLN))], label = 'TregMLN')
    # plt.plot(T, [np.sum(mTH17MLN[i])*size_cube**3 for i in range(0, len(mTH17MLN))], label = 'mTH17MLN')
    # plt.plot(T, [np.sum(mTregMLN[i])*size_cube**3 for i in range(0, len(mTregMLN))], label = 'mTregMLN')
    # plt.plot(T, [np.sum(fTH17MLN[i])*size_cube**3 for i in range(0, len(fTH17MLN))], label = 'fTH17MLN')
    # plt.plot(T, [np.sum(fTregMLN[i])*size_cube**3 for i in range(0, len(fTregMLN))], label = 'fTregMLN')
    # plt.title('T cells MLN')
    # plt.xlabel('Time in days')
    # plt.ylabel('Cell number (x1000)')
    # plt.legend()
    # plt.savefig(Folder + '/TcellsMLN.png')
    # plt.close()

    # plt.plot(T, [eDCLP[i]*size_cube**3 for i in range(0, len(eDCLP))], label = 'eDCLP')
    # plt.plot(T, [tDCLP[i]*size_cube**3 for i in range(0, len(tDCLP))], label = 'tDCLP')
    # plt.plot(T, [Mon[i]*size_cube**3 for i in range(0, len(Mon))], label = 'Mon')
    # plt.plot(T, [Mac[i]*size_cube**3 for i in range(0, len(Mac))], label = 'Mac')
    # plt.plot(T, [ImDC[i]*size_cube**3 for i in range(0, len(ImDC))], label = 'ImDC')
    # plt.title('DCs and monocytes')
    # plt.xlabel('Time in days')
    # plt.ylabel('Cell number (x1000)')
    # plt.legend()
    # plt.savefig(Folder + '/DCsMonocytes.png')
    # plt.close()

    plt.plot(T, [eDCMLN[i]*size_cube**3 for i in range(0, len(eDCMLN))], label = 'eDCMLN')
    plt.plot(T, [tDCMLN[i]*size_cube**3 for i in range(0, len(tDCMLN))], label = 'tDCMLN')
    plt.title('DCs MLN')
    plt.xlabel('Time in days')
    plt.ylabel('Cell number (x1000)')
    plt.legend()
    plt.savefig(Folder + '/DCsMLN.png')
    plt.close()

    plt.plot(T, [nTc[i]*size_cube**3 for i in range(0, len(nTc))], label = 'nTc')
    plt.plot(T, [nTe[i]*size_cube**3 for i in range(0, len(nTe))], label = 'nTe')
    plt.plot(T, [nTt[i]*size_cube**3 for i in range(0, len(nTt))], label = 'nTt')
    plt.title('naive T cells MLN')
    plt.xlabel('Time in days')
    plt.ylabel('Cell number (x1000)')
    plt.legend()
    plt.savefig(Folder + '/naiveTcellsMLN.png')
    plt.close()

    # plt.plot(T, [IL1Beta23LP[i]/336 for i in range(0, len(IL1Beta23LP))], label = 'IL1Beta23LP', color = 'green')
    # plt.plot(T, [IL10LP[i]/336 for i in range(0, len(IL10LP))], label = 'IL10LP', color = 'red')
    # plt.plot(T, [np.sum(IL17LP[i])/336 for i in range(0, len(IL17LP))], label = 'IL17LP', color = 'blue')
    # plt.plot(T, [IL22LP[i]/336 for i in range(0, len(IL22LP))], label = 'IL22LP')
    # plt.plot(T, [np.sum(TGFBetaLP[i])/336 for i in range(0, len(TGFBetaLP))], label = 'TGFBetaLP')
    # plt.plot(T, [IL10LP_Obj for i in range(0, len(SAALP))], linestyle = '--', label = 'IL10 Obj', color = 'red')
    # plt.plot(T, [IL17LP_Obj for i in range(0, len(SAALP))], linestyle = '--', label = 'IL17 Obj', color = 'blue')
    # plt.plot(T, [IL1Beta23_Obj for i in range(0, len(SAALP))], linestyle = '--', label = 'IL1Beta-23 Obj', color = 'green')
    # plt.title('Cytokines LP 1')
    # plt.xlabel('Time in days')
    # plt.ylabel('Cytokine concentration (pg/mm^3)')
    # plt.legend(loc = 'best')
    # plt.savefig(Folder + '/CytokinesLP1.png')
    # plt.close()


    # plt.plot(T, [CCL2LP[i]/336 for i in range(0, len(CCL2LP))], label = 'CCL2LP')
    # plt.plot(T, [CCL25LP[i]/336 for i in range(0, len(CCL25LP))], label = 'CCL25LP')
    # plt.plot(T, [IL2LP[i]/336 for i in range(0, len(IL2LP))], label = 'IL2LP')
    # plt.plot(T, [np.sum(SAALP[i])/336 for i in range(0, len(SAALP))], label = 'SAALP')
    # plt.plot(T, [Csf2LP[i]/336 for i in range(0, len(Csf2LP))], label = 'Csf2LP')
    # plt.plot(T, [IL10LP_Obj for i in range(0, len(SAALP))], linestyle = '--', label = 'IL10 Obj', color = 'red')
    # plt.plot(T, [IL17LP_Obj for i in range(0, len(SAALP))], linestyle = '--', label = 'IL17 Obj', color = 'blue')
    # plt.plot(T, [IL1Beta23_Obj for i in range(0, len(SAALP))], linestyle = '--', label = 'IL1Beta-23 Obj', color = 'green')
    # plt.title('Cytokines LP 2')
    # plt.xlabel('Time in days')
    # plt.ylabel('Cytokine concentration (pg/mm^3)')
    # plt.legend(loc = 'best')
    # plt.savefig(Folder + '/CytokinesLP2.png')
    # plt.close()

    # plt.plot(T, [IL10MLN[i]/633 for i in range(0, len(IL10MLN))], label = 'IL10MLN', color = 'red')
    # plt.plot(T, [TGFBetaMLN[i]/633 for i in range(0, len(TGFBetaMLN))], label = 'TGFBetaMLN')
    # plt.plot(T, [IL6MLN[i]/633 for i in range(0, len(IL6MLN))], label = 'IL6MLN')
    # plt.plot(T, [IL2MLN[i]/633 for i in range(0, len(IL2MLN))], label = 'IL2MLN')
    # plt.plot(T, [IL21MLN[i]/633 for i in range(0, len(IL21MLN))], label = 'IL21MLN')
    # plt.plot(T, [IL10MLN_Obj for i in range(0, len(SAALP))], linestyle = '--', label = 'IL10 Obj', color = 'red')
    # plt.title('Cytokines MLN')
    # plt.xlabel('Time in days')
    # plt.ylabel('Cytokine concentration (pg/mm^3)')
    # plt.legend()
    # plt.savefig(Folder + '/CytokinesMLN.png')
    # plt.close()





    plt.plot(T, [((Epi[i][0, 3, z_Epi]) * L_Crypt[i][0, 3, z_Epi] + sEpi[i][0, 3, z_Epi]) * size_cube**2 for i in range(0, len(TregMLN))], color = 'red')
    plt.plot(T, [((Epi[i][0, 7, z_Epi]) * L_Crypt[i][0, 7, z_Epi] + sEpi[i][0, 7, z_Epi]) * size_cube**2 for i in range(0, len(TregMLN))], color = 'coral')
    plt.plot(T, [((Epi[i][1, 0, z_Epi]) * L_Crypt[i][1, 0, z_Epi] + sEpi[i][1, 0, z_Epi]) * size_cube**2 for i in range(0, len(TregMLN))], color = 'orange')
    plt.plot(T, [((Epi[i][1, 2, z_Epi]) * L_Crypt[i][1, 2, z_Epi] + sEpi[i][1, 2, z_Epi]) * size_cube**2 for i in range(0, len(TregMLN))], color = 'goldenrod')
    plt.plot(T, [((Epi[i][1, 4, z_Epi]) * L_Crypt[i][1, 4, z_Epi] + sEpi[i][1, 4, z_Epi]) * size_cube**2 for i in range(0, len(TregMLN))], color = 'olive')
    plt.plot(T, [((Epi[i][1, 6, z_Epi]) * L_Crypt[i][1, 6, z_Epi] + sEpi[i][1, 6, z_Epi]) * size_cube**2 for i in range(0, len(TregMLN))], color = 'yellow')
    plt.plot(T, [((Epi[i][2, 1, z_Epi]) * L_Crypt[i][2, 1, z_Epi] + sEpi[i][2, 1, z_Epi]) * size_cube**2 for i in range(0, len(TregMLN))], color = 'forestgreen')
    plt.plot(T, [((Epi[i][2, 5, z_Epi]) * L_Crypt[i][2, 5, z_Epi] + sEpi[i][2, 5, z_Epi]) * size_cube**2 for i in range(0, len(TregMLN))], color = 'turquoise')
    plt.plot(T, [((Epi[i][3, 0, z_Epi]) * L_Crypt[i][3, 0, z_Epi] + sEpi[i][3, 0, z_Epi]) * size_cube**2 for i in range(0, len(TregMLN))], color = 'darkcyan')
    plt.plot(T, [((Epi[i][3, 2, z_Epi]) * L_Crypt[i][3, 2, z_Epi] + sEpi[i][3, 2, z_Epi]) * size_cube**2 for i in range(0, len(TregMLN))], color = 'deepskyblue')
    plt.plot(T, [((Epi[i][3, 4, z_Epi]) * L_Crypt[i][3, 4, z_Epi] + sEpi[i][3, 4, z_Epi]) * size_cube**2 for i in range(0, len(TregMLN))], color = 'steelblue')
    plt.plot(T, [((Epi[i][3, 6, z_Epi]) * L_Crypt[i][3, 6, z_Epi] + sEpi[i][3, 6, z_Epi]) * size_cube**2 for i in range(0, len(TregMLN))], color = 'royalblue')
    plt.plot(T, [((Epi[i][4, 3, z_Epi]) * L_Crypt[i][4, 3, z_Epi] + sEpi[i][4, 3, z_Epi]) * size_cube**2 for i in range(0, len(TregMLN))], color = 'navy')
    plt.plot(T, [((Epi[i][4, 7, z_Epi]) * L_Crypt[i][4, 7, z_Epi] + sEpi[i][4, 7, z_Epi]) * size_cube**2 for i in range(0, len(TregMLN))], color = 'indigo')
    plt.plot(T, [((Epi[i][5, 0, z_Epi]) * L_Crypt[i][5, 0, z_Epi] + sEpi[i][5, 0, z_Epi]) * size_cube**2 for i in range(0, len(TregMLN))], color = 'purple')
    plt.plot(T, [((Epi[i][5, 2, z_Epi]) * L_Crypt[i][5, 2, z_Epi] + sEpi[i][5, 2, z_Epi]) * size_cube**2 for i in range(0, len(TregMLN))], color = 'blue')
    plt.plot(T, [((Epi[i][5, 4, z_Epi]) * L_Crypt[i][5, 4, z_Epi] + sEpi[i][5, 4, z_Epi]) * size_cube**2 for i in range(0, len(TregMLN))], color = 'crimson')
    plt.plot(T, [((Epi[i][5, 6, z_Epi]) * L_Crypt[i][5, 6, z_Epi] + sEpi[i][5, 6, z_Epi]) * size_cube**2 for i in range(0, len(TregMLN))], color = 'magenta')
    plt.plot(T, [((Epi[i][6, 1, z_Epi]) * L_Crypt[i][6, 1, z_Epi] + sEpi[i][6, 1, z_Epi]) * size_cube**2 for i in range(0, len(TregMLN))], color = 'wheat')
    plt.plot(T, [((Epi[i][6, 5, z_Epi]) * L_Crypt[i][6, 5, z_Epi] + sEpi[i][6, 5, z_Epi]) * size_cube**2 for i in range(0, len(TregMLN))], color = 'grey')
    plt.plot(T, [((Epi[i][7, 0, z_Epi]) * L_Crypt[i][7, 0, z_Epi] + sEpi[i][7, 0, z_Epi]) * size_cube**2 for i in range(0, len(TregMLN))], color = 'aquamarine')
    plt.plot(T, [((Epi[i][7, 2, z_Epi]) * L_Crypt[i][7, 2, z_Epi] + sEpi[i][7, 2, z_Epi]) * size_cube**2 for i in range(0, len(TregMLN))], color = 'tan')
    plt.plot(T, [((Epi[i][7, 4, z_Epi]) * L_Crypt[i][7, 4, z_Epi] + sEpi[i][7, 4, z_Epi]) * size_cube**2 for i in range(0, len(TregMLN))], color = 'peru')
    plt.plot(T, [((Epi[i][7, 6, z_Epi]) * L_Crypt[i][7, 6, z_Epi] + sEpi[i][7, 6, z_Epi]) * size_cube**2 for i in range(0, len(TregMLN))], color = 'chocolate')
    plt.plot(T, [Crypts_Cells for i in range(0, len(TregMLN))], color = 'black', label = 'Crypt objective')
    #plt.title('Crypts cells')
    plt.ylim(0, 0.1)
    plt.xlabel('Time in days', fontsize = 15)
    plt.ylabel('Cell number (x1000)', fontsize= 15)
    plt.legend()
    plt.savefig(Folder + '/CryptsObj.png', dpi =300)
    plt.close()


    plt.plot(T, [((np.sum(Epi[i][0, 1, z_Epi + 1: z_max_LP + 2])) * size_cube**2) for i in range(0, len(TregMLN))], color = 'steelblue')
    plt.plot(T, [((np.sum(Epi[i][0, 5, z_Epi + 1: z_max_LP + 2])) * size_cube**2) for i in range(0, len(TregMLN))], color = 'blue')
    plt.plot(T, [((np.sum(Epi[i][2, 3, z_Epi + 1: z_max_LP + 2])) * size_cube**2) for i in range(0, len(TregMLN))], color = 'red')
    plt.plot(T, [((np.sum(Epi[i][2, 7, z_Epi + 1: z_max_LP + 2])) * size_cube**2) for i in range(0, len(TregMLN))], color = 'firebrick')
    plt.plot(T, [((np.sum(Epi[i][4, 1, z_Epi + 1: z_max_LP + 2])) * size_cube**2) for i in range(0, len(TregMLN))], color = 'forestgreen')
    plt.plot(T, [((np.sum(Epi[i][4, 5, z_Epi + 1: z_max_LP + 2])) * size_cube**2) for i in range(0, len(TregMLN))], color = 'goldenrod')
    plt.plot(T, [((np.sum(Epi[i][6, 3, z_Epi + 1: z_max_LP + 2])) * size_cube**2) for i in range(0, len(TregMLN))], color = 'brown')
    plt.plot(T, [((np.sum(Epi[i][6, 7, z_Epi + 1: z_max_LP + 2])) * size_cube**2) for i in range(0, len(TregMLN))], color = 'purple')
    plt.plot(T, [Villus_Cells for i in range(0, len(TregMLN))], color = 'black', label = 'Villus objective')
    #plt.title('Villus')
    plt.ylim(0, 8.5)
    plt.ylabel('Cell number (x1000)', fontsize = 15)
    plt.xlabel('Time in days', fontsize = 15)
    plt.legend()
    plt.savefig(Folder + '/VillusObj.png', dpi = 300)
    plt.close()

    # for io in range(Model['N_chemical'] + 1):
    #     plt.plot(T, [(np.sum(TregLP[i], axis=(1, 2, 3))[io] + np.sum(sTregLP[i], axis=(1,2,3))[io])*size_cube**3 for i in range(0, len(L_A_Treg_MLN))], label = str(io))
    # plt.title('T reg in the LP Ag')
    # # plt.ylim(0, 100)
    # plt.xlabel('Time in days')
    # plt.ylabel('Cell number (x1000)')
    # plt.legend()
    # plt.savefig(Folder + '/TregAgLP.png')
    # plt.close()

    # for io in range(Model['N_chemical'] + 1):
    #     plt.plot(T, [(np.sum(TH17LP[i], axis=(1, 2,3))[io] + np.sum(sTH17LP[i], axis=(1, 2, 3))[io])*size_cube**3 for i in range(0, len(L_A_Treg_MLN))], label = str(io))
    # plt.title('TH17 in the LP Ag')
    # # plt.ylim(0, 100)
    # plt.xlabel('Time in days')
    # plt.ylabel('Cell number (x1000)')
    # plt.legend()
    # plt.savefig(Folder + '/TH17AgLP.png')
    # plt.close()

    # for io in range(Model['N_chemical'] + 1):
    #     plt.plot(T[1:], [(np.sum(eAg[i], axis=(1, 2,3))[io] + np.sum(tAg[i], axis=(1, 2, 3))[io])*size_cube**3 for i in range(1, len(L_A_Treg_MLN))], label = str(io))
    # plt.title('Ag in the Lumen Ag')
    # # plt.ylim(0, 100)
    # plt.xlabel('Time in days')
    # plt.ylabel('Cell number (x1000)')
    # plt.legend()
    # plt.savefig(Folder + '/AgLumen.png')
    # plt.close()


    # plt.plot(T, [(np.sum(TregLP[i]) + np.sum(sTregLP[i]) + np.sum(tTregLP[i]))*size_cube**3 for i in range(0, len(TregMLN))], color = 'steelblue', label = 'Treg')
    # plt.plot(T, [(np.sum(TH17LP[i]) + np.sum(sTH17LP[i]))*size_cube**3 for i in range(0, len(L_A_Treg_MLN))], color = 'firebrick', label = 'TH17')
    # plt.plot(T, [(np.sum(TH17LP[i]) + np.sum(TregLP[i]) + np.sum(sTH17LP[i]) + np.sum(sTregLP[i]) + np.sum(tTregLP[i]))*size_cube**3 for i in range(0, len(TregMLN))], color = 'purple', label = 'Tcells')
    # plt.plot(T, [TcellsLP for i in range(0, len(TregMLN))], color = 'black', label='ObjTcells')
    # plt.plot(T, [TcellsLP/7.7 for i in range(0, len(TregMLN))], color = 'gray', label = 'ObjTH17')
    # plt.plot(T, [TcellsLP/1.15 for i in range(0, len(TregMLN))], color = 'grey', label = 'ObjTreg')
    # plt.title('T cells in the LP')
    # plt.ylim(0, 100)
    # plt.xlabel('Time in days')
    # plt.ylabel('Cell number (x1000)')
    # plt.legend()
    # plt.savefig(Folder + '/TcellsObj.png')
    # plt.close()

    # plt.plot(T, [(np.sum(TregLP[i]) + np.sum(sTregLP[i]) + np.sum(sTH17LP[i]) + np.sum(TH17LP[i]))*size_cube**3 for i in range(0, len(TregMLN))], color = 'steelblue', label = 'Tcells')
    # plt.title('T cells in the LP')
    # plt.ylim(0, 100)
    # plt.xlabel('Time in days')
    # plt.ylabel('Cell number (x1000)')
    # plt.legend()
    # plt.savefig(Folder + '/FullTcells.png')
    # plt.close()


    # plt.plot(T, [(np.sum(TregLP[i][:, 0:4, 0:4, :]) + np.sum(sTregLP[i][:, 0:4, 0:4, :]))*size_cube**3 for i in range(0, len(TregMLN))], label = 'Zone 1')
    # plt.plot(T, [(np.sum(TregLP[i][:, 0:4, 4:, :]) + np.sum(sTregLP[i][:, 0:4, 4:, :]))*size_cube**3 for i in range(0, len(TregMLN))], label = 'Zone 2')
    # plt.plot(T, [(np.sum(TregLP[i][:, 4:, 0:4, :]) + np.sum(sTregLP[i][:, 4:, 0:4, :]))*size_cube**3 for i in range(0, len(TregMLN))], label = 'Zone 3')
    # plt.plot(T, [(np.sum(TregLP[i][:, 4:, 4:, :]) + np.sum(sTregLP[i][:, 4:, 4:, :]))*size_cube**3 for i in range(0, len(TregMLN))], label = 'Zone 4')
    # plt.xlabel('Time in days')
    # plt.ylabel('Cell number (x1000)')
    # plt.legend()
    # plt.savefig(Folder + '/TregZone.png')
    # plt.close()

    # plt.plot(T, [(np.sum(TH17LP[i][:, 0:4, 0:4, :]) + np.sum(sTH17LP[i][:, 0:4, 0:4, :]))*size_cube**3 for i in range(0, len(TregMLN))], label = 'Zone 1')
    # plt.plot(T, [(np.sum(TH17LP[i][:, 0:4, 4:, :]) + np.sum(sTH17LP[i][:, 0:4, 4:, :]))*size_cube**3 for i in range(0, len(TregMLN))], label = 'Zone 2')
    # plt.plot(T, [(np.sum(TH17LP[i][:, 4:, 0:4, :]) + np.sum(sTH17LP[i][:, 4:, 0:4, :]))*size_cube**3 for i in range(0, len(TregMLN))], label = 'Zone 3')
    # plt.plot(T, [(np.sum(TH17LP[i][:, 4:, 4:, :]) + np.sum(sTH17LP[i][:, 4:, 4:, :]))*size_cube**3 for i in range(0, len(TregMLN))], label = 'Zone 4')
    # plt.xlabel('Time in days')
    # plt.ylabel('Cell number (x1000)')
    # plt.legend()
    # plt.savefig(Folder + '/TH17Zone.png')
    # plt.close()

    # plt.plot(T, [(np.sum(IL17LP[i][0:4, 0:4, :]))*size_cube**3 for i in range(0, len(TregMLN))], label = 'Zone 1')
    # plt.plot(T, [(np.sum(IL17LP[i][0:4, 4:, :]))*size_cube**3 for i in range(0, len(TregMLN))], label = 'Zone 2')
    # plt.plot(T, [(np.sum(IL17LP[i][4:, 0:4, :]))*size_cube**3 for i in range(0, len(TregMLN))], label = 'Zone 3')
    # plt.plot(T, [(np.sum(IL17LP[i][4:, 4:, :]))*size_cube**3 for i in range(0, len(TregMLN))], label = 'Zone 4')
    # plt.xlabel('Time in days')
    # plt.ylabel('Molecular amount (pg)')
    # plt.legend()
    # plt.savefig(Folder + '/IL17Zone.png')
    # plt.close()

    # plt.plot(T, [(np.sum(SAALP[i][0:4, 0:4, :]))*size_cube**3 for i in range(0, len(TregMLN))], label = 'Zone 1')
    # plt.plot(T, [(np.sum(SAALP[i][0:4, 4:, :]))*size_cube**3 for i in range(0, len(TregMLN))], label = 'Zone 2')
    # plt.plot(T, [(np.sum(SAALP[i][4:, 0:4, :]))*size_cube**3 for i in range(0, len(TregMLN))], label = 'Zone 3')
    # plt.plot(T, [(np.sum(SAALP[i][4:, 4:, :]))*size_cube**3 for i in range(0, len(TregMLN))], label = 'Zone 4')
    # plt.xlabel('Time in days')
    # plt.ylabel('Molecular amount (pg)')
    # plt.legend()
    # plt.savefig(Folder + '/SAAZone.png')
    # plt.close()

    # plt.plot(T, [(np.sum(TGFBetaLP[i][0:4, 0:4, :]))*size_cube**3 for i in range(0, len(TregMLN))], label = 'Zone 1')
    # plt.plot(T, [(np.sum(TGFBetaLP[i][0:4, 4:, :]))*size_cube**3 for i in range(0, len(TregMLN))], label = 'Zone 2')
    # plt.plot(T, [(np.sum(TGFBetaLP[i][4:, 0:4, :]))*size_cube**3 for i in range(0, len(TregMLN))], label = 'Zone 3')
    # plt.plot(T, [(np.sum(TGFBetaLP[i][4:, 4:, :]))*size_cube**3 for i in range(0, len(TregMLN))], label = 'Zone 4')
    # plt.xlabel('Time in days')
    # plt.ylabel('Molecular amount (pg)')
    # plt.legend()
    # plt.savefig(Folder + '/TGFBetaZone.png')
    # plt.close()

    ###--- modified by RLim ---###
    save_data['IL-10_LP'] = [IL10LP[i]/336 for i in range(0, len(IL10LP))]
    plt.plot(T, [IL10LP[i]/336 for i in range(0, len(IL10LP))], label = 'IL10LP', color = 'red')
    plt.plot(T, [IL10LP_Obj for i in range(0, len(SAALP))], linestyle = '--', label = 'IL10 Obj', color = 'red')
    plt.ylim(bottom=0.0)
    plt.title('Cytokine, IL-10_LP')
    plt.xlabel('Time in days')
    plt.ylabel('Cytokine concentration (pg/mm^3)')
    plt.legend(loc = 'best')
    plt.savefig(Folder + '/Cytokines_IL10_LP.png')
    plt.close()
    
    save_data['IL-17_LP'] = [np.sum(IL17LP[i])/336 for i in range(0, len(IL17LP))]
    plt.plot(T, [np.sum(IL17LP[i])/336 for i in range(0, len(IL17LP))], label = 'IL17LP', color = 'blue')
    plt.plot(T, [IL17LP_Obj for i in range(0, len(SAALP))], linestyle = '--', label = 'IL17 Obj', color = 'blue')
    plt.ylim(bottom=0.0)
    plt.title('Cytokine, IL-17_LP')
    plt.xlabel('Time in days')
    plt.ylabel('Cytokine concentration (pg/mm^3)')
    plt.legend(loc = 'best')
    plt.savefig(Folder + '/Cytokines_IL17_LP.png')
    plt.close()

    save_data['IL-22_LP'] = [IL22LP[i]/336 for i in range(0, len(IL22LP))]
    plt.plot(T, [IL22LP[i]/336 for i in range(0, len(IL22LP))], label = 'IL22LP')
    plt.ylim(bottom=0.0)
    plt.title('Cytokine, IL-22_LP')
    plt.xlabel('Time in days')
    plt.ylabel('Cytokine concentration (pg/mm^3)')
    plt.legend(loc = 'best')
    plt.savefig(Folder + '/Cytokines_IL22_LP.png')
    plt.close()

    save_data['IL-1beta/23_LP'] = [IL1Beta23LP[i]/336 for i in range(0, len(IL1Beta23LP))]
    plt.plot(T, [IL1Beta23LP[i]/336 for i in range(0, len(IL1Beta23LP))], label = 'IL1Beta23LP', color = 'green')
    plt.plot(T, [IL1Beta23_Obj for i in range(0, len(SAALP))], linestyle = '--', label = 'IL1Beta-23 Obj', color = 'green')
    plt.ylim(bottom=0.0)
    plt.title('Cytokine, IL-1beta/23_LP')
    plt.xlabel('Time in days')
    plt.ylabel('Cytokine concentration (pg/mm^3)')
    plt.legend(loc = 'best')
    plt.savefig(Folder + '/Cytokines_IL1Beta23_LP.png')
    plt.close()

    save_data['SAA_LP'] = [np.sum(SAALP[i])/336 for i in range(0, len(SAALP))]
    plt.plot(T, [np.sum(SAALP[i])/336 for i in range(0, len(SAALP))], label = 'SAALP')
    plt.title('Cytokine, SAA_LP')
    plt.xlabel('Time in days')
    plt.ylabel('Cytokine concentration (pg/mm^3)')
    plt.legend(loc = 'best')
    plt.savefig(Folder + '/Cytokines_SAA_LP.png')
    plt.close()

    save_data['Treg_LP'] = [(np.sum(TregLP[i]) + np.sum(sTregLP[i]) + np.sum(tTregLP[i]))*size_cube**3 for i in range(0, len(TregMLN))]
    plt.plot(T, [(np.sum(TregLP[i]) + np.sum(sTregLP[i]) + np.sum(tTregLP[i]))*size_cube**3 for i in range(0, len(TregMLN))], color = 'steelblue', label = 'Treg')
    plt.plot(T, [TcellsLP/1.15 for i in range(0, len(TregMLN))], color = 'grey', label = 'ObjTreg')
    plt.title('Treg in the LP')
    plt.ylim(bottom=0.0)
    plt.xlabel('Time in days')
    plt.ylabel('Cell number (x1000)')
    plt.legend()
    plt.savefig(Folder + '/TcellsObj_Treg_LP.png')
    plt.close()
    
    
    plt.plot(T, [(np.sum(TregMLN[i]))*size_cube**3 for i in range(0, len(TregMLN))], color = 'steelblue', label = 'Treg')
    plt.plot(T, [(np.sum(TH17MLN[i]))*size_cube**3 for i in range(0, len(TregMLN))], color = 'orangered', label = 'TH17')
    plt.title('Treg in the MLN')
    plt.ylim(bottom=0.0)
    plt.xlabel('Time in days')
    plt.ylabel('Cell number (x1000)')
    plt.legend()
    plt.savefig(Folder + '/TcellsObj_TMLN_LP.png')
    plt.close()
    
    plt.plot(T, [(np.sum(mTregMLN[i]))*size_cube**3 for i in range(0, len(TregMLN))], color = 'purple', label = 'mTreg')
    plt.plot(T, [(np.sum(mTH17MLN[i]))*size_cube**3 for i in range(0, len(TregMLN))], color = 'green', label = 'mTH17')
    plt.title('Treg in the MLN')
    plt.ylim(bottom=0.0)
    plt.xlabel('Time in days')
    plt.ylabel('Cell number (x1000)')
    plt.legend()
    plt.savefig(Folder + '/TcellsObj_mTMLN_LP.png')
    plt.close()

    save_data['TH17_LP'] = [(np.sum(TH17LP[i]) + np.sum(sTH17LP[i]))*size_cube**3 for i in range(0, len(L_A_Treg_MLN))]
    plt.plot(T, [(np.sum(TH17LP[i]) + np.sum(sTH17LP[i]))*size_cube**3 for i in range(0, len(L_A_Treg_MLN))], color = 'firebrick', label = 'TH17')
    plt.plot(T, [TcellsLP/7.7 for i in range(0, len(TregMLN))], color = 'gray', label = 'ObjTH17')
    plt.title('TH17 in the LP')
    plt.ylim(bottom=0.0)
    plt.xlabel('Time in days')
    plt.ylabel('Cell number (x1000)')
    plt.legend()
    plt.savefig(Folder + '/TcellsObj_TH17_LP.png')
    plt.close()

    for io in range(Model['N_chemical'] + 1):
        eAg_tmp_list = [eAg_eval[i][io]*size_cube**3 for i in range(0, len(T))]
        save_data['eAg_' + str(io)] = eAg_tmp_list
        tAg_tmp_list = [tAg_eval[i][io]*size_cube**3 for i in range(0, len(T))]
        save_data['tAg_' + str(io)] = tAg_tmp_list
        plt.plot(T,eAg_tmp_list, label = 'eAg')
        plt.plot(T,tAg_tmp_list, label = 'tAg')
        plt.plot(T, [Ag_threshold[io] for i in range(0, len(T))], color = 'gray', label = 'Thomas')
        plt.title('Antigen ' + str(io))
        plt.xlabel('Time in days')
        plt.ylabel('Density')
        plt.ylim(bottom=0.0)
        plt.legend()
        plt.savefig(Folder + '/Ag_load_' + str(io) + '_LP.png')
        plt.close()
    #

    save_data['eMac_LP'] = [np.sum(eAgMac[i])*size_cube**3 for i in range(0, len(TH17LP))]
    save_data['tMac_LP'] = [np.sum(tAgMac[i])*size_cube**3 for i in range(0, len(TH17LP))]
    plt.plot(T, [np.sum(eAgMac[i])*size_cube**3 for i in range(0, len(TH17LP))], label = 'eMac')
    plt.plot(T, [np.sum(tAgMac[i])*size_cube**3 for i in range(0, len(TregLP))], label = 'tMac')
    plt.ylim(bottom=0.0)
    plt.title('Macrophages')
    plt.xlabel('Time in days')
    plt.ylabel('Cell number (x1000)')
    plt.legend()
    plt.savefig(Folder + '/Macrophages_LP.png')
    plt.close()

    save_data['eEpi_LP'] = [np.sum(eAgEpi[i])*size_cube**3 for i in range(0, len(sTH17LP))]
    save_data['tEpi_LP'] = [np.sum(tAgEpi[i])*size_cube**3 for i in range(0, len(sTregLP))]
    plt.plot(T, [np.sum(eAgEpi[i])*size_cube**3 for i in range(0, len(sTH17LP))], label = 'eEpi')
    plt.plot(T, [np.sum(tAgEpi[i])*size_cube**3 for i in range(0, len(sTregLP))], label = 'tEpi')
    plt.title('Epithelial cells')
    plt.xlabel('Time in days')
    plt.ylabel('Cell number (x1000)')
    plt.legend()
    plt.savefig(Folder + '/Epithelial_cells_LP.png')
    plt.close()

    save_data['Time'] = T
    save_data['T_threshold'] = T_threshold
    with open('json_Model_' + Folder + '.json','w',encoding='utf-8') as json_file:
        json.dump(save_data,json_file,ensure_ascii=False,indent='\t')
    #
    ###--- modified by RLim ---###

    end_time = time.time()
    total_time = end_time - start_time

    print('for Chemical number ', N_chemical + 1, ' time is ', total_time, ' seconds')

###--- modified by RLim ---###
Folder = 'result'
if not os.path.exists(Folder):
        os.mkdir(Folder)
else:
    shutil.rmtree(Folder,ignore_errors=True)
    os.makedirs(Folder)
#
dt = 0.002
T = 14
N_chemical = 0
nu_Ag = 5.0
Ag_coeff = 5
save_data = dict()
eAg_eval = list()
tAg_eval = list()
eAg_eval.append(list(np.zeros((N_chemical + 1,))))
tAg_eval.append(list(np.zeros((N_chemical + 1,))))
Main(dt, T, SubT, N_chemical, Folder)
# shutil.rmtree('__pycache__',ignore_errors=True)
###--- modified by RLim ---###
