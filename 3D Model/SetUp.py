# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 11:46:22 2022

@author: User
"""
import numpy as np
import pandas as pd
import ast



#This function creates all 3D numpy arrays for the structure of the model, for LP, MLN and epithelium
#The Ind arrays are just matrix of 1 and 0, 1 if the coordinate is in the zone, 0 else
#The N_Epi function describes how many of the surfaces of one cube or occupy buy epithelium, 1 for crypt and basic epithelium, 4 for climbing and 5 for top of villi
def Topological_Parameter(Model):
    Model['size_cube'] = 0.3
    ###########################################################################
    #Lamina Propria & Epithelium
    ###########################################################################
    
    Model['x_max_LP'] = 7
    Model['y_max_LP'] = 7
    Model['z_max_LP'] = 5
    Model['z_Epi'] = 4
    
    
    #Creation of the Volume containing blood vessels in the LP
    Model['Volume_blood_LP'] = []
    for x in range(0, Model['x_max_LP'] + 1):
        for y in range(0, Model['y_max_LP'] + 1):
            Model['Volume_blood_LP'].append((x, y, Model['z_Epi'] - 2))
    for y in range(1, Model['y_max_LP'] + 1, 4):
            for x in range(0, Model['x_max_LP'] + 1, 4):
                for z in range(Model['z_Epi'] - 1, Model['z_max_LP'] + 1):
                    Model['Volume_blood_LP'].append((x, y, z))
    for y in range(3, Model['y_max_LP'] + 1, 4):
        for x in range(2, Model['x_max_LP'] + 1, 4):
            for z in range(Model['z_Epi'] - 1, Model['z_max_LP'] + 1):
                    Model['Volume_blood_LP'].append((x, y, z))
                    
    Model['Ind_Volume_blood_LP'] = np.array([[[float(0)] * (Model['z_max_LP'] + 3)] * (Model['y_max_LP'] + 2)] * (Model['x_max_LP'] + 2))
    for x in range(0, Model['x_max_LP'] + 1):
        for y in range(0, Model['y_max_LP'] + 1):
            for z in range(0, Model['z_max_LP'] + 1):
                if (x, y, z) in Model['Volume_blood_LP']:
                    Model['Ind_Volume_blood_LP'][x, y, z] = 1
                    
    Model['V_blood_LP'] = len(Model['Volume_blood_LP'])
    
    
    #Creation of the Volume containing lymphatic vessels in the LP
    Model['Volume_Lymph_adj_LP'] = []
    for x in range(0, Model['x_max_LP'] + 1):
        for y in range(0, Model['y_max_LP'] + 1):
            Model['Volume_Lymph_adj_LP'].append((x, y, Model['z_Epi'] - 1))
    for y in range(0, Model['y_max_LP'] + 1, 4):
        for x in range(1, Model['x_max_LP'] + 1, 4):
            for z in range(0, Model['z_max_LP'] - 1):
                Model['Volume_Lymph_adj_LP'].append((x, y, z))
        for x in range(3, Model['x_max_LP'] + 1, 4):
            for z in range(0, Model['z_max_LP'] - 1):
                Model['Volume_Lymph_adj_LP'].append((x, y, z))
    for y in range(1, Model['y_max_LP'] + 1, 4):
        for x in range(2, Model['x_max_LP'] + 1, 4):
            for z in range(0, Model['z_max_LP'] - 1):
                Model['Volume_Lymph_adj_LP'].append((x, y, z))
    for y in range(2, Model['y_max_LP'] + 1, 4):
        for x in range(1, Model['x_max_LP'] + 1, 4):
            for z in range(0, Model['z_max_LP'] - 1):
                Model['Volume_Lymph_adj_LP'].append((x, y, z))
        for x in range(3, Model['x_max_LP'] + 1, 4):
            for z in range(0, Model['z_max_LP'] - 1):
                Model['Volume_Lymph_adj_LP'].append((x, y, z))
    for y in range(3, Model['y_max_LP'] + 1, 4):
        for x in range(0, Model['x_max_LP'] + 1, 4):
            for z in range(0, Model['z_max_LP'] - 1):
                Model['Volume_Lymph_adj_LP'].append((x, y, z))
                
    Model['Ind_Volume_Lymph_adj_LP'] = np.array([[[float(0)] * (Model['z_max_LP'] + 3)] * (Model['y_max_LP'] + 2)] * (Model['x_max_LP'] + 2))
    for x in range(0, Model['x_max_LP'] + 1):
        for y in range(0, Model['y_max_LP'] + 1):
            for z in range(0, Model['z_max_LP'] + 1):
                if (x, y, z) in Model['Volume_blood_LP']:
                    Model['Ind_Volume_Lymph_adj_LP'][x, y, z] = 1
    
    
    #Creation of Volumes for LP and the screening sites
    Model['Volume_Screening'] = []
    Model['Volume_Lamina_Propria'] = []
    for x in range(0, Model['x_max_LP'] + 1):
        for y in range(0, Model['y_max_LP'] + 1):
            for z in range(0, Model['z_Epi'] + 1):
                Model['Volume_Lamina_Propria'].append((x, y, z))
    for y in range(1, Model['y_max_LP'] + 1, 4):
            for x in range(0, Model['x_max_LP'] + 1, 4):
                for z in range(Model['z_Epi'] + 1, Model['z_max_LP'] + 2):
                    Model['Volume_Screening'].append((x, y, z))
                    Model['Volume_Lamina_Propria'].append((x, y, z))
    for y in range(3, Model['y_max_LP'] + 1, 4):
        for x in range(2, Model['x_max_LP'] + 1, 4):
            for z in range(Model['z_Epi'] + 1, Model['z_max_LP'] + 2):
                Model['Volume_Screening'].append((x, y, z))
                Model['Volume_Lamina_Propria'].append((x, y, z))
                
    #Creation of Epithelium and Crypts Volumes                
    Model['Volume_Epithelium'] = []
    Model['Volume_Crypt'] = []
    for y in range(0, Model['y_max_LP'] + 1, 4):
        for x in range(0, Model['x_max_LP'] + 1, 4):
            #Basic Epithelium
            Model['Volume_Epithelium'].append((x, y, Model['z_Epi']))            
        for x in range(1, Model['x_max_LP'] + 1, 4):
            #Crypt
            Model['Volume_Epithelium'].append((x, y, Model['z_Epi']))
            Model['Volume_Crypt'].append((x, y, Model['z_Epi']))
        for x in range(2, Model['x_max_LP'] + 1, 4):
            #Basic Epithelium
            Model['Volume_Epithelium'].append((x, y, Model['z_Epi']))
        for x in range(3, Model['x_max_LP'] + 1, 4):
            #Crypt
            Model['Volume_Epithelium'].append((x, y, Model['z_Epi']))
            Model['Volume_Crypt'].append((x, y, Model['z_Epi']))
            
    for y in range(1, Model['y_max_LP'] + 1, 4):
        for x in range(0, Model['x_max_LP'] + 1, 4):
            for z in range(Model['z_Epi'] + 1, Model['z_max_LP'] + 2):
                #Climbing and Top
                Model['Volume_Epithelium'].append((x, y, z))
        for x in range(1, Model['x_max_LP'] + 1, 4):
            #Basic Epithelium
            Model['Volume_Epithelium'].append((x, y, Model['z_Epi']))
        for x in range(2, Model['x_max_LP'] + 1, 4):
            #Crypt
            Model['Volume_Epithelium'].append((x, y, Model['z_Epi']))
            Model['Volume_Crypt'].append((x, y, Model['z_Epi']))
        for x in range(3, Model['x_max_LP'] + 1, 4):
            #Basic Epithelium
            Model['Volume_Epithelium'].append((x, y, Model['z_Epi']))
            
    for y in range(2, Model['y_max_LP'] + 1, 4):
        for x in range(0, Model['x_max_LP'] + 1, 4):
            #Basic Epithelium
            Model['Volume_Epithelium'].append((x, y, Model['z_Epi']))
        for x in range(1, Model['x_max_LP'] + 1, 4):
            #Crypt
            Model['Volume_Epithelium'].append((x, y, Model['z_Epi']))
            Model['Volume_Crypt'].append((x, y, Model['z_Epi']))
        for x in range(2, Model['x_max_LP'] + 1, 4):
            #Basic Epithelium
            Model['Volume_Epithelium'].append((x, y, Model['z_Epi']))
        for x in range(3, Model['x_max_LP'] + 1, 4):
            #Crypt
            Model['Volume_Epithelium'].append((x, y, Model['z_Epi']))
            Model['Volume_Crypt'].append((x, y, Model['z_Epi']))
            
    for y in range(3, Model['y_max_LP'] + 1, 4):
        for x in range(0, Model['x_max_LP'] + 1, 4):
            #Crypt
            Model['Volume_Epithelium'].append((x, y, Model['z_Epi']))
            Model['Volume_Crypt'].append((x, y, Model['z_Epi']))
        for x in range(1, Model['x_max_LP'] + 1, 4):
            #Basic Epithelium
            Model['Volume_Epithelium'].append((x, y, Model['z_Epi']))
        for x in range(2, Model['x_max_LP'] + 1, 4):
            for z in range(Model['z_Epi'] + 1, Model['z_max_LP'] + 2):
                #Climbing and Top
                Model['Volume_Epithelium'].append((x, y, z))
        for x in range(3, Model['x_max_LP'] + 1, 4):
            #Basic Epithelium
            Model['Volume_Epithelium'].append((x, y, Model['z_Epi']))

    
                
    #Full Volume            
    Model['Volume_Lamina_Propria_Epi'] = Model['Volume_Lamina_Propria'] + Model['Volume_Epithelium']
    
    
    #Creation of Complementary Volumes                
    Model['Volume_Lamina_Propria_Complementary'] = []
    Model['Volume_Epithelium_Complementary'] = []
    Model['Volume_Crypt_Complementary'] = []
    for x in range(0, Model['x_max_LP'] + 2):
        for y in range(0, Model['y_max_LP'] + 2):
            for z in range(0, Model['z_max_LP'] + 3):
                if (x, y, z) not in Model['Volume_Lamina_Propria']:
                    Model['Volume_Lamina_Propria_Complementary'].append((x, y, z))
                if (x, y, z) not in Model['Volume_Epithelium']:
                    Model['Volume_Epithelium_Complementary'].append((x, y, z))
                if (x, y, z) not in Model['Volume_Crypt']:
                    Model['Volume_Crypt_Complementary'].append((x, y, z))
                    
    Model['Volume_Lamina_Propria_Epi_Complementary'] = []
    for x in range(0, Model['x_max_LP'] + 2):
        for y in range(0, Model['y_max_LP'] + 2):
            for z in range(0, Model['z_max_LP'] + 3):
                if (x, y, z) not in Model['Volume_Lamina_Propria_Epi']:
                    Model['Volume_Lamina_Propria_Epi_Complementary'].append((x, y, z))
                    
    Model['Volume_Lumen'] = []
    for x in range(0, Model['x_max_LP'] + 1):
        for y in range(0, Model['y_max_LP'] + 2):
            for z in range(0, Model['z_max_LP'] + 3):
                if (x, y, z) not in Model['Volume_Lamina_Propria_Epi']:
                    Model['Volume_Lamina_Propria_Epi_Complementary'].append((x, y, z))
        
                    
    
    ###########################################################################
    #Mesenteric Lymph Node
    ###########################################################################
                    
    Model['x_max_MLN'] = 9
    Model['y_max_MLN'] = 9
    Model['z_max_MLN'] = 9
    Model['x_enter1_MLN'] = 0
    Model['x_enter2_MLN'] = 9
    Model['y_enter1_MLN'] = 0
    Model['y_enter2_MLN'] = 0
    Model['z_enter1_MLN'] = 0
    Model['z_enter2_MLN'] = 9
    Model['x_out1_MLN'] = 0
    Model['x_out2_MLN'] = 9
    Model['y_out1_MLN'] = 9
    Model['y_out2_MLN'] = 9
    Model['z_out1_MLN'] = 0
    Model['z_out2_MLN'] = 9
    Model['x_hole1_MLN'] = 3
    Model['x_hole2_MLN'] = 6
    Model['y_hole1_MLN'] = 3
    Model['y_hole2_MLN'] = 6
    Model['z_hole1_MLN'] = 3
    Model['z_hole2_MLN'] = 6
    
    
    #Creation of the Volumes of MLN
    
    Model['Volume_vess_in_MLN'] = (Model['x_enter2_MLN'] + 1 - Model['x_enter1_MLN']) * (Model['y_enter2_MLN'] + 1 - Model['y_enter1_MLN']) * (Model['z_enter2_MLN'] + 1 - Model['z_enter1_MLN'])
    Volume_vess_out_MLN = (Model['x_out2_MLN'] + 1 - Model['x_out1_MLN']) * (Model['y_out2_MLN'] + 1 - Model['y_out1_MLN']) * (Model['z_out2_MLN'] + 1 - Model['z_out1_MLN'])
    
    Model['Volume_vess_MLN'] = []
    Model['Volume_hole_MLN'] = []
    Model['Volume_free_MLN'] = []
    Model['Volume_Complementary_MLN'] = []
    
    Model['Ind_Volume_vess_in_MLN'] = np.array([[[float(0)]*(Model['z_max_MLN'] + 2)]*(Model['y_max_MLN'] + 3)]*(Model['x_max_MLN'] + 2))
    Model['Ind_Volume_vess_out_MLN'] = np.array([[[float(0)]*(Model['z_max_MLN'] + 2)]*(Model['y_max_MLN'] + 3)]*(Model['x_max_MLN'] + 2))
                    
    
    for x in range(0, Model['x_max_MLN'] + 1):
        for y in range(0, Model['y_max_MLN'] + 1):
            for z in range(0, Model['z_max_MLN'] + 1):
                if Model['x_hole1_MLN'] <= x <= Model['x_hole2_MLN'] and Model['y_hole1_MLN'] <= y <= Model['y_hole2_MLN'] and Model['z_hole1_MLN'] <= z <= Model['z_hole2_MLN']:
                    Model['Volume_hole_MLN'].append((x, y, z))
                    Model['Volume_Complementary_MLN'].append((x, y, z))
                else:
                    Model['Volume_free_MLN'].append((x, y, z))
                if Model['x_enter1_MLN'] <= x <= Model['x_enter2_MLN'] and Model['y_enter1_MLN'] <= y <= Model['y_enter2_MLN'] and Model['z_enter1_MLN'] <= z <= Model['z_enter2_MLN']:
                    Model['Volume_vess_MLN'].append((x, y, z))
                    Model['Ind_Volume_vess_in_MLN'][x, y, z] = 1
                if Model['x_out1_MLN'] <= x <= Model['x_out2_MLN'] and Model['y_out1_MLN'] <= y <= Model['y_out2_MLN'] and Model['z_out1_MLN'] <= z <= Model['z_out2_MLN']:
                    Model['Volume_vess_MLN'].append((x, y, z))
                    Model['Ind_Volume_vess_out_MLN'][x, y, z] = 1
    
    for x in range(0, Model['x_max_MLN'] + 2):
        for y in range(0, Model['y_max_MLN'] + 3):
            for z in range(0, Model['z_max_MLN'] + 2):
                if (x, y, z) not in Model['Volume_free_MLN']:
                    Model['Volume_Complementary_MLN'].append((x, y, z))
                    
    
    #Number of Epithelial surfeces per cube
    Model['N_Epi'] = np.array([[[float(0)]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2))
    Model['Ind_crypt'] = np.array([[[float(0)]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2))
    Model['Ind_Top'] = np.array([[[float(0)]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2))
    for y in range(0, Model['y_max_LP'] + 1, 4):
        for x in range(0, Model['x_max_LP'] + 1, 4):
    #Basic Epithelium
            Model['N_Epi'][x, y, Model['z_Epi']] = 1    
        for x in range(1, Model['x_max_LP'] + 1, 4):
    #Crypt
            Model['N_Epi'][x, y, Model['z_Epi']] = 1
            Model['Ind_crypt'][x, y, Model['z_Epi']] = 1
        for x in range(2, Model['x_max_LP'] + 1, 4):
    #Basic Epithelium
            Model['N_Epi'][x, y, Model['z_Epi']] = 1
        for x in range(3, Model['x_max_LP'] + 1, 4):
    #Crypt
            Model['N_Epi'][x, y, Model['z_Epi']] = 1 
            Model['Ind_crypt'][x, y, Model['z_Epi']] = 1       
    for y in range(1, Model['y_max_LP'] + 1, 4):
        for x in range(0, Model['x_max_LP'] + 1, 4):
    #Climbing and Top of Villus
            Model['N_Epi'][x, y, Model['z_Epi'] + 1] = 4
            Model['N_Epi'][x, y, Model['z_max_LP'] + 1] = 5
            Model['Ind_Top'][x, y, Model['z_max_LP'] + 1] = 1
            for z in range(Model['z_Epi'] + 2, Model['z_max_LP'] + 1):
                Model['N_Epi'][x, y, z] = 4
        for x in range(1, Model['x_max_LP'] + 1, 4):
    #Basic Epithelium
            Model['N_Epi'][x, y, Model['z_Epi']] = 1
        for x in range(2, Model['x_max_LP'] + 1, 4):
    #Crypt
            Model['N_Epi'][x, y, Model['z_Epi']] = 1
            Model['Ind_crypt'][x, y, Model['z_Epi']] = 1
        for x in range(3, Model['x_max_LP'] + 1, 4):
    #Basic Epithelium
            Model['N_Epi'][x, y, Model['z_Epi']] = 1
    for y in range(2, Model['y_max_LP'] + 1, 4):
        for x in range(0, Model['x_max_LP'] + 1, 4):
    #Basic Epithelium
            Model['N_Epi'][x, y, Model['z_Epi']] = 1
        for x in range(1, Model['x_max_LP'] + 1, 4):
    #Crypt
            Model['N_Epi'][x, y, Model['z_Epi']] = 1
            Model['Ind_crypt'][x, y, Model['z_Epi']] = 1
        for x in range(2, Model['x_max_LP'] + 1, 4):
    #Basic Epithelium
            Model['N_Epi'][x, y, Model['z_Epi']] = 1
        for x in range(3, Model['x_max_LP'] + 1, 4):
    #Crypt
            Model['N_Epi'][x, y, Model['z_Epi']] = 1
            Model['Ind_crypt'][x, y, Model['z_Epi']] = 1
    for y in range(3, Model['y_max_LP'] + 1, 4):
        for x in range(0, Model['x_max_LP'] + 1, 4):
    #Crypt
            Model['N_Epi'][x, y, Model['z_Epi']] = 1
            Model['Ind_crypt'][x, y, Model['z_Epi']] = 1
        for x in range(1, Model['x_max_LP'] + 1, 4):
    #Basic Epithelium
            Model['N_Epi'][x, y, Model['z_Epi']] = 1
        for x in range(2, Model['x_max_LP'] + 1, 4):
    #Climbing and Top of Villus
            Model['N_Epi'][x, y, Model['z_Epi'] + 1] = 4
            Model['N_Epi'][x, y, Model['z_max_LP'] + 1] = 5
            Model['Ind_Top'][x, y, Model['z_max_LP'] + 1] = 1
            for z in range(Model['z_Epi'] + 2, Model['z_max_LP'] + 1):
                Model['N_Epi'][x, y, z] = 4
        for x in range(3, Model['x_max_LP'] + 1, 4):
    #Basic Epithelium
            Model['N_Epi'][x, y, Model['z_Epi']] = 1


#This function sets up biological parameters using the excel file Parameter.xlsx (that's df = pd.read_csv('Parameter_v2.csv', delimiter=',') in the main function), it requires the number of chemical group for antigens N_chemical, note that the actual number of chemical groups is going to be N_chemical + 1 and as in this context we want half of them effector and half of them tolerogenic, we want N_chemical+1 to be even so put N_chemical odd
#This function also create arrays for the agents with constant value, tTreg, ILC3 and stromal cells              
def Biological_Parameter(Model, df, N_chemical):
    Model['N_chemical'] = N_chemical
    for index, row in df.iterrows():
        if row[5]:
            Model[row[0]] = int(row[1])
        elif row[4]:
            Model[row[0]] = ast.literal_eval(row[1])
        else:
            Model[row[0]] = float(row[1])
            
    Model['phi_fTH17_mTH17'] = 1/Model['Gamma_eRA_fTH17']
    Model['phi_fTreg_mTreg'] = 1/Model['Gamma_eRA_fTreg']

    
    Model['A_ILC3'] = np.array([[[Model['ILC3_in_LP']]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2))
    Model['A_tTreg_LP'] = np.array([[[Model['tTreg_density_LP']]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2))
    Model['A_Stro_LP'] = np.array([[[Model['Stromal_cells']]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2))
    
    for (x, y, z) in Model['Volume_Lamina_Propria_Complementary']:
        Model['A_ILC3'][x, y, z] = 0
        Model['A_tTreg_LP'][x, y, z] = 0
    
    Model['A_tTreg_MLN'] = np.array([[[Model['tTreg_density_MLN']]*(Model['z_max_MLN'] + 2)]*(Model['y_max_MLN'] + 3)]*(Model['x_max_MLN'] + 2))
    Model['A_Stro_MLN'] = np.array([[[Model['Stromal_cells']]*(Model['z_max_MLN'] + 2)]*(Model['y_max_MLN'] + 3)]*(Model['x_max_MLN'] + 2))

#This function creates arrays with initial populations for each agents and cells, basically, all induced immune cells (Tcells) are set up at 0.
def Initial_Population(Model, dt):
    #Agents in the Lamina Propria
    
    
    Model['A_Mon_LP'] = np.random.rand(Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Lamina_Propria_Complementary']:
        Model['A_Mon_LP'][x, y, z] = 0
    Model['A_Mon_LP'] = np.copy(Model['A_Mon_LP'] * Model['D_max_LP']/100)  
    
    Model['A_Mac_LP'] = np.random.rand(Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Lamina_Propria_Complementary']:
        Model['A_Mac_LP'][x, y, z] = 0
    Model['A_Mac_LP'] = np.copy(Model['A_Mac_LP'] * Model['D_max_LP']/100)
    
    Model['A_ImDC_LP'] = np.random.rand(Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Lamina_Propria_Complementary']:
        Model['A_ImDC_LP'][x, y, z] = 0
    Model['A_ImDC_LP'] = np.copy(Model['A_ImDC_LP'] * Model['D_max_LP']/2)
    
    Model['A_eDC_LP'] = np.random.rand(Model['N_chemical'] + 1, Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Lamina_Propria_Complementary']:
        Model['A_eDC_LP'][:, x, y, z] = 0
    Model['A_eDC_LP'] = np.copy(Model['A_eDC_LP']*0 * (Model['D_max_LP']/100) / (Model['N_chemical'] + 1))
    #Model['A_eDC_LP'][:, :, :, 4] = 0
    
    Model['A_tDC_LP'] = np.random.rand(Model['N_chemical'] + 1, Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Lamina_Propria_Complementary']:
        Model['A_tDC_LP'][:, x, y, z] = 0 
    Model['A_tDC_LP'] = np.copy(Model['A_tDC_LP']*0 * (Model['D_max_LP']/100) / (Model['N_chemical'] + 1))
    #Model['A_tDC_LP'][:, :, :, 9] = 0
    
    Model['A_TH17_LP'] = np.random.rand(Model['N_chemical'] + 1, Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Lamina_Propria_Complementary']:
        Model['A_TH17_LP'][:, x, y, z] = 0
    Model['A_TH17_LP'] = np.copy(0*Model['A_TH17_LP'] * (Model['D_max_LP']/100) / (Model['N_chemical'] + 1))
    #Model['A_TH17_LP'][:, :, :, 4] = 0
    
    Model['A_Treg_LP'] = np.random.rand(Model['N_chemical'] + 1, Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Lamina_Propria_Complementary']:
        Model['A_Treg_LP'][:, x, y, z] = 0
    Model['A_Treg_LP'] = np.copy(0*Model['A_Treg_LP'] * (Model['D_max_LP']/100) / (Model['N_chemical'] + 1))
    #Model['A_Treg_LP'][:, :, :, 9] = 0
    
    Model['A_sTH17_LP'] = np.random.rand(Model['N_chemical'] + 1, Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Lamina_Propria_Complementary']:
        Model['A_sTH17_LP'][:, x, y, z] = 0
    Model['A_sTH17_LP'] = np.copy(0*Model['A_sTH17_LP'] * (Model['D_max_LP']/100) / (Model['N_chemical'] + 1))
    #Model['A_sTH17_LP'][:, :, :, 4] = 0
    
    Model['A_sTreg_LP'] = np.random.rand(Model['N_chemical'] + 1, Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Lamina_Propria_Complementary']:
        Model['A_sTreg_LP'][:, x, y, z] = 0
    Model['A_sTreg_LP'] = np.copy(0*Model['A_sTreg_LP'] * (Model['D_max_LP']/100) / (Model['N_chemical'] + 1))
    #Model['A_sTreg_LP'][:, :, :, 9] = 0
     
    Model['A_eAg_Mac_LP'] = np.random.rand(Model['N_chemical'] + 1, Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3) /1.11111 + 0.1 
    for (x, y, z) in Model['Volume_Lamina_Propria_Complementary']:
        Model['A_eAg_Mac_LP'][:, x, y, z] = 0
    Model['A_eAg_Mac_LP'] = np.copy(0*Model['A_eAg_Mac_LP'] * (Model['D_max_LP']/400) / (Model['N_chemical'] + 1))
    
    Model['A_tAg_Mac_LP'] = np.random.rand(Model['N_chemical'] + 1, Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Lamina_Propria_Complementary']:
        Model['A_tAg_Mac_LP'][:, x, y, z] = 0
    Model['A_tAg_Mac_LP'] = np.copy(0*Model['A_tAg_Mac_LP'] * (Model['D_max_LP']/400) / (Model['N_chemical'] + 1)) 
    
    # Model['A_eAg'] = np.random.rand(Model['N_chemical'] + 1, Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3)  /1.11111 + 0.1
    # for (x, y, z) in Model['Volume_Lamina_Propria']:
    #     Model['A_eAg'][:, x, y, z] = 0
    # Model['A_eAg'] = np.copy(0*Model['A_eAg'] / (Model['N_chemical'] + 1))
    
    # Model['A_tAg'] = np.random.rand(Model['N_chemical'] + 1, Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3) /1.11111 + 0.1
    # for (x, y, z) in Model['Volume_Lamina_Propria']:
    #     Model['A_tAg'][:, x, y, z] = 0
    # Model['A_tAg'] = np.copy(0*Model['A_tAg'] / (Model['N_chemical'] + 1)) 
    
    Model['F_tot_LP'] = [[[[float(0)]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2)]*(Model['N_chemical'] + 1)
    Model['F_tot_LP'] = np.copy(Model['F_tot_LP'])
     
    Model['S_TH17_LP_ini'] = np.random.rand(int(Model['delta_mTH17'] / dt) + 1, Model['N_chemical'] + 1) /1.11111 + 0.1
    Model['S_TH17_LP_ini'] = np.copy(Model['S_TH17_LP_ini'] * 0 / (Model['N_chemical'] + 1))
    Model['S_TH17_LP'] = np.copy(Model['S_TH17_LP_ini'][-1])
    VS_TH17_LP = np.copy(Model['S_TH17_LP_ini'][-1])
    
    Model['S_Treg_LP_ini'] = np.random.rand(int(Model['delta_mTreg'] / dt) + 1, Model['N_chemical'] + 1) /1.11111 + 0.1
    Model['S_Treg_LP_ini'] = np.copy(Model['S_Treg_LP_ini'] * 0 / (Model['N_chemical'] + 1))
    Model['S_Treg_LP'] = np.copy(Model['S_Treg_LP_ini'][-1])
    VS_Treg_LP = np.copy(Model['S_Treg_LP_ini'][-1])
    
    L_out_TH17 = []
    L_out_Treg = []
    
    
    #Agents in the Epithelium
    
    Model['A_sEpi'] = np.random.rand(Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3)/1.25 + 0.2
    Model['A_Epi'] = np.random.rand(Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3)/1.25 + 0.2
    Model['A_eAg_Epi'] = np.random.rand(Model['N_chemical'] + 1, Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3)
    Model['A_tAg_Epi'] = np.random.rand(Model['N_chemical'] + 1, Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3)
    Model['A_eAg_sEpi'] = np.random.rand(Model['N_chemical'] + 1, Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3)
    Model['A_tAg_sEpi'] = np.random.rand(Model['N_chemical'] + 1, Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3)
       
    
    for (x, y, z) in Model['Volume_Epithelium_Complementary']:
        Model['A_Epi'][x, y, z] = 0
        Model['A_eAg_Epi'][:, x, y, z] = 0
        Model['A_tAg_Epi'][:, x, y, z] = 0
    for (x, y, z) in Model['Volume_Crypt_Complementary']:
        Model['A_sEpi'][x, y, z] = 0
        Model['A_eAg_sEpi'][:, x, y, z] = 0
        Model['A_tAg_sEpi'][:, x, y, z] = 0
    for (x, y, z) in Model['Volume_Crypt']:
        Model['A_Epi'][x, y, z] = 0.5
        Model['A_eAg_Epi'][:, x, y, z] = 0
        Model['A_tAg_Epi'][:, x, y, z] = 0
        
    Model['A_sEpi'] = np.copy(Model['A_sEpi'] * Model['D_max_Epi']*Model['Crypt_portion']*0.1)
    Model['A_Epi'] = np.copy(Model['A_Epi'] * Model['D_max_Epi'] * Model['N_Epi'])
    Model['A_eAg_Epi'] = np.copy(Model['A_eAg_Epi'] * 0)
    Model['A_tAg_Epi'] = np.copy(Model['A_tAg_Epi'] * 0)
    Model['A_eAg_sEpi'] = np.copy(Model['A_eAg_sEpi'] * 0)
    Model['A_tAg_sEpi'] = np.copy(Model['A_tAg_sEpi'] * 0)

    Model['Storage_IL'] = [0]*len(Model['Volume_Crypt'])
    
    
    #Agents in the Mesenteric Lymph Node
    
    
    Model['A_eDC_MLN'] = np.random.rand(Model['N_chemical'] + 1, Model['x_max_MLN'] + 2, Model['y_max_MLN'] + 3, Model['z_max_MLN'] +2) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Complementary_MLN']:
        Model['A_eDC_MLN'][:, x, y, z] = 0
    Model['A_eDC_MLN'] = np.copy(Model['A_eDC_MLN']*0 * (Model['D_max_MLN']/400) / (Model['N_chemical'] + 1))
    #Model['A_eDC_MLN'][:, :, :, 5:] = 0
    
    Model['A_tDC_MLN'] = np.random.rand(Model['N_chemical'] + 1, Model['x_max_MLN'] + 2, Model['y_max_MLN'] + 3, Model['z_max_MLN'] +2) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Complementary_MLN']:
        Model['A_tDC_MLN'][:, x, y, z] = 0
    Model['A_tDC_MLN'] = np.copy(Model['A_tDC_MLN']*0 * (Model['D_max_MLN']/400) / (Model['N_chemical'] + 1))
    #Model['A_tDC_MLN'][:, :, :, :5] = 0
    
    Model['A_nTc_MLN'] = np.random.rand(Model['N_chemical'] + 1, Model['x_max_MLN'] + 2, Model['y_max_MLN'] + 3, Model['z_max_MLN'] +2) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Complementary_MLN']:
        Model['A_nTc_MLN'][:, x, y, z] = 0
    for x in range(0, Model['x_max_MLN'] + 1):
        for y in range(2, 7):
            for z in range(0, Model['z_max_MLN'] + 1):
                Model['A_nTc_MLN'][:, x, y, z] = Model['A_nTc_MLN'][:, x, y, z] * (Model['D_max_MLN']*0/100) / (Model['N_chemical'] + 1)
      
    Model['A_nTe_MLN'] = np.random.rand(Model['N_chemical'] + 1, Model['x_max_MLN'] + 2, Model['y_max_MLN'] + 3, Model['z_max_MLN'] +2) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Complementary_MLN']:
        Model['A_nTe_MLN'][:, x, y, z] = 0
    Model['A_nTe_MLN'] = np.copy(Model['A_nTe_MLN']*0 * (Model['D_max_MLN']/400) / (Model['N_chemical'] + 1))
    #Model['A_nTe_MLN'][:, :, :, 4] = 0

    
    Model['A_nTt_MLN'] = np.random.rand(Model['N_chemical'] + 1, Model['x_max_MLN'] + 2, Model['y_max_MLN'] + 3, Model['z_max_MLN'] +2) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Complementary_MLN']:
        Model['A_nTt_MLN'][:, x, y, z] = 0
    Model['A_nTt_MLN'] = np.copy(Model['A_nTt_MLN']*0 * (Model['D_max_MLN']/400) / (Model['N_chemical'] + 1))
    #Model['A_nTt_MLN'][:, :, :, 9] = 0
    
    Model['A_TH17_MLN'] = np.random.rand(Model['N_chemical'] + 1, Model['x_max_MLN'] + 2, Model['y_max_MLN'] + 3, Model['z_max_MLN'] +2) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Complementary_MLN']:
        Model['A_TH17_MLN'][:, x, y, z] = 0
    Model['A_TH17_MLN'] = np.copy(Model['A_TH17_MLN']*0 * (Model['D_max_MLN']/400) / (Model['N_chemical'] + 1))
    #Model['A_TH17_MLN'][:, :, :, 4] = 0
    
    Model['A_fTH17_MLN'] = np.random.rand(Model['N_chemical'] + 1, Model['x_max_MLN'] + 2, Model['y_max_MLN'] + 3, Model['z_max_MLN'] +2) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Complementary_MLN']:
        Model['A_fTH17_MLN'][:, x, y, z] = 0
    Model['A_fTH17_MLN'] = np.copy(Model['A_fTH17_MLN']*0 * (Model['D_max_MLN']/400) / (Model['N_chemical'] + 1))
    #Model['A_fTH17_MLN'][:, :, :, 4] = 0

    Model['A_mTH17_MLN'] = np.random.rand(Model['N_chemical'] + 1, Model['x_max_MLN'] + 2, Model['y_max_MLN'] + 3, Model['z_max_MLN'] +2) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Complementary_MLN']:
        Model['A_mTH17_MLN'][:, x, y, z] = 0
    Model['A_mTH17_MLN'] = np.copy(Model['A_mTH17_MLN']*0 * (Model['D_max_MLN']/400) / (Model['N_chemical'] + 1))
    #Model['A_mTH17_MLN'][:, :, :, 4] = 0
    
    Model['A_Treg_MLN'] = np.random.rand(Model['N_chemical'] + 1, Model['x_max_MLN'] + 2, Model['y_max_MLN'] + 3, Model['z_max_MLN'] +2) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Complementary_MLN']:
        Model['A_Treg_MLN'][:, x, y, z] = 0
    Model['A_Treg_MLN'] = np.copy(Model['A_Treg_MLN']*0 * (Model['D_max_MLN']/400) / (Model['N_chemical'] + 1))
    #Model['A_Treg_MLN'][:, :, :, 9] = 0
    
    Model['A_fTreg_MLN'] = np.random.rand(Model['N_chemical'] + 1, Model['x_max_MLN'] + 2, Model['y_max_MLN'] + 3, Model['z_max_MLN'] +2) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Complementary_MLN']:
        Model['A_fTreg_MLN'][:, x, y, z] = 0
    Model['A_fTreg_MLN'] = np.copy(Model['A_fTreg_MLN']*0 * (Model['D_max_MLN']/400) / (Model['N_chemical'] + 1))
    #Model['A_fTreg_MLN'][:, :, :, 9] = 0
     
    Model['A_mTreg_MLN'] = np.random.rand(Model['N_chemical'] + 1, Model['x_max_MLN'] + 2, Model['y_max_MLN'] + 3, Model['z_max_MLN'] +2) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Complementary_MLN']:
        Model['A_mTreg_MLN'][:, x, y, z] = 0
    Model['A_mTreg_MLN'] = np.copy(Model['A_mTreg_MLN']*0 * (Model['D_max_MLN']/400) / (Model['N_chemical'] + 1)) 
    #Model['A_mTreg_MLN'][:, :, :, 9] = 0
     
    F_eDC_MLN = [[[[float(0)]*(Model['z_max_MLN'] + 2)]*(Model['y_max_MLN'] + 3)]*(Model['x_max_MLN'] + 2)]*(Model['N_chemical'] + 1)
    F_eDC_MLN = np.copy(F_eDC_MLN)
    
    F_tDC_MLN = [[[[float(0)]*(Model['z_max_MLN'] + 2)]*(Model['y_max_MLN'] + 3)]*(Model['x_max_MLN'] + 2)]*(Model['N_chemical'] + 1)
    F_tDC_MLN = np.copy(F_tDC_MLN)
    
    Model['S_eDC_MLN_ini'] = np.random.rand(int(Model['delta_DC'] / dt) + 1, Model['N_chemical'] + 1) /1.11111 + 0.1
    Model['S_eDC_MLN_ini'] = np.copy(Model['S_eDC_MLN_ini'] * 0 / (Model['N_chemical'] + 1)) 
    Model['S_eDC_MLN'] = np.copy(Model['S_eDC_MLN_ini'][-1])
    VS_eDC_MLN = np.copy(Model['S_eDC_MLN_ini'][-1])
    
    Model['S_tDC_MLN_ini'] = np.random.rand(int(Model['delta_DC'] / dt) + 1, Model['N_chemical'] + 1)  /1.11111 + 0.1
    Model['S_tDC_MLN_ini'] = np.copy(Model['S_tDC_MLN_ini'] * 0 / (Model['N_chemical'] + 1))
    Model['S_tDC_MLN'] = np.copy((Model['S_tDC_MLN_ini'][-1]))
    VS_tDC_MLN = np.copy((Model['S_tDC_MLN_ini'][-1]))
    
    L_out_eDC = []
    L_out_tDC = []
    
    
    #Soluble Mediators in the Lamina Propria
    
    
    Model['C_IL10_LP'] = np.random.rand(Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Lamina_Propria_Epi_Complementary']:
        Model['C_IL10_LP'][x, y, z] = 0
    Model['C_IL10_LP'] = np.copy(Model['C_IL10_LP'] * 0)
     
    Model['C_TGFBeta_LP'] = np.random.rand(Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Lamina_Propria_Epi_Complementary']:
        Model['C_TGFBeta_LP'][x, y, z] = 0
    Model['C_TGFBeta_LP'] = np.copy(Model['C_TGFBeta_LP'] * 0)
    
    Model['C_IL22_LP'] = np.random.rand(Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Lamina_Propria_Epi_Complementary']:
        Model['C_IL22_LP'][x, y, z] = 0
    Model['C_IL22_LP'] = np.copy(Model['C_IL22_LP'] * 0)
    
    Model['C_IL2_LP'] = np.random.rand(Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Lamina_Propria_Epi_Complementary']:
        Model['C_IL2_LP'][x, y, z] = 0
    Model['C_IL2_LP'] = np.copy(Model['C_IL2_LP'] * 0)
    
    Model['C_IL17_LP'] = np.random.rand(Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Lamina_Propria_Epi_Complementary']:
        Model['C_IL17_LP'][x, y, z] = 0
    Model['C_IL17_LP'] = np.copy(Model['C_IL17_LP'] * 0)
    
    Model['C_Csf2_LP'] = np.random.rand(Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Lamina_Propria_Epi_Complementary']:
        Model['C_Csf2_LP'][x, y, z] = 0
    Model['C_Csf2_LP'] = np.copy(Model['C_Csf2_LP'] * 0)
    
    Model['C_IL1Beta_23_LP'] = np.random.rand(Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Lamina_Propria_Epi_Complementary']:
        Model['C_IL1Beta_23_LP'][x, y, z] = 0
    Model['C_IL1Beta_23_LP'] = np.copy(Model['C_IL1Beta_23_LP'] * 0)
    
    Model['C_CCL2_LP'] = np.random.rand(Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Lamina_Propria_Epi_Complementary']:
        Model['C_CCL2_LP'][x, y, z] = 0
    Model['C_CCL2_LP'] = np.copy(Model['C_CCL2_LP'] * 0)
    
    Model['C_CCL25_LP'] = np.random.rand(Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Lamina_Propria_Epi_Complementary']:
        Model['C_CCL25_LP'][x, y, z] = 0
    Model['C_CCL25_LP'] = np.copy(Model['C_CCL25_LP'] * 0)
    
    Model['C_SAA_LP'] = np.random.rand(Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Lamina_Propria_Epi_Complementary']:
        Model['C_SAA_LP'][x, y, z] = 0
    Model['C_SAA_LP'] = np.copy(Model['C_SAA_LP'] * 0)
    
    Model['C_eRA_LP'] = np.random.rand(Model['N_chemical'] + 1, Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Lamina_Propria_Epi_Complementary']:
        for i in range(Model['N_chemical'] + 1):
            Model['C_eRA_LP'][i, x, y, z] = 0
    Model['C_eRA_LP'] = np.copy(Model['C_eRA_LP'] * 0)
    
    Model['C_tRA_LP'] = np.random.rand(Model['N_chemical'] + 1, Model['x_max_LP'] + 2, Model['y_max_LP'] + 2, Model['z_max_LP'] + 3) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Lamina_Propria_Epi_Complementary']:
        for i in range(Model['N_chemical'] + 1):
            Model['C_tRA_LP'][i, x, y, z] = 0
    Model['C_tRA_LP'] = np.copy(Model['C_tRA_LP'] * 0)
    
     
    #Soluble Mediators in the Mesenteric Lymph Node
    
    
    Model['C_IL21_MLN'] = np.random.rand(Model['x_max_MLN'] + 2, Model['y_max_MLN'] + 3, Model['z_max_MLN'] + 2) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Complementary_MLN']:
        Model['C_IL21_MLN'][x, y, z] = 0
    Model['C_IL21_MLN'] = np.copy(Model['C_IL21_MLN'] * 0)
    
    Model['C_IL2_MLN'] = np.random.rand(Model['x_max_MLN'] + 2, Model['y_max_MLN'] + 3, Model['z_max_MLN'] + 2) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Complementary_MLN']:
        Model['C_IL2_MLN'][x, y, z] = 0
    Model['C_IL2_MLN'] = np.copy(Model['C_IL2_MLN'] * 0)
    
    Model['C_IL10_MLN'] = np.random.rand(Model['x_max_MLN'] + 2, Model['y_max_MLN'] + 3, Model['z_max_MLN'] + 2) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Complementary_MLN']:
        Model['C_IL10_MLN'][x, y, z] = 0
    Model['C_IL10_MLN'] = np.copy(Model['C_IL10_MLN'] * 0)
    
    Model['C_IL6_MLN'] = np.random.rand(Model['x_max_MLN'] + 2, Model['y_max_MLN'] + 3, Model['z_max_MLN'] + 2) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Complementary_MLN']:
        Model['C_IL6_MLN'][x, y, z] = 0
    Model['C_IL6_MLN'] = np.copy(Model['C_IL6_MLN'] * 0)
    
    Model['C_TGFBeta_MLN'] = np.random.rand(Model['x_max_MLN'] + 2, Model['y_max_MLN'] + 3, Model['z_max_MLN'] + 2) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Complementary_MLN']:
        Model['C_TGFBeta_MLN'][x, y, z] = 0
    Model['C_TGFBeta_MLN'] = np.copy(Model['C_TGFBeta_MLN'] * 0)
    
    Model['C_eRA_MLN'] = np.random.rand(Model['N_chemical'] + 1, Model['x_max_MLN'] + 2, Model['y_max_MLN'] + 3, Model['z_max_MLN'] + 2) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Complementary_MLN']:
        for i in range(Model['N_chemical'] + 1):
            Model['C_eRA_MLN'][i, x, y, z] = 0
    Model['C_eRA_MLN'] = np.copy(Model['C_eRA_MLN'] * 0)
    
    Model['C_tRA_MLN'] = np.random.rand(Model['N_chemical'] + 1, Model['x_max_MLN'] + 2, Model['y_max_MLN'] + 3, Model['z_max_MLN'] + 2) /1.11111 + 0.1
    for (x, y, z) in Model['Volume_Complementary_MLN']:
        for i in range(Model['N_chemical'] + 1):
            Model['C_tRA_MLN'][i, x, y, z] = 0
    Model['C_tRA_MLN'] = np.copy(Model['C_tRA_MLN'] * 0)
    
    return(VS_TH17_LP, VS_Treg_LP, L_out_TH17, L_out_Treg, VS_eDC_MLN, VS_tDC_MLN, L_out_eDC, L_out_tDC)

#This function creates arrays for the movement and diffusion of agents and soluble mediators    
def Diffusion_Maps(Model):
    #Agents in the Lamina Propria
            
    Model['MapMov_Mac_LP'] = np.array([[[Model['M_Mac_LP']]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2))
    Model['MapMov_Mon_LP'] = np.array([[[Model['M_Mon_LP_0']]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2))
    Model['MapMov_ImDC_LP'] = np.array([[[Model['M_ImDC_LP']]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2))
    Model['MapMov_Treg_LP'] = np.array([[[Model['M_Treg_LP_0']]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2))
    Model['MapMov_TH17_LP'] = np.array([[[Model['M_TH17_LP_0']]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2))
    Model['MapMov_sTreg_LP'] = np.array([[[Model['M_sTreg_LP_0']]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2))
    Model['MapMov_sTH17_LP'] = np.array([[[Model['M_sTH17_LP_0']]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2))
    Model['MapMov_eDC_LP'] = np.array([[[Model['M_eDC_LP']]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2))
    Model['MapMov_tDC_LP'] = np.array([[[Model['M_tDC_LP']]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2))
    
    for (x, y, z) in Model['Volume_Lamina_Propria_Complementary']:
        Model['MapMov_Mac_LP'][x, y, z] = 0
        Model['MapMov_Mon_LP'][x, y, z] = 0
        Model['MapMov_ImDC_LP'][x, y, z] = 0
        Model['MapMov_Treg_LP'][x, y, z] = 0
        Model['MapMov_TH17_LP'][x, y, z] = 0
        Model['MapMov_sTreg_LP'][x, y, z] = 0
        Model['MapMov_sTH17_LP'][x, y, z] = 0
        Model['MapMov_eDC_LP'][x, y, z] = 0
        Model['MapMov_tDC_LP'][x, y, z] = 0
        
    #Agents in the Mesenteric Lymph Node
        
    Model['MapMov_DC_MLN'] = np.array([[[Model['M_DC_MLN']]*(Model['z_max_MLN'] + 2)]*(Model['y_max_MLN'] + 3)]*(Model['x_max_MLN'] + 2))
    Model['MapMov_nTc_MLN'] = np.array([[[Model['M_nTc_MLN']]*(Model['z_max_MLN'] + 2)]*(Model['y_max_MLN'] + 3)]*(Model['x_max_MLN'] + 2))
    Model['MapMov_TH17_MLN'] = np.array([[[Model['M_TH17_MLN']]*(Model['z_max_MLN'] + 2)]*(Model['y_max_MLN'] + 3)]*(Model['x_max_MLN'] + 2))
    Model['MapMov_Treg_MLN'] = np.array([[[Model['M_Treg_MLN']]*(Model['z_max_MLN'] + 2)]*(Model['y_max_MLN'] + 3)]*(Model['x_max_MLN'] + 2))
    Model['MapMov_mTH17_MLN'] = np.array([[[Model['M_mTH17_MLN']]*(Model['z_max_MLN'] + 2)]*(Model['y_max_MLN'] + 3)]*(Model['x_max_MLN'] + 2))
    Model['MapMov_mTreg_MLN'] = np.array([[[Model['M_mTreg_MLN']]*(Model['z_max_MLN'] + 2)]*(Model['y_max_MLN'] + 3)]*(Model['x_max_MLN'] + 2))
    
    for (x, y, z) in Model['Volume_Complementary_MLN']:
        Model['MapMov_DC_MLN'][x, y, z] = 0
        Model['MapMov_nTc_MLN'][x, y, z] = 0
        Model['MapMov_TH17_MLN'][x, y, z] = 0
        Model['MapMov_Treg_MLN'][x, y, z] = 0
        Model['MapMov_mTH17_MLN'][x, y, z] = 0
        Model['MapMov_mTreg_MLN'][x, y, z] = 0
        
    for x in range(0, Model['x_max_MLN'] + 1):
        for y in range(0, Model['y_max_MLN'] + 1):
            for z in range(0, Model['z_max_MLN'] + 1):
                Model['MapMov_DC_MLN'][x, y, z] = Model['MapMov_DC_MLN'][x, y, z]
                Model['MapMov_mTH17_MLN'][x, y, z] = Model['MapMov_mTH17_MLN'][x, y, z] * y*10
                Model['MapMov_mTreg_MLN'][x, y, z] = Model['MapMov_mTreg_MLN'][x, y, z] * y*10
        
    #Soluble Mediators in the Lamina Propria/Epithelium
        
    Model['MapDiff_IL10_LP'] = np.array([[[Model['d_IL10_LP']]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2))
    Model['MapDiff_TGFBeta_LP'] = np.array([[[Model['d_TGFBeta_LP']]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2))
    Model['MapDiff_IL17_LP'] = np.array([[[Model['d_IL17_LP']]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2))
    Model['MapDiff_IL22_LP'] = np.array([[[Model['d_IL22_LP']]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2))
    Model['MapDiff_Csf2_LP'] = np.array([[[Model['d_Csf2_LP']]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2))
    Model['MapDiff_CCL2_LP'] = np.array([[[Model['d_CCL2_LP']]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2))
    Model['MapDiff_CCL25_LP'] = np.array([[[Model['d_CCL25_LP']]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2))
    Model['MapDiff_IL2_LP'] = np.array([[[Model['d_IL2_LP']]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2))
    Model['MapDiff_IL1Beta_23_LP'] = np.array([[[Model['d_IL1Beta_23_LP']]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2))
    Model['MapDiff_SAA_LP'] = np.array([[[Model['d_SAA_LP']]*(Model['z_max_LP'] + 3)]*(Model['y_max_LP'] + 2)]*(Model['x_max_LP'] + 2))
    
    for (x, y, z) in Model['Volume_Lamina_Propria_Complementary']:
        Model['MapDiff_IL10_LP'][x, y, z] = 0
        Model['MapDiff_TGFBeta_LP'][x, y, z] = 0
        Model['MapDiff_IL17_LP'][x, y, z] = 0
        Model['MapDiff_IL22_LP'][x, y, z] = 0
        Model['MapDiff_Csf2_LP'][x, y, z] = 0
        Model['MapDiff_CCL2_LP'][x, y, z] = 0
        Model['MapDiff_CCL25_LP'][x, y, z] = 0
        Model['MapDiff_IL2_LP'][x, y, z] = 0
        Model['MapDiff_IL1Beta_23_LP'][x, y, z] = 0
        Model['MapDiff_SAA_LP'][x, y, z] = 0
        
    
    #Soluble Mediators in the Mesenteric Lymph Node
    
    Model['MapDiff_IL2_MLN'] = np.array([[[Model['d_IL2_MLN']]*(Model['z_max_MLN'] + 2)]*(Model['y_max_MLN'] + 3)]*(Model['x_max_MLN'] + 2))
    Model['MapDiff_IL6_MLN'] = np.array([[[Model['d_IL6_MLN']]*(Model['z_max_MLN'] + 2)]*(Model['y_max_MLN'] + 3)]*(Model['x_max_MLN'] + 2))
    Model['MapDiff_IL21_MLN'] = np.array([[[Model['d_IL21_MLN']]*(Model['z_max_MLN'] + 2)]*(Model['y_max_MLN'] + 3)]*(Model['x_max_MLN'] + 2))
    Model['MapDiff_IL10_MLN'] = np.array([[[Model['d_IL10_MLN']]*(Model['z_max_MLN'] + 2)]*(Model['y_max_MLN'] + 3)]*(Model['x_max_MLN'] + 2))
    Model['MapDiff_TGFBeta_MLN'] = np.array([[[Model['d_TGFBeta_MLN']]*(Model['z_max_MLN'] + 2)]*(Model['y_max_MLN'] + 3)]*(Model['x_max_MLN'] + 2))
    
    for (x, y, z) in Model['Volume_Complementary_MLN']:
        Model['MapDiff_IL2_MLN'][x, y, z] = 0
        Model['MapDiff_IL6_MLN'][x, y, z] = 0
        Model['MapDiff_IL21_MLN'][x, y, z] = 0
        Model['MapDiff_IL10_MLN'][x, y, z] = 0
        Model['MapDiff_TGFBeta_MLN'][x, y, z] = 0
    
    
    for z in range(Model['z_enter1_MLN'], Model['z_enter2_MLN'] + 1):         #Open boundary condition 
        for x in range(Model['x_enter1_MLN'], Model['x_enter2_MLN'] + 1):     #Lymphatic and blood vessels
            Model['MapDiff_IL2_MLN'][x, y, z] = 3.8
            Model['MapDiff_IL6_MLN'][x, y, z] = 3.8
            Model['MapDiff_IL21_MLN'][x, y, z] = 3.8
            Model['MapDiff_IL10_MLN'][x, y, z] = 3.8
            Model['MapDiff_TGFBeta_MLN'][x, y, z] = 3.8
            
def Overpopulation_Test(Model, dt):
    ###############################################################################
    #Condition for Negative Populations
    ###############################################################################
    
    #Agents Local Lamina Propria
    if dt*Model['mu_Mac'] >= 1:
        print("error over Mac")
    if dt*(Model['phi_Mon_Mac'] * Model['Gamma_TGFBeta_Mon'] + Model['mu_Mon']) >= 1:
        print("error over Mon")
    if dt*(Model['phi_ImDC_eDC_Ag_Mac'] * Model['D_max_LP'] + Model['mu_Mac'] + Model['mu_eAg_Mac']) >= 1:
        print("error over eAg")
    if dt*(Model['phi_ImDC_tDC_Ag_Mac'] * Model['D_max_LP'] + Model['mu_Mac'] + Model['mu_tAg_Mac']) >= 1:
        print("error over tAg")
    if dt*((Model['phi_ImDC_eDC_Ag_Mac'] + Model['phi_ImDC_tDC_Ag_Mac']) * Model['D_max_LP'] * Model['n_Mac_0'] +  (Model['phi_ImDC_eDC_Ag_Epi'] + Model['phi_ImDC_tDC_Ag_Epi'] + Model['phi_ImDC_eDC_Ag_sEpi'] + Model['phi_ImDC_tDC_Ag_sEpi']) * Model['D_max_LP'] * Model['n_Epi'] + Model['mu_ImDC']) >= 1:
        print("error over ImDC")
    if dt*(Model['M_outLP_eDC'] + Model['mu_DC_LP']) >= 1:
        print("error over eDC")
    if dt*(Model['M_outLP_tDC'] + Model['mu_DC_LP']) >= 1:
        print("error over tDC")
    if dt*(Model['phi_TH17_Treg_TGFBeta'] * Model['Gamma_TGFBeta_TH17'] + Model['phi_TH17_Treg_RA'] * ((Model['Gamma_eRA_TH17'] + Model['Gamma_tRA_TH17']) * Model['n_DC'] * Model['D_max_LP']) + Model['phi_TH17_Treg_RA_Epi'] * (max(Model['Gamma_eRA_Epi_TH17'], Model['Gamma_tRA_Epi_TH17']) * Model['n_Epi'] * 5 * Model['D_max_Epi']) + Model['phi_TH17_Treg_RA_sEpi'] * (max(Model['Gamma_eRA_sEpi_TH17'], Model['Gamma_tRA_sEpi_TH17']) * Model['n_Epi'] * Model['D_max_Epi'] * 5) + Model['phi_TH17_sTH17_IL1Beta_23'] * Model['Gamma_IL1Beta_23_TH17'] + Model['mu_TH17_LP']) >= 1:
        print("error over TH17 in LP")
    if dt*(Model['phi_Treg_sTreg_TGFBeta'] * Model['Gamma_TGFBeta_Treg'] + Model['phi_Treg_sTreg_RA'] * ((Model['Gamma_eRA_Treg'] + Model['Gamma_tRA_Treg']) * Model['n_DC'] * Model['D_max_LP']) + Model['phi_Treg_sTreg_RA_Epi'] * (max(Model['Gamma_eRA_Epi_Treg'], Model['Gamma_tRA_Epi_Treg']) * Model['n_Epi'] * 5 * Model['D_max_Epi']) + Model['phi_Treg_sTreg_RA_sEpi'] * (max(Model['Gamma_eRA_sEpi_Treg'], Model['Gamma_tRA_sEpi_Treg']) * Model['n_Epi'] * Model['D_max_Epi'] * 5) + Model['phi_Treg_TH17'] * Model['Gamma_IL1Beta_23_TH17'] + Model['mu_Treg_LP']) >= 1:
        print("error over Treg in LP")    
    if dt*Model['mu_sTH17_LP'] >= 1:
        print("error over sTH17")
    if dt*Model['mu_sTreg_LP'] >= 1:
        print("error over sTreg")
    if dt*(Model['M_inLP_mTH17'] * Model['Gamma_CCL25_sTH17'] + Model['mu_mTH17_T']) >= 1:
        print("error over waiting TH17")
    if dt*(Model['M_inLP_mTreg'] * Model['Gamma_CCL25_sTreg'] + Model['mu_mTreg_T']) >= 1:
        print("error over waiting Treg")
    
    #Agents Local Epithelium
    if dt*(Model['mu_Epi_0'] + Model['mu_Ejec']) >= 1:
        print("error over Epi")
    if dt*(Model['mu_sEpi_0']) >= 1:
        print("error over sEpi")
        
    #Agents Local Mesenteric Lymph Nodes
    if dt*(Model['xi_nTc_DC_0'] + Model['mu_nTc']) >= 1:
        print("error over nTc")
    if dt*Model['mu_DC_MLN'] >= 1:
        print("error over eDC MLN")
    if dt*Model['mu_DC_MLN'] >= 1:
        print("error over tDC MLN")
    if dt*(Model['phi_Tcell'] * Model['Gamma_TGFBeta_nTe'] + Model['mu_nTe'] + Model['mu_DC_MLN']) >= 1:
        print("error over nTe")
    if dt*(Model['phi_Tcell'] * Model['Gamma_TGFBeta_nTt'] + Model['mu_nTt'] + Model['mu_DC_MLN']) >= 1:
        print("error over nTt")
    if dt*(Model['barXi_DC_fTH17'] + Model['mu_fTH17'] + Model['mu_DC_MLN']) >= 1:
        print("error over fTH17 in MLN")
    if dt*(Model['barXi_DC_fTreg'] + Model['mu_fTreg'] + Model['mu_DC_MLN']) >= 1:
        print("error over fTreg in MLN")
    if dt*(Model['xi_TH17_DC_0'] + Model['mu_TH17']) >= 1:
        print("error over TH17 in MLN")
    if dt*(Model['xi_Treg_DC_0'] + Model['mu_Treg']) >= 1:
        print("error over Treg in MLN")
    if dt*(Model['mu_mTH17'] + Model['M_outMLN_mTH17']) >= 1:
        print("error over mTH17")
    if dt*(Model['mu_mTreg'] + Model['M_outMLN_mTreg']) >= 1:
        print("error over mTreg")
    if dt*(Model['M_inMLN_eDC'] + Model['mu_DC_T']) >= 1:
        print("error over waiting eDC")
    if dt*(Model['M_inMLN_tDC'] + Model['mu_DC_T']) >= 1:
        print("error over waiting tDC")
    if dt*(Model['D_max_MLN'] * max(Model['xi_nTc_DC']) * Model['xi_nTc_DC_0'] / Model['K_nTc']) >= 1:
        print("error over F_eDC")
    if dt*(Model['D_max_MLN'] * max(Model['xi_nTc_DC']) * Model['xi_nTc_DC_0'] / Model['K_nTc']) >= 1:
        print("error over F_tDC")
    
    
    #Movements and Diffusion
    if max(np.max(Model['MapMov_eDC_LP']), np.max(Model['MapMov_tDC_LP']), np.max(Model['MapMov_ImDC_LP']), np.max(Model['MapMov_Treg_LP']) * (1 + Model['M_Treg_chem'] * Model['Gamma_CCL25_Treg']), np.max(Model['MapMov_Mac_LP']), np.max(Model['MapMov_Mon_LP']) * (1 + Model['M_Mon_chem'] * Model['Gamma_CCL2_Mon']), np.max(Model['MapMov_TH17_LP']) * (1 + Model['M_TH17_chem'] * Model['Gamma_CCL25_TH17']), np.max(Model['MapMov_sTreg_LP']) * (1 + Model['M_sTreg_chem'] * Model['Gamma_CCL25_sTreg']), np.max(Model['MapMov_sTH17_LP']) * (1 + Model['M_sTH17_chem'] * Model['Gamma_CCL25_sTH17']))*dt*6 >= 1:
        print("over Movement")
    if max(np.max(Model['MapDiff_IL22_LP']), np.max(Model['MapDiff_IL17_LP']), np.max(Model['MapDiff_Csf2_LP']), np.max(Model['MapDiff_IL10_LP']), np.max(Model['MapDiff_TGFBeta_LP']), np.max(Model['MapDiff_CCL2_LP']), np.max(Model['MapDiff_CCL25_LP']), np.max(Model['MapDiff_IL2_LP']), np.max(Model['MapDiff_IL1Beta_23_LP']))*dt*6 >= 1:
        print("over diffusion")
    if Model['M_Epi'] * dt >= 1:
        print("over Movement Epi")
    if max(np.max(Model['MapMov_DC_MLN']), np.max(Model['MapMov_DC_MLN']), np.max(Model['MapMov_nTc_MLN']), np.max(Model['MapMov_TH17_MLN']), np.max(Model['MapMov_Treg_MLN']), np.max(Model['MapMov_mTH17_MLN']), np.max(Model['MapMov_mTreg_MLN']))*dt*6 >= 1:
        print("over Movement")
    if max(np.max(Model['MapDiff_IL2_MLN']), np.max(Model['MapDiff_IL21_MLN']), np.max(Model['MapDiff_IL6_MLN']), np.max(Model['MapDiff_IL10_MLN']), np.max(Model['MapDiff_TGFBeta_MLN']))*dt*6 >= 1:
        print("over diffusion")
        
    #Soluble Mediators  in Lamina Propria
    if dt*((Model['Gamma_IL10_Treg']/Model['K_IL10_Treg'] + Model['Gamma_IL10_TH17']/Model['K_IL10_TH17'] + Model['Gamma_IL10_eDC']/Model['K_IL10_eDC'] + Model['Gamma_IL10_Mac']/Model['K_IL10_Mac'] + Model['Gamma_IL10_sTreg']/Model['K_IL10_sTreg'] + Model['Gamma_IL10_sTH17']/Model['K_IL10_sTH17'] + Model['Gamma_IL10_tTreg']/Model['K_IL10_tTreg']) * Model['D_max_LP'] + (Model['Gamma_IL10_sEpi']/Model['K_IL10_sEpi']) * Model['D_max_Epi'] + Model['c_delta_IL10']) >= 1:
        print("error over IL-10 LP")
    if dt*((Model['Gamma_TGFBeta_TH17']/Model['K_TGFBeta_TH17'] + Model['Gamma_TGFBeta_Treg']/Model['K_TGFBeta_Treg'] + Model['Gamma_TGFBeta_Mon']/Model['K_TGFBeta_Mon']) * Model['D_max_LP'] + Model['c_delta_TGFBeta']) >= 1:
        print("error over TGF-Beta LP")
    if dt*((Model['Gamma_IL17_Epi']/1) * 5 * Model['D_max_Epi'] + (Model['Gamma_IL17_sEpi']/Model['K_IL17_sEpi']) * Model['D_max_Epi'] + Model['c_delta_IL17']) >= 1:
        print("error over IL-17")
    if dt*((Model['Gamma_IL22_Epi']/1) * 5 * Model['D_max_Epi'] + (Model['Gamma_IL22_sEpi']/Model['K_IL22_sEpi']) * Model['D_max_Epi'] + Model['c_delta_IL22']) >= 1:
        print("error over IL-22")
    if dt*((Model['Gamma_IL2_tTreg']/1 + Model['Gamma_IL2_sTreg']/Model['K_IL2_sTreg'] + Model['Gamma_IL2_Treg']/Model['K_IL2_Treg']) * Model['D_max_LP'] + Model['c_delta_IL2']) >= 1:
        print("error over IL-2")
    if dt*((Model['Gamma_CCL2_Mon']/1) * Model['D_max_LP'] + Model['L_CCL2_ves'] + Model['c_delta_CCL2']) >= 1:
        print("error over CCL2")
    if dt*((Model['Gamma_CCL25_TH17']/1 + Model['Gamma_CCL25_Treg']/1 + Model['Gamma_CCL25_sTreg']/Model['K_CCL25_sTreg'] + Model['Gamma_CCL25_sTH17']/Model['K_CCL25_sTH17']) * Model['D_max_LP'] + Model['L_CCL25_ves'] + Model['c_delta_CCL25']) >= 1:
        print("error over CCL25")
    if dt*((Model['Gamma_IL1Beta_23_TH17']/Model['K_IL1Beta_23_TH17'] + Model['Gamma_IL1Beta_23_Treg']/Model['K_IL1Beta_23_Treg'] + Model['Gamma_IL1Beta_23_ILC3']/Model['K_IL1Beta_23_ILC3']) * Model['D_max_LP'] + Model['c_delta_IL1Beta_23']) >= 1:
        print("error over IL-1Beta/23")    
    if dt*((Model['Gamma_Csf2_eDC']/Model['K_Csf2_eDC'] + Model['Gamma_Csf2_tDC']/Model['K_Csf2_tDC']) * Model['D_max_LP'] + Model['c_delta_Csf2']) >= 1:
        print("error over CSF-2")
    if dt*((Model['Gamma_SAA_Mac']/Model['K_SAA_Mac'] + Model['Gamma_SAA_TH17']/Model['K_SAA_TH17']) * Model['D_max_LP'] + Model['c_delta_SAA']) >= 1:
        print("error over SAA")
    if dt*((Model['Gamma_eRA_TH17']/Model['K_eRA_TH17'] + Model['Gamma_eRA_Treg']/Model['K_eRA_Treg'])*Model['D_max_LP'] + Model['c_delta_RA'] + Model['mu_DC_LP'] + Model['M_outLP_eDC']) >= 1:
        print("error over eRA in LP")
    if dt*((Model['Gamma_tRA_TH17']/Model['K_tRA_TH17'] + Model['Gamma_tRA_Treg']/Model['K_tRA_Treg'])*Model['D_max_LP'] + Model['c_delta_RA'] + Model['mu_DC_LP'] + Model['M_outLP_tDC']) >= 1:
        print("error over tRA in LP")
    
    #Soluble Mediators in Mesenteric Lymph Nodes
    if dt*((Model['Gamma_IL21_nTe']/Model['K_IL21_nTe'] + Model['Gamma_IL21_nTt']/Model['K_IL21_nTt'] + Model['Gamma_IL21_fTH17']/Model['K_IL21_fTH17'])*Model['D_max_MLN'] + Model['c_delta_IL21']) >= 1:
        print("error over IL-21")
    if dt*((Model['Gamma_IL2_nTt']/Model['K_IL2_nTt'] + Model['Gamma_IL2_nTe']/Model['K_IL2_nTe'] + Model['Gamma_IL2_Treg']/Model['K_IL2_Treg'] + Model['Gamma_IL2_fTreg']/Model['K_IL2_fTreg'] + Model['Gamma_IL2_tTreg']/Model['K_IL2_tTreg'] + Model['Gamma_IL2_mTreg']/Model['K_IL2_mTreg'])*Model['D_max_MLN'] + Model['c_delta_IL2']) >= 1:
        print("error over IL-2")
    if dt*((Model['Gamma_eRA_nTe']/Model['K_eRA_nTe'] + Model['Gamma_eRA_fTH17']/Model['K_eRA_fTH17'] + Model['Gamma_eRA_fTreg']/Model['K_eRA_fTreg'])*Model['n_DC'] + Model['c_delta_RA'] + Model['mu_DC_MLN']) >= 1:
        print("error over eRA in MLN")
    if dt*((Model['Gamma_tRA_nTt']/Model['K_tRA_nTt'] + Model['Gamma_tRA_fTH17']/Model['K_tRA_fTH17'] + Model['Gamma_tRA_fTreg']/Model['K_tRA_fTreg'])*Model['n_DC'] + Model['c_delta_RA'] + Model['mu_DC_MLN']) >= 1:
        print("error over tRA in MLN")
    if dt*((Model['Gamma_IL6_nTe']/Model['K_IL6_nTe'] + Model['Gamma_IL6_nTt']/Model['K_IL6_nTt'])*Model['D_max_MLN'] + Model['c_delta_IL6']) >= 1:
        print("error over IL-6")
    if dt*((Model['Gamma_TGFBeta_nTe']/Model['K_TGFBeta_nTe'] + Model['Gamma_TGFBeta_nTt']/Model['K_TGFBeta_nTt'])*Model['D_max_MLN'] + Model['c_delta_TGFBeta']) >= 1:
        print("error over TGF-Beta MLN")
    if dt*((Model['Gamma_IL10_Treg']/Model['K_IL10_Treg'] + Model['Gamma_IL10_eDC']/Model['K_IL10_eDC'] + Model['Gamma_IL10_tTreg']/Model['K_IL10_tTreg'] + Model['Gamma_IL10_mTreg']/Model['K_IL10_mTreg'] + Model['Gamma_IL10_TH17']/Model['K_IL10_TH17'] + Model['Gamma_IL10_mTH17']/Model['K_IL10_mTH17'] + Model['Gamma_IL10_fTH17']/Model['K_IL10_fTH17'] + Model['Gamma_IL10_fTreg']/Model['K_IL10_fTreg'])*Model['D_max_MLN'] + Model['c_delta_IL10']) >= 1:
        print("error over IL-10 MLN")

    

