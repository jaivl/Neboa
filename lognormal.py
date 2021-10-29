#
# Autor: Jairo Valea
#
# Distribución lognormal
#

import pandas as pd
import numpy as np
import scipy.stats as stats
import time
import natsort
import csv
import math
import random
import datetime
import seaborn as sns
import matplotlib.patches as mpatches
from scipy.optimize import curve_fit
from sklearn import metrics
from IPython.utils import io
from natsort import natsorted, index_natsorted
from tqdm.notebook import trange # barra de progreso
from matplotlib import pyplot as plt
from matplotlib import gridspec

def promedio(lst):
    return sum(lst) / len(lst)

def filtrador(datos,ensayos='8.',inicio = '01/06/2021', fin = '31/07/2021',cap = 150):
    datos = datos.dropna()
    datos.drop(datos[datos['Visibilidad corregida (m)'] == 0].index, inplace=True)
    datos.drop(datos[datos['Prec_mensual'] == -9999].index, inplace=True)
    
    datos.Hora = pd.to_datetime(datos.Hora, format = '%d/%m/%Y %H:%M')
    inicio = datetime.datetime.strptime(inicio,'%d/%m/%Y')
    fin = datetime.datetime.strptime(fin,'%d/%m/%Y')
    
    if ((ensayos == 8) | (ensayos[0:2] == '8.')):
        datos.drop(datos[datos['Ensayo'].astype(str).str[0] != '8'].index, inplace = True)
    datos['Dia'] = datos['Hora'].dt.month*30 + datos['Hora'].dt.day
    
    # rango de tiempos:
    datos.drop(datos[(datos['Hora'] < inicio) | (datos['Hora'] > fin)].index, inplace = True)
    # diferencia de vis. corregida y real mayor que 20
    datos.drop(datos[abs(datos['Visibilidad corregida (m)']-datos['Visibilidad (m)']) > 19].index, inplace = True)
        
    for i in (datos.index):
        if (datos.loc[i,'Visibilidad corregida (m)'] > cap):
            datos.loc[i,'Visibilidad corregida (m)'] = cap
            
    return datos

def lognorm_fit(diams_g,curva):
    fit = curve_fit(stats.lognorm.cdf, diams_g,curva, p0=[diams_g[0],curva[0]])
    par1, par2 = fit[0]
    errores = fit[1]
    frozen_lognorm = stats.lognorm(par1, par2)
    
    return frozen_lognorm

def generar_curvas(datos,rangos):
    curvas = []
    diam_medios = []
    
    for v in range(len(rangos)-1):
        vis = datos[(datos['Visibilidad corregida (m)'] >= rangos[v]) &
        (datos['Visibilidad corregida (m)'] < rangos[v+1])]
        
        if (len(vis) > 0):
            gruesos = np.array(vis.iloc[:,54:85])
            unidades = np.empty((gruesos.shape[0],gruesos.shape[1]))
            masas_ac_brut = np.empty((gruesos.shape[0],gruesos.shape[1]))
            masas_ac = np.empty((gruesos.shape[0],gruesos.shape[1]))
            pesos = []
            for k in range(gruesos.shape[0]):
                for m in range(gruesos.shape[1]):
                    unidades[k,m] = np.divide(gruesos[k,m],dx[m+42])        
                masas_ac_brut[k] = np.cumsum(unidades[k,:])
                masas_ac[k] = masas_ac_brut[k]/masas_ac_brut[k][-1]
            for m in range(masas_ac_brut.shape[1]):
                pesos.append(promedio(unidades[:,m]))
        curvas.append(np.mean(masas_ac, axis = 0))
        diam_medios.append(np.average(diams_g, weights = pesos))
        
        ###
        ### Aquí se opera con la distribución log-normal
        ###
        
        t = np.linspace(2,18,100)
        
        frozen_lognorm = lognorm_fit(diams_g,curvas[v])
        
        #sigma = np.sqrt(np.average(diams_g, weights = pesos))
        sigma = math.sqrt(np.average(((np.log(diams_g) - np.log(diam_medios[v]))**2), weights = pesos))
        gamma = stats.gamma.cdf(t,0.9*diam_medios[v],sigma)
        otra_lognorm = stats.lognorm.cdf(t,s=sigma,scale=0.9*diam_medios[v],loc=0)

        ###
        ### Fin de operaciones con log-normal (excepto graficado)
        ###
        
        fig = plt.figure(figsize = (15,8))
        gs = gridspec.GridSpec(1,2,height_ratios=[1],width_ratios=[1,1])
        
        plt.suptitle('Visibilidad '+str(rangos[v]) + ' a ' + str(rangos[v+1]) + ' m - '
                              + str(len(vis)) + ' registros', size=18)
        
        ax1 = plt.subplot(gs[0,0]); plt.grid(which='both')
#        ax1.set_title('Ajuste lognormal // media = ' + str(round(d_medios[v],3)) + ', sigma = '
 #                    + str(round(sigma1,3)))
        ax1.set_title('Dm = ' + str(round(diam_medios[v],3)))
        ax1.set_xscale('log')
        ax1.set_xlim(2,18); ax1.set_ylim(0,100);
        ax1.set_xlabel('Diámetro (um)');
        ax1.plot(diams_g,100*curvas[v],lw=2,color='red',label='Media')
        #ax1.plot(t, stats.norm.cdf(t, mu1, sigma1), color='forestgreen',ls='--')
        #ax1.plot(t, 100*frozen_lognorm.cdf(t), color = 'coral',ls='--',
         #        label='Log-normal, u=')
        ax1.plot(t,100*gamma,color='blue',ls='--',
                 label='Gamma, $\mu=0.9*Dm$, $\sigma$='+str(round(sigma,3)))
        ax1.plot(t,100*otra_lognorm,color='forestgreen',ls='--',
                 label='Lognorm, $\mu=0.9*Dm$, $\sigma$='+str(round(sigma,3)))
        ax1.legend(loc='lower right')

        ax2 = plt.subplot(gs[0,1]); ax2.grid(which='both')
        ax2.set_title('Dm = ' + str(round(diam_medios[v],3)))
        ax2.set_xscale('log'); ax2.set_xlim(2,18); ax2.set_ylim(0,100);
        for i in range(masas_ac.shape[0]):
            ax2.plot(diams_g,100*masas_ac[i], color = 'blue', alpha = 0.35, lw = 0.5)
        ax2.set_xlabel('Diámetro (um)');
        ax2.plot(diams_g,100*curvas[v],lw=2,color='red')
        
    return diam_medios, curvas


def lognormal(datos,rangos):
    
    #vols = (np.pi/6)*(diams_g**3)
    vols = (diams_g**0)
    
    for v in range(len(rangos)-1):
        medias = []
        mues = []
        sigmas = []
        pesos = []
        vis = datos[(datos['Visibilidad corregida (m)'] >= rangos[v]) &
        (datos['Visibilidad corregida (m)'] < rangos[v+1])]
        
        if (len(vis) > 0):
            gruesos = np.array(vis.iloc[:,54:85])
            unidades = np.empty((gruesos.shape[0],gruesos.shape[1]))
            masas_ac_brut = np.empty((gruesos.shape[0],gruesos.shape[1]))
            masas_ac = np.empty((gruesos.shape[0],gruesos.shape[1]))
            
            for k in range(gruesos.shape[0]):
                for m in range(gruesos.shape[1]):
                    unidades[k,m] = (vols[m])*np.divide(gruesos[k,m],dx[m+42])
                    #unidades[k,m] = (vols[m])*gruesos[k,m]
                masas_ac_brut[k] = np.cumsum(unidades[k,:])
                mu = np.average(diams_g, weights = vols*unidades[k,:])
                mues.append(mu)
                pesos.append(max(masas_ac_brut[k]))
            for k in range(masas_ac.shape[0]):
                masas_ac[k] = (masas_ac_brut[k])/pesos[k]
            for m in range(masas_ac.shape[1]):
                medias.append(promedio(masas_ac[:,m]))
                
            mu = np.average(np.log(mues), weights = pesos) # mu = media
            sigma = math.sqrt(np.average((np.log(mues - mu)**2), weights = pesos)) # sigma = desviación típica
            print('mu =',mu,'sigma =',sigma)
        
        t = np.linspace(2,18,100)
        
        mu,sigma = curve_fit(stats.lognorm.cdf, diams_g,medias, p0=[2,10])[0]
        frozen_lognorm = stats.lognorm(s=sigma, scale=math.exp(mu))
    
        # acumulada:
        mu1,sigma1 = curve_fit(stats.norm.cdf, diams_g,medias, p0=[2,10])[0]
        
        fig = plt.figure(figsize = (15,8))
        gs = gridspec.GridSpec(1,2,height_ratios=[1],width_ratios=[1,1])
        
        plt.suptitle('Visibilidad '+str(rangos[v]) + ' a ' + str(rangos[v+1]) + ' m - '
                              + str(len(vis)) + ' registros', size=14)
        
        ax1 = plt.subplot(gs[0,0]); plt.grid(which='both')
        ax1.set_title('Ajuste Gamma // media = ' + str(round(mu1,3)) + ', sigma = '
                     + str(round(sigma1,3)))
        ax1.set_xscale('log')
        ax1.set_xlim(2,18); ax1.set_ylim(0,1);
        ax1.set_xlabel('Diámetro (um)');
        ax1.plot(diams_g,medias,lw=1.5,color='red')
        ax1.plot(t, stats.norm.cdf(t, mu1, sigma1), color='forestgreen',ls='--')
        ax1.plot(t, frozen_lognorm.cdf(t), color = 'blue', ls='--')
        
        ax2 = plt.subplot(gs[0,1]); ax2.grid(which='both')
        ax2.set_title('Dm = ' + str(round(mu,3)))
        ax2.set_xscale('log'); ax2.set_xlim(2,18); ax2.set_ylim(0,1);
        for i in range(masas_ac.shape[0]):
            ax2.plot(diams_g,masas_ac[i], color = 'blue', alpha = 0.35, lw = 0.5)
        ax2.set_xlabel('Diámetro (um)');
        ax2.plot(diams_g,medias,lw=1.5,color='red')
        
##########################
##########################
###### PROGRAMA PRINCIPAL
##########################
##########################

diams_g = np.array([2.13,2.289,2.46,2.643,2.841,3.053,3.28,3.525,3.788,4.071,4.374,4.701,5.051,5.428,5.833,6.268,6.736,7.239,
    7.779,8.359,8.983,9.653,10.373,11.147,11.979,12.872,13.833,14.865,15.974,17.165,18.446])
diams = np.array([0.104,0.111,0.12,0.129,0.138,0.149,0.16,0.172,0.184,0.198,0.213,0.229,0.246,0.264,0.284,0.305,0.328,0.352,0.379,
    0.407,0.437,0.47,0.505,0.543,0.583,0.627,0.674,0.724,0.778,0.836,0.898,0.965,1.037,1.115,1.198,1.287,1.383,1.486,
    1.597,1.717,1.845,1.982,2.13,2.289,2.46,2.643,2.841,3.053,3.28,3.525,3.788,4.071,4.374,4.701,5.051,5.428,5.833,
    6.268,6.736,7.239,7.779,8.359,8.983,9.653,10.373,11.147,11.979,12.872,13.833,14.865,15.974,17.165,18.446])
dx = np.array([0.007,0.008,0.009,0.009,0.01,0.011,0.011,0.012,0.013,0.014,0.015,0.016,0.018,0.019,0.02,0.022,0.024,0.025,0.027
,0.029,0.031,0.034,0.036,0.039,0.042,0.045,0.048,0.052,0.056,0.06,0.065,0.069,0.075,0.08,0.086,0.093,0.099,0.107,0.115
,0.123,0.133,0.143,0.153,0.165,0.177,0.19,0.204,0.22,0.236,0.254,0.272,0.293,0.315,0.338,0.363,0.39,0.42,0.451,0.484
,0.521,0.559,0.601,0.646,0.694,0.746,0.802,0.862,0.926,0.995,1.069,1.149,1.235,1.327])

######
###### 

ruta_proces = 'C:\\Users\\miguel.anton\\Desktop\\NIEBLA\\Ensayos procesados\\'
ruta_machine = 'C:\\Users\\miguel.anton\\Desktop\\NIEBLA\\Machine_learning\\'

inicio = ['01/06/2021', '01/08/2021']
fin = ['31/07/2021', '31/10/2021']
rangos = [15,30,45,60,75,100,200,1000]

datos = pd.read_csv(ruta_proces + 'database_modif.csv', delimiter = ";", decimal = ".")

datos = filtrador(datos,'8.',inicio=inicio[0],fin=fin[0],cap=1000)
d_medios, curvas = generar_curvas(datos,rangos)