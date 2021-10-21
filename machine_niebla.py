# Autor: Jairo Valea López
#
# Cousiñas con machine learning aplicado á niebla do Fiouco

# Importado de librerías habituales

import os
import math
import pandas as pd
import numpy as np
import scipy.stats as stats
import time
import natsort
import csv
import datetime
import seaborn as sns
import matplotlib.patches as mpatches
from scipy.stats import spearmanr
from sklearn import metrics
from IPython.utils import io
from natsort import natsorted, index_natsorted
from tqdm.notebook import tqdm, trange # barra de progreso
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance

def promedio(lst):
    return sum(lst) / len(lst)

def correlaciones(ruta_proces, datos, modo = 0):
    if ((modo == 8) | (modo == '8.')):
        datos.drop(datos[datos['Ensayo'].astype(str).str[0] != '8'].index, inplace = True)
    if ((modo == 'cs') | (modo == 'CS')):
        datos.drop(datos[datos['Ensayo'].astype(str).str[0:2] != 'cs'].index, inplace = True)
    if (modo == 9):
        print('')

    correl = []; orden = []
    for i in range(len(datos.columns)):
        correl.append(spearmanr(datos.iloc[:,i],datos['Visibilidad corregida (m)']))
        orden.append(abs(correl[i][0]))

    indices = index_natsorted(orden, reverse = True)
    f = open(ruta_proces + 'correlaciones_'+ str(modo) + '.txt','w')
    f.write('CorrelaciÃ³n de Spearman\n')
    f.write('NÃºmero de datos = ' + str(len(datos)) + '\n')
    f.write('Ensayos: ' + str(datos['Ensayo'].unique()) +'\n\n')
    for i in indices:
        f.write(f"{datos.columns[i]:<28}" + 'r^2 = ' + f"{round(correl[i][0],4):<8}" + ' | pvalor = ' + str(round(correl[i][1],4))+'\n')
    f.close()
    return datos

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    x_vals = np.array(plt.gca.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.gca.plot(x_vals, y_vals, '--', color = 'red')
    
def machine_fiouco(datos,ensayos='8.',modo = 'clima',inicio = '01/06/2021', fin = '31/07/2021',cap = 150,oper='_'):
    ruta_machine = 'C:\\Users\\miguel.anton\\Desktop\\NIEBLA\\Machine_learning\\'
    
    for i in datos.index:
        datos.loc[i,'Hora'] = datetime.datetime.strptime(datos.loc[i,'Hora'], '%d/%m/%Y %H:%M')
    inicio = datetime.datetime.strptime(inicio,'%d/%m/%Y')
    fin = datetime.datetime.strptime(fin,'%d/%m/%Y')
    
    if ((ensayos == 8) | (ensayos[0:2] == '8.')):
        datos.drop(datos[datos['Ensayo'].astype(str).str[0] != '8'].index, inplace = True)
    
    if (modo == 'clima'):       #
        posiciones = [2,3,4,5,6,7,9,10,11]
    if (modo == 'climaoptimo'): # 38.58
        posiciones = [2,5,6,7,9,11]
    if (modo == 'mixto'):
        posiciones = [2,3,4,5,6,7,9,10,11,13,14]
    if (modo == 'optimo'):      # 16.23 [2,5,11,90,91,92]
        posiciones = [2,5,11,90,91,92]
    if (modo == 'todo'):        # 17.12 normal, 18.13 acumulado
        posiciones = [2,3,4,5,6,7,9,10,11,13,14,15,16,17,18,19,20,21,22,23,
                      24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,
                      42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,
                      60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,
                      78,79,80,81,82,83,84,85,86,87,90,91,92]
    
    # rango de tiempos:
    datos.drop(datos[(datos['Hora'] < inicio) | (datos['Hora'] > fin)].index, inplace = True)
    # diferencia de vis. corregida y real mayor que 20
    datos.drop(datos[abs(datos['Visibilidad corregida (m)']-datos['Visibilidad (m)']) > 19].index, inplace = True)
        
    for i in (datos.index):
        if (datos.loc[i,'Visibilidad corregida (m)'] > cap):
            datos.loc[i,'Visibilidad corregida (m)'] = cap
            
    #np.divide(datos.iloc[:,15:88],dx)
            
    granos = np.array(datos.iloc[:,15:88])
    unidades = np.empty((granos.shape[0],granos.shape[1]))
    masas_ac = np.empty((granos.shape[0],granos.shape[1]))
    masas_ac_norm = np.empty((granos.shape[0],granos.shape[1]))
    vols = (np.pi/6)*(diams**3)
    
    for k in range(granos.shape[0]):
        for m in range(granos.shape[1]):
            unidades[k,m] = (vols[m])*np.divide(granos[k,m],dx[m])
            masas_ac[k] = np.cumsum(unidades[k,:])
        for i in range(masas_ac.shape[0]):
            masas_ac_norm[i] = 100*(masas_ac[i])/max(masas_ac[i])
    #
    # Nuevos campos de caracterizaci�n:
    #
    datos['Acumulado_2um'] = masas_ac_norm[:,41]
    datos['Acumulado_6um'] = masas_ac_norm[:,57]
    datos['Volumen_+9um'] = masas_ac[:,72] - masas_ac[:,62]
            
    X = datos.iloc[:, posiciones].values
    y = datos['Visibilidad corregida (m)'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state = 1)
    
    # evitar diferencias de escala
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # random forest
    bosque = RandomForestRegressor(n_estimators=200,random_state=0,min_samples_split=5,min_samples_leaf=2,max_depth=50,bootstrap=True)
    bosque.fit(X_train, y_train)
    y_pred_bosque = bosque.predict(X_test)
    
    # gradient booster
    '''boost = AdaBoostRegressor(n_estimators=200, learning_rate=0.1,random_state=0,loss='square')
    boost.fit(X_train, y_train)
    y_pred_gradient = boost.predict(X_test)'''
    
    # neural network c/ backpropagation
    '''neural = MLPRegressor(random_state=1, max_iter=1000, warm_start = True)
    neural.fit(X_train, y_train)
    y_pred_neural = neural.predict(X_test)'''
    
    s_bosque = cross_val_score(bosque, X, y, cv=10, scoring='neg_root_mean_squared_error')
    #s_neural = cross_val_score(neural, X, y, cv=5, scoring='neg_root_mean_squared_error')
    
    print('Al bosque aleatorio le importan las siguientes variables:') 
    print("%0.2f accuracy with a standard deviation of %0.2f" % (s_bosque.mean(), s_bosque.std()))
    print('')
    r = permutation_importance(bosque, X_test, y_test, n_repeats=10, random_state=0)
    for i in r.importances_mean.argsort()[::-1]:
        print(posiciones[i],'-',
        f"{datos.columns[posiciones[i]]:<14}"
        f"{r.importances_mean[i]:.3f}"
        f" +/- {r.importances_std[i]:.3f}")
    print('')
    '''print('A la red neuronal le importan las siguientes variables:')
    print("%0.2f accuracy with a standard deviation of %0.2f" % (s_neural.mean(), s_neural.std()))
    print('')
    r = permutation_importance(neural, X_test, y_test, n_repeats=10, random_state=0)
    for i in r.importances_mean.argsort()[::-1]:
        print(posiciones[i],'-',
        f"{datos.columns[posiciones[i]]:<14}"
        f"{r.importances_mean[i]:.3f}"
        f" +/- {r.importances_std[i]:.3f}")'''
        
    equis = range(len(y_test))

    void = pd.DataFrame(y_test)
    ind = void.sort_values(by=0).index
    void = []
    
    fig = plt.figure(figsize = (16,9))
    gs = gridspec.GridSpec(1, 2, width_ratios = [3,1])
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[0,1])
    
    plt.suptitle("Predicciones de visibilidad,"+' del ' + inicio.strftime("%d/%m/%Y") +
                             ' a ' + fin.strftime("%d/%m/%Y"))
    
    ax1.plot(equis,y_test[ind], color = "blue",label="Real")
    ax1.plot(equis, y_pred_bosque[ind], color = "green", marker = ",",label="Predecido",lw=0.5)
    ax1.set_ylim(bottom=0); ax1.grid(True)
    ax1.set_ylabel("Visibilidad (m)",size=10)
    ax1.set_title('Bosque aleatorio - RMSE = ' + str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred_bosque)),3)),size=12)
    ax1.legend(loc="upper left",prop={'size': 10})
    
    ax2.annotate(str(round(s_bosque.mean(),3)), xy=(0.2,0.6))
    '''ax2.plot(equis,y_test[ind], color = "blue",label="Real")
    ax2.plot(equis, y_pred_neural[ind], color = "red", marker = ",",label="Predecido",lw=0.5)
    ax2.set_ylim(bottom=0); ax2.grid(True)
    ax2.set_ylabel("Visibilidad (m)",size=8)
    ax2.set_title('Red neuronal - RMSE = ' + str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred_neural)),3)),size=10)
    ax2.legend(loc="upper left",prop={'size': 8})'''
    
    fig.tight_layout()
    plt.savefig(ruta_machine + 'preds_' + modo + '.png')
    plt.show()
    
    return datos
    

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

ruta_proces = 'C:\\Users\\miguel.anton\\Desktop\\NIEBLA\\Ensayos procesados\\'

datos = pd.read_csv(ruta_proces + 'database_modif.csv', delimiter = ";", decimal = ".")
datos = datos.dropna()
datos.drop(datos[datos['Visibilidad corregida (m)'] == 0].index, inplace=True)
datos.drop(datos[datos['Prec_mensual'] == -9999].index, inplace=True)

machine_fiouco(datos,'8.','optimo',inicio='01/06/2021',
               fin='31/07/2021',cap=150)
