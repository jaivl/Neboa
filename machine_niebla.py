# Autor: Jairo Valea LÃ³pez
#
# CousiÃ±as con machine learning aplicado Ã¡ niebla do Fiouco

# Importado de librerÃ­as habituales

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
from scipy.stats import spearmanr
from sklearn import metrics
from IPython.utils import io
from natsort import natsorted, index_natsorted
from tqdm.notebook import trange # barra de progreso
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance

def promedio(lst):
    return sum(lst) / len(lst)

def correlaciones(datos, modo = 0):
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
    f = open(ruta_machine + 'correlaciones_'+ str(modo) + '.txt','w')
    f.write('Correlacion de Spearman\n')
    f.write('Numero de datos = ' + str(len(datos)) + '\n')
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

def scorer_fiouco(bosque,X,y,posiciones):
    pesos = posiciones
    
    s_bosque = cross_val_score(bosque, X, y, cv=5,
                               scoring='neg_mean_absolute_percentage_error')
    pesos.append(round(s_bosque.mean(),2))
    
    return pesos
    

def regres_fiouco(datos,modo = 'clima',oper='_'):
    
    if (modo == 'clima'):       #
        posiciones = [2,3,4,5,6,7,9,10,11]
    if (modo == 'climaoptimo'): # 38.58
        posiciones = [2,5,6,7,9,11,90]
    if (modo == 'mixto'):
        posiciones = [2,3,4,5,6,7,9,10,11,13,14]
    if (modo == 'optimo'):      # 16.23 [2,5,11,90,91,92]
        posiciones = [2,13,14,90]
    if (modo == 'grano'):      # 
        posiciones = [13,14,90,91,92,93,94]
    if (modo == 'random'):
        posiciones = [11,13,14]
    if (modo == 'antiguo'):
        posiciones = [21,25,29,33,37,41,45,49,53,56,61,65,69,73,77]
    if (modo == 'todo'):        # 17.12 normal, 18.13 acumulado
        posiciones = [2,3,4,5,6,7,9,10,11,13,14,31,32,33,34,35,36,37,38,39,40,41,
                      42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,
                      60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,
                      78,79,80,81,82,83,84,85,86,87,90,91,92,93]
             
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
    # Nuevos campos de caracterización:
    #
    datos['Acumulado_2um'] = masas_ac_norm[:,41]
    datos['Acumulado_4um'] = masas_ac_norm[:,51]
    datos['Acumulado_6um'] = masas_ac_norm[:,57]
    datos['Acumulado_8um'] = masas_ac_norm[:,61]
    datos['Acumulado_10um'] = masas_ac_norm[:,64]
            
    X = datos.iloc[:, posiciones].values
    y = datos['Visibilidad corregida (m)'].values
    
    bosque = RandomForestRegressor(n_estimators=500,random_state=0,
    min_samples_split=5,min_samples_leaf=2,max_depth=50,warm_start=False)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    
    # evitar diferencias de escala
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # random forest
    bosque.fit(X_train, y_train)
    
    s_bosque = cross_val_score(bosque, X, y, cv=5, scoring='neg_root_mean_squared_error')
    s_bosque2 = cross_val_score(bosque, X, y, cv=5, scoring='neg_mean_absolute_percentage_error')
    
    print('Error medio (%): '+str(round(100*s_bosque2.mean(),2)) +
    '\nRaíz del error cuadrático medio: '+str(round(s_bosque.mean(),2)))
    print('')
    r = permutation_importance(bosque, X_test, y_test, n_repeats=10, random_state=0)
    
    fig = plt.figure(figsize = (16,9))
    gs = gridspec.GridSpec(2, 3, height_ratios = [1,1], width_ratios = [1.5,1.5,1])
    ax1 = plt.subplot(gs[0,0])
    ax1b = plt.subplot(gs[0,1],sharey=ax1)
    ax1c = plt.subplot(gs[1,0],sharex=ax1)
    ax1d = plt.subplot(gs[1,1],sharex=ax1b,sharey=ax1c)
    ax2 = plt.subplot(gs[0:2,2])
    
    axiter = [ax1,ax1b,ax1c,ax1d]
    cont = 0
    
    for z in axiter:
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                            random_state = cont)
        
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        y_pred_bosque = bosque.predict(X_test)
        
        equis = range(len(y_test))
        void = pd.DataFrame(y_test)
        ind = void.sort_values(by=0).index
        void = []
    
        plt.suptitle("Predicciones de visibilidad,"+' del ' + inicio[0] +
                                 ' a ' + fin[0])
        
        z.plot(equis,y_test[ind], color = "blue",label="Real")
        z.plot(equis, y_pred_bosque[ind], color = "green", marker = ",",label="Predecido",lw=0.5)
        z.set_ylim(bottom=0); z.grid(True)
        z.set_ylabel("Visibilidad (m)",size=10)
        #ax1.set_title('Bosque aleatorio - RMSE = ' + str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred_bosque)),3)),size=12)
        z.set_title('Error medio (%) = ' +
            str(round(100*metrics.mean_absolute_percentage_error(y_test, y_pred_bosque),2)) +
            ' || RMSE = ' +
            str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred_bosque)),2)), size = 10)
        z.legend(loc="upper left",prop={'size': 10})
        
        cont = cont + 1
    
    ax2.annotate(inicio[0]+' a ' + fin[0]
                 , xy=(0.01,0.95),size=10)
    ax2.annotate('Error medio (%): '+str(round(100*s_bosque2.mean(),2)) +
    '\nRaíz del error cuadrático medio: '+str(round(s_bosque.mean(),2)), xy=(0.01,0.90))
    ax2.annotate('El bosque aleatorio ha utilizado \nlas siguientes variables:', xy=(0.01,0.8))
    
    colocar = 0
    for i in r.importances_mean.argsort()[::-1]:
        ax2.annotate(f"{posiciones[i]:<2} - "
        f"{datos.columns[posiciones[i]]:<15}"
        f"{r.importances_mean[i]:.3f}"
        f" +/- {r.importances_std[i]:.3f}", xy=(0.01,0.75-0.03*colocar),
        fontfamily = 'monospace')
        colocar = colocar + 1
    
    ax2.set_axis_off()
    
    fig.tight_layout()
    plt.savefig(ruta_machine + 'preds_' + '_' + modo + '.png')
    plt.show()
    
    return bosque

def optimizador_fiouco(datos):

    pesos = []
    bosque = RandomForestRegressor(n_estimators=200,random_state=0,
    min_samples_split=5,min_samples_leaf=2,max_depth=20,warm_start=False)
    
    combinaciones = [[13,14,90],
                     [2,5,9,11],
                     [2,5,9,11,90],
                     [2,13,14,90],
                     [2,5,6,7,9,11],
                     [2,3,4,5,6,7,9,10,11,13,14],
                     [5,6,7,9,11,90]]
    
    posiciones = [18,22,26,30,34,38,42,46,50,54,58,62,66,70,74]
    
    for c in combinaciones:        
        X = datos.iloc[:, c].values
        y = datos['Visibilidad corregida (m)'].values        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        bosque.fit(X_train, y_train)
        pesos.append(scorer_fiouco(bosque,X,y,c))
    
    return pesos

def predictor_fiouco(bosque,entrada): 
    pre = bosque.predict(entrada)
    return pre
    

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
###### filtrador() para filtrar los datos
###### optimizador_fiouco() y regres_fiouco() para machine-learning
######

ruta_proces = 'C:\\Users\\miguel.anton\\Desktop\\NIEBLA\\Ensayos procesados\\'
ruta_machine = 'C:\\Users\\miguel.anton\\Desktop\\NIEBLA\\Machine_learning\\'

inicio = ['01/06/2021', '01/08/2021']
fin = ['31/07/2021', '31/10/2021']

datos = pd.read_csv(ruta_proces + 'database_modif.csv', delimiter = ";", decimal = ".")

datos = filtrador(datos,'8.',inicio=inicio[0],fin=fin[0],cap=1000)
#regres_fiouco(datos,'climaoptimo')