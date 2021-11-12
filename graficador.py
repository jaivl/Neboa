# Autor: Jairo Valea LÃ³pez

# Documentación:

# ensayos_disipacion(datos,inicio='01/07/2021',fin='31/10/2021',rangos)
# función experimental no completada

# visib(datos,inicio,fin,rangos,rampa)
# Grafica las funciones de densidad granulométricas entre las fechas señaladas,
# con los rangos de visibilidad aportados.
# 'rampa' hace referencia a la rampa de color
#   - 'ensayo' - un color para cada ensayo
#   - 'temp' - gradiente de color por temperatura
#   - 'tempdisc' - un color para cada temperatura
#   - 'mes' - un color para cada mes

# separar_ensayos(datos, inicio, fin, rangos, niebla, ensayos)
# Similar a visib, pero en vez de hacer una gráfica por cada rango de
# visibilidad extrae una gráfica destacando cada ensayo.
# 'niebla' y 'ensayos son valores booleanos:
#   - para extraer solo los 8., niebla = True y ensayos = False
#   - para extraer los 8. + 10 primeros min de niebla, niebla = True y 
#       ensayos = True
#   - para extraer los 10., niebla = False y ensayos = True

# distribuciones_cajas(datos,ensayos = '8.',inicio,fin,rangos)
# Extrae los boxplot caracterizadores de la niebla
# 'ensayos' es un argumento cadena que puede tomar múltiples valores:
#   - '8.' para usar solo los ensayos de caracterización
#       - '8. + cs12 t=8'
#   - '10' para usar solo los ensayos de disipación
#   - 'cs' para usar solo los ensayos de disipación con cloruro sódico
#
#   - 'solo b28 min 7' para representar solo un minuto concreto de un ensayo
#


# Importado de librerÃ­as habituales

import os
import csv
import pandas as pd
import numpy as np
import scipy.stats as stats
import datetime # formato fecha
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from IPython.utils import io
from natsort import natsorted, index_natsorted
from matplotlib import pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm, trange # barra de progreso

def promedio(lst):
    return sum(lst) / len(lst)

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)
    
def construyebase(ruta):
    carpeta = natsorted(os.listdir(ruta_proces))
    procesados = []
    nombres = []
    
    for f in carpeta:
        name, ext = os.path.splitext(f)
        if ((ext == '.txt') & (name[0] == 'E')):
            procesados.append(pd.read_csv(ruta_proces + name + ext, delimiter = ",", decimal = "."))
            nombres.append(name + ext)
    for i in range(len(nombres)):
        procesados[i] = procesados[i].apply(lambda col:pd.to_numeric(col, errors='coerce'))
        procesados[i]['Ensayo'] = nombres[i][6:-14]
    procesados_total = pd.concat(procesados,ignore_index = True)
    g = open(ruta_proces + 'database_todo.csv', 'w', encoding='UTF8', newline='')
    writer = csv.writer(g, delimiter=';')
    writer.writerow(procesados_total.columns)
    for i in range(len(procesados_total)):
        writer.writerow(procesados_total.iloc[i,:])
    g.close()
    
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def localizardiam(ac, d):
    diams_g = np.array([2.13,2.289,2.46,2.643,2.841,3.053,3.28,3.525,3.788,4.071,4.374,4.701,5.051,5.428,5.833,6.268,6.736,7.239,
                   7.779,8.359,8.983,9.653,10.373,11.147,11.979,12.872,13.833,14.865,15.974,17.165,18.446])
    cerca = find_nearest(ac, d)
    res = np.where(ac == cerca)[0][0]
    if (cerca > d):
        while (ac[res] == ac[res-1]):
            res = res-1        
        diam_tipo = diams_g[res]-((diams_g[res] - diams_g[res-1])*(ac[res] - d) / (ac[res] - ac[res-1]))
    if (cerca < d):
        while (ac[res] == ac[res+1]):
            res = res+1
        diam_tipo = diams_g[res+1]-((diams_g[res+1] - diams_g[res])*(ac[res+1] - d) / (ac[res+1] - ac[res]))
    if (cerca == d):
        diam_tipo = diams_g[res]
    return diam_tipo

def rosinrammler(ac, m):
    diams_g = np.array([2.13,2.289,2.46,2.643,2.841,3.053,3.28,3.525,3.788,4.071,4.374,4.701,5.051,5.428,5.833,6.268,6.736,7.239,
                   7.779,8.359,8.983,9.653,10.373,11.147,11.979,12.872,13.833,14.865,15.974,17.165,18.446])
    x = np.linspace (0, 20, 100)
    cerca = find_nearest(ac, 0.632)
    res = np.where(ac == cerca)[0][0]
    if (cerca > 0.632):
        while (ac[res] == ac[res-1]):
            res = res-1        
        diam_tipo = diams_g[res]-((diams_g[res] - diams_g[res-1])*(ac[res] - 0.632) / (ac[res] - ac[res-1]))
    if (cerca < 0.632):
        while (ac[res] == ac[res+1]):
            res = res+1
        diam_tipo = diams_g[res+1]-((diams_g[res+1] - diams_g[res])*(ac[res+1] - 0.632) / (ac[res+1] - ac[res]))
    if (cerca == 0.632):
        diam_tipo = diams_g[res]
    F = 1 - 2.71828**(-(x/diam_tipo)**m)
    return F    

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

#########################
### Función descontinuada
#########################

def graficado(datos, modo = '8.', grafico = 'gra-vis', grupos = 5, vmax = 2000, normalizado = False, rampa = False,
              v_rampa = 'Prec_mensual', alfa = 0.3):
    if ((modo == 8) | (modo == '8.')):
        datos.drop(datos[datos['Ensayo'].astype(str).str[0] != '8'].index, inplace = True)
    if ((modo[0:2] == 'cs') | (modo[0:2] == 'CS')):
        datos.drop(datos[datos['Ensayo'].astype(str).str[0:2] != 'cs'].index, inplace = True)
        if ((modo[3:] == '10min') | (modo[3:] == '10 min')):
            datos.drop(datos[datos['Tiempo (min)'] > 9.5].index, inplace = True)
        if ((modo[3:] == 'generada') | (modo[3:] == 'gen')):
            datos.drop(datos[(datos['Tiempo (min)'] > 9.5) & (datos['Tiempo (min)'] < 30.5)].index, inplace = True)
    if ((modo[0:2] == '10') | (modo[0:2] == '!8') | (modo[0:2] == 'n8')):
        datos.drop(datos[datos['Ensayo'].astype(str).str[0] == '8'].index, inplace = True)
        if ((modo[3:] == '10min') | (modo[3:] == '10 min')):
            datos.drop(datos[datos['Tiempo (min)'] > 9.5].index, inplace = True)
        if ((modo[3:] == 'generada') | (modo[3:] == 'gen')):
            datos.drop(datos[(datos['Tiempo (min)'] < 9.5) | (datos['Tiempo (min)'] > 30.5)].index, inplace = True)
    if (modo[0:4] == 'solo'):
        datos.drop(datos[datos['Ensayo'] != modo[5:]].index, inplace = True)
    
    datos.drop(datos[datos['Visibilidad corregida (m)'] > vmax].index, inplace = True)
    
    if rampa:
        color_labels = natsorted(datos[v_rampa].unique())
        rgb_values = sns.color_palette("ch:start=.2,rot=-.3", len(color_labels)) # cubehelix // alternativa: 'crest'
        #colorinchos = ListedColormap(sns.color_palette(rgb_values).as_hex())
        color_map = dict(zip(color_labels, rgb_values))
        hand = []
        for i in range(len(rgb_values)):
            hand.append(mpatches.Patch(color=rgb_values[i], label=str(color_labels[i])))
    else:
        color_map = 'blue'
    if ((grafico[0:3] == 'gra') | (grafico[0:3] == 'mas')):
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
        limites = []
        vols = (np.pi/8)*(diams**3)
        if (grafico[4:7] == 'vis'):
            for v in range(grupos):
                limites.append(round(np.quantile(datos['Visibilidad corregida (m)'],v/grupos),1))
            limites.append(2000)
            for i in range(grupos):
                fig, ax = plt.subplots(figsize = (12,7))
                vis = datos[(datos['Visibilidad corregida (m)'] >= limites[i]) & (datos['Visibilidad corregida (m)'] < limites[i+1])]
                if normalizado:
                    suma_gruesos = []
                    gruesos = np.array(vis.iloc[:,54:85])
                    for n in range(gruesos.shape[0]): # el mismo para ambos
                        suma_gruesos.append(np.sum(gruesos[n,:]))
                    densidad_gruesos = np.empty((gruesos.shape[0],gruesos.shape[1]))
                    for k in range(gruesos.shape[0]):
                        for m in range(gruesos.shape[1]):
                            densidad_gruesos[k,m] = (1/suma_gruesos[k])*(gruesos[k,m])*(1/dx[m+42])
                    ax.set_xlim(2,18);ax.set_ylim(0,0.6); ax.grid(True)
                    media = []
                    for k in range(31):
                        media.append(promedio(densidad_gruesos[:,k]))
                    ax.set_ylabel('dN/N/dX')
                else:
                    gruesos = np.array(vis.iloc[:,54:85])
                    densidad_gruesos = np.empty((gruesos.shape[0],gruesos.shape[1]))
                    if (grafico[0:3] == 'mas'):
                        ax.set_ylabel('Masa (ug/cm3)'); ax.set_ylim(0,10000)
                        for k in range(gruesos.shape[0]):
                            for m in range(gruesos.shape[1]):
                                densidad_gruesos[k,m] = (vols[m+42])*(gruesos[k,m])*(1/dx[m+42])
                    else:
                        ax.set_ylabel('N * dN/N/dX'); ax.set_ylim(0,150)
                        for k in range(gruesos.shape[0]):
                            for m in range(gruesos.shape[1]):
                                densidad_gruesos[k,m] = (gruesos[k,m])*(1/dx[m+42])
                    ax.set_xlim(2,18); ax.grid(True)
                    media = []
                    for k in range(31):
                        media.append(promedio(densidad_gruesos[:,k]))
                    masa_media = 'Masa media = ' + str(int(np.sum(media))) + ' ug/cm3'
                    ax.annotate(masa_media, xy = (10,9750), ha = 'center')
                for j in range(len(densidad_gruesos)):
                    if rampa:
                        ax.plot(diams_g, densidad_gruesos[j], alpha = alfa, lw = 0.5, color = vis[v_rampa].map(color_map)[vis.index[j]])
                    else:
                        ax.plot(diams_g, densidad_gruesos[j], alpha = alfa, lw = 0.5, color = 'blue')
                ax.plot(diams_g, media, alpha = 1, lw = 2, color = 'red')
                ax.set_xlabel('Diámetro (um)')
                if rampa:
                    ax.legend(handles = hand, title=str(v_rampa), loc = 'upper right')
                ax.set_title(str(modo) + ', visibilidad ' + str(limites[i]) + ' a ' + str(limites[i+1]) + ' m')
    

    return '0'
                #for k in datos_cp.index:
                #    tiempos.append(datetime.datetime.strptime(datos_cp['Hora'][k], '%d/%m/%Y %H:%M'))
                #axes[i,j].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))


def sinoutlier(datos):
    Q1 = np.percentile(datos, 25)
    Q3 = np.percentile(datos, 75)
    gap = 1.6*(Q3 - Q1)
    filtrado = []
    for i in datos:
        if ((i < Q3 + gap) & (i > Q1 - gap)):
            filtrado.append(i)
    return filtrado


#######################
#######################
### Función principal gráficos de cajas
#######################
#######################

def distribuciones_cajas(datos, ensayos = '8.', inicio = '01/06/2021', fin = '31/08/2021', 
                   vmax = 2000, rangos = []):
    
    #####################
    ####  Menú y filtrado
    #####################
    
    vector = ['particulas','superficie','volumen']
    
    inicio = datetime.datetime.strptime(inicio,'%d/%m/%Y')
    fin = datetime.datetime.strptime(fin,'%d/%m/%Y')
    for i in datos.index:
        datos.loc[i,'Hora'] = datetime.datetime.strptime(datos.loc[i,'Hora'], '%d/%m/%Y %H:%M')
    if ((ensayos == 8) | (ensayos[0:2] == '8.')):
        if (len(ensayos) > 2):
            if (ensayos[3] == '+'):
                especial = datos[(datos['Ensayo'].astype(str) == ensayos[5:-6]) & (datos['Tiempo (min)'] == float(ensayos[-1]))]
        datos.drop(datos[datos['Ensayo'].astype(str).str[0] != '8'].index, inplace = True)
    if ((ensayos[0:2] == 'cs') | (ensayos[0:2] == 'CS')):
        datos.drop(datos[datos['Ensayo'].astype(str).str[0:2] != 'cs'].index, inplace = True)
        if ((ensayos[3:] == '10min') | (ensayos[3:] == '10 min')):
            datos.drop(datos[datos['Tiempo (min)'] > 9.5].index, inplace = True)
        if ((ensayos[3:] == 'generada') | (ensayos[3:] == 'gen')):
            datos.drop(datos[(datos['Tiempo (min)'] > 9.5) & (datos['Tiempo (min)'] < 30.5)].index, inplace = True)
    if ((ensayos[0:2] == '10') | (ensayos[0:2] == '!8') | (ensayos[0:2] == 'n8')):
        datos.drop(datos[datos['Ensayo'].astype(str).str[0] == '8'].index, inplace = True)
        if ((ensayos[3:] == '10min') | (ensayos[3:] == '10 min')):
            datos.drop(datos[datos['Tiempo (min)'] > 9.5].index, inplace = True)
        if ((ensayos[3:] == 'generada') | (ensayos[3:] == 'gen')):
            datos.drop(datos[(datos['Tiempo (min)'] < 9.5) | (datos['Tiempo (min)'] > 30.5)].index, inplace = True)
    if (ensayos[0:4] == 'solo'):
        datos.drop(datos[datos['Ensayo'].astype(str) != ensayos[5:-6]].index, inplace = True)
        if (ensayos[-5:-2] == 'min'):
            datos.drop(datos[datos['Tiempo (min)'] != int(ensayos[-1])].index, inplace = True)

    datos.drop(datos[datos['Visibilidad corregida (m)'] > vmax].index, inplace = True)
    # rango de tiempos:
    datos.drop(datos[(datos['Hora'] < inicio) | (datos['Hora'] > fin)].index, inplace = True)
    # diferencia de vis. corregida y real mayor que 20
    datos = filtrar_corregida(datos,rangos)

    if rangos:
        pass
    else: 
        for v in range(6):
            rangos.append(round(np.quantile(datos['Visibilidad corregida (m)'],v/6),1))
            rangos.append(2000)
            
    #####################
    ####  Datos ctes.
    #####################        
    
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
    
    #####################
    ####  Bucle de operaciones
    #####################   
    for s in tqdm(vector):
        
        grafico = s
        medias = [[] for v in range(len(rangos) - 1)]
        pend = []
        valores15 = [[] for v in range(len(rangos) - 1)]
        valores60 = [[] for v in range(len(rangos) - 1)]
        valores50 = [[] for v in range(len(rangos) - 1)]
        valores85 = [[] for v in range(len(rangos) - 1)]
        
        if (s == 'volumen'):
            vols = (np.pi/6)*(diams_g**3)
        if (s == 'superficie'):
            vols = (np.pi)*(diams_g**2)
        if (s == 'particulas'):   
            vols = (diams_g**0)
            
        if (len(ensayos) > 2):
            if (ensayos[3] == '+'):
                spec = np.array(especial.iloc[:,54:85])
                for m in range(len(spec)):
                    spec[m] = (vols)*np.divide(spec[m],dx[m+42])
                especialac = np.cumsum(spec)
                especialac = 100*especialac/max(especialac)

        for v in range(len(rangos)-1):
            vis = datos[(datos['Visibilidad corregida (m)'] >= rangos[v]) &
            (datos['Visibilidad corregida (m)'] < rangos[v+1])]
            
            if (len(ensayos) > 2):
                if (ensayos[3] == '+'):
                    if ((especial['Visibilidad corregida (m)'][especial.index[0]] >= rangos[v]) &
                        (especial['Visibilidad corregida (m)'][especial.index[0]] < rangos[v+1])):
                        dondeespecial = v
    
            if (len(vis) > 0):
                gruesos = np.array(vis.iloc[:,54:85])
                unidades = np.empty((gruesos.shape[0],gruesos.shape[1]))
                masas_ac = np.empty((gruesos.shape[0],gruesos.shape[1]))
                    
                for k in range(gruesos.shape[0]):
                    for m in range(gruesos.shape[1]):
                        unidades[k,m] = (vols[m])*np.divide(gruesos[k,m],dx[m+42])
                    masas_ac[k] = np.cumsum(unidades[k,:])
                for i in range(masas_ac.shape[0]):
                    masas_ac[i] = 100*(masas_ac[i])/max(masas_ac[i])
                for i in range(masas_ac.shape[1]):
                    medias[v].append(promedio(masas_ac[:,i]))
                
                
                #logaritmos = (1-(masas_ac/100))
                logaritmos = (-np.log(1.0001-masas_ac/100))
                interm = 0;
                if (logaritmos.shape[0] != 0):
                    for i in range(logaritmos.shape[0]):
                        pendiente, orden, _, _, _ = stats.linregress(np.log(diams_g/10)[5:14], np.log(logaritmos[i])[5:14])
                        interm = interm + pendiente
                    pend.append(round((interm/(i+1)),3))
                else:
                    pend.append(0)
            
            #########################
                #### Rampa de color
                #####################
                
                color_labels = []
                for w in range(len(rangos)-1):
                    color_labels.append(str(rangos[w]) + '_' + str(rangos[w+1]))
                
                #rgb_values = sns.color_palette("ch:start=.2,rot=-.3", len(color_labels)) # cubehelix // alternativa: 'crest', 'PuBu'
                rgb_values = sns.color_palette("hls", len(color_labels))
                #colorinchos = ListedColormap(sns.color_palette(rgb_values).as_hex())
                #color_map = dict(zip(color_labels, rgb_values))
                hand = []
                for i in range(len(rgb_values)-1,-1,-1):
                    hand.append(mpatches.Patch(color=rgb_values[i], label=str(color_labels[i])))
                
            #########################
                ####  Graficado
                #####################
                
                labelsd = [2,3,4,5,6,8,10,12,15]; labelsi = [0.2,0.3,0.4,0.6,0.8,1.2,1.8]

                for i in range(masas_ac.shape[0]):
                    valores15[v].append(localizardiam(masas_ac[i],15))
                    valores50[v].append(localizardiam(masas_ac[i],50))
                    valores60[v].append(localizardiam(masas_ac[i],60))
                    valores85[v].append(localizardiam(masas_ac[i],85))
                
                #
                # AQUÍ SE QUITAN LOS OUTLIERS (*1.5 VECES EL IQR)
                #
                
                #valores15[v] = sinoutlier(valores15[v])
                #valores50[v] = sinoutlier(valores50[v])
                #valores60[v] = sinoutlier(valores60[v])
                #valores85[v] = sinoutlier(valores85[v])
                    
        # Gráfico de comparación de medias
        with io.capture_output() as captured:
            fig = plt.figure(figsize = (16,9))
        gs = gridspec.GridSpec(4, 2, width_ratios = [1,1.5], height_ratios = [1,1,1,1])
        ax1 = plt.subplot(gs[0:3,1])
        ax2 = plt.subplot(gs[3,0], sharex = ax1)
        ax3 = plt.subplot(gs[2,0], sharex = ax1)
        ax4 = plt.subplot(gs[1,0], sharex = ax1)
        ax5 = plt.subplot(gs[0,0], sharex = ax1)
        ax2.boxplot(valores15, vert = False, positions = range(len(valores15)))
        ax3.boxplot(valores50, vert = False, positions = range(len(valores50)))
        ax4.boxplot(valores60, vert = False, positions = range(len(valores60)))
        ax5.boxplot(valores85, vert = False, positions = range(len(valores85)))
        if (len(ensayos) > 2):
            if (ensayos[3] == '+'):
                especial15 = localizardiam(especialac,15)
                especial50 = localizardiam(especialac,50)
                especial60 = localizardiam(especialac,60)
                especial85 = localizardiam(especialac,85)
                ax2.scatter(especial15, dondeespecial, marker = '*', s=150, color='red')
                ax3.scatter(especial50, dondeespecial, marker = '*', s=150, color='red')
                ax4.scatter(especial60, dondeespecial, marker = '*', s=150, color='red')
                ax5.scatter(especial85, dondeespecial, marker = '*', s=150, color='red')
                ax2.axvline(especial15, color = 'red', lw = 0.5)
                ax3.axvline(especial50, color = 'red', lw = 0.5)
                ax4.axvline(especial60, color = 'red', lw = 0.5)
                ax5.axvline(especial85, color = 'red', lw = 0.5)

        for z in range(len(medias)):
            if medias[z]:
                ax1.plot(diams_g,medias[z], color = rgb_values[z])
        ax1.set_xlabel('Diámetro (um)');
        ax2.set_ylabel('Visibilidad (m)'); ax2.set_xlabel('Diámetro (um)'); 
        ax3.set_ylabel('Visibilidad (m)'); ax3.set_xlabel('Diámetro (um)'); 
        ax4.set_ylabel('Visibilidad (m)'); ax4.set_xlabel('Diámetro (um)'); 
        ax5.set_ylabel('Visibilidad (m)'); ax5.set_xlabel('Diámetro (um)'); 
        if ((grafico == 'volumen') | (grafico == 'vol')):
            ax1.set_ylabel('% Volumen')
        if ((grafico == 'superficie') | (grafico == 'sup')):
            ax1.set_ylabel('% Superficie')
        if ((grafico == 'particulas') | (grafico == 'part')):   
            ax1.set_ylabel('% Particulas')
        ax1.set_xscale('log'); ax1.set_xlim(2,18); ax1.set_ylim(0,100);
        ax2.set_xscale('log'); ax2.set_xlim(2,18)
        ax1.grid(True); ax2.grid(True); ax3.grid(True); ax4.grid(True); ax5.grid(True)
        ax1.legend(handles = hand, title = 'Visibilidad',loc='lower right')
        ax1.set_xticks(labelsd,minor=False); ax1.set_xticklabels(labelsd)
        ys = [0,15,25,50,60,75,85,100]
        ax1.set_yticks(ys); ax1.set_yticklabels(ys)
        ax2.set_yticks(range(len(valores15)),minor=False); ax2.set_yticklabels(color_labels)
        ax3.set_yticks(range(len(valores50)),minor=False); ax3.set_yticklabels(color_labels)
        ax4.set_yticks(range(len(valores60)),minor=False); ax4.set_yticklabels(color_labels)
        ax5.set_yticks(range(len(valores85)),minor=False); ax5.set_yticklabels(color_labels)

        plt.suptitle('Ensayos de caracterización' + ', del ' + inicio.strftime("%d/%m/%Y") +
                     ' a ' + fin.strftime("%d/%m/%Y"),size=14)
        ax1.set_title('Curvas acumuladas medias (' + str(grafico) + ')')
        ax2.set_title('D(acum = 15%) por cada rango de visibilidad')
        ax3.set_title('D(acum = 50%) por cada rango de visibilidad')
        ax4.set_title('D(acum = 60%) por cada rango de visibilidad')
        ax5.set_title('D(acum = 85%) por cada rango de visibilidad')
        nom = inicio.strftime('%d%m') + '_' + fin.strftime('%d%m')+'_'+ensayos+'_'+str(grafico)+'_comparacion'
        plt.tight_layout()
        with io.capture_output() as captured:
            plt.savefig(ruta_proces + 'Gráficos/' + nom + '.png')

    return datos


#######################
#######################
### Gráficos "clásicos" de visibilidad
#######################
#######################

def filtrar_corregida(datos,rangos):
    lim = [15,15,15,15,25,25,50,50,50]
    for v in range(len(rangos)-1):
        datos.drop(datos[(datos['Visibilidad corregida (m)'] >= rangos[v]) &
            (datos['Visibilidad corregida (m)'] < rangos[v+1]) &
            (abs(datos['Visibilidad corregida (m)']-datos['Visibilidad (m)']) > lim[v])]
                   .index, inplace = True)
    return datos

def niebla_only(datos,rangos,corregido=False):
    datos.dropna(inplace=True)
    datos.drop(datos[datos['Visibilidad corregida (m)'] == 0].index, inplace=True)
    datos.drop(datos[(datos['Ensayo'].astype(str).str[0] != '8') & (datos['Tiempo (min)'] > 9)].index, inplace = True)
    if corregido:
        filtrar_corregida(datos,rangos)
    return datos

def caract_only(datos,rangos,corregido=True):
    datos.dropna(inplace=True)
    datos.drop(datos[datos['Visibilidad corregida (m)'] == 0].index, inplace=True)
    datos.drop(datos[(datos['Ensayo'].astype(str).str[0] != '8')].index, inplace = True)
    if corregido:
        filtrar_corregida(datos,rangos)
    return datos

def ensayos_only(datos,rangos,corregido=False):
    datos.dropna(inplace=True)
    datos.drop(datos[datos['Visibilidad corregida (m)'] == 0].index, inplace=True)
    datos.drop(datos[(datos['Ensayo'].astype(str).str[0] == '8')
                     | (datos['Ensayo'].astype(str).str[0] == '9')
                     | (datos['Ensayo'].astype(str).str[0:2] == 'EC')].index,
               inplace = True)
    if corregido:
        filtrar_corregida(datos,rangos)
    return datos

def preparar(datos,inicio='01/06/2021',fin='31/07/2021',rangos=[],
             niebla=False, ensayos=False):
    if niebla:
        if ensayos == False:
            datos = caract_only(datos,rangos,corregido=True)
        else:
            datos = niebla_only(datos,rangos,corregido=True)
    else:
        if ensayos:
            datos = ensayos_only(datos,rangos,corregido=True)
            
    inicio = datetime.datetime.strptime(inicio,'%d/%m/%Y')
    fin = datetime.datetime.strptime(fin,'%d/%m/%Y')
    for i in datos.index:
        datos.loc[i,'Hora'] = datetime.datetime.strptime(datos.loc[i,'Hora'], '%d/%m/%Y %H:%M')
    datos.drop(datos[(datos['Hora'] < inicio) | (datos['Hora'] > fin)].index, inplace = True)
    datos.drop(datos[datos['Prec_mensual'] == -9999].index, inplace=True)
    return datos

def separar_ensayos(datos, inicio, fin, rangos=[],niebla=True,ensayos=True):
    if rangos:
        pass
    else: 
        for v in range(6):
            rangos.append(round(np.quantile(datos['Visibilidad corregida (m)'],v/6),1))
            rangos.append(2000)
            
    datos = preparar(datos,inicio,fin,rangos,niebla,ensayos)
            
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
    
    if (len(datos) > 0):
        suma_gruesos = np.empty((len(datos),1))
        unidades = np.array(datos.iloc[:,57:88])
        unidades_norm = np.array(datos.iloc[:,57:88])
        for k in range(unidades.shape[0]):
            suma_gruesos[k] = np.sum(unidades[k,:])
            unidades_norm[k,:] = np.divide(unidades[k,:],suma_gruesos[k])
        for m in range(unidades.shape[1]):
            unidades[:,m] = np.divide(unidades[:,m],dx[m+42])
            unidades_norm[:,m] = np.divide(unidades_norm[:,m],dx[m+42])
    acumulado = np.cumsum(unidades,axis=1)
    acumulado = (acumulado.T/acumulado[:,-1]).T
    
    for i in datos.Ensayo.unique():
        fig = plt.figure(figsize = (16,9))
        gs = gridspec.GridSpec(3, 3, height_ratios=[1,1,1])
        ax15 = plt.subplot(gs[0,0])
        ax30 = plt.subplot(gs[0,1])
        ax45 = plt.subplot(gs[0,2])
        ax60 = plt.subplot(gs[1,0])
        ax75 = plt.subplot(gs[1,1])
        ax100 = plt.subplot(gs[1,2])
        ax200 = plt.subplot(gs[2,0])
        
        ejes = [ax15,ax30,ax45,ax60,ax75,ax100,ax200]
        labels = [2,3,4,5,6,8,10,12,15,18]
        c = 0

        plt.suptitle('Ensayo ' + i + ' - ' +pd.DatetimeIndex(datos
        [datos['Ensayo'] == i].Hora).strftime('%d/%m, %H:%M')[0], size=16)
        
        for eje in ejes:
            ploteo = unidades_norm[(datos['Visibilidad corregida (m)'] >= rangos[c]) &
                            (datos['Visibilidad corregida (m)'] < rangos[c+1])]
            ensayo = unidades_norm[(datos['Visibilidad corregida (m)'] >= rangos[c]) &
                            (datos['Visibilidad corregida (m)'] < rangos[c+1]) &
                            (datos['Ensayo'] == i)]
            
            eje.set_title('Visibilidad de ' + str(rangos[c]) + ' a ' +
                          str(rangos[c+1]) + ' m')
            eje.set_xscale('log'); eje.grid(which = 'both')
            eje.set_xlim(2,18);
            eje.set_xlabel('Diámetro ($\mu$m)'); eje.set_ylabel('dN/N/dx');
            eje.set_xticks(labels); eje.set_xticklabels(labels)
            for j in range(ploteo.shape[0]):
                eje.plot(diams_g,ploteo[j],
                    c = 'black',alpha = 0.1, lw = 0.5)
            for j in range(ensayo.shape[0]):
                eje.plot(diams_g,ensayo[j],
                    c = 'blue', alpha = 0.6, lw = 0.75)
            c = c+1
        plt.tight_layout()
        plt.savefig(ruta_proces + 'Gráficos/Ensayos_Junio/' + i + '_norm.png')

    return 

def visib(datos, inicio, fin, rangos=[], rampa = 'mes'):
    if rangos:
        pass
    else: 
        for v in range(6):
            rangos.append(round(np.quantile(datos['Visibilidad corregida (m)'],v/6),1))
            rangos.append(2000)
            
    datos = preparar(datos,inicio,fin,rangos)
            
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
      
    if rampa == 'tempdisc':
        datos['Temp_corr'] = datos['Temp_est'].apply(np.floor)
    
    for v in range(len(rangos)-1):
        vis = datos[(datos['Visibilidad corregida (m)'] >= rangos[v]) &
        (datos['Visibilidad corregida (m)'] < rangos[v+1])]
        
        ## RAMPA DE COLOR
        if rampa == 'mes':
            color_labels = pd.DatetimeIndex(vis['Hora']).month.unique()
            rgb_values = sns.color_palette("tab10", len(color_labels))
        if rampa == 'ensayo':
            color_labels = vis['Ensayo'].unique()
            rgb_values = sns.color_palette("tab10", len(color_labels))
        if rampa == 'temp':
            color_labels = natsorted(datos['Temp_est'].unique())
            rgb_values = sns.color_palette("flare_r", len(color_labels))
        if rampa == 'tempdisc':
            vis['Temp_corr'] = vis['Temp_est'].apply(np.floor)
            color_labels = natsorted(datos['Temp_corr'].unique())
            rgb_values = sns.color_palette("flare", len(color_labels))
        color_map = dict(zip(color_labels, rgb_values))
        hand = []
        if (rampa != 'temp') & (rampa != 'tempdisc'):
            for i in range(len(rgb_values)):
                hand.append(mpatches.Patch(color=rgb_values[i], label=color_labels[i]))
        else:
            if rampa == 'tempdisc':
                for i in range(len(rgb_values)-1):
                    hand.append(mpatches.Patch(color=rgb_values[i],
                        label=str(color_labels[i]) + ' a ' + str(color_labels[i+1])))
                hand.append(mpatches.Patch(color=rgb_values[i+1],
                        label='>'+str(color_labels[i])))
            if rampa == 'temp':
                muestras = np.linspace(0,len(rgb_values)-1, 10,dtype=int)
                for i in muestras:
                    hand.append(mpatches.Patch(color=rgb_values[i],
                        label=str(round(color_labels[i],2))))
        ##
        
        if (len(vis) > 0):
            suma_gruesos = np.empty((len(vis),1))
            unidades = np.array(vis.iloc[:,57:88])
            unidades_norm = np.array(vis.iloc[:,57:88])
            for k in range(unidades.shape[0]):
                suma_gruesos[k] = np.sum(unidades[k,:])
                unidades_norm[k,:] = np.divide(unidades[k,:],suma_gruesos[k])
            for m in range(unidades.shape[1]):
                unidades[:,m] = np.divide(unidades[:,m],dx[m+42])
                unidades_norm[:,m] = np.divide(unidades_norm[:,m],dx[m+42])
        acumulado = np.cumsum(unidades,axis=1)
        acumulado = (acumulado.T/acumulado[:,-1]).T

        media_norm = np.mean(unidades_norm, axis = 0)
        media = np.mean(unidades, axis = 0)
        media_acum = np.mean(acumulado,axis=0)
        
        fig = plt.figure(figsize = (14,20))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1,1.5,1.5])
        ax = plt.subplot(gs[1,0:2])
        ax_sin = plt.subplot(gs[2,0:2],sharex=ax)
        ax2 = plt.subplot(gs[0,1],sharex=ax)
        plt.suptitle('Visibilidad '+str(rangos[v]) + ' a ' + str(rangos[v+1]) + ' m - '
                              + str(len(vis)) + ' registros\n'+
                              inicio + ' a ' + fin, size=14)
        ax.set_xscale('log'); ax.grid(which='both')
        ax_sin.grid(which='both')
        ax2.grid(which='both'), ax2.set_ylim(0,100)
        ax.set_xlim(2,18); ax.set_ylim(0,0.6);
        ax_sin.set_ylim(0,140)
        ax.set_xlabel('Diámetro ($\mu$m)'); ax.set_ylabel('dN/N/dx');
        ax_sin.set_xlabel('Diámetro ($\mu$m)'); ax_sin.set_ylabel('N* dN/N/dx');
        for i in range(unidades.shape[0]):
            if rampa == 'mes':
                ax.plot(diams_g,unidades_norm[i],
                        c = pd.DatetimeIndex(vis['Hora']).month.map(color_map)[i],
                        alpha = 0.7, lw = 0.5)
                ax_sin.plot(diams_g,unidades[i],
                    c = pd.DatetimeIndex(vis['Hora']).month.map(color_map)[i],
                    alpha = 0.7, lw = 0.5)
                ax2.plot(diams_g,100*acumulado[i],
                    c = pd.DatetimeIndex(vis['Hora']).month.map(color_map)[i],
                    alpha = 0.7, lw = 0.5)       
            if rampa == 'ensayo':
                ax.plot(diams_g,unidades_norm[i],
                        c = vis['Ensayo'].map(color_map)[vis.index[i]],
                        alpha = 0.7, lw = 0.5)
                ax_sin.plot(diams_g,unidades[i],
                    c = vis['Ensayo'].map(color_map)[vis.index[i]],
                    alpha = 0.7, lw = 0.5)
                ax2.plot(diams_g,100*acumulado[i],
                    c = vis['Ensayo'].map(color_map)[vis.index[i]],
                    alpha = 0.7, lw = 0.5)
            if rampa == 'tempdisc':
                ax.plot(diams_g,unidades_norm[i],
                        c = vis['Temp_corr'].map(color_map)[vis.index[i]],
                        alpha = 0.7, lw = 0.5)
                ax_sin.plot(diams_g,unidades[i],
                    c = vis['Temp_corr'].map(color_map)[vis.index[i]],
                    alpha = 0.7, lw = 0.5)
                ax2.plot(diams_g,100*acumulado[i],
                    c = vis['Temp_corr'].map(color_map)[vis.index[i]],
                    alpha = 0.7, lw = 0.5)
            if rampa == 'temp':
                ax.plot(diams_g,unidades_norm[i],
                        c = vis['Temp_est'].map(color_map)[vis.index[i]],
                        alpha = 0.7, lw = 0.5)
                ax_sin.plot(diams_g,unidades[i],
                    c = vis['Temp_est'].map(color_map)[vis.index[i]],
                    alpha = 0.7, lw = 0.5)
                ax2.plot(diams_g,100*acumulado[i],
                    c = vis['Temp_est'].map(color_map)[vis.index[i]],
                    alpha = 0.7, lw = 0.5)
        if rampa == 'mes':
            ax.legend(handles=hand,title='Mes',loc='upper right')
        if rampa == 'ensayo':
            ax.legend(handles=hand,title='Ensayo',loc='upper right')
        if rampa == 'temp':
            ax.legend(handles=hand,title='Temperatura (ºC)',loc='upper right')
            ax_sin.legend(handles=hand,title='Temperatura (ºC)',loc='upper right')
        ax.plot(diams_g,media_norm,lw=2,color='red',label='Media')
        ax_sin.plot(diams_g,media,lw=2,color='red',label='Media')
        ax2.plot(diams_g,100*media_acum,lw=2,color='red',label='Media')
        if rampa == 'mes':
            ax_sin.legend(handles=hand,title='Mes',loc='upper right')
        if rampa == 'ensayo':
            ax_sin.legend(handles=hand,title='Ensayo',loc='upper right')
        labels = [2,3,4,5,6,8,10,12,15,18]
        ax.set_xticks(labels); ax.set_xticklabels(labels)
        plt.savefig(ruta_proces + 'Gráficos/' + str(rangos[v]) + '_'
                    + str(rangos[v+1])+ '.png')

def ensayos_disipacion(datos, inicio, fin, rangos=[]):
    if rangos:
        pass
    else: 
        for v in range(6):
            rangos.append(round(np.quantile(datos['Visibilidad corregida (m)'],v/6),1))
            rangos.append(2000)
            
    datos = preparar(datos,inicio,fin,rangos,ensayos=True)
            
    diams_g = np.array([2.13,2.289,2.46,2.643,2.841,3.053,3.28,3.525,3.788,
                        4.071,4.374,4.701,5.051,5.428,5.833,6.268,6.736,7.239,
                        7.779,8.359,8.983,9.653,10.373,11.147,11.979,12.872,
                        13.833,14.865,15.974,17.165,18.446
                        ])
    diams = np.array([0.104,0.111,0.12,0.129,0.138,0.149,0.16,0.172,0.184,
                      0.198,0.213,0.229,0.246,0.264,0.284,0.305,0.328,0.352,
                      0.379,0.407,0.437,0.47,0.505,0.543,0.583,0.627,0.674,
                      0.724,0.778,0.836,0.898,0.965,1.037,1.115,1.198,1.287,
                      1.383,1.486,1.597,1.717,1.845,1.982,2.13,2.289,2.46,
                      2.643,2.841,3.053,3.28,3.525,3.788,4.071,4.374,4.701,
                      5.051,5.428,5.833,6.268,6.736,7.239,7.779,8.359,8.983,
                      9.653,10.373,11.147,11.979,12.872,13.833,14.865,15.974,
                      17.165,18.446
                      ])
    dx = np.array([0.007,0.008,0.009,0.009,0.01,0.011,0.011,0.012,0.013,0.014,
                   0.015,0.016,0.018,0.019,0.02,0.022,0.024,0.025,0.027,0.029,
                   0.031,0.034,0.036,0.039,0.042,0.045,0.048,0.052,0.056,0.06,
                   0.065,0.069,0.075,0.08,0.086,0.093,0.099,0.107,0.115,0.123,
                   0.133,0.143,0.153,0.165,0.177,0.19,0.204,0.22,0.236,0.254,
                   0.272,0.293,0.315,0.338,0.363,0.39,0.42,0.451,0.484,0.521,
                   0.559,0.601,0.646,0.694,0.746,0.802,0.862,0.926,0.995,1.069,
                   1.149,1.235,1.327
                   ])
    
    if (len(datos) > 0):
        suma_gruesos = np.empty((len(datos),1))
        unidades = np.array(datos.iloc[:,57:88])
        unidades_norm = np.array(datos.iloc[:,57:88])
        for k in range(unidades.shape[0]):
            suma_gruesos[k] = np.sum(unidades[k,:])
            unidades_norm[k,:] = np.divide(unidades[k,:],suma_gruesos[k])
        for m in range(unidades.shape[1]):
            unidades[:,m] = np.divide(unidades[:,m],dx[m+42])
            unidades_norm[:,m] = np.divide(unidades_norm[:,m],dx[m+42])
    acumulado = np.cumsum(unidades,axis=1)
    acumulado = (acumulado.T/acumulado[:,-1]).T
    
    for i in datos.Ensayo.unique():
        fig = plt.figure(figsize = (16,7))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1,1])
        eje = plt.subplot(gs[0,0])
        non = plt.subplot(gs[1,0])
        comp = plt.subplot(gs[0:2,1])
        
        ejes = [eje,non,comp]
        labels = [2,3,4,5,6,8,10,12,15,18]

        plt.suptitle('Ensayo ' + i + ' - ' +pd.DatetimeIndex(datos
        [datos['Ensayo'] == i].Hora).strftime('%d/%m, %H:%M')[0], size=16)
        
        e_ploteo = unidades_norm[(datos['Visibilidad corregida (m)'] >= 10)
                        & (datos['Visibilidad corregida (m)'] < 150)]
        e_nie = unidades_norm[(datos['Visibilidad corregida (m)'] >= 10)
                        & (datos['Visibilidad corregida (m)'] < 150) &
                        (datos['Ensayo'] == i) & (datos['Tiempo (min)'] < 9.5)]
        pe_inic = np.mean(e_nie,axis=0)
        e_dur = unidades_norm[(datos['Visibilidad corregida (m)'] >= 10)
                        & (datos['Visibilidad corregida (m)'] < 150) &
                        (datos['Ensayo'] == i) & (datos['Tiempo (min)'] > 9.5)
                        & (datos['Tiempo (min)'] < 29.5)]
        pe_durante = np.mean(e_dur,axis=0)
        e_post = unidades_norm[(datos['Visibilidad corregida (m)'] >= 10)
                        & (datos['Visibilidad corregida (m)'] < 150) &
                        (datos['Ensayo'] == i) & (datos['Tiempo (min)'] > 29.5)]
        pe_post = np.mean(e_post,axis=0)
        
        n_ploteo = unidades[(datos['Visibilidad corregida (m)'] >= 10)
                        & (datos['Visibilidad corregida (m)'] < 150)]
        n_nie = unidades[(datos['Visibilidad corregida (m)'] >= 10)
                        & (datos['Visibilidad corregida (m)'] < 150) &
                        (datos['Ensayo'] == i) & (datos['Tiempo (min)'] < 9.5)]
        pn_inic = np.mean(e_nie,axis=0)
        n_dur = unidades[(datos['Visibilidad corregida (m)'] >= 10)
                        & (datos['Visibilidad corregida (m)'] < 150) &
                        (datos['Ensayo'] == i) & (datos['Tiempo (min)'] > 9.5)
                        & (datos['Tiempo (min)'] < 29.5)]
        pn_durante = np.mean(e_dur,axis=0)
        n_post = unidades[(datos['Visibilidad corregida (m)'] >= 10)
                        & (datos['Visibilidad corregida (m)'] < 150) &
                        (datos['Ensayo'] == i) & (datos['Tiempo (min)'] > 29.5)]
        pn_post = np.mean(e_post,axis=0)
        
        nieblainic = acumulado[(datos['Visibilidad corregida (m)'] >= 10)
                        & (datos['Visibilidad corregida (m)'] < 150) &
                        (datos['Ensayo'] == i) & (datos['Tiempo (min)'] < 9.5)]
        durante = acumulado[(datos['Visibilidad corregida (m)'] >= 10)
                        & (datos['Visibilidad corregida (m)'] < 150) &
                        (datos['Ensayo'] == i) & (datos['Tiempo (min)'] > 9.5)
                        & (datos['Tiempo (min)'] < 29.5)]
        post = acumulado[(datos['Visibilidad corregida (m)'] >= 10)
                        & (datos['Visibilidad corregida (m)'] < 150) &
                        (datos['Ensayo'] == i) & (datos['Tiempo (min)'] > 29.5)]
        
        eje.set_ylim(0,0.5); non.set_ylim(0,120)
        eje.set_ylabel('dN/N/dx'); non.set_ylabel('N * dN/N/dx')
        
        for a in ejes[0:2]:
            a.set_title('Granulometría - visibilidad de 10 a 150 m')
            a.set_xscale('log'); eje.grid(which = 'both')
            a.set_xlim(2,18);
            a.set_xlabel('Diámetro ($\mu$m)');
            a.set_xticks(labels)
            a.set_xticklabels(labels)
            
        for j in range(e_ploteo.shape[0]):
            eje.plot(diams_g,e_ploteo[j],
                c = 'black',alpha = 0.05, lw = 0.5)
        for j in range(e_nie.shape[0]):
            eje.plot(diams_g,e_nie[j],
                c = 'blue',alpha = 0.5, lw = 0.5)
        for j in range(e_dur.shape[0]):
            eje.plot(diams_g,e_dur[j],
                c = 'red',alpha = 0.5, lw = 0.5)

        for j in range(n_ploteo.shape[0]):
            non.plot(diams_g,n_ploteo[j],
                c = 'black',alpha = 0.05, lw = 0.5)
        for j in range(n_nie.shape[0]):
            non.plot(diams_g,n_nie[j],
                c = 'blue',alpha = 0.5, lw = 0.5)
        for j in range(n_dur.shape[0]):
            non.plot(diams_g,n_dur[j],
                c = 'red',alpha = 0.5, lw = 0.5)
        
        comp.set_title('Comparación de variables durante el ensayo')
        
        
        comp.plot(nieblainic['Tiempo (min)'], 1-nieblainic[:,21],
                  c = 'blue', lw = 1)
        comp.plot(durante['Tiempo (min)'], 1-durante[:,21],
                  c = 'red', lw = 1)
        comp.plot(post['Tiempo (min)'], 1-post[:,21],
                  c = 'blue', lw = 1)
        
        
        plt.tight_layout()
        plt.savefig(ruta_proces + 'Gráficos/Ensayos_disipacion/' + i + '_norm.png')

    return

def valores_tipicos(datos,inicio='01/06/2021',fin='30/09/2021',rangos=[]):
    
    datos = preparar(datos,inicio,fin,rangos,niebla=True,ensayos=True)
            
    diams_g = np.array([2.13,2.289,2.46,2.643,2.841,3.053,3.28,3.525,3.788,
                        4.071,4.374,4.701,5.051,5.428,5.833,6.268,6.736,7.239,
                        7.779,8.359,8.983,9.653,10.373,11.147,11.979,12.872,
                        13.833,14.865,15.974,17.165,18.446
                        ])
    diams = np.array([0.104,0.111,0.12,0.129,0.138,0.149,0.16,0.172,0.184,
                      0.198,0.213,0.229,0.246,0.264,0.284,0.305,0.328,0.352,
                      0.379,0.407,0.437,0.47,0.505,0.543,0.583,0.627,0.674,
                      0.724,0.778,0.836,0.898,0.965,1.037,1.115,1.198,1.287,
                      1.383,1.486,1.597,1.717,1.845,1.982,2.13,2.289,2.46,
                      2.643,2.841,3.053,3.28,3.525,3.788,4.071,4.374,4.701,
                      5.051,5.428,5.833,6.268,6.736,7.239,7.779,8.359,8.983,
                      9.653,10.373,11.147,11.979,12.872,13.833,14.865,15.974,
                      17.165,18.446
                      ])
    dx = np.array([0.007,0.008,0.009,0.009,0.01,0.011,0.011,0.012,0.013,0.014,
                   0.015,0.016,0.018,0.019,0.02,0.022,0.024,0.025,0.027,0.029,
                   0.031,0.034,0.036,0.039,0.042,0.045,0.048,0.052,0.056,0.06,
                   0.065,0.069,0.075,0.08,0.086,0.093,0.099,0.107,0.115,0.123,
                   0.133,0.143,0.153,0.165,0.177,0.19,0.204,0.22,0.236,0.254,
                   0.272,0.293,0.315,0.338,0.363,0.39,0.42,0.451,0.484,0.521,
                   0.559,0.601,0.646,0.694,0.746,0.802,0.862,0.926,0.995,1.069,
                   1.149,1.235,1.327
                   ])
    
    vols = (np.pi/6000000)*(diams_g**3)
    volsfinos = (np.pi/6000000)*(diams**3)

    for v in range(len(rangos)-1):
        vis = datos[(datos['Visibilidad corregida (m)'] >= rangos[v]) &
        (datos['Visibilidad corregida (m)'] < rangos[v+1])]
        
        if (len(datos) > 0):
            suma_gruesos = np.empty((len(vis),1))
            suma_finos = np.empty((len(vis),1))
            unidades = np.array(vis.iloc[:,57:88])
            masas = np.array(vis.iloc[:,57:88])
            masasfinas = np.array(vis.iloc[:,15:57])
            for k in range(unidades.shape[0]):
                masas[k,:] = vols * masas[k,:]
                masasfinas[k,:] = volsfinos[0:42] * masasfinas[k,:]
                suma_gruesos[k] = np.sum(unidades[k,:])
                suma_finos[k] = np.sum(vis.iloc[k,15:57])
            for m in range(unidades.shape[1]):
                unidades[:,m] = np.divide(unidades[:,m],dx[m+42])
                masas[:,m] = np.divide(masas[:,m],dx[m+42])
        acumulado = np.cumsum(unidades,axis=1)
        #acumulado = (acumulado.T/acumulado[:,-1]).T
        
        print('LWC',rangos[v],'a',rangos[v+1],'m:',round(
            np.mean(vis['LWC (g/m3)']),5))
        print('N(finas)',rangos[v],'a',rangos[v+1],'m:',round(
            np.mean(suma_finos),2),'- sigma',round(np.std(suma_finos),2))
        print('N(gruesas)',rangos[v],'a',rangos[v+1],'m:',round(
            np.mean(suma_gruesos),2),'- sigma',round(np.std(suma_gruesos),2))
        print('g/m3 finas',rangos[v],'a',rangos[v+1],'m:',round(
            np.mean(masasfinas),5))
        print('g/m3 gruesas',rangos[v],'a',rangos[v+1],'m:',round(
            np.mean(masas),5))
        print('ug por p. gruesa',rangos[v],'a',rangos[v+1],'m:',round(
            10000000*np.mean(masas)/np.mean(suma_gruesos),6),'\n')
        
    return

########################
########################
#### Programa principal
########################
########################

ruta_proces = 'C:\\Users\\miguel.anton\\Desktop\\NIEBLA\\Ensayos procesados\\'
datos = pd.read_csv(ruta_proces + 'database_modif.csv', delimiter = ";", decimal = ".")
datos.dropna(inplace=True)

rang = [15,30,45,60,75,100,200,1000,1500,2000]

#ensayos_disipacion(datos,inicio='01/07/2021',fin='31/10/2021',rangos=rang)

#visib(datos,inicio='01/06/2021',fin='30/09/2021',rangos=rang,rampa='ensayo')

#datos=distribuciones_cajas(datos,ensayos = '8.',inicio='20/08/2021',
#                           fin='31/08/2021',rangos=rang)

valores_tipicos(datos,inicio='01/06/2021',fin='30/09/2021',rangos=rang)