"""
Autor: Jairo Valea López

Programa de apoyo al análisis de fotografías en el entorno del prototipo
de disipación de niebla en la autovía A-8 (Fiouco) del GSJ.

Se valora también la detección automática del contorno del cartel.
"""

# Cargado de librerías necesarias

import os
import csv
import time
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import datetime  # formato fecha
# import seaborn as sbs
from natsort import natsorted  # ordenación de las imágenes
from tqdm.notebook import tqdm, trange, tnrange  # barra de progreso

ruta_proces = 'C:/Users/miguel.anton/Desktop/NIEBLA/Videos grabados/'

carpeta = natsorted(os.listdir(ruta_proces))
procesados = []
nombres = []

for f in carpeta:
    name, ext = os.path.splitext(f)
    if ext == '.csv':
        procesados.append(pd.read_csv(ruta_proces + name +
                          ext, delimiter=";", decimal="."))
        nombres.append(name + ext)

j = 10

procesados[j].drop([0], inplace=True)
for i in (procesados[j].index):
    procesados[j].loc[i,'Hora'] = datetime.datetime.strptime(procesados[j].loc[i,'Hora'], '%H:%M:%S')


fig, ax= plt.subplots(figsize = (14, 5))
plt.plot(procesados[j]['Hora'], procesados[j]['Mejora_m'], label='Diferencia de visibilidad (estimada)')
plt.plot(procesados[j]['Hora'], procesados[j]['Vis_fuera_m'], c='red',ls='--',lw=1,
         label="Visibilidad fuera (estimada)")
plt.plot(procesados[j]['Hora'], procesados[j]['Vis_dentro_m'], c='green',ls='--',lw=1,
         label="Visibilidad dentro (estimada)")

plt.axhline(y=0, color='black', linewidth=2, linestyle='--')

plt.axvline(datetime.datetime(1900, 1, 1, 9, 45, 0),label="Inicio difusi�n continua",c='purple',ls='-',lw=1.5)
plt.axvline(datetime.datetime(1900, 1, 1, 9, 48, 12),label="Inicio difusi�n a intervalos",c='purple',ls='--',lw=1.5)
#plt.axvline(datetime.datetime(1900, 1, 1, 10, 5, 0), label="Fin difusi�n a intervalos", color='purple', linestyle='--',linewidth=1.5)

ax.legend(loc="upper left")
ax.set_title(nombres[j][0:-4])
ax.set_xlabel('Hora')
ax.set_ylabel('Visibilidad (m)')
ax.set_ylim(-20, 120)
ax.grid(True)
# formato correcto de hora:minuto en los ejes
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.savefig(ruta_proces + nombres[j][0:-4] +  '.png')
plt.show()
