###############################
##### Autor: Jairo Valea LÃ³pez
###############################

import os
import csv
import time
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import datetime # formato fecha
import cv2 # AKA openCV
import seaborn as sbs
from PIL import ImageFont, ImageDraw, Image 
from natsort import natsorted # ordenaciÃ³n de las imÃ¡genes
from random import random, randint # aleatoriedad

# Funciones propias

def promedio(lst):
    a = sum(lst)/(len(lst)+1)
    return a

def sig_rand():
    y = random()
    if (y < 0.5):
        s = -1
    else: s = 1
    return s

def coordenadas(images,numpuntos):
    c_neg_fuera = []   # lista de coord de los puntos tomados de la zona negra del cartel de fuera
    c_bla_fuera = []   # etc
    c_neg_dentro = []  # 
    c_bla_dentro = []
     #
    for i in range(numpuntos):
        c_neg_fuera.append([centro_fuera[1]+randint(-16,-3),centro_fuera[0]+sig_rand()*randint(25,29)])
        c_bla_fuera.append([centro_fuera[1]+randint(-14,0),centro_fuera[0]+sig_rand()*randint(12,16)])
        c_neg_dentro.append([centro_dentro[1]+sig_rand()*randint(0,24),centro_dentro[0]+sig_rand()*randint(22,27)])
        c_bla_dentro.append([centro_dentro[1]+randint(-13,5),centro_dentro[0]+sig_rand()*randint(9,15)])
    return c_bla_fuera,c_neg_fuera,c_bla_dentro,c_neg_dentro

def visibilidad(blanco,negro):
    v_claro = 181.67 # obtenida a partir de la Ãºltima imagen del Ensayo 15 (tomada 11 Ago 2021 - 12:39:26)
    # distancia estimada de la foto al cartel
    d = 58
    promedios = np.clip(promedio(blanco) - promedio(negro), a_min = 0.001, a_max = 255)
    viz = np.clip(round(3/((-1/d)*np.log((promedios)/v_claro)),2), a_min = 10, a_max = 2000)
    return viz
    
def analizador(images,c_bla_fuera,c_neg_fuera,c_bla_dentro,c_neg_dentro):
    neg_fuera = []     # valores de la luminosidad de los puntos tomados de c_neg_fuera
    bla_fuera = []     # etc
    neg_dentro = []    #
    bla_dentro = []
    for i in range(len(neg_fuera)):
        neg_fuera.append(images[c_neg_fuera[i][0],c_neg_fuera[i][1]])
        bla_fuera.append(images[c_bla_fuera[i][0],c_bla_fuera[i][1]])
        neg_dentro.append(images[c_neg_dentro[i][0],c_neg_dentro[i][1]])
        bla_dentro.append(images[c_bla_dentro[i][0],c_bla_dentro[i][1]])  
    visi_fuera = visibilidad(bla_fuera,neg_fuera)    
    visi_dentro = visibilidad(bla_dentro,neg_dentro)
    return visi_fuera,visi_dentro


###############################
##### Cargado del vÃ­deo
###############################

ruta = 'C:/Users/miguel.anton/Desktop/NIEBLA/Videos grabados/'

carpeta = natsorted(os.listdir(ruta))
nombres = []

for f in carpeta:
    name, ext = os.path.splitext(f)
    if ext == '.mp4':
        nombres.append(name + ext)
        
###############################
##### Cambiar para cambiar el vídeo a procesar
n = 1
###############################

if (n==0): # CS25 inicio
    centro_fuera = (607,363)
    centro_dentro = (1061,385)
if (n==1): # CS27 inicio
    centro_fuera = (659,474)
    centro_dentro = (1113,495)
if (n==2): # CS27 final
    centro_fuera = (658,287)
    centro_dentro = (1114,323)
if (n==3): # CS28 inicial (fusionado)
    centro_fuera = (589,512)
    centro_dentro = (1044,533)
if (n==4): # CS28 final
    centro_fuera = (601,463)
    centro_dentro = (1055,489)
if (n==5): # CS29 inicial
    centro_fuera = (594,456)
    centro_dentro = (1049,470)
if (n==6): # CS29 final
    centro_fuera = (623,490)
    centro_dentro = (1075,504)
if (n==7): # CS30 inicial
    centro_fuera = (617,406)
    centro_dentro = (1072,420)
if (n==8): # CS30 final
    centro_fuera = (622,479)
    centro_dentro = (1074,498)
if (n==9): # CS31 inicial
    centro_fuera = (737,536)
    centro_dentro = (1187,540)
if (n==10): # CS31 final
    centro_fuera = (668,411)
    centro_dentro = (1121,428)

video = cv2.VideoCapture(ruta + nombres[n])
tot_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
ancho_orig = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
alto_orig = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
iniciovideo = datetime.datetime.strptime(nombres[n][5:-4], '%Y%m%d_%H%M%S')
cv2.namedWindow("Ventana")

ancho = int(ancho_orig)
alto = int(alto_orig)
nframe = 0
vfuera = []
vdentro = []
vf = []
vd = []
dif = []
out = cv2.VideoWriter(ruta + nombres[n][:-4] + '.avi',
                     cv2.VideoWriter_fourcc('M','J','P','G'), 25, (ancho,alto))

# Framerate original: 30 fps
# Multiplicado x8 por los frames escogidos

f = open(ruta + nombres[n][:-4] + '_vis_est.csv', 'w', encoding='UTF8', newline='')
writer = csv.writer(f, delimiter=';')
writer.writerow(['Tiempo_min','Vis_fuera_m','Vis_dentro_m','Mejora_m'])

###############################
##### Bucle de visionado
###############################

while(video.isOpened()):
    nframe = nframe+1
    rval, imagen = video.read() # video.read obtiene el frame actual y avanza al siguiente
    if ((rval == True)):
        im = cv2.resize(imagen, (ancho, alto))
        im_copia = im.copy()
    
    if ((nframe>149) & (nframe<tot_frames-200)): # ignoramos los primeros segundos
        imbn = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        bf,nf,bd,nd = coordenadas(imbn,10) # nÃºmero de puntos
        neg_fuera = []     # valores de la luminosidad de los puntos tomados de c_neg_fuera
        bla_fuera = []     # etc
        neg_dentro = []    #
        bla_dentro = []
        for i in range(len(bf)):
            neg_fuera.append(imbn[nf[i][0],nf[i][1]])
            bla_fuera.append(imbn[bf[i][0],bf[i][1]])
            neg_dentro.append(imbn[nd[i][0],nd[i][1]])
            bla_dentro.append(imbn[bd[i][0],bd[i][1]])
        vf.append(visibilidad(bla_fuera,neg_fuera))
        vd.append(visibilidad(bla_dentro,neg_dentro))
        # AnotaciÃ³n de las imÃ¡genes y mostrado por pantalla
        if (nframe % 150 == 0): # a 25 fps, cada 10 segundos
            vfuera.append(round(promedio(vf),1))
            vdentro.append(round(promedio(vd),1))
            dif.append(round(vdentro[-1]-vfuera[-1],1))
            writer.writerow([round(video.get(cv2.CAP_PROP_POS_MSEC)/60000,2),vfuera[-1],vdentro[-1],dif[-1]])
            vf = []
            vd = []
        texto_fuera = 'Visibilidad estimada fuera: ' + str(vfuera[-1])
        texto_dentro = 'Visibilidad estimada dentro: ' + str(vdentro[-1])
        texto_medio = 'Mejora: ' + str(dif[-1]) + ' m'
        fuera = (int(centro_fuera[0]-180),int(centro_fuera[1] + 140))
        dentro = (int(centro_dentro[0]-180),int(centro_fuera[1] + 140))
        medio = (int(centro_fuera[0]+100),int(centro_fuera[1] + 280))
        cv2.rectangle(im_copia, (fuera[0]-10,fuera[1]-25), (fuera[0]+380,fuera[1]+15), (0,0,0), -1)
        cv2.putText(im_copia, texto_fuera, fuera, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.7, color = (255,255,255))
        cv2.rectangle(im_copia, (dentro[0]-10,dentro[1]-25), (dentro[0]+390,dentro[1]+15), (0,0,0), -1)
        cv2.putText(im_copia, texto_dentro, dentro, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.7, color = (255,255,255))
        cv2.rectangle(im_copia, (medio[0]-10,medio[1]-25), (medio[0]+225,medio[1]+15), (0,0,0), -1)
        cv2.putText(im_copia, texto_medio, medio, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.8, color = (255,255,255))
        for z in range(len(bf)):
            cv2.circle(im_copia, (bf[z][1],bf[z][0]), 1, (0,255,0), thickness=-3, lineType=cv2.LINE_AA)
            cv2.circle(im_copia, (nf[z][1],nf[z][0]), 1, (0,255,255), thickness=-3, lineType=cv2.LINE_AA)
            cv2.circle(im_copia, (bd[z][1],bd[z][0]), 1, (0,255,0), thickness=-3, lineType=cv2.LINE_AA)
            cv2.circle(im_copia, (nd[z][1],nd[z][0]), 1, (0,255,255), thickness=-3, lineType=cv2.LINE_AA)
        cv2.circle(im_copia, centro_fuera, 2, (255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(im_copia, centro_dentro, 2, (255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
        
        cv2.rectangle(im_copia, (10,10), (320,70), (0,0,0), -1)
        
        msec = video.get(cv2.CAP_PROP_POS_MSEC)/1000
        horaactual = iniciovideo + datetime.timedelta(seconds = msec)
        tiempo_inst = horaactual.strftime("%H:%M:%S")
        cv2.putText(im_copia, tiempo_inst, (20,60), fontFace = cv2.FONT_HERSHEY_TRIPLEX, fontScale = 2, color = (255,255,255))
        
        if (nframe % 4 == 0): # a efectos prácticos aumenta velocidad x4
            out.write(im_copia)
    if ((rval == True)):
        cv2.imshow('Ventana',im_copia)
        if cv2.waitKey(1) & 0xFF == ord('q'):
        # Velocidad
        # waitKey(20) = 20 ms de espera entre frames = 50 fps
        # waitKey(5) = 5 ms de espera entre frames = 200 fps
        # waitKey(2) = 2 ms de espera entre frames = 500 fps
            out.release()
            f.close()
            break
    else:
        break
video.release()
cv2.destroyAllWindows()
out.release()
f.close()