# Proxecto Néboa Fiouco - Fase 2
Compendio de códigos utilizados no proxecto de dispersión de néboa do Grupo Sanjosé (no entorno da autovía A-8 ao seu paso polo Alto do Fiouco).

Isto inclúe, entre outros, programas de graficado, de análise de espectrometría e volcado en formato vídeo, ou de aprendizaxe automático.

**ajuste_curva:** Intenta axustar as curvas de granulometría obtidas nos ensaios a unha distribución conocida (Gamma, lognormal). Paquetes principais: stats. Formato Jupyter Python Notebook.

**analizador_video:** Analiza a visibilidade teórica dos vídeos de ensaio dispoñibles, exporta un vídeo acelerado x8 anotando a visibilidade instantánea cada 10 segundos, ademáis dunha táboa cos resultados. Paquetes principais: opencv (cv2). Formato iPython.

Exemplos dos resultados obtidos en https://drive.google.com/drive/u/0/folders/1mvqNDyd0jeS0wm13GLxfEWV74Ron2kU1

**fractal:** Calcula a distribución fractal para cada minuto de ensaios de só-néboa, exporta gráficos duales cos gráficos log/log e a evolución da dimensión fractal no tempo. Paquetes principais: pandas, matplotlib, seaborn. Formato Jupyter Python Notebook.

**graficador:** Programa estándar de graficado, multifunción:
  - debuxa os gráficos de distribución de partícula dos ensaios agrupados pola súa visibilidade
  - constrúe gráficos de caixas coa caracterización granulométrica dos ensaios
  - debuxa gráficos de masas, superficies, etc, con rampas de cor clasificatorias
  - etc

**machine_niebla:** Aplicacións de aprendizaxe automática ao análise das néboas do Fiouco - predicción mediante bosques aleatorios e gradient-boosters, clasificación en clústeres mediante algoritmo k-fold, etc. Paquetes principais: sklearn.

**rosinrammler:** Calcula a distribución de Rosin-Rammler para cada minuto de ensaios de só-néboa, exporta gráficos duales coas pendentes do gráfico log/loglog (sí, dous logs no eixo y) e coa distribución teórica. Paquetes principais: pandas, matplotlib, seaborn. Formato Jupyter Python Notebook.

**video_distribucion:** tbc
