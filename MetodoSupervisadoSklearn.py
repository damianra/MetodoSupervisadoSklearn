# -*- coding: utf-8 -*-
"""
Damian Ramirez

Se utilizara la libreria de Sklearn KNeighborsClassifier de aprendisaje 
supervisado para analizar conjuntos de datos provenientes de capturas de 
trafico de red infectada por el malware 2017-01-19-pseudoDarkleech-Rig-V-sends-Cerber 
(https://www.malware-traffic-analysis.net/2017/01/19/index2.html), el cual 
tiene un comportamiento que podria ser clasificado a simple vista por el ser 
humano. Se utilizaron dos dataframes uno para test 
(se clasifico el malware como M y normal como N) y otro para 
entrenamiento (no esta clasificado).
"""
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import MinMaxScaler
import mglearn
from sklearn.decomposition import PCA

# Obtenemos los datos del archivo csv con pandas y se crea una nueva columna

dataframe = pd.read_csv('trafico.csv', index_col = 'Time')

dataframe['Count'] = np.nan

# Se filtran los datos con groupby() que contengan los mismos valores 
# y se guarda el conteo de los mismos en la nueva columna.
df = dataframe.groupby(['Time', 'Src Port', 'Dst Port', 'Source', 'Protocol', 'Length', 'Type']).size().reset_index(name='counts')
dfc= df.copy()

"""Con la libreria LabelEncoder() codificamos nuestros datos creando relaciones
 entre los mismos devolviendonos una tabla de numeros donde cada numero 
 identifica a un dato unico en la tabla permitiendonos utilizar los algoritmos
 de Machine Learning sin tener problemas de formato y que pueda entenderlos.
 Tambien con el objeto MinMaxScaler escalamos nuestros datos para poder 
 visualizar el comportamiento"""

encoder = LabelEncoder()

dfc['Time'] = encoder.fit_transform(dfc['Time'])
dfc['Src Port'] = encoder.fit_transform(dfc['Src Port'])
dfc['Dst Port'] = encoder.fit_transform(dfc['Dst Port'])
dfc['Source'] = encoder.fit_transform(dfc['Source'])
dfc['Protocol'] = encoder.fit_transform(dfc['Protocol'])

x = dfc.iloc[:, [ 0, 1, 2, 3, 4, 5, 7]]
y = dfc.iloc[:, -2]


# datos que son consecutivos
esc = dfc.iloc[:, [ 0, 1, 2, 3, 4, 5, 7]]
escala = MinMaxScaler()
escala.fit(esc)
escalada = escala.transform(esc)
pca=PCA(n_components=2)
pca.fit(escalada)
transformada=pca.transform(escalada)

mglearn.discrete_scatter(transformada[:,0], transformada[:,1], dfc['Type'])

# Creamos un objeto de la clase KNeighborsClassifier() con una cantidad de vecinos en 2
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x, y)

# Scrore de acierto, con 2 llega a 100%, hasta ahora con las pruebas que 
# realice no demuestra un sobre ajuste

knn.score(x, y)

# Se cargan los datos para la prueba
dataframe2 = pd.read_csv('trafico_prueba_2016.csv', index_col = 'Time')

dataframe2['Count'] = np.nan

# Se reagrupan los datos como en el test y se guarda una copia del dataframe para la prediccion
df2 = dataframe2.groupby(['Time', 'Src Port', 'Dst Port', 'Source', 'Protocol', 'Length']).size().reset_index(name='counts')
dfpredict = df2.copy()

# Codificacion de los datos con el mismo objeto LabelEncoder(), 
# tambien creamos una escala para analizar el comportamiento graficamente

dfpredict['Time'] = encoder.fit_transform(dfpredict['Time'])
dfpredict['Src Port'] = encoder.fit_transform(dfpredict['Src Port'])
dfpredict['Dst Port'] = encoder.fit_transform(dfpredict['Dst Port'])
dfpredict['Source'] = encoder.fit_transform(df2['Source'])
dfpredict['Protocol'] = encoder.fit_transform(dfpredict['Protocol'])

x2 = dfpredict

esc2 = dfpredict
escala2 = MinMaxScaler()
escala2.fit(esc2)
escalada2 = escala.transform(esc2)
pca2=PCA(n_components=2)
pca2.fit(escalada2)
transformada2=pca2.transform(escalada2)

mglearn.discrete_scatter(transformada[:,0], transformada[:,1])

# Creamos una variable que gardara el array con la prediccion y guarda esos datos en una columna del dataframe

knnpred = knn.predict(x2)

df2['Prediction'] = knnpred
print(df2[-50:])

