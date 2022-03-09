# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 14:13:25 2021

@author: Luis Vicente
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense
from tensorflow.keras.layers import BatchNormalization
plt.style.use('Solarize_Light2')

#Semillas para replicar los resultados: 9,2,4,5,7
np.random.seed(9)
tf.random.set_seed(9)


ipc = pd.read_csv('C:/Users/Luis Vicente/Documents/Tesis/Proyecto Tesis/Prediccion-ML/datos_IPC.csv')
ipc['Date']=pd.to_datetime(ipc['Date'])
ipc=ipc.drop(['Adj Close','Volume','High','Open','Low'],axis=1)

a_movil = pd.read_csv('C:/Users/Luis Vicente/Documents/Tesis/Proyecto Tesis/Prediccion-ML/datos_A_Movil.csv')
a_movil['Date']=pd.to_datetime(a_movil['Date'])
a_movil=a_movil.drop(['Adj Close','Volume','High','Open','Low'],axis=1)

walmex = pd.read_csv('C:/Users/Luis Vicente/Documents/Tesis/Proyecto Tesis/Prediccion-ML/datos_WALMEX.csv')
walmex['Date']=pd.to_datetime(walmex['Date'])
walmex=walmex.drop(['Adj Close','Volume','High','Open','Low'],axis=1)

femsa = pd.read_csv('C:/Users/Luis Vicente/Documents/Tesis/Proyecto Tesis/Prediccion-ML/datos_Femsa.csv')
femsa['Date']=pd.to_datetime(femsa['Date'])
femsa=femsa.drop(['Adj Close','Volume','High','Open','Low'],axis=1)

### EL FEMSA TIENE UN VALOR NAN, SUSTITUIRLO CON EL PROMEDIO DE LOS PRECIOS ANTERIORES Y POSTERIORES
#null_columns=femsa.columns[femsa.isnull().any()]
#femsa[null_columns].isnull().sum()
#print(femsa[femsa.isnull().any(axis=1)][null_columns].head())
femsa.iloc[49,1]=((femsa.iloc[47:49,1].sum()+femsa.iloc[50:52,1].sum())/4)

televisa = pd.read_csv('C:/Users/Luis Vicente/Documents/Tesis/Proyecto Tesis/Prediccion-ML/datos_Televisa.csv')
televisa['Date']=pd.to_datetime(televisa['Date'])
televisa=televisa.drop(['Adj Close','Volume','High','Open','Low'],axis=1)

gfnorte = pd.read_csv('C:/Users/Luis Vicente/Documents/Tesis/Proyecto Tesis/Prediccion-ML/datos_GFNorte.csv')
gfnorte['Date']=pd.to_datetime(gfnorte['Date'])
gfnorte=gfnorte.drop(['Adj Close','Volume','High','Open','Low'],axis=1)


from functools import reduce
dfs = [ipc,a_movil,walmex,femsa,televisa,gfnorte]
df = reduce(lambda left,right: pd.merge(left,right,on='Date'), dfs)
df.columns=['Date','IPC','America Movil','Walmex','Femsa','Televisa','GFNorte']


df_final= df.set_index('Date')


dataset = df_final.values
print(dataset[:10])
print(dataset.shape)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
#index_train=ipc.index[ipc['Date']=='2014-12-31'].tolist()
scaled_train=scaler.fit_transform(df_final[:'2014-12-31'])
scaled_validation=scaler.transform(df_final['2015-01-01':'2015-12-31'])
scaled_test=scaler.transform(df_final['2016-01-01':])

#La función MinMaxScaler hace esto
#min(df_final.iloc[:,0])
#max(df_final.iloc[:,0])
#(df_final.iloc[0,0]-min(df_final.iloc[:,0]))/(max(df_final.iloc[:,0])-min(df_final.iloc[:,0]))


def split_sequence(sequence, n_steps):
    """Función que dividie el dataset en datos de entrada y datos que
    funcionan como etiquetas."""
    X, Y = [], []
    for i in range(sequence.shape[0]):
      if (i + n_steps) >= sequence.shape[0]:
        break
      # Divide los datos entre datos de entrada (input) y etiquetas (output)
      seq_X, seq_Y = sequence[i: i + n_steps], sequence[i + n_steps, 0]
      X.append(seq_X)
      Y.append(seq_Y)
    return np.array(X), np.array(Y)

#dataset_size = scaled_dataset.shape[0]
#index_train=df.index[df['Date']=='2014-12-31'].tolist()
#index_validation=df.index[df['Date']=='2015-12-31'].tolist()

#Entrenamiento y validación
x_train, y_train = split_sequence(scaled_train, 30)
x_val, y_val = split_sequence(scaled_validation, 30)


print("dataset.shape: {}".format(dataset.shape))
print("df.shape: {}".format(df.shape))
print("x_train.shape: {}".format(x_train.shape))
print("y_train.shape: {}".format(y_train.shape))
print("x_val.shape: {}".format(x_val.shape))
print("y_val.shape: {}".format(y_val.shape))
print('=======================')


batch_size = 100
buffer_size = x_train.shape[0]
# Crea un conjunto de datos que incrementa de acuerdo a lo que se necesite
train_iterator = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size).batch(batch_size).repeat()
# Crea un conjunto de datos que incrementa de acuerdo a lo que se necesite
val_iterator = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).repeat()


n_steps = x_train.shape[-2]
n_features = x_train.shape[-1]

#Se define el modelo
model = Sequential()
model.add((LSTM(50, activation='tanh',return_sequences=False,input_shape=(n_steps, n_features))))
#model.add(BatchNormalization(center=False))
model.add(Dense(1))


# Compilación del modelo 
model.compile(optimizer='adam', loss='mse')


epochs = 60
steps_per_epoch = 95
validation_steps = 45
# Entrenar y validar el modelo
history = model.fit(train_iterator, epochs=epochs,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=val_iterator,
                    validation_steps=validation_steps)



def plot_1(history, title):
  """función que grafica los errores obtenidos del modelo"""
  plt.figure(figsize=(8,6))
  plt.plot(history.history['loss'], 'o-', mfc='none', markersize=10, 
  label='Train',color='deepskyblue')
  plt.plot(history.history['val_loss'], 'o-', mfc='none', 
  markersize=10, label='Validation',color='green')
  plt.title('Curva de aprendizaje')
  plt.xlabel('Epocas')
  plt.ylabel('Error cuadrático medio')
  plt.legend()
  plt.show()
  
# Gráfica de la curva de aprendizaje del modelo en los conjuntos de 
#entrenamiento y validación
plot_1(history, 'Training / Validation Losses from History')


x_test,y_test=split_sequence(scaled_test, 30)
print(x_test.shape)
print(y_test.shape)


predictions = model.predict(x_test)

y_plot=np.delete(y_test,-1)
predictions_plot=np.delete(predictions,0)
plt.figure(figsize=(15,8))
plt.plot(y_plot)
plt.plot(predictions_plot)
xlabels = np.arange(len(y_plot))
plt.plot(xlabels, y_plot, label= 'Actual',color='deepskyblue')
plt.plot(xlabels, predictions_plot, label = 'Predicción',color='green')
plt.legend()


#MSE

from sklearn.metrics import mean_squared_error

mean_squared_error(y_test,predictions)


#Predecir el siguiente día 

a=[]
a.append(scaled_test[222:252])
#scaler.inverse_transform(a)
a=np.array(a)
a.shape
prediction_1=model.predict(a)
(prediction_1*(max(df_final[:'2014-12-31']['IPC'])-min(df_final[:'2014-12-31']['IPC'])))+min(df_final[:'2014-12-31']['IPC'])


#Graficas de barras
configuraciones = ['LSTM Bidireccional \ncon BN', 'LSTM\n Bidireccional', 'LSTM con BN', 'LSTM']
predicciones = [45058.062,45447.117,45975.86,45457.445]
plt.bar(configuraciones,predicciones,color=['red','blue','orange','brown'])
plt.xticks( configuraciones,  ha='center', rotation=0, fontsize=9.5, fontname='monospace')
plt.axhline(y=45695.10, color='black', linestyle='-')
plt.ylim(44000,46000)
plt.show()


#Graficas de puntos 
plt.plot(45695.10,marker="o",label='Real',color='red')
plt.plot(45699.902,marker="x",label='LSTM Bidireccional con BN',color='royalblue',markeredgewidth=2,markersize=8.5)
plt.plot(45751.055,marker="x",label='LSTM Bidireccional',color='black',markersize=8.5)
plt.plot(44931.34,marker="x",label='LSTM con BN',color='orange',markersize=8.5)
plt.plot(45619.64,marker="x",label='LSTM',color='green',markersize=8.5)
plt.legend(fontsize=8)


#seed(9)
#Bidirectional y BatchNormalization
#0.0008707808349343966
#El predicho es 45058.062

#Bidirectional 
#0.0025379629895917787
#El predicho es 45447.117

#Unidireccional y BatchNormalization
#0.001438217204221353
#El predicho es 45975.86

#Unidireccional 
#0.0012667130808765852
#El predicho es 45457.445

#seed(2)
#Bidirectional y BatchNormalization
#0.008523466138093681
#El predicho es 46269.285

#Bidirectional 
#0.001520438242504173
#El predicho es 44783.652

#Unidireccional y BatchNormalization
#0.013384371911337825
#El predicho es 43845.812

#Unidireccional 
#0.002381212465281249
#El predicho es 45184.297

#seed(4)
#Bidirectional y BatchNormalization
#0.0016696801035729907
#El predicho es 44645.08

#Bidirectional 
#0.0016062857815340383
#El predicho es 45435.55

#Unidireccional y BatchNormalization
#0.0024235519046709314
#El predicho es 44812.117

#Unidireccional 
#0.002026809279027949
#El predicho es 45151.004

#seed(5)
#Bidirectional y BatchNormalization
#0.001140110452950318
#El predicho es 45016.297

#Bidirectional 
#0.0025490979456039738
#El predicho es 44939.812

#Unidireccional y BatchNormalization
#0.001689180543661203
#El predicho es 45201.492

#Unidireccional 
#0.0007931982984805833
#El predicho es 45185.6

#seed(7)
#Bidirectional y BatchNormalization
#0.0009431551826209658
#El predicho es 44989.555

#Bidirectional 
#0.0016273854987611963
#El predicho es 44953.953

#Unidireccional y BatchNormalization
#0.0036129981379982965
#El predicho es 46386.332

#Unidireccional 
#0.0008125055532174477
#El predicho es 45440.547


#Media
np.mean((0.000871,0.008523,0.00167,0.00114,0.000943))
np.mean((0.002538,0.00152,0.001606,0.00254,0.001627))
np.mean((0.001438,0.013384,0.002424,0.001689,0.003613))
np.mean((0.001267,0.002381,0.002027,0.000793,0.000812))

#Desviación estándar
np.std((0.000871,0.008523,0.00167,0.00114,0.000943))
np.std((0.002538,0.00152,0.001606,0.00254,0.001627))
np.std((0.001438,0.013384,0.002424,0.001689,0.003613))
np.std((0.001267,0.002381,0.002027,0.000793,0.000812))
