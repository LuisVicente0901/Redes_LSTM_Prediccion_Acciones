# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 19:17:10 2021

@author: Luis Vicente
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense
from tensorflow.keras.layers import BatchNormalization
#from keras.layers.normalization import BatchNormalization
plt.style.use('Solarize_Light2')

#Semillas para replicar los resultados: 9,2,4,5,7
np.random.seed(9)
tf.random.set_seed(9)

ipc = pd.read_csv('C:/Users/Luis Vicente/Documents/Tesis/Proyecto Tesis/Prediccion-ML/datos_IPC.csv')
ipc['Date']=pd.to_datetime(ipc['Date'])
ipc=ipc.drop(['Adj Close','Volume','High','Open','Low'],axis=1)
ipc['Date'] = pd.to_datetime(ipc['Date'])

df_final= ipc.set_index('Date')


dataset = df_final.values
print(dataset[:10])
print(dataset.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
#index_train=ipc.index[ipc['Date']=='2014-12-31'].tolist()
scaled_train=scaler.fit_transform(df_final[:'2014-12-31'])
scaled_validation=scaler.transform(df_final['2015-01-01':'2015-12-31'])
scaled_test=scaler.transform(df_final['2016-01-01':])

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
#index_train=ipc.index[ipc['Date']=='2014-12-31'].tolist()
#index_validation=ipc.index[ipc['Date']=='2015-12-31'].tolist()

#Entrenamiento y validación
x_train, y_train = split_sequence(scaled_train, 30)
x_val, y_val = split_sequence(scaled_validation, 30)


print("dataset.shape: {}".format(dataset.shape))
print("df.shape: {}".format(ipc.shape))
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



def graficar_estrategia(datos,numero_veces):
    comprar=[]
    vender=[]
    indice_c=[]
    indice_v=[]
    for i in range(1,len(datos)):
        if (i*numero_veces) >= len(datos):
          break
        if (datos[(i*numero_veces)-1])<(datos[i*numero_veces]):
            comprar.append(datos[(i*numero_veces)-1])
            indice_c.append((i*numero_veces)-1)
        else:
            vender.append(datos[(i*numero_veces)-1])
            indice_v.append((i*numero_veces)-1)
    if len(datos)>800:
        plt.figure(figsize=(20,10))
    else:
        plt.figure(figsize=(15,8))
    plt.plot(datos,color='green')
    #plt.plot(predictions_plot)
    #plt.plot(xlabels, y_plot, label= 'Actual',color='deepskyblue')
    #plt.plot(xlabels, predictions_plot, label = 'Predicción',color='green')
    plt.plot(indice_c,comprar,'ro',label='comprar')
    plt.plot(indice_v,vender,'ko',label='vender')
    plt.title('Estrategia de inversión')
    plt.legend()

graficar_estrategia(y_train,50)
graficar_estrategia(y_val,20)
graficar_estrategia(y_plot,2)    

#Plot de la comparación entre lo predicho y los valores actuales no escalados
# y_test=y_test.reshape(y_test.shape[0],1)
# plt.plot(scaler.inverse_transform(y_test))
# plt.plot(scaler.inverse_transform(predictions))
# xlabels = np.arange(len(y_test))
# plt.plot(xlabels, scaler.inverse_transform(y_test), label= 'Actual')
# plt.plot(xlabels, scaler.inverse_transform(predictions), label = 'Pred')
# plt.legend()


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
scaler.inverse_transform(prediction_1)


#Graficas de barras
configuraciones = ['LSTM Bidireccional \ncon BN', 'LSTM\n Bidireccional', 'LSTM con BN', 'LSTM']
predicciones = [45345.48,45673.17,46327.68,45708.73]
plt.bar(configuraciones,predicciones,color=['red','blue','orange','brown'])
plt.xticks( configuraciones,  ha='center', rotation=0, fontsize=9.5, fontname='monospace')
plt.axhline(y=45695.10, color='black', linestyle='-')
plt.ylim(45000,46600)
plt.show()

#seed(9)
#Bidirectional y BatchNormalization
#0.0009340155405005492
#El predicho es 45345.484

#Bidirectional 
#0.0005719862860689206
#El predicho es 45673.168

#Unidireccional y BatchNormalization
#0.002699896851938102
#El predicho es 46327.68

#Unidireccional 
#0.0005809741081875289
#El predicho es 45708.727

#seed(2)
#Bidirectional y BatchNormalization
#0.001450946196315766
#El predicho es 45157.363

#Bidirectional 
#0.0005766674708193246
#El predicho es 45589.176

#Unidireccional y BatchNormalization
#0.0008587485350082119
#El predicho es 45865.254

#Unidireccional 
#0.0005814921134298409
#El predicho es 45543.605

#seed(4)
#Bidirectional y BatchNormalization
#0.0014206335351908338
#El predicho es 45171.62

#Bidirectional 
#0.0005953588014149803
#El predicho es 45739.137

#Unidireccional y BatchNormalization
#0.001250360161472491
#El predicho es 45210.305

#Unidireccional 
#0.0005711297748409037
#El predicho es 45575.277

#seed(5)
#Bidirectional y BatchNormalization
#0.0008575452235002183
#El predicho es 45877.816

#Bidirectional 
#0.0006111861180329707
#El predicho es 45525.15

#Unidireccional y BatchNormalization
#0.0016723078008160316
#El predicho es 45123.332

#Unidireccional 
#0.0005905067994229348
#El predicho es 45587.24

#seed(7)
#Bidirectional y BatchNormalization
#0.009158661545603379
#El predicho es 44231.492

#Bidirectional 
#0.000679705080898848
#El predicho es 45471.215

#Unidireccional y BatchNormalization
#0.001815310894900897
#El predicho es 45126.684

#Unidireccional 
#0.0006477012451894649
#El predicho es 45479.375



#Media
np.mean((0.000934,0.00145,0.00142,0.000857,0.009159))
np.mean((0.000572,0.000577,0.000595,0.000611,0.00068))
np.mean((0.0027,0.000859,0.00125,0.001672,0.001815))
np.mean((0.000581,0.000581,0.000571,0.000591,0.000648))

#Desviación estándar
np.std((0.000934,0.00145,0.00142,0.000857,0.009159))
np.std((0.000572,0.000577,0.000595,0.000611,0.00068))
np.std((0.0027,0.000859,0.00125,0.001672,0.001815))
np.std((0.000581,0.000581,0.000571,0.000591,0.000648))
