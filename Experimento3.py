# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 17:52:27 2021

@author: Luis Vicente
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense
from keras.layers.normalization import BatchNormalization
plt.style.use('Solarize_Light2')

#Semillas para replicar los resultados: 9,2,4,5,7
np.random.seed(9)
tf.random.set_seed(9)

a_movil = pd.read_csv('C:/Users/Luis Vicente/Documents/Tesis/Proyecto Tesis/Prediccion-ML/datos_A_Movil.csv')
a_movil['Date']=pd.to_datetime(a_movil['Date'])
a_movil=a_movil.drop(['Adj Close','Volume','High','Open','Low'],axis=1)
a_movil['Date'] = pd.to_datetime(a_movil['Date'])

df_final= a_movil.set_index('Date')


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
#index_train=a_movil.index[a_movil['Date']=='2014-12-31'].tolist()
#index_validation=a_movil.index[a_movil['Date']=='2015-12-31'].tolist()

#Entrenamiento y validación
x_train, y_train = split_sequence(scaled_train, 30)
x_val, y_val = split_sequence(scaled_validation, 30)

print("dataset.shape: {}".format(dataset.shape))
print("df.shape: {}".format(a_movil.shape))
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
model.add(Bidirectional(LSTM(50, activation='tanh',return_sequences=False,input_shape=(n_steps, n_features))))
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



#Plot de la comparación entre lo predicho y los valores actuales no escalados
#y_test=y_test.reshape(y_test.shape[0],1)
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

#scaler.inverse_transform(predictions)


#Grafica de barras
configuraciones = ['LSTM Bidireccional \ncon BN', 'LSTM\n Bidireccional', 'LSTM con BN', 'LSTM']
predicciones = [13.161911,13.062797,12.934747,13.081673]
plt.bar(configuraciones,predicciones,color=['red','blue','orange','brown'])
plt.xticks( configuraciones,  ha='center', rotation=0, fontsize=9.5, fontname='monospace')
plt.axhline(y=13.03, color='black', linestyle='-')
plt.ylim(12.5,13.5)
plt.show()


#Grafica de puntos 
plt.plot(13.03,marker="o",label='Real',color='red',markersize=7)
plt.plot(13.190034,marker="x",label='LSTM Bidireccional con BN',color='royalblue',markersize=8.5)
plt.plot(13.078709,marker="x",label='LSTM Bidireccional',color='black',markersize=8.5)
plt.plot(12.978207,marker="x",label='LSTM con BN',color='orange',markersize=8.5)
plt.plot(13.101506,marker="x",label='LSTM',color='green',markeredgewidth=2,markersize=8.5)
plt.legend(fontsize=8)


#seed(9)
#Bidirectional y BatchNormalization
#0.0012625377416969029
#El predicho es 13.161911

#Bidirectional 
#0.0010052217594582725
#El predicho es 13.062797

#Unidireccional y BatchNormalization
#0.0018577684876894354
#El predicho es 12.934747

#Unidireccional 
#0.0014465077311408268
#El predicho es 13.081673

#seed(2)
#Bidirectional y BatchNormalization
#0.0011224394321314752
#El predicho es 13.078352

#Bidirectional 
#0.0010072334582396326
#El predicho es 13.079373

#Unidireccional y BatchNormalization
#0.0009763069964437682
#El predicho es 13.031533

#Unidireccional 
#0.0010256277135851583
#El predicho es 13.070409

#seed(4)
#Bidirectional y BatchNormalization
#0.0009674974256861539
#El predicho es 13.026612

#Bidirectional 
#0.0011905514271887746
#El predicho es 13.045404

#Unidireccional y BatchNormalization
#0.0010022867473834397
#El predicho es 12.956666

#Unidireccional 
#0.00098497589161442
#El predicho es 13.064922

#seed(5)
#Bidirectional y BatchNormalization
#0.001050248507725014
#El predicho es 13.088875

#Bidirectional 
#0.0011674072129175819
#El predicho es 13.066069

#Unidireccional y BatchNormalization
#0.0010813573308923363
#El predicho es 13.120932

#Unidireccional 
#0.0010043642435139197
#El predicho es 13.045475

#seed(7)
#Bidirectional y BatchNormalization
#0.001294513151568159
#El predicho es 13.167522

#Bidirectional 
#0.0011316770835805695
#El predicho es 13.089217

#Unidireccional y BatchNormalization
#0.0010563985947873267
#El predicho es 13.023595

#Unidireccional 
#0.0011371708181644854
#El predicho es 13.1015005



#Media
np.mean((0.001262,0.001122,0.000967,0.00105,0.001294))
np.mean((0.001005,0.001007,0.001191,0.001167,0.001132))
np.mean((0.001858,0.000976,0.001002,0.001081,0.001056))
np.mean((0.001446,0.001026,0.000985,0.001004,0.001137))

#Desviación estándar
np.std((0.001262,0.001122,0.000967,0.00105,0.001294))
np.std((0.001005,0.001007,0.001191,0.001167,0.001132))
np.std((0.001858,0.000976,0.001002,0.001081,0.001056))
np.std((0.001446,0.001026,0.000985,0.001004,0.001137))
