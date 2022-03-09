# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 10:05:21 2021

@author: Luis Vicente
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense
#from keras.layers.normalization import BatchNormalization
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


from functools import reduce
dfs = [ipc,a_movil]
df = reduce(lambda left,right: pd.merge(left,right,on='Date'), dfs)
df.columns=['Date','IPC','America Movil']


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
      seq_X, seq_Y = sequence[i: i + n_steps], sequence[i + n_steps, 1]
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
model.add(LSTM(50, activation='tanh',return_sequences=False,input_shape=(n_steps, n_features)))
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

(prediction_1*(max(df_final[:'2014-12-31']['America Movil'])-min(df_final[:'2014-12-31']['America Movil'])))+min(df_final[:'2014-12-31']['America Movil'])

#Grafica de barras
configuraciones = ['LSTM Bidireccional \ncon BN', 'LSTM\n Bidireccional', 'LSTM con BN', 'LSTM']
predicciones = [13.334793,13.157218,12.678058,13.061586]
plt.bar(configuraciones,predicciones,color=['red','blue','orange','brown'])
plt.xticks( configuraciones,  ha='center', rotation=0, fontsize=9.5, fontname='monospace')
plt.axhline(y=13.03, color='black', linestyle='-')
plt.ylim(12.3,13.3)
plt.show()


#Grafica de puntos
plt.plot(13.03,marker="o",label='Real',color='red')
plt.plot(13.053839,marker="x",label='LSTM Bidireccional con BN',color='royalblue',markersize=8.5)
plt.plot(13.144364,marker="x",label='LSTM Bidireccional',color='black',markersize=8.5)
plt.plot(12.54138,marker="x",label='LSTM con BN',color='orange',markersize=8.5)
plt.plot(13.066681,marker="x",label='LSTM',color='green',markersize=8.5)
plt.legend(fontsize=8)



#seed(9)
#Bidirectional y BatchNormalization
#0.0030085687016157144
#El predicho es 13.334793

#Bidirectional 
#0.001351595734362939
#El predicho es 13.157218

#Unidireccional y BatchNormalization
#0.0023906722657505274
#El predicho es 12.678058

#Unidireccional 
#0.0009711861156608885
#El predicho es 13.061586

#seed(2)
#Bidirectional y BatchNormalization
#0.001975979549329719
#El predicho es 12.866392

#Bidirectional 
#0.0010095596754321607
#El predicho es 13.056463

#Unidireccional y BatchNormalization
#0.001034768050101839
#El predicho es 13.065011

#Unidireccional 
#0.0010293016470795233
#El predicho es 13.07408

#seed(4)
#Bidirectional y BatchNormalization
#0.0012362750006849213
#El predicho es 12.898989

#Bidirectional 
#0.0012127848213947516
#El predicho es 13.084521

#Unidireccional y BatchNormalization
#0.003653920211737734
#El predicho es 13.337511

#Unidireccional 
#0.0011389074581713526
#El predicho es 13.086459

#seed(5)
#Bidirectional y BatchNormalization
#0.004830043403850272
#El predicho es 13.468104

#Bidirectional 
#0.0010078105217921128
#El predicho es 13.068851

#Unidireccional y BatchNormalization
#0.000944168685293424
#El predicho es 12.944334

#Unidireccional 
#0.0011404439294809786
#El predicho es 13.10107

#seed(7)
#Bidirectional y BatchNormalization
#0.013318443277186365
#El predicho es 12.353634

#Bidirectional 
#0.00099732253136159
#El predicho es 13.052393

#Unidireccional y BatchNormalization
#0.003874461134822199
#El predicho es 13.39516

#Unidireccional 
#0.0009852004180301462
#El predicho es 12.9989195



#Media
np.mean((0.003009,0.001976,0.001236,0.00483,0.013318))
np.mean((0.001352,0.00101,0.001213,0.001008,0.000997))
np.mean((0.002391,0.001035,0.003654,0.000944,0.003874))
np.mean((0.000971,0.001029,0.001139,0.00114,0.000985))

#Desviación estándar
np.std((0.003009,0.001976,0.001236,0.00483,0.013318))
np.std((0.001352,0.00101,0.001213,0.001008,0.000997))
np.std((0.002391,0.001035,0.003654,0.000944,0.003874))
np.std((0.000971,0.001029,0.001139,0.00114,0.000985))
