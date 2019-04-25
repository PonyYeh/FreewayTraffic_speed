# -*- coding: utf-8 -*-

from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
import time
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import time

FEATURE = 'Total_Flow'

column = ['Timestamp','Station_ID','Freeway','Direction_Travel',
           'Station_Type','Samples','Observed','Total_Flow',
           'Av_Occupancy','Av_Speed']

df = pd.read_csv("C:/Users/ting/Desktop/San Diego Dataset/San Diego Dataset/Freeways-5Minab/Freeways-5Minab.csv",header=None, skipinitialspace=True, usecols=[0,1,2,3,4,5,6,7,8,9],)
df.columns = column
df = df.dropna(axis=0, subset=[FEATURE])
df_sta = df.loc[df['Station_ID']==1108687]
df_sta.sort_values(by=['Timestamp'])


df_sta['Timestamp'] = pd.to_datetime(df_sta['Timestamp'],format="%m/%d/%Y %H:%M:%S")

#mask = (df_sta['Timestamp'] > '2/15/2010') & (df_sta['Timestamp'] <= '2/23/2010')
mask = (df_sta['Timestamp'] > '1/1/2010') 
df_time = df_sta.loc[mask]

df_time.index = df_time['Timestamp']
c0 = df_time.groupby(by=[df_time.index.day,df_time.index.hour])[FEATURE].count()
sum0 = df_time.groupby(by=[df_time.index.day,df_time.index.hour])[FEATURE].sum()
mean0 = sum0/c0
#
#fig, ax = plt.subplots(figsize=(15,7))
#mean0.plot(ax=ax)
#plt.xticks(list(mean0.index))
#plt.show()
#plt.close()


#--------------------------------------------------------------------------
TIMESTEP = 5

def supervised(dataset, lag=1):
    df = pd.DataFrame(dataset)
    df_shift = df.shift()
    sumdf = pd.concat([df_shift,df], axis=1)
    sumdf.fillna(0, inplace=True)
    sumdf = np.array(sumdf)
    return sumdf

raw = df_time[FEATURE].values
raw_supercised = supervised(raw)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(raw_supercised[:,0], raw_supercised[:,1], test_size = 0.2, random_state = 0)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train_scaled = np.reshape(X_train,(-1,1))  #先變成 N*1 
y_train_scaled = np.reshape(y_train,(-1,1))
X_train_scaled = sc.fit_transform(X_train_scaled)
y_train_scaled = sc.fit_transform(y_train_scaled)
X_train_scaled = np.reshape(X_train_scaled, (-1,TIMESTEP,1)) #未知列數,5代表timestep,1代表維度
y_train_scaled = np.reshape(y_train_scaled, (-1,TIMESTEP))

X_test_scaled = np.reshape(X_test,(-1,1))
X_test_scaled = sc.transform(X_test_scaled)
X_test_scaled = np.reshape(X_test_scaled, (-1, TIMESTEP, 1))

#------------------------------------------------------------------------------------------------------------------------

# Function to create model, required for KerasClassifier
def create_model(neurons=1):   
	# create model
    regressor = Sequential()
    regressor.add(LSTM(units = neurons, 
                   activation = 'relu', 
                   return_sequences=True,
                   input_shape = (  X_train_scaled.shape[1],
                                    X_train_scaled.shape[2])))
    regressor.add(LSTM(units = neurons, 
                   activation = 'relu'))
    regressor.add(Dense(units = TIMESTEP))
    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')
    return regressor
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

neurons = [1, 5, 10] 
#learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
#momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

#model = create_model()
# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters

param_grid = dict(neurons = neurons)
grid = GridSearchCV(estimator = model, param_grid = param_grid, n_jobs=-1)
grid_result = grid.fit(X_train_scaled, y_train_scaled)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


