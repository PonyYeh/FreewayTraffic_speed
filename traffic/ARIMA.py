# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 17:49:43 2018

@author: ting
"""

import time
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,TimeDistributed
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from pandas.tools.plotting import autocorrelation_plot

t0 = time.time()
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
df_sta.index = df_sta['Timestamp']
mask = (df_sta['Timestamp'] > '1/1/2010') 
df_time = df_sta.loc[mask]
series_time= df_time[FEATURE]
pd.to_numeric(series_time, errors='coerce').fillna(0)
series_time.astype('float64')
print(series_time.dtype)

#c0 = df_time.groupby(by=[df_time.index.day,df_time.index.hour])[FEATURE].count()
#sum0 = df_time.groupby(by=[df_time.index.day,df_time.index.hour])[FEATURE].sum()
#mean0 = sum0/c0

#autocorrelation_plot(series_time)
#plt.show()

predictions=[]
from statsmodels.tsa.arima_model import ARIMA
size = int(len(series_time)*0.66)
train , test = series_time[:size],series_time[size:]
history = train.tolist()
#for t in range(len(series_time)):
model = ARIMA(history, order=(2,1,0))   
model_fit = model.fit(disp=0)     
    output = model_fit.predict()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
    