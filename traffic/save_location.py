# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 21:34:48 2018

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
import os
import glob
#np.seterr(divide='ignore', invalid='ignore')

t0 = time.time()


column = ['Timestamp','Station_ID','Freeway','Direction_Travel',
           'Station_Type','Samples','Observed','Total_Flow',
           'Av_Occupancy','Av_Speed']
#dfaa_freeway = pd.read_csv('C:/Users/ting/Desktop/Freeway/Freeways-Rawar/Freeways-Rawaa.csv', engine='python')
#print(dfaa_freeway)



#read dataset
RESERVE_FEATURE= ['Timestamp','Station_ID','Total_Flow','Station_Type', 'Av_Occupancy','Av_Speed']
def set_readDataset(location):
    frame = pd.DataFrame()
    _list = []
    for file in glob.glob("D:/San Diego Dataset/San Diego Dataset/Freeways-5Min/*.csv"):
        print("read file")
        dataset = pd.read_csv(file,header=None, skipinitialspace=True, usecols=[0,1,2,3,4,5,6,7,8,9])
        dataset.columns = column
        dataset = dataset.dropna(axis=0, subset=['Av_Speed','Av_Occupancy'])
        dataset= dataset[RESERVE_FEATURE]
        dataset = dataset.loc[dataset['Station_ID']==location]
        dataset.index = dataset['Timestamp']
        _list.append(dataset)
        
    frame = pd.concat(_list)
    return frame

df = set_readDataset(1108687)
print(df)
df.to_csv ("location_1108687.csv" , index = False)

location = pd.read_csv("location_1108687.csv")
location['Timestamp'] = pd.to_datetime(location['Timestamp'],format="%m/%d/%Y %H:%M:%S")
location.index = location['Timestamp']
mask1 = location['Timestamp'] > '11/1/2010'
mask2 = location['Timestamp'] < '11/10/2010'
location= location[mask1 & mask2]
location['Av_Speed'].plot()
