# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 15:33:31 2018

@author: ting
"""

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
import os
import glob
#np.seterr(divide='ignore', invalid='ignore')

path = "D:/Freeway/traffic/location_1108687_all.csv"
dataset = pd.read_csv(path)
#dataset = dataset.loc[dataset['Station ID']==1108687]

dataset.index = dataset['Timestamp']
dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'],format="%m/%d/%Y %H:%M:%S")
#mask1 = dataset['Timestamp'] >= '7/1/2010 00:00:00'
#mask2 = dataset['Timestamp'] < '8/1/2010 00:00:00'
#dataset= dataset[mask1 & mask2]

#mask1 = (dataset['Timestamp'] >= '2/1/2010 00:00:00')
#mask2 = dataset['Timestamp'] < '2/2/2010 00:00:00'
#mask3 = dataset['Timestamp'] >= '4/1/2010 00:00:00'
#mask4 = dataset['Timestamp'] < '4/2/2010 00:00:00'
#mask5 = dataset['Timestamp'] >= '6/1/2010 00:00:00'
#mask6 = dataset['Timestamp'] < '6/2/2010 00:00:00'
#mask7 = dataset['Timestamp'] >= '8/1/2010 00:00:00'
#mask8 = dataset['Timestamp'] < '8/2/2010 00:00:00'
#R= dataset[(mask1&mask2)|(mask3&mask4)|(mask5&mask6)|(mask7&mask8)]

mask1 = (dataset['Timestamp'] >= '1/1/2010 00:00:00')
mask2 = dataset['Timestamp'] < '2/1/2010 00:00:00'
mask3 = dataset['Timestamp'] >= '3/1/2010 00:00:00'
mask4 = dataset['Timestamp'] < '4/1/2010 00:00:00'
mask5 = dataset['Timestamp'] >= '5/1/2010 00:00:00'
mask6 = dataset['Timestamp'] < '6/1/2010 00:00:00'
mask7 = dataset['Timestamp'] >= '7/1/2010 00:00:00'
mask8 = dataset['Timestamp'] < '8/1/2010 00:00:00'
R= dataset[(mask1&mask2)|(mask3&mask4)|(mask5&mask6)|(mask7&mask8)]





R.to_csv ("traffic_TrainData.csv" , index = False )