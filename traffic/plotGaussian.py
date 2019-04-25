# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 11:06:19 2018

@author: ting
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_csv("./traffic/location_1108687.csv")
#df_2 = pd.read_csv("./traffic/location_1108659.csv")
print(df)
#df = df[:100000]
#df_a = pd.read_csv("D:/San Diego Dataset/San Diego Dataset/Freeways-5Min/Freeways-5Minaa.csv",header=None, skipinitialspace=True, usecols=[0,1,2,3,4,5,6,7,8,9],)
#print(df_a.shape)

def processDataset(dataset):
    dataset.sort_values(by = ['Timestamp'])
    dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'],format="%m/%d/%Y %H:%M:%S") # dataframe tp datetime
    dataset['weekday'] = dataset['Timestamp'].dt.dayofweek  #dayweek = 0-6  sereis.dt.dayofweek
    dataset['flag_weekday'] = dataset['weekday'].apply(lambda x:  0 if x<5  else 1)
    dataset['min_index'] = dataset['Timestamp'].apply(lambda d: (d.hour*60 + d.minute) / 5)
#    dataset['hour'] = dataset['Timestamp'].dt.hour
    #mask = (df_sta['Timestamp'] > '2/15/2010') & (df_sta['Timestamp'] <= '2/23/2010')
    mask = (dataset['Timestamp'] > '1/1/2010') 
    dataset = dataset.loc[mask]
    dataset.index = dataset['Timestamp']
    dataset['Av_Speed_t1']= dataset['Av_Speed'].shift(-1)
    dataset['Av_Speed_t2']= dataset['Av_Speed'].shift(-2)
    dataset['Av_Speed_t3']= dataset['Av_Speed'].shift(-3)
    dataset['Av_Speed_t4']= dataset['Av_Speed'].shift(-4)
    dataset['Av_Speed_t5']= dataset['Av_Speed'].shift(-5)
    
    return dataset

df = processDataset(df)
eight_day = df[(df.index.hour==12 )& (df.index.minute==0)]
nine_day = df[(df.index.hour==18 )& (df.index.minute==0)]
eight_day = eight_day.Av_Speed
nine_day = nine_day.Av_Speed
plt.scatter(eight_day,nine_day)
plt.xlabel("speed of 12 am")
plt.ylabel("speed of 18 pm")
plt.show()

df = processDataset(df)
df_week = df[(df.weekday==4)]
eight_day = df[(df.index.hour==8 )& (df.index.minute==0)]
week_eight_day = df_week[(df_week.index.hour==8 )& (df_week.index.minute==0)]
eight_day = eight_day.Av_Speed
week_eight_day = week_eight_day.Av_Speed
plt.scatter(eight_day,week_eight_day)
plt.xlabel("speed of 8 am")
plt.ylabel("speed of 9 pm")
plt.show()
#mask1 = df.index>= "1/1/2010 08:00:00"
#mask2 = df.index< "1/1/2010 09:00:00"
#mask3 = df.index>= "1/1/2010 09:00:00"
#mask4 = df.index< "1/1/2010 10:00:00"
#eight = df[mask1 & mask2].Av_Speed


eight.reset_index(drop=True, inplace=True)
nine.reset_index(drop=True, inplace=True)
sumdf = pd.concat([eight,nine], axis=1)
cov = np.cov(eight,nine)
mean = np.mean(sumdf).values

from scipy.stats import multivariate_normal    
x, y = np.mgrid[-1:1:.01, -1:1:.01]
pos = np.dstack((x, y))
rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.contourf(x, y, rv.pdf(pos))




mean = [0, 0]
cov = [[1, 0], [0, 100]]  # diagonal covariance

import matplotlib.pyplot as plt
x, y = np.random.multivariate_normal(mean, cov, 5000).T
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()