# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 21:32:59 2018

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
#np.seterr(divide='ignore', invalid='ignore')

t0 = time.time()


column = ['Timestamp','Station_ID','Freeway','Direction_Travel',
           'Station_Type','Samples','Observed','Total_Flow',
           'Av_Occupancy','Av_Speed']
#dfaa_freeway = pd.read_csv('C:/Users/ting/Desktop/Freeway/Freeways-Rawar/Freeways-Rawaa.csv', engine='python')
#print(dfaa_freeway)
df_a = pd.read_csv("D:/San Diego Dataset/San Diego Dataset/Freeways-5Min/Freeways-5Minaa.csv",header=None, skipinitialspace=True, usecols=[0,1,2,3,4,5,6,7,8,9],) #不會讀第一行'
df_b = pd.read_csv("D:/San Diego Dataset/San Diego Dataset/Freeways-5Min/Freeways-5Minab.csv",header=None, skipinitialspace=True, usecols=[0,1,2,3,4,5,6,7,8,9],)
df_c = pd.read_csv("D:/San Diego Dataset/San Diego Dataset/Freeways-5Min/Freeways-5Minac.csv",header=None, skipinitialspace=True, usecols=[0,1,2,3,4,5,6,7,8,9],)
df_d = pd.read_csv("D:/San Diego Dataset/San Diego Dataset/Freeways-5Min/Freeways-5Minad.csv",header=None, skipinitialspace=True, usecols=[0,1,2,3,4,5,6,7,8,9],)
df_e = pd.read_csv("D:/San Diego Dataset/San Diego Dataset/Freeways-5Min/Freeways-5Minae.csv",header=None, skipinitialspace=True, usecols=[0,1,2,3,4,5,6,7,8,9],)
df_f = pd.read_csv("D:/San Diego Dataset/San Diego Dataset/Freeways-5Min/Freeways-5Minaf.csv",header=None, skipinitialspace=True, usecols=[0,1,2,3,4,5,6,7,8,9],)
df_g = pd.read_csv("D:/San Diego Dataset/San Diego Dataset/Freeways-5Min/Freeways-5Minag.csv",header=None, skipinitialspace=True, usecols=[0,1,2,3,4,5,6,7,8,9],)
df_h = pd.read_csv("D:/San Diego Dataset/San Diego Dataset/Freeways-5Min/Freeways-5Minah.csv",header=None, skipinitialspace=True, usecols=[0,1,2,3,4,5,6,7,8,9],)
df.columns = column
df = df.dropna(axis=0, subset=['Av_Speed','Av_Occupancy'])


RESERVE_FEATURE= ['Timestamp','Station_ID','Total_Flow', 'Av_Occupancy','Av_Speed']
df= df[RESERVE_FEATURE]

#c0 = df_time.groupby(by=[df_time.index.day,df_time.index.hour])[FEATURE].count()
#sum0 = df_time.groupby(by=[df_time.index.day,df_time.index.hour])[FEATURE].sum()
#mean0 = sum0/c0
#
#fig, ax = plt.subplots(figsize=(15,7))
#mean0.plot(ax=ax)
#plt.xticks(list(mean0.index))
#plt.show()
#plt.close()

df.sort_values(by = ['Timestamp'])
df['Timestamp'] = pd.to_datetime(df['Timestamp'],format="%m/%d/%Y %H:%M:%S") # dataframe tp datetime
mask = (df['Timestamp'] > '1/1/2010') 
df = df.loc[mask]

def set_location(dataset , location):
    
    df_sta = dataset.loc[dataset['Station_ID']==location]
    df_sta.sort_values(by = ['Timestamp'])
    df_sta['Timestamp'] = pd.to_datetime(df_sta['Timestamp'],format="%m/%d/%Y %H:%M:%S") # dataframe tp datetime
    df_sta['weekday'] = df_sta['Timestamp'].dt.dayofweek  #dayweek = 0-6  sereis.dt.dayofweek
    df_sta['weekday'] = df_sta['weekday'].apply(lambda x:  0 if x<5  else 1)
    df_sta['min_index'] = df_sta['Timestamp'].apply(lambda d: (d.hour*60 + d.minute) / 5)
    #mask = (df_sta['Timestamp'] > '2/15/2010') & (df_sta['Timestamp'] <= '2/23/2010')
    mask = (df_sta['Timestamp'] > '1/1/2010') 
    df_sta = df_sta.loc[mask]
    df_sta.index = df_sta['Timestamp']
    df_sta['Av_Speed_t1']= df_sta['Av_Speed'].shift(-1)
    df_sta['Av_Speed_t2']= df_sta['Av_Speed'].shift(-2)
    df_sta['Av_Speed_t3']= df_sta['Av_Speed'].shift(-3)
    df_sta['Av_Speed_t4']= df_sta['Av_Speed'].shift(-4)
    df_sta['Av_Speed_t5']= df_sta['Av_Speed'].shift(-5)
    return df_sta

df_sta2 = set_location(df,1108678)
df_sta1 = set_location(df,1108659)
df_sta3 = set_location(df,1119921)
df_sta4 = set_location(df,1119928)
df_sta5 = set_location(df,1119934)


df_combineLoc= pd.concat([df_sta1, df_sta2], axis=1,join_axes=[df_sta2.index])  #以第一張dataframe為index ，若合併後有缺就是NAN

df_combineLoc.columns = ['Timestamp','Station_ID','Total_Flow','Av_Occupancy','Av_Speed','weekday','min_index',
                         'Av_Speed_t1','Av_Speed_t2','Av_Speed_t3','Av_Speed_t4','Av_Speed_t5',
                         'Timestamp2','Station_ID2','Total_Flow2','Av_Occupancy2','Av_Speed2','weekday2','min_index2']
df_combineLoc = df_combineLoc.drop(['Timestamp2','Station_ID2','weekday2','min_index2'], 1)


#僅測試用看是否合併後還有NA存在
df_combineLoc = df_combineLoc.fillna(method='ffill') 
df_combineLoc.isnull().sum()
#df_combineLoc.isnull().values.any()
#--------------------------------------------------------------------------

TIMESTEP = 5
LAG = 6
output_unit = 5
TRAIN_FEATURE = ['Av_Speed','Av_Occupancy','weekday','min_index',
                 'Av_Speed_t1','Av_Speed_t2','Av_Speed_t3','Av_Speed_t4','Av_Speed_t5']
#TRAIN_FEATURE = ['Av_Speed','Av_Occupancy','Av_Speed2','Av_Occupancy2','weekday','min_index']
LABEL_FEATURE = 'Av_Speed'
#def supervised(dataset, lag=1):
#    df = pd.DataFrame(dataset)
#    df_shift = df.shift()
#    sumdf = pd.concat([df_shift,df], axis=1)
#    sumdf.fillna(0, inplace=True)
#    sumdf = np.array(sumdf)
#    return sumdf

def multi_supervised(dataset):
    df_shift = dataset[LABEL_FEATURE].shift(-LAG)
    sumdf = pd.concat([dataset[TRAIN_FEATURE],df_shift], axis=1)
    sumdf.fillna(0, inplace=True)
#    result= sumdf.values
#    sumdf = np.array(sumdf)
    return sumdf


raw = multi_supervised(df_sta2)
#raw = multi_supervised(df_combineLoc)
raw= raw.values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(raw[:,:-1], raw[:,-1], test_size = 0.2 ,shuffle = False, stratify = None)   #shuffle 不要洗牌(因為時序資料要依序列出) stratify train set/ test set 當中的比例相同
#from sklearn.preprocessing import StandardScaler
#scaled_features = StandardScaler().fit_transform(df.values)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
#X_train_scaled = np.reshape(X_train,(-1,1))  #先變成 N*1 

X_train_scaled = sc.fit_transform(X_train)
y_train_scaled = np.reshape(y_train,(-1,1))
y_train_scaled = sc.fit_transform(y_train_scaled)
X_train_scaled = np.reshape(X_train_scaled, (-1,TIMESTEP,len(TRAIN_FEATURE))) #未知列數,5代表timestep,1代表維度
y_train_scaled = np.reshape(y_train_scaled, (-1,output_unit))

#X_test_scaled = np.reshape(X_test,(-1,1))
X_test_scaled = sc.transform(X_test)
X_test_scaled = np.reshape(X_test_scaled, (-1, TIMESTEP, len(TRAIN_FEATURE)))


#-------------------------------------------------------------------------------
save_path = "./traffic_modelsave/"+time.strftime('%m%d%H%M')   #save的資料夾
os.mkdir(save_path)



#training model 


history = []
patience=30
epochs =600
batch_size = 32
hidden_units = 8


earlystop = EarlyStopping(patience = patience)
modelsave = ModelCheckpoint(filepath= save_path+'/model.h5', save_best_only=True, verbose=1)
#opt = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
regressor = Sequential()
regressor.add(LSTM(units = hidden_units, 
                   activation = 'relu', 
                   return_sequences=True,  #有這行就是多對多
#                   stateful=stateful,
#                   dropout=0.2,
                   input_shape = (
                                  X_train_scaled.shape[1],
                                  X_train_scaled.shape[2])))

regressor.add(LSTM(units = hidden_units, 
                   activation = 'relu', 
#                   stateful=stateful,
#                   return_sequences=True,  
#                   dropout=0.2,
                   ))
#regressor.add(TimeDistributed(Dense(TIMESTEP)))
regressor.add(Dense(units = output_unit)) #最後一層要輸出幾個

regressor.compile(optimizer = 'adam',
                  loss = 'mean_squared_error')
his= regressor.fit(X_train_scaled,
              y_train_scaled, 
              batch_size = batch_size, 
              epochs = epochs, 
              verbose = 1,
              validation_split=0.2,
              callbacks=[earlystop,modelsave])

history.append(his)
regressor.summary()

#-------------------------------------------------------------------------------------------------
#print the result
train_pred = regressor.predict(X_train_scaled)
train_pred = sc.inverse_transform(train_pred)
train_pred = train_pred.reshape(-1)

y_pred = regressor.predict(X_test_scaled)
y_pred = sc.inverse_transform(y_pred)
y_pred = y_pred.reshape(-1)


#print the evaluation    
from sklearn.metrics import mean_squared_error,mean_absolute_error


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

train_mse = mean_squared_error(y_train, train_pred)
print("Train_mse",train_mse)
train_mae= mean_absolute_error(y_train, train_pred)
print("Train_mae=",train_mae)
train_mape = mean_absolute_percentage_error(y_train, train_pred)
print("Train_ MAPE:",train_mape)

test_mse = mean_squared_error(y_test, y_pred)
print("Test_mse",test_mse)
test_mae= mean_absolute_error(y_test, y_pred)
print("Test_mae=",test_mae)
test_mape = mean_absolute_percentage_error(y_test, y_pred)
print("Test_ MAPE:",test_mape)

#write log file
with open(save_path + './Log.txt','w') as fh:
    regressor.summary(print_fn = lambda x: fh.write(x + '\n'))
    fh.write("TIMESTEP:"+str(TIMESTEP)+
             "\ndimension:"+str(len(TRAIN_FEATURE))+
             "\nTRAIN_FEATURE:"+str(TRAIN_FEATURE)+
             "\nlag:"+str(LAG)+
             "\nunits:"+str(hidden_units)+
             "\noutput_units:"+str(output_unit )+
             "\nbatch_size:"+str(batch_size)+
             "\npatience:"+str(patience)+
             "\nepochs:"+str(epochs)+
             "\nTrain_mse="+str(train_mse)+
             "\nTrain_mae="+str(train_mae)+
             "\nTrain_mape="+str(train_mape)+
             "\ntest_mse="+str(test_mse)+
             "\ntest_mae="+str(test_mae)+
             "\ntest_mape="+str(test_mape)
             )

#draw the ground truth and prediction  
def show_predict(y_pred,y_test):
    plt.plot(y_pred)
    plt.plot(y_test)
    plt.ylabel("speed")
    plt.xlabel("time")
    plt.xlim(0,100)
    plt.ylim(0, 80)
    plt.legend(['y_pred', 'y_test'], loc='best')  
    plt.savefig(save_path+'/speed.png')
    plt.show() 
    plt.close()

show_predict(y_pred,y_test)

#draw the loss function  
def show_train_history(i,train_history):
    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xlim(0, 600)
    plt.ylim(0, 0.005)
    plt.legend(['train_loss', 'val_loss'], loc='best')  
    plt.savefig(save_path+'/loss.png')
    plt.show() 
    plt.close()
    
for i in range(len(history)):
    show_train_history(i,history[i])
    
#os.makedirs("./traffic/traffic_modelsave/"+time.strftime('%m%d%H%M'))    
#text_file = open(save_path+"./Log.txt", "w")
#text_file.write("TIMESTEP:"+str(TIMESTEP)+"\ndimension:"+str(len(TRAIN_FEATURE))+
#                "\nlag:"+str(LAG)+
#                "\nTrain_mae="+str(train_mae)+"\ntrain_mape="+str(train_mape)+
#                "\ntest_mae="+str(test_mae)+"\ntest_mape="+str(test_mape)+
#                "\nstateful = "+str(stateful))


