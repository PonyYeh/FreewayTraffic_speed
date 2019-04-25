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
import glob

import tensorflow as tf
from tensorflow.python.client import device_lib

from sklearn.metrics import mean_squared_error,mean_absolute_error


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(device_lib.list_local_devices())
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess)
os.getcwd()  #返回目前目錄

#-------------------------------------------------------------------------------------------------

#c0 = df_time.groupby(by=[df_time.index.day,df_time.index.hour])[FEATURE].count()
#sum0 = df_time.groupby(by=[df_time.index.day,df_time.index.hour])[FEATURE].sum()
#mean0 = sum0/c0
#
#fig, ax = plt.subplots(figsize=(15,7))
#mean0.plot(ax=ax)
#plt.xticks(list(mean0.index))
#plt.show()
#plt.close()


#dataset preprocess
df = pd.read_csv("./traffic/location_1108687.csv")
df_2 = pd.read_csv("./traffic/location_1108659.csv")

#df = df[:100000]
#df_a = pd.read_csv("D:/San Diego Dataset/San Diego Dataset/Freeways-5Min/Freeways-5Minaa.csv",header=None, skipinitialspace=True, usecols=[0,1,2,3,4,5,6,7,8,9],)
#print(df_a.shape)

def processDataset(dataset):
    dataset.sort_values(by = ['Timestamp'])
    dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'],format="%m/%d/%Y %H:%M:%S") # dataframe tp datetime
    dataset['weekday'] = dataset['Timestamp'].dt.dayofweek  #dayweek = 0-6  sereis.dt.dayofweek
    dataset['weekday'] = dataset['weekday'].apply(lambda x:  0 if x<5  else 1)
    dataset['min_index'] = dataset['Timestamp'].apply(lambda d: (d.hour*60 + d.minute) / 5)
#    dataset['hour'] = dataset['Timestamp'].dt.hour
    #mask = (df_sta['Timestamp'] > '2/15/2010') & (df_sta['Timestamp'] <= '2/23/2010')
#    mask = (dataset['Timestamp'] > '10/23/2010') 
#    dataset = dataset.loc[mask]
    dataset.index = dataset['Timestamp']
    dataset['Av_Speed_t1']= dataset['Av_Speed'].shift(-1)
    dataset['Av_Speed_t2']= dataset['Av_Speed'].shift(-2)
    dataset['Av_Speed_t3']= dataset['Av_Speed'].shift(-3)
    dataset['Av_Speed_t4']= dataset['Av_Speed'].shift(-4)
    dataset['Av_Speed_t5']= dataset['Av_Speed'].shift(-5)
    return dataset

df = processDataset(df)
df_2 = processDataset(df_2)

#------------------------------------------------------------------
weather = pd.read_csv("./traffic/San_Diego_Miramar_Wscmo.csv")
#print(np.where(weather['Visibility']=='N' ))
def processWeather(weather,df):
    
    weather['Timestamp'] = pd.to_datetime(weather['Timestamp'],format="%Y/%m/%d %H:%M:%S") 
    weather.index = weather['Timestamp']
    weather = weather[weather.Visibility != 'N']
    weather = weather['Visibility'] 
    weather = pd.to_numeric(weather)
    visibility = weather.resample('5T').pad()
    df_combineWeather = pd.concat([visibility,df],axis=1,join_axes=[df.index])
    
    return df_combineWeather

df_combineWeather = processWeather(weather,df)
df2_combineWeather = processWeather(weather,df_2)
#df_combineWeather = df_combineWeather.fillna(method='ffill') 
#df_combineWeather.isnull().sum()

#--------------------------------------------------------------------------------------------------------------------------------
# if you want to combine two location
df2_combineWeather = df2_combineWeather.drop(['Visibility','Timestamp','Station_ID','weekday','min_index'], 1)
df_combineLoc= pd.concat([df_combineWeather, df2_combineWeather], axis=1,join_axes=[df_combineWeather.index])  #以第一張dataframe為index ，若合併後有缺就是NAN
#
df_combineLoc.columns = ['Visibility','Timestamp','Station_ID',
                         'Total_Flow','Av_Occupancy','Av_Speed','weekday','min_index',
                         'Av_Speed_t1','Av_Speed_t2','Av_Speed_t3','Av_Speed_t4','Av_Speed_t5',
                         'Total_Flow2','Av_Occupancy2','Av_Speed2',
                         'Av_Speed2_t1','Av_Speed2_t2','Av_Speed2_t3','Av_Speed2_t4','Av_Speed2_t5',
                         ]
#df_combineLoc.columns = ['Visibility','Timestamp','Station_ID',
#                         'Total_Flow','Av_Occupancy','Av_Speed','weekday','min_index',
#                         'Av_Speed_t1','Av_Speed_t2','Av_Speed_t3',
#                         'Total_Flow2','Av_Occupancy2','Av_Speed2',
#                         'Av_Speed2_t1','Av_Speed2_t2','Av_Speed2_t3',
#                         ]


#僅測試用看是否合併後還有NA存在
df_combineLoc = df_combineLoc.fillna(method='ffill') 
df_combineLoc.isnull().sum()
#df_combineLoc.isnull().values.any()
#--------------------------------------------------------------------------

TIMESTEP = 5
LAG = 4
output_unit = 5

#one location+weather
#TRAIN_FEATURE = ['Visibility','weekday','min_index',
#                 'Av_Speed','Av_Occupancy','Total_Flow',
#                 'Av_Speed_t1','Av_Speed_t2','Av_Speed_t3','Av_Speed_t4','Av_Speed_t5',
#                 ]

#two location and weather
#TRAIN_FEATURE = ['weekday','Visibility',
#                 'Av_Speed','Av_Occupancy',
#                 'Av_Speed_t1','Av_Speed_t2','Av_Speed_t3','Av_Speed_t4','Av_Speed_t5',
#                 'Av_Occupancy2','Av_Speed2',
#                 'Av_Speed2_t1','Av_Speed2_t2','Av_Speed2_t3','Av_Speed2_t4','Av_Speed2_t5',
#                 ]

#two location and weather
TRAIN_FEATURE = ['Visibility','weekday',
                         'Av_Occupancy','Av_Speed',
                         'Av_Speed_t1','Av_Speed_t2','Av_Speed_t3','Av_Speed_t4','Av_Speed_t5',
                         'Av_Occupancy2','Av_Speed2',
                         'Av_Speed2_t1','Av_Speed2_t2','Av_Speed2_t3','Av_Speed2_t4','Av_Speed2_t5',
                 ]

LABEL_FEATURE = 'Av_Speed'



def multi_supervised(dataset):
    df_shift = dataset[LABEL_FEATURE].shift(-LAG)
    sumdf = pd.concat([dataset[TRAIN_FEATURE],df_shift], axis=1)
    sumdf.fillna(0, inplace=True)
    return sumdf

raw = multi_supervised(df_combineLoc)

#df_combineWeather.reset_index(drop=True, inplace=True)
test_mask1 = raw.index>'11/1/2010'
test_mask2 = raw.index<'11/30/2010'
train_mask1 = raw.index<'11/1/2010'
train_mask2 = raw.index>'1/1/2010'
numberofsize= raw.shape[1]
raw.columns=range(numberofsize)
testraw= raw[test_mask1 & test_mask2]
trainraw= raw[train_mask1 & train_mask2]
mulNum=25
q_test= testraw.shape[0]/mulNum
testraw= testraw[:int(q_test)*mulNum]
q_train= trainraw.shape[0]/mulNum
trainraw= trainraw[:int(q_train)*mulNum]

X_test=testraw[testraw.columns[0:numberofsize-1]]
Y_test=testraw[testraw.columns[numberofsize-1:]]
X_train=trainraw[trainraw.columns[0:numberofsize-1]]
Y_train=trainraw[trainraw.columns[numberofsize-1:]]




#raw= raw.values
#from sklearn.model_selection import train_test_split #raw 取出除了標籤以外的當作 training data ；只取出最後一欄位當作標籤
#X_train, X_test, y_train, y_test = train_test_split(raw[:,:-1], raw[:,-1], test_size = 0.2 ,shuffle = False, stratify = None)   #shuffle 不要洗牌(因為時序資料要依序列出) stratify train set/ test set 當中的比例相同

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
#X_train_scaled = np.reshape(X_train,(-1,1))  #先變成 N*1 

X_train_scaled = sc.fit_transform(X_train)
#y_train_scaled = np.reshape(y_train,(-1,1))
Y_train_scaled = sc.fit_transform(Y_train)

X_train_scaled = np.reshape(X_train_scaled, (-1,TIMESTEP,len(TRAIN_FEATURE))) #未知列數,5代表timestep,1代表維度
Y_train_scaled = np.reshape(Y_train_scaled, (-1,output_unit))

#X_test_scaled = np.reshape(X_test,(-1,1))
X_test_scaled = sc.transform(X_test)
X_test_scaled = np.reshape(X_test_scaled, (-1, TIMESTEP, len(TRAIN_FEATURE)))


#-------------------------------------------------------------------------------

#training model 
history = []
patience=25
epochs =2000
batch_size = 32
hidden_units = 8

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
regressor.summary()

#---------------------------------------------------------------------------------------------

save_path = "./traffic/traffic_modelsave/"+time.strftime('%m%d%H%M')   #save的資料夾
os.mkdir(save_path)

earlystop = EarlyStopping(patience = patience)
modelsave = ModelCheckpoint(filepath= save_path+'/model.h5', save_best_only=True, verbose=1)

his= regressor.fit(X_train_scaled,
              Y_train_scaled, 
              batch_size = batch_size, 
              epochs = epochs, 
              verbose = 1,
              validation_split=0.2,
              callbacks=[earlystop,modelsave])

history.append(his)
regressor.summary()

#-------------------------------------------------------------------------------------------------
#print the result

Load = True
model_file = "06111944"
if(Load): 
    model_path = "./traffic/traffic_modelsave/"+model_file
    regressor.load_weights(model_path+"/model.h5")
    print("Loaded model from disk")
    train_pred = regressor.predict(X_train_scaled)
    train_pred = sc.inverse_transform(train_pred)
    train_pred = train_pred.reshape(-1)
    
    y_pred = regressor.predict(X_test_scaled)
    y_pred = sc.inverse_transform(y_pred)
    y_pred = y_pred.reshape(-1)
else:
    train_pred = regressor.predict(X_train_scaled)
    train_pred = sc.inverse_transform(train_pred)
    train_pred = train_pred.reshape(-1)
    
    y_pred = regressor.predict(X_test_scaled)
    y_pred = sc.inverse_transform(y_pred)
    y_pred = y_pred.reshape(-1)
    
#--------------------------------------------------------------------------------------------------------------

#print the all evaluation    
train_mse = mean_squared_error(Y_train, train_pred)
print("Train_mse",train_mse)
train_mae= mean_absolute_error(Y_train, train_pred)
print("Train_mae=",train_mae)
#train_mape = mean_absolute_percentage_error(y_train, train_pred)
#print("Train_ MAPE:",train_mape)

test_mse = mean_squared_error(Y_test, y_pred)
print("Test_mse",test_mse)
test_mae= mean_absolute_error(Y_test, y_pred)
print("Test_mae=",test_mae)
test_mape = mean_absolute_percentage_error(Y_test, y_pred)
print("Test_ MAPE:",test_mape)



#write log file
with open(save_path + './Log.txt','w') as fh:
    regressor.summary(print_fn = lambda x: fh.write(x + '\n'))
    fh.write("runtime:"+save_path+
             "\nTIMESTEP:"+str(TIMESTEP)+
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
#             "\nTrain_mape="+str(train_mape)+
             "\ntest_mse="+str(test_mse)+
             "\ntest_mae="+str(test_mae)+
             "\ntest_mape="+str(test_mape)
             )
    
def write_evaluation(i,mae,mse):
    if os.path.exists(save_path+ '/Log.txt'): 
        with open(save_path + '/Log.txt','a') as fh:  #繼續寫下一行
            fh.write("\ntime in "+str(i+1)+" a day "+
                     "\ntest_mse="+str(mse)+
                     "\ntest_mae="+str(mae)
            
                     
                     )
    else:
        with open(save_path + '/Log.txt','w') as fh: #不存在檔案就寫檔
            fh.write("\ntime in "+str(i+1)+" a day"+
                     "\ntest_mse="+str(mse)+
                     "\ntest_mae="+str(mae)
                     
                     )   
            
def evaluation_test(truth,ym):
    
    truth=truth.values
    ym = ym.values
    mse = mean_squared_error(ym, truth)
    print("Test_mse",mse)
    mae = mean_absolute_error(ym, truth)
    print("Test_mae=",mae)
    test_mape = mean_absolute_percentage_error(ym, truth)
    print("Test_ MAPE:",test_mape)
    return (mae,mse)
    
def show_abnormal(y_test_draw,y_pred_draw,i): #平均每個小時的評估誤差是多少
    plt.title(str(i+1)+" in day, abnormal")
    plt.xticks(rotation=30) 
    truth_abnormal = y_test_draw[(y_test_draw.index.hour==(i+1) )& (y_test_draw.index.minute==0)]
    #    truth_normal = y_test_draw[(y_test_draw.index.hour==10 )& (y_test_draw.index.minute==0)]
    pre_abnormal = y_pred_draw[(y_pred_draw.index.hour==(i+1) )& (y_pred_draw.index.minute==0)]
    #    pre_normal = y_pred_draw[(y_pred_draw.index.hour==10 )& (y_pred_draw.index.minute==0)]
    plt.plot(truth_abnormal.index, truth_abnormal,'-')
    plt.plot(pre_abnormal.index, pre_abnormal ,'-')   
#    plt.ylim(0, 80) 
    plt.ylabel("speed")
    plt.legend(['truth', 'prediction'])    
    plt.savefig(save_path+'/speed'+str(i+1)+'.png')
    plt.show()
    m = evaluation_test(truth_abnormal,pre_abnormal)
    write_evaluation(i,m[0],m[1])
    
def show_normal(y_test_draw,y_pred_draw):
    plt.title("11 p.m normal")
    plt.xticks(rotation=30)  
    truth_normal = y_test_draw[(y_test_draw.index.hour==23 )& (y_test_draw.index.minute==0)]    
    pre_normal = y_pred_draw[(y_pred_draw.index.hour==23 )& (y_pred_draw.index.minute==0)]
    plt.plot(truth_normal.index, truth_normal,'-')
    plt.plot(pre_normal.index, pre_normal ,'-')   
#    plt.ylim(0, 80) 
    plt.ylabel("speed")
    plt.legend(['truth', 'prediction'])      
    plt.show()
    evaluation_test(truth_normal,pre_normal)
    
#draw the ground truth and prediction  
def show_predict(y_pred,y_test):
#    y_pred = pd.DataFrame(y_pred,index=y_test.index,columns =['prediction'])
    y_test.columns=['truth']
    ax = y_test.plot()
    y_pred.plot(ax=ax)
    ax.set_ylabel("speed")
    ax.set_xlabel("time ")
    ax.set_ylim([50, 75])
    plt.show() 
    fig = ax.get_figure()
    fig.savefig(save_path+'/speed3.png')
    evaluation_test(y_pred,y_test)
    


A = Y_test.index >'11/1/2010 00:00:00'
B = Y_test.index <'11/30/2010 00:00:00'
y_test_draw = Y_test[A&B]
y_pred = pd.DataFrame(y_pred,index=Y_test.index,columns =['prediction'])
y_pred_draw = y_pred[A&B]
show_predict(y_pred_draw,y_test_draw)

for i in range(24):  
    show_abnormal(y_test_draw,y_pred_draw,i)
    
show_normal(y_test_draw,y_pred_draw)


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
    


