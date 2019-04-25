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
import tensorflow as tf
from tensorflow.contrib import rnn
import time

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
from sklearn import cross_validation

tf.set_random_seed(20160612)
sc = MinMaxScaler()

X_train_scaled = np.reshape(X_train,(-1,1))  #先變成 N*1 
y_train_scaled = np.reshape(y_train,(-1,1))
X_train_scaled = sc.fit_transform(X_train_scaled)
y_train_scaled = sc.fit_transform(y_train_scaled)
X_train_scaled = np.reshape(X_train_scaled, (-1,TIMESTEP,1)) #未知列數,5代表timestep,1代表維度
y_train_scaled = np.reshape(y_train_scaled, (-1,TIMESTEP))

X_train_scaled,X_val_scaled,y_train_scaled,y_val_scaled = cross_validation.train_test_split(X_train_scaled,y_train_scaled,test_size=0.1,random_state=None)

X_test_scaled = np.reshape(X_test,(-1,1))
X_test_scaled = sc.transform(X_test_scaled)
X_test_scaled = np.reshape(X_test_scaled, (-1, TIMESTEP, 1))
y_test_scaled = np.reshape(y_test,(-1,1))
y_test_scaled = sc.fit_transform(y_test_scaled)
y_test_scaled = np.reshape(y_test_scaled, (-1,TIMESTEP))

#-----------------------------------------------------------------------------------------------------------------
#training model
tf.reset_default_graph() 
learning_rate= 0.001
training_steps = 50
batch_size = 32
Bol_load = True

TIMESTEP = 5
DIMENSION = 1
num_hidden = 8
num_output = 5

X = tf.placeholder(tf.float32 , shape=(None,TIMESTEP,DIMENSION),name='X') 
Y = tf.placeholder(tf.float32 , shape=(None,num_output),name='Y' )


weight = {
        'out':tf.Variable(tf.random_normal([num_hidden,num_output]))
        } 
bias = {
        'out':tf.Variable(tf.random_normal([num_output]))
        }

def LSTM(x, weight, biases):
    x = tf.unstack(x, TIMESTEP, 1)
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    output , states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(output[-1],weight['out']) + bias['out']

output = LSTM(X,weight,bias)

loss = tf.losses.mean_squared_error(labels = Y, predictions= output)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001,
                                    beta1=0.9,
                                    beta2=0.999,
                                    epsilon=1e-08,)
train_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
        
    sess.run(init)
    if(Bol_load):
    
        for step in range(1,training_steps+1):
            for index, offset in enumerate(range(0, X_train_scaled.shape[0], batch_size)):
                xs, ys= X_train_scaled[offset:offset+batch_size],y_train_scaled[offset:offset+batch_size]
                
                _ ,train_lo,train_pre = sess.run([train_op,loss,output], feed_dict={X: xs, Y: ys} )
                if(index % 100 == 0):
                    _ ,val_lo,val_pre = sess.run([train_op,loss,output], feed_dict={X: X_val_scaled, Y: y_val_scaled} )
                    print("step "+str(step)+", Train_Loss="+"{:.6f}".format(train_lo)+", Val_Loss="+"{:.6f}".format(val_lo))
                
            if(step == training_steps):  
                tran_pre = sc.inverse_transform(train_pre)
                tran_pre= tran_pre.reshape(-1)
    #            num_y= y_train.shape[0]
                for i in range(50):   
                    print('Month=%d, Predicted=%f, Expected=%f' % (i+1, tran_pre[i], y_train[i]))
        
        save_path = saver.save(sess, "./tf_model/model.ckpt")
        print("Model saved in path: %s" % save_path)
        
        _ , test_lo, test_pre = sess.run([train_op,loss,output], feed_dict={X:X_test_scaled ,Y: y_test_scaled} )
        print("Test_Loss="+"{:.4f}".format(test_lo))
#        test_pre = sc.inverse_transform(test_pre)
#        test_pre= test_pre.reshape(-1)
#        for i in range(50):   
#                    print('Month=%d, Predicted=%f, Expected=%f' % (i+1, test_pre[i], y_test[i]))
    else:
        saver.restore(sess, "./tf_model/model.ckpt")
        print("Model restored.")
        _ , test_lo, test_pre = sess.run([train_op,loss,output], feed_dict={X:X_test_scaled ,Y: y_test_scaled} )
        print("Test_Loss="+"{:.4f}".format(test_lo))
        
#-------------------------------------------------------------------------------------------------
#train_pred = regressor.predict(X_train_scaled)
#train_pred = sc.inverse_transform(train_pred)
#train_pred = train_pred.reshape(-1)
#y_pred = regressor.predict(X_test_scaled)
#y_pred = sc.inverse_transform(y_pred)
#y_pred = y_pred.reshape(-1)
#
#for i in range(len(X_test)):   
#    print('Month=%d, Predicted=%f, Expected=%f' % (i+1, y_pred[i], y_test[i]))
#    
#    
#from sklearn.metrics import mean_squared_error,mean_absolute_error
#from math import sqrt
#mse = mean_squared_error(y_train, train_pred)
#print("Train_mse",mse)
#mae= mean_absolute_error(y_train, train_pred)
#print("Train_mae=",mae)
#mape = np.mean(np.abs((y_train - train_pred) / y_train)) * 100
#print("Train_ MAPE:",mape)
#
#mse = mean_squared_error(y_test, y_pred)
#print("Test_mse",mse)
#mae= mean_absolute_error(y_test, y_pred)
#print("Test_mae=",mae)
#mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
#print("Train_ MAPE:",mape)
#
#def show_predict(y_pred,y_test):
#    plt.plot(y_pred)
#    plt.plot(y_test)
#    plt.ylabel("flow")
#    plt.xlabel("time")
#    plt.xlim(0,20)
#    plt.ylim(0, 600)
#    plt.legend(['y_pred', 'y_test'], loc='best')  
#    plt.show() 
#    plt.close()
#
#show_predict(y_pred,y_test)
#
#def show_train_history(i,train_history):
#    plt.plot(train_history.history['loss'])
#    plt.plot(train_history.history['val_loss'])
#    plt.ylabel('loss')
#    plt.xlabel('epoch')
#    plt.xlim(0, 1000)
#    plt.ylim(0, 0.1)
#    plt.legend(['train_loss', 'val_loss'], loc='best')  
#    plt.savefig('./traffic_modelsave/'+time.strftime('%m%d%H%M')+' loss.png')
#    plt.show() 
#    plt.close()
#    
#for i in range(len(history)):
#    show_train_history(i,history[i])