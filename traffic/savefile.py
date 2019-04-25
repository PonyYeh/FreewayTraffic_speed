# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 14:12:30 2018

@author: ting
"""

import numpy as np
import pandas as pd
import glob
COLUMN = []
def set_readDataset(location):
    frame = pd.DataFrame()
    _list = []
    for i,file in enumerate (glob.glob("D:/San Diego Dataset/San Diego Dataset/Freeways-5Min/*.csv")):
        print("read file")
        if i ==0:         
            dataset = pd.read_csv(file)
            COLUMN = dataset.columns
        else:
            dataset = pd.read_csv(file, header=None)
            dataset.columns = COLUMN
            
        dataset = dataset.loc[dataset['Station ID']==location]
        _list.append(dataset) 
        
    frame = pd.concat(_list)
    return frame

df = set_readDataset(1108687)
df.to_csv ("location_1108687_all.csv" , index = False)