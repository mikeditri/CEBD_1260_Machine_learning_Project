#!/usr/bin/env python
#import libraries

import numpy as np   # import numpy
import pandas as pd  # import pandas
import os
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import glob

#def functions for pre-processing , feature engineering etc

#pre-processing
##type conversions dtype conversion (int64 to int32)

def dtype_conver(Dataframe):
    for col in Dataframe:
        if Dataframe[col].dtype == 'float':
            Dataframe[col] = Dataframe[col].astype(np.float16)
        if Dataframe[col].dtype == 'int64':
            Dataframe[col] = Dataframe[col].astype(np.int16)

##NaN fill with fill with means or medians? Or -1
##Data label encoding (SCIKIT LEARN) for call categorical features that are float or number with decimal .

## Check memory usage
def mem_use(Dataframe):
    mem_use = Dataframe.memory_usage().sum() / 1024 ** 3  # this data set takes >1.7G RAM, we should optimize it after
    print('Memory usage of dataframe is {:.2f} GB'.format(mem_use))

#Feature Engineering
##Convert timestamp and split day month year day of week etc.

#read data
#define your data path or specify your path (on windows add 'r' at the beginning to allow single slashes for path)
DATA_PATH = r'C:/Users/t891199/Desktop/Big_Data_Diploma/CEBD_1260_Machine_learning/Data Files/Class_3/'
file_name = os.path.join(DATA_PATH,'train.csv')

train_df = pd.read_csv(file_name)
print(train_df.shape)


#run functions

mem_use(train_df)
dtype_conver(train_df)
mem_use(train_df)