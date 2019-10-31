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
##NaN fill with fill with means or medians? Or -1
##Data label encoding (SCIKIT LEARN) for call categorical features that are float or number with decimal .

#Feature Engineering
##Convert timestamp and split day month year day of week etc.

#read data
#define your data path or specify your path (on windows add 'r' at the beginning to allow single slashes for path)
DATA_PATH = r'C:/Users/t891199/Desktop/Big_Data_Diploma/CEBD_1260_Machine_learning/Data Files/Class_3/'
file_name = os.path.join(DATA_PATH,'train.csv')

train_df = pd.read_csv(file_name)
print(train_df.shape)



#run functions
