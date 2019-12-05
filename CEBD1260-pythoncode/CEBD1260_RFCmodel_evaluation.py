import numpy as np
import pandas as pd
from CEBD1260_preprocessing import ohe
from CEBD1260_preprocessing import master_pipe_RFC
from CEBD1260_cleaning import dtype_conver
from scipy.sparse import coo_matrix, hstack

# to display maximum rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# train.csv file is already in this project folder..so...
# pandas reads in csv file
old_train_df = pd.read_csv('train.csv')
print("Shape of train dataset is:")
print(old_train_df.shape)

#original_quote_date is time-series
#Feature Engineering
old_train_df['Original_Quote_Date'] = pd.to_datetime(old_train_df['Original_Quote_Date'])
old_train_df['year'] = old_train_df['Original_Quote_Date'].dt.year
old_train_df['month'] = old_train_df['Original_Quote_Date'].dt.month
old_train_df['day'] = old_train_df['Original_Quote_Date'].dt.day
train_df = old_train_df.drop(["Original_Quote_Date"], axis = 1)

# Convert all numerical value cols to int16 or float16 to save on memory use
dtype_conver(train_df)
# lets see how many NaN or Null values are in each column
nan_info = pd.DataFrame(train_df.isnull().sum()).reset_index()
nan_info.columns = ['col','nan_cnt']

#sort them in descending order and print 1st 10
nan_info.sort_values(by = 'nan_cnt',ascending=False,inplace=True)
nan_info.head(10)

# extract column names with NaNs and Nulls
# in numerical cols
num_cols_with_missing = ['PersonalField84','PropertyField29']

# extract column names with NaNs and Nulls
# in boolean type cols
bool_cols_with_missing = ['PropertyField3','PropertyField4','PersonalField7','PropertyField32',
                          'PropertyField34','PropertyField36','PropertyField38']

# fill in null and NaN values with 'U' in boolean type cols ('Y','N')
for cols in bool_cols_with_missing:
    train_df[cols].fillna('U',inplace=True)

# fill in null and NaN values with -1 in numerical missing values
for cols in num_cols_with_missing:
    train_df[cols].fillna(-1, inplace=True)

# define target
y = old_train_df["QuoteConversion_Flag"].values

# drop target column from data
# and static columns GeographicField10A & PropertyField6
X = train_df.drop(["QuoteConversion_Flag","GeographicField10A","PropertyField6"], axis = 1)

#QuoteNumber setting as index
X = X.set_index("QuoteNumber")

# select all columns that are categorical i.e with unique categories less than 40 in our case
X_for_ohe = [cols for cols in X.columns if X[cols].nunique() < 40 or X[cols].dtype in['object']]
X_not_ohe = [cols for cols in X.columns if X[cols].nunique() > 40 and X[cols].dtype not in['object']]

#numerical column that we will not encode
print("Numerical columns not to be encoded: {}".format(X[X_not_ohe].head()))

#to keep track of our columns, how many are remaining after we removed 4 so far?
print("Current length of X_for_ohe is: {}".format(len(X_for_ohe)))

#print(X['SalesField8'].head())

# Check to see if we still have null or NaN values
nan_info = pd.DataFrame(X[X_for_ohe].isnull().sum()).reset_index()
nan_info.columns = ['col','nan_cnt']

#sort them in descending order and print 1st 10
nan_info.sort_values(by = 'nan_cnt',ascending=False,inplace=True)
print("Top 10 of remaning null or NaN values per col: ")
print(nan_info.head(10))

# apply OneHotEncoder on categorical feature columns and return a csr_matrix
X_ohe = ohe.fit_transform(X[X_for_ohe])

# print the csr_matrix
print("csr_matrix format will be: ")
print(X_ohe)

# SalesField8 was kept out of sparse matrix, now we need to bring it back
# scale down SalesField8 for easy handling using log(), then convert to float16
SF8 = np.log(X['SalesField8']).astype(np.float16)

# put SalesField8 back in
hstack((X_ohe,np.array(SF8)[:,None]))

# lets get the model k-fold scores and print feature importance graphs
print("Feature importance graphs will be printed to local folder /plots")
print("5-fold cross-validation scores are:")

master_pipe_RFC(X_ohe,y)
