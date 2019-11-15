# CEBD_1260_Machine_learning_Project
Project folder for us to share our code and reports

Project:

***NO SHUFFLING of data because we have time wont line up**
Research insurance fields typically used in analytics

Inpycharm need to link to anaconda 

Import numpy as np etc etc etc
Assigment - preprocessing in pycharm (def fuction for preprocessing)
DF feature generation (def feat engineering)

Return DF

Then read data
Then run your new functions 


Setup github repo for sharing & setup outline for code project
Check unique values.
NaN fill with fill with means or medians? Or -1 

Convert timestamp and split day month year day of week etc. 
Preprocessing type conversions dtype conversion (int64 to int32)
Data label encoding (SCIKIT LEARN) for call categorical features that are float or number with decimal . 

Feature important selection (questions should this be done after feature engineering -confirmed.)

Split your data into train and test and validate (do we do this after all this preprocessing and feature eng or in parralel? Confirmed to do cleaning before splitting)

There is 9 columns that have random values in them, other than what the model expected. They are :
'PropertyField3','PropertyField4','PropertyField32','PropertyField34','PropertyField36',
                           'PropertyField38','PersonalField7','PersonalField4A',
                           'PersonalField4B'
  We need to pick through these ans see what their individual issues are. For the sake of progress, we dropped them from the dataset for now


