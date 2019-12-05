import numpy as np
# function to set all numerical data to int16 or float16, to save on memory use
def dtype_conver(Dataframe):
    for col in Dataframe:
        if Dataframe[col].dtype in ['float32','float64']:
            Dataframe[col] = Dataframe[col].astype(np.float16)
        if Dataframe[col].dtype in ['int32','float64']:
            Dataframe[col] = Dataframe[col].astype(np.int16)
