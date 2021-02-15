import pandas as pd
import numpy as np
import copy 

# FUNCTION TO RETURN DF WITH ONLY MISSING VALUES
def missing_val_df_split(data_frame):
    missing_data_rows = data_frame[data_frame.isnull().any(axis=1)]
    data_rows = data_frame.dropna()
    return missing_data_rows, data_rows

# RANDOMLY GETTING VALUES FROM ONE DF AND ADDING TO ANOTHER
def add_random_df(data_frame1, data_frame2, amount = 0):
    if amount == 0:
        amount = len(data_frame1.index)
    random_rows = data_frame2.sample(amount)
    joint_dfs = pd.concat([data_frame1, random_rows])
    return joint_dfs

# FILL NA WITH VALUE
def populate_df_na_val(data_frame, fill_value = "", mode = False):
    data_frame_fill = data_frame.copy()
    if mode == False and len(fill_value) > 0:
        print("triggered missing")
        data_frame_fill.fillna(fill_value, inplace=True)
    else:
        print("triggered mode")
        columns = data_frame_fill.columns
        for column in data_frame_fill.columns:
            data_frame_fill[column] = data_frame_fill[column].fillna(data_frame_fill[column].mode().iat[0])
    
    return data_frame_fill




