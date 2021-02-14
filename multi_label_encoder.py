import pandas as pd
import numpy as np
from sklearn import preprocessing
from prettytable import PrettyTable

def cols_encoding(data_frame, to_print = True):
    num_columns = len(data_frame.columns)
    column_names = data_frame.columns     
    # List of lists of all the encoded column values
    encoded_cols = []
    # Initialising LabelEncoder
    le = preprocessing.LabelEncoder()
    # Iterating over columns and encoding them.
    for iter in range(num_columns):
        if data_frame[column_names[iter]].dtypes == object:
            encoded = list(set(le.fit_transform(data_frame[column_names[iter]].astype(str))))
        else:
            encoded = list(set(le.fit_transform(data_frame[column_names[iter]])))
        encoded_cols.append(encoded)

    if(to_print):
        # Formatting information into a PrettyTable
        table = PrettyTable()
        table.field_names = ["ATTRIBUTE","NOMINAL VALUE"]
        table.align["ATTRIBUTE"] = "l"
        table.align["NOMINAL VALUE"] = "l"
        for cols in range(len(encoded_cols)):table.add_row([data_frame.columns[cols] , encoded_cols[cols]])
        print(table)
    
    return encoded_cols

def x_y_encoding(data_frame, target_column_index):
    # Calculate number of columns ie: rows  
    num_columns = len(data_frame.columns)
    # Storing all the column names
    column_names = data_frame.columns
    # List of lists of all the encoded column values
    encoded_rows = []
    # Initialising LabelEncoder
    le = preprocessing.LabelEncoder()
    # Iterating over columns and encoding them.
    for iter in range(num_columns):
        if data_frame[column_names[iter]].dtypes == object:
            encoded = list(le.fit_transform(data_frame[column_names[iter]].astype(str)))
        else:
            encoded = list(le.fit_transform(data_frame[column_names[iter]]))
        encoded_rows.append(encoded)
    
    target_encoded_row = encoded_rows[target_column_index]
    target_encoded_row = list(map(lambda el:[el], target_encoded_row))

    del encoded_rows[target_column_index]
    encoded_rows = np.array(encoded_rows)
    transpose_rows = encoded_rows.T
    encoded_rows = transpose_rows.tolist()

    return encoded_rows, target_encoded_row