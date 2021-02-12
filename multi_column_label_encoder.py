from sklearn import preprocessing
from prettytable import PrettyTable

def cols_encoding(data_frame, column_names):     
    # List of lists of all the encoded column values
    encoded_cols = []
    # Initialising LabelEncoder
    le = preprocessing.LabelEncoder()
    # Iterating over columns and encoding them.
    for iter in column_names:
        if data_frame[iter].dtypes == object:
            encoded = list(set(le.fit_transform(data_frame[iter].astype(str))))
        else:
            encoded = list(set(le.fit_transform(data_frame[iter])))
        encoded_cols.append(encoded)

    # Formatting information into a PrettyTable
    table = PrettyTable()
    table.field_names = ["ATTRIBUTE","NOMINAL VALUE"]
    table.align["ATTRIBUTE"] = "l"
    table.align["NOMINAL VALUE"] = "l"
    for cols in range(len(encoded_cols)):table.add_row([column_names[cols] , encoded_cols[cols]])
    print(table)