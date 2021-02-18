import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import preprocessing

# The read_data() function reads in csv and outputs a dataframe.
# Additionally a df with a specific column dropped can also be created.
def read_data(data_path, data_cols, excluded_columns = ''):
    df = pd.read_csv(data_path, usecols=data_cols) # Reading in csv, specifiying which columns in particular
    if len(excluded_columns): # If the exluded_columns input is not equal to 0 then:
        df_dropped = df.drop([excluded_columns], axis = 1) # dropping the specified column and saving it to df_dropped
        return df, df_dropped 
    else:
        return df

# FUNCTION WHICH RETURNS INFORMATION ABOUT THE DATA WITHIN THE DATAFRAME
# THE INFORMATION BEING:
# - Number of instances
# - Number of missing values
# - Fraction of missing values over all attribute values
# - Number of instances with missing values
# - Fraction of instances with missing values over all instances
def data_information_gathering(data_frame):
    num_missing_values = sum(data_frame.isnull().values.ravel())
    total_values = data_frame.size
    frac_missing_values = num_missing_values / total_values
    num_instance = len(data_frame)
    num_values = data_frame.count(1)
    num_instances_missing_values = (num_values < len(data_frame.columns)).sum()
    frac_missing_instances = num_instances_missing_values / num_instance

    # Formatting information into a PrettyTable
    table = PrettyTable()
    table.field_names = ["INFORMATION","VALUE"]
    table.align["INFORMATION"] = "l"
    table.add_row(["Number of instances:" , num_instance])
    table.add_row(["Number of missing values:", num_missing_values])
    table.add_row(["Fraction of missing values over all attribute values:", frac_missing_values])
    table.add_row(["Number of instances with missing values:", num_instances_missing_values])
    table.add_row(["Fraction of instances with missing values over all instances:", frac_missing_instances])
    print(table)

# ENCODES, THE EACH ATTRIBUTE AND RETURNS THE SET OF THE ENCODED VALUES
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

def create_decision_tree(data_frame, target_column, test_data_frame = pd.DataFrame(), ignore_missing = True):
    # IGNORE INSTANCES WITH MISSING VALUES
    if ignore_missing == True: 
        new_df = data_frame.dropna(axis = 0, how = 'any')
        new_test_df = test_data_frame.dropna(axis = 0, how = 'any')
    else:
        new_df = data_frame
        new_test_df = test_data_frame

    # CREATE A TEST AND TRAINING SET
    target_column_index = data_frame.columns.get_loc(target_column) 

    if test_data_frame.empty:
        #CALLING THE ROW_ENCODING FUNCTION WHICH ENCODES BY LABEL AND THEN TRANSPOSES
        X, Y = x_y_encoding(new_df, target_column_index)
        # Creating a test and training dataset, the test size is 30% of the total dataset.
        X_train, X_test, y_train, y_test =  train_test_split(X, Y, test_size=0.20, random_state = 0)
    else:
        X_train, y_train = x_y_encoding(new_df, target_column_index)
        testing_data = new_test_df.loc[~new_test_df.index.isin(new_df.index)]
        X_test, y_test = x_y_encoding(testing_data, target_column_index)
        
    # CREATE A DECISION TREE CLASSIFIER
    # Initialising the decision tree classifier
    clf = tree.DecisionTreeClassifier(random_state = 0)
    # Fitting the tree model to the training data
    clf.fit(X_train, y_train)

    # COMPUTE THE ERROR RATE OF THE RESULTING TREE
    # Predicting the class for the test data
    y_hat = clf.predict(X_test)
    # Counting the number of correctly predicted classes
    count = 0
    len_test_data = len(X_test)
    
    for index in range(len_test_data):
        if y_hat[index] == y_test[index]:
            count += 1
    score = 100 - ((count/len_test_data) * 100)
    incorrect_predict = len_test_data - count

    print("Out of " + str(len_test_data) + " values, " + str(count) + " were correctly predicted and " + str(incorrect_predict) + " were incorrectly predicted, resulting in " + str(round(score,2)) + "%" + " error rate")

# FUNCTION TO RETURN DF WITH ONLY MISSING VALUES
def missing_val_df_split(data_frame):
    missing_data_rows = data_frame[data_frame.isnull().any(axis=1)]
    full_data_rows = data_frame.dropna()
    return missing_data_rows, full_data_rows

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
        data_frame_fill.fillna(fill_value, inplace=True)
    else:
        columns = data_frame_fill.columns
        for column in data_frame_fill.columns:
            data_frame_fill[column] = data_frame_fill[column].fillna(data_frame_fill[column].mode().iat[0])
    
    return data_frame_fill

# # # # # # # # # # # # # 
# CALLING THE FUNCTIONS #
# # # # # # # # # # # # #  

data_cols = ['age','workclass','education','education-num','marital-status','occupation','relationship','race','sex','capitalgain','capitalloss','hoursperweek','native-country','class']
df, df_dropped = read_data("data/adult.csv",data_cols, 'class')
data_information_gathering(df)

cols_encoding(df_dropped)
create_decision_tree(df,'class')

rows_with_missing_vals, full_data_rows = missing_val_df_split(df)
joint_dfs = add_random_df(rows_with_missing_vals, full_data_rows)

filled_missing = populate_df_na_val(joint_dfs,"missing")
filled_mode = populate_df_na_val(joint_dfs,mode = True)

create_decision_tree(filled_missing, 'class', df)
create_decision_tree(filled_mode, 'class', df)



 







 