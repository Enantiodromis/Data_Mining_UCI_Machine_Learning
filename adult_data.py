# AUTHOR: GEORGE BRADLEY
# LAST EDIT: 18/02/2021
# TITLE: CW_PART_1.PY

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

# Function which returns information about the data within the dataframe
# THE INFORMATION BEING:
# - Number of instances
# - Number of missing values
# - Fraction of missing values over all attribute values
# - Number of instances with missing values
# - Fraction of instances with missing values over all instances
def data_information_gathering(data_frame):
    num_missing_values = sum(data_frame.isnull().values.ravel()) # Storing number of missing values
    total_values = data_frame.size # Storing the total number of instances
    frac_missing_values = num_missing_values / total_values # Calculating the fraction of missing values to total values
    num_instance = len(data_frame) # Storing number of instances within the dataframe
    num_values = data_frame.count(1)
    num_instances_missing_values = (num_values < len(data_frame.columns)).sum() # Calculating the number of instances which have missing values
    frac_missing_instances = num_instances_missing_values / num_instance # Fraction of missing instances

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

# Encodes for each attribute and returns the set of encoded values
def cols_encoding(data_frame, to_print = True):
    num_columns = len(data_frame.columns) # Storing number of columns
    column_names = data_frame.columns # Storing the column names    
    encoded_cols = [] # List of lists of all the encoded column values
    le = preprocessing.LabelEncoder() # Initialising LabelEncoder
    for iter in range(num_columns):  # Iterating over columns and encoding them.
        # Checking the type of the column and handling accordingly
        if data_frame[column_names[iter]].dtypes == object: 
            encoded = list(set(le.fit_transform(data_frame[column_names[iter]].astype(str))))
        else:
            encoded = list(set(le.fit_transform(data_frame[column_names[iter]])))
        encoded_cols.append(encoded)
    # If print is True the set of nominal values for each of the attributes will be printed in a table
    if(to_print):
        # Formatting information into a PrettyTable
        table = PrettyTable()
        table.field_names = ["ATTRIBUTE","NOMINAL VALUE"]
        table.align["ATTRIBUTE"] = "l"
        table.align["NOMINAL VALUE"] = "l"
        for cols in range(len(encoded_cols)):table.add_row([data_frame.columns[cols] , encoded_cols[cols]])
        print(table)
    return encoded_cols

# A function called from create_decision_tree() function and used to encode the data used for training and testing.
def x_y_encoding(data_frame, target_column_index):  
    num_columns = len(data_frame.columns) # Calculate number of columns ie: rows
    column_names = data_frame.columns # Storing all the column names
    encoded_rows = [] # List of lists of all the encoded column values
    le = preprocessing.LabelEncoder()  # Initialising LabelEncoder
    # Iterating over columns and encoding them.
    for iter in range(num_columns): # Looping through the number of columns
        # Checking the type of the column and handling accordingly
        if data_frame[column_names[iter]].dtypes == object:
            encoded = list(le.fit_transform(data_frame[column_names[iter]].astype(str)))
        else:
            encoded = list(le.fit_transform(data_frame[column_names[iter]]))
        encoded_rows.append(encoded)
    target_encoded_row = encoded_rows[target_column_index] # From the encoded columns targetting the target column
    target_encoded_row = list(map(lambda el:[el], target_encoded_row)) # Making the target column into a list of lists.

    del encoded_rows[target_column_index] # Deleting the target column from the encoded rows
    encoded_rows = np.array(encoded_rows) # Converting to a np.array
    transpose_rows = encoded_rows.T # Transposing the array
    encoded_rows = transpose_rows.tolist() # Converting transposed to a list.

    return encoded_rows, target_encoded_row

# Function which creates trains and tests a decision tree classifier
def create_decision_tree(data_frame, target_column, test_data_frame = pd.DataFrame(), ignore_missing = True):
    # If neede ignores instances with missing values
    if ignore_missing == True: 
        new_df = data_frame.dropna(axis = 0, how = 'any')
        new_test_df = test_data_frame.dropna(axis = 0, how = 'any')
    else:
        new_df = data_frame
        new_test_df = test_data_frame

    target_column_index = data_frame.columns.get_loc(target_column) # Identifies the target column index within the dataframe.

    if test_data_frame.empty:
        X, Y = x_y_encoding(new_df, target_column_index) #Calling the x_y encoding fucntion which encodes the X Y data.
        # Creating a test and training dataset, the test size is 30% of the total dataset.
        X_train, X_test, y_train, y_test =  train_test_split(X, Y, test_size=0.20, random_state = 0)
    else:
        X_train, y_train = x_y_encoding(new_df, target_column_index)
        testing_data = new_test_df.loc[~new_test_df.index.isin(new_df.index)] # Ensuring that the no data which appears in training data is in test data 
        X_test, y_test = x_y_encoding(testing_data, target_column_index)
        
    clf = tree.DecisionTreeClassifier(random_state = 0) # Initialising the decision tree classifier
    clf.fit(X_train, y_train) # Fitting the tree model to the training data

    # Computing the error rate of the decision tree
    y_hat = clf.predict(X_test) # Predicting the class for the test data
    count = 0 # Counting the number of correctly predicted classes
    len_test_data = len(X_test)
    
    for index in range(len_test_data):
        if y_hat[index] == y_test[index]:
            count += 1
    score = 100 - ((count/len_test_data) * 100)
    incorrect_predict = len_test_data - count

    print("Out of " + str(len_test_data) + " values, " + str(count) + " were correctly predicted and " + str(incorrect_predict) + " were incorrectly predicted, resulting in " + str(round(score,2)) + "%" + " error rate")

# Function to return df with only missing values
def missing_val_df_split(data_frame):
    missing_data_rows = data_frame[data_frame.isnull().any(axis=1)]
    full_data_rows = data_frame.dropna()
    return missing_data_rows, full_data_rows

# Randomly gets values from one df and adds to another used for creating D'
def add_random_df(data_frame1, data_frame2, amount = 0):
    if amount == 0:
        amount = len(data_frame1.index)
    random_rows = data_frame2.sample(amount)
    joint_dfs = pd.concat([data_frame1, random_rows])
    return joint_dfs

# Filling missing values withing a dataframe with a specified value or with the mode value
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
 
# Reading in the required csv file and creating the dataframes used for the rest of the solutions.
data_cols = ['age','workclass','education','education-num','marital-status','occupation','relationship','race','sex','capitalgain','capitalloss','hoursperweek','native-country','class']
df, df_dropped = read_data("data/adult.csv",data_cols, 'class')

# Question 1.1:
# Create a table in the report stating the following information about the adult data set: 
# (i) number of instances, 
# (ii) number of missing values, 
# (iii) fraction of missing values over all attribute values, 
# (iv) number of instances with missing values and 
# (v) fraction of instances with missing values over all instances.
data_information_gathering(df)

# Question 1.2:
#Convert all 13 attributes into nominal using a Scikit-learn LabelEncoder. 
# Then, print the set of all possible discrete values for each attribute.
cols_encoding(df_dropped)

# Question 1.3:
# Ignore any instance with missing value(s) and use Scikit-learn to build a decision tree
# for classifying an individual to one of the <= 50K and > 50K categories. 
# Compute the error rate of the resulting tree
create_decision_tree(df,'class')

# Question 1.3:
#construct a smaller data set D0 from the original data set D, containing 
# (i) all instances with at least one missing value and 
# (ii) an equal number of randomly selected instances without missing values. 
rows_with_missing_vals, full_data_rows = missing_val_df_split(df)
joint_dfs = add_random_df(rows_with_missing_vals, full_data_rows)

# Construct D1 by creating a new value “missing” for each attribute and using this value for every missing value in D0
filled_missing = populate_df_na_val(joint_dfs,"missing")
# Construct D2 by using the most popular value for all missing values of each attribute
filled_mode = populate_df_na_val(joint_dfs,mode = True)

# Train two decision trees with these two data sets and compare their error rates using instances from D for testing. 
create_decision_tree(filled_missing, 'class', df)
create_decision_tree(filled_mode, 'class', df)



 







 