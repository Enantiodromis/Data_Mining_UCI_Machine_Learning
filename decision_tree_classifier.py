import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix
from multi_label_encoder import cols_encoding, x_y_encoding

#def create_decision_tree(data_frame, predicted_attribute, index_of_attribute):
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
        X_train, X_test, y_train, y_test =  train_test_split(X, Y, test_size=0.30, random_state = 0)
    else:
        X_train, y_train = x_y_encoding(new_test_df, target_column_index)
        testing_len = int(len(new_df.index))
        testing_data = new_df.sample(testing_len)
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
    count = 0.0
    len_test_data = len(X_test)
    
    for index in range(len_test_data):
        if y_hat[index] == y_test[index]:
            count += 1
    score = 100 - ((count/len_test_data) * 100)

    print("Out of " + str(len_test_data) + " values, " + str(count) + " were correctly predicted, resulting in " + str(round(score,2)) + "%" + " error rate")

    
