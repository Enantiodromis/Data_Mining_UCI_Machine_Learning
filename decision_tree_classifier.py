import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix
from multi_label_encoder import cols_encoding, x_y_encoding
import graphviz

#def create_decision_tree(data_frame, predicted_attribute, index_of_attribute):
def create_decision_tree(data_frame, target_column):
    # IGNORE INSTANCES WITH MISSING VALUES
    new_df = data_frame.dropna(axis = 0, how = 'any')

    # CREATE A TEST AND TRAINING SET
    target_column_index = data_frame.columns.get_loc(target_column) 
    #CALLING THE ROW_ENCODING FUNCTION WHICH ENCODES BY LABEL AND THEN TRANSPOSES
    X, Y = x_y_encoding(new_df, target_column_index)

    # Creating a test and training dataset, the test size is 30% of the total dataset.
    X_train, X_test, y_train, y_test =  train_test_split(X, Y, random_state = 0)
    
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
    score = (count/len_test_data) * 100

    print("Out of " + str(len_test_data) + " values, " + str(count) + " were correctly predicted, resulting in " + str(score) + "%" + " accuracy")
 
    # DISPLAY THE DECISION TREE
    decision_path = clf.decision_path(X)
    print(decision_path)

    
