from data_information import read_data, data_information_gathering
from multi_label_encoder import cols_encoding
from decision_tree_classifier import create_decision_tree


# Creating a table stating the following information about the adult dataset:
# - number of instances
# - number of mising values
# - fraction of missing values over all attribute values
# - number of instances with missing values
# - fraction of isntances with missing values overa all instances
data_cols = ['age','workclass','education','education-num','marital-status','occupation','relationship','race','sex','capitalgain','capitalloss','hoursperweek','native-country','class']
df = read_data("data/adult.csv",data_cols)
#data_information_gathering(df)

# Using Scikit-learn LabelEncoder converting the 13 attributes to nominal and printing the set of all discrete
# values for each attribute.
#cols_encoding(df_dropped)

# Ignoring any values with mising value(s) and use Scikit-learn to build a decision tree for classifying an individual
# to one fo the <=50K and > 50K categories. Compute the error rate of the resulting tree. The data which we are focusing on has
# 14 attributes, 13 of which are feature attributes and 1 as the target attribute. Here we want to build a classifier for
# predicting the class attribute.
create_decision_tree(df,'class')



 