from data_information import read_data, data_information_gathering
from multi_label_encoder import cols_encoding
from decision_tree_classifier import create_decision_tree
from df_manipulation import missing_val_df_split, add_random_df, populate_df_na_val

# Creating a table stating the following information about the adult dataset:
# - number of instances
# - number of mising values
# - fraction of missing values over all attribute values
# - number of instances with missing values
# - fraction of isntances with missing values overa all instances
data_cols = ['age','workclass','education','education-num','marital-status','occupation','relationship','race','sex','capitalgain','capitalloss','hoursperweek','native-country','class']
df, df_dropped = read_data("data/adult.csv",data_cols, 'class')
data_information_gathering(df)

# Using Scikit-learn LabelEncoder converting the 13 attributes to nominal and printing the set of all discrete
# values for each attribute.
cols_encoding(df_dropped)

# Ignoring any values with mising value(s) and use Scikit-learn to build a decision tree for classifying an individual
# to one fo the <=50K and > 50K categories. Compute the error rate of the resulting tree. The data which we are focusing on has
# 14 attributes, 13 of which are feature attributes and 1 as the target attribute. Here we want to build a classifier for
# predicting the class attribute.
create_decision_tree(df,'class')

# Construct a smaller data set D' from the original data set D, containing (i) all instances with at least one missing value and (ii)
# an equal number of randomly selected instances without missing values. That is, if the number of instances with missing values is v in D,
# then D' should countain these v instances and additional v instances without any missing values, which are randomly selected from D. 
# Then using D' construct two modified data sets D'1 and D'2 to handle missing values, In particular:
# - Construct D'1 by creating a new value "missing" for each attribute and using this value for every missing value in D'
# - Construct D'2 by using the most popular value for all missing values for each attribute.
# Train two decision trees with these two data sets and compare their error rates using instances from D for testing.
rows_with_missing_vals, full_data_rows = missing_val_df_split(df)
joint_dfs = add_random_df(rows_with_missing_vals, full_data_rows)

filled_missing = populate_df_na_val(joint_dfs,"missing")
filled_mode = populate_df_na_val(joint_dfs,mode = True)

print(filled_missing)
print(filled_mode)

create_decision_tree(filled_missing, 'class', df)
create_decision_tree(filled_mode, 'class', df)



 