from data_information import read_data, data_information_gathering, mean_and_ranges_df
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

create_decision_tree(filled_missing, 'class', df)
create_decision_tree(filled_mode, 'class', df)

# For this part of the coursework, the attributes CHANNEL and REGION should be droppped. The following 6 numeric attributes should be considered:
# ---------------------------------------------------------
# Attribute:    -   Description:                          -
# ---------------------------------------------------------
# FRESH         - Annual expenses on fresh products       -
# MILK          - Annual expenses on milk products        -
# GROCERY       - Annual expenses on grocery products     -
# FROZEN        - Annual expenses on frozen products      -
# DETERGENTS    - Annual expenses on detergent products   -
# DELICATESSEN  - Annual expenses on delicatessen products-
#----------------------------------------------------------

# Create table in the report with the mean and range for each attribute j, where xij is the attribute j value of instances i and Xminj Xmaxj are the
# minimum and maximum attribute j values among all instances.
data_cols_1 = ['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicassen']
df_1 = read_data("data/wholesale_customers.csv",data_cols_1)
mean_and_ranges_df(df_1)

# Run K-means with k=3 and constrcut a scatterplot for each pair of attributes using Pyplot. Therefore, 15 scatter plots should be constructed in total.
# Different clusters should appear with different colors in the scatter plot. All scatter plots should be included in the report, using no more than two 
# pages for them. What do you observe?


# Run k-means for each possible value of k in the set {3,5,10}. Complete the following table with the between cluster distance BC, within cluster distance WC
# and ration BC/WC of the set of clusters obtained for each K. Briefly commment on the obtained results.









 