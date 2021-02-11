from data_information import read_data, data_information_gathering

# Creating a table stating the following information about the adult dataset:
# - number of instances
# - number of mising values
# - fraction of missing values over all attribute values
# - number of instances with missing values
# - fraction of isntances with missing values overa all instances
data_cols = ['age','workclass','education','education-num','marital-status','occupation','relationship','race','sex','capitalgain','capitalloss','hoursperweek','native-country']
df = read_data("data/adult.csv", data_cols)
data_information_gathering(df)

# Using Scikit-learn LabelEncoder converting the 13 attributes to nominal and printing the set of all discrete
# values for each attribute. 