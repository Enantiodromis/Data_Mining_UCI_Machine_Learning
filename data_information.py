import pandas as pd 

def read_data(data_path, data_cols):
    df = pd.read_csv(data_path, usecols=data_cols)
    return df

def data_information_gathering(data_frame):
    total_values_missing = data_frame.isnull().sum().sum()
    instance_count = df.index
    total_values = total_values_missing + df.count(axis=0, level=None, numeric_only=False)
    

data_cols = ['age','workclass','education','education-num','marital-status','occupation','relationship','race','sex','capitalgain','capitalloss','hoursperweek','native-country']
df = read_data("data/adult.csv", data_cols)
data_information_gathering(df)