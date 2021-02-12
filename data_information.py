import pandas as pd
import numpy as np
from fractions import Fraction as frac
from prettytable import PrettyTable

def read_data(data_path, data_cols):
    df = pd.read_csv(data_path, usecols=data_cols)
    return df

def data_information_gathering(data_frame):
    num_missing_values = sum(data_frame.isnull().values.ravel())
    total_values = data_frame.size
    frac_missing_values = frac(num_missing_values, total_values)
    
    num_instance = len(data_frame)
    num_instances_missing_values = sum([True for idx,row in data_frame.iterrows() if any(row.isnull())])
    frac_missing_instances = frac(num_instances_missing_values, num_instance)

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