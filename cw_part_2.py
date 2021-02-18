# AUTHOR: GEORGE BRADLEY
# LAST EDIT: 18/02/2021
# TITLE: CW_PART_2.PY

import numpy as np
import pandas as pd
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

# The read_data() function reads in csv and outputs a dataframe.
# Additionally a df with a specific column dropped can also be created.
def read_data(data_path, data_cols, excluded_columns = ''):
    df = pd.read_csv(data_path, usecols=data_cols) # Reading in csv, specifiying which columns in particular
    if len(excluded_columns): # If the exluded_columns input is not equal to 0 then:
        df_dropped = df.drop([excluded_columns], axis = 1) # dropping the specified column and saving it to df_dropped
        return df, df_dropped 
    else:
        return df
# The mean_and_ranges_df() function is used to retrieve the means and ranges
# for every column of a specified dataframe
def mean_and_ranges_df(data_frame):
    table = PrettyTable() # Creating a table using prettyTable
    table.field_names = ["COLUMN","MEAN","MIN","MAX"] # Defining column names
    table.align["COLUMN"] = "l" #
    table.align["MEAN"] = "c"   # Formatting the position
    table.align["MIN"] = "c"    # of column names
    table.align["MAX"] = "c"    #
    for col in data_frame.columns: # Iterating over all the columns within a specified 
        col_mean = data_frame[col].mean() # Calculating the mean for the column
        col_max = data_frame[col].max() # Calculating the max for the column
        col_min = data_frame[col].min() # Calculating the min for the column
        table.add_row([col, col_mean, col_min, col_max]) # Adding all the above calculated data as a row in the table
    print(table) # Displaying the table

# The bc_distance_calc() function is used to calculate the between cluster score.
def bc_distance_calc(centroids): 
    distance_matrix = euclidean_distances(centroids) # Calculating the euclidean distance between centroids results in a distance matrix
    uniq_distances = np.unique(np.around(distance_matrix, decimals=4)) # The values are rounded to prevent float errors and the unique values are kept
    sqr_distances = np.square(uniq_distances) # Squaring the values
    bc_score = np.sum(sqr_distances) # Summing all the values, which returns the bc score
    return bc_score

# The display_cluster_data() function prints the within cluster and between cluster distances and scores
def display_cluster_data(wc, bc, clusters):
    if not isinstance(clusters,list): # If there is only one cluster value, the below line will be printed
        print("K:" + str(clusters) + " WC:" + str(wc) + " BC:" + str(bc) + " BC/WC:" + str(bc/wc))
    else:
        for el in range(len(clusters)): # Iterating through all the cluster sizes and for each one the WC, BC and BC/WC values are printed
            print("K:" + str(clusters[el]) + " WC:" + str(wc[el]) + " BC:" + str(bc[el]) + " BC/WC:" + str(bc[el]/wc[el]))

# The scatter_plotting() function is used to plot the scatterplots when triggered.
def scatter_plotting(data_frame, labels, centroids, clusters):
    nrows = 5 # Specifying the rows for the plots
    ncols = 3 # Specifying the columns for the plots
    cols = data_frame.columns # Storing the names of all the columns within the dataframe
    num_columns = len(cols) # Storing the number of columns within the dataframe
    scaler = clusters * 2 # Creating a scaler to ensure the fig is not too small
    # Intitialising a series of colours so that every cluster up to 13 clusters will have a unique colour.
    colors = ["blue","orange","purple","yellow","pink","magenta","beige","brown","gray","cyan","black","red","green",] 
    # Creating the figure
    fig = plt.figure(figsize=(scaler*ncols, scaler*nrows)) # Specifying the figure size
    fig.subplots_adjust(wspace=0.4) # Width spacing between subplots
    fig.subplots_adjust(hspace=0.4) # Vertical spacing between subplots
    sub_idx = 1 # Initialising the index of the subplots
    for col_1 in range(num_columns): # First element of the pair
        for col_2 in range(col_1+1, num_columns): # Second element of the pair
            for cluster_num in range(clusters): # Iterating over the number of clusters
                fig.add_subplot(nrows, ncols, sub_idx) # Adding the subplot
                Label_col_1 = data_frame[labels == cluster_num].iloc[:,col_1] # Accessing the data for X
                Label_col_2 = data_frame[labels == cluster_num].iloc[:,col_2] # Accessing the data for Y
                # Plotting the scatter whilst specifying certain stylizing variables
                plt.scatter(x=Label_col_1, y=Label_col_2, c=colors[cluster_num],s=30,linewidths=0,alpha=0.5, label="Cluster " + str(cluster_num)) 
                plt.scatter(centroids[:,col_1],centroids[:,col_2], marker='s', s=30, color='k') # Plotting the centroids
                plt.legend(fontsize='8',) # Specifiying the legend fontsize
                # Setting subplot X and Y labels
                plt.title(str(cols[col_1]) + " & " + str(str(cols[col_2])) + " pair", fontsize = 10) # Setting the subplot title
                plt.xlabel(str(cols[col_1]), fontsize = 10)
                plt.ylabel(str(cols[col_2]), fontsize = 10)
            sub_idx += 1
    fig.savefig('scatter'+str(cluster_num)+'.png', dpi=250) # Saving the scatter plot figure as png with dpi of 250 

# The k_means_pairs_run() function is used run the kmeans algorithm with every possible pair of attributes from the dataframe
def k_means_pairs_run(data_frame, clusters, plotting = False, cluster_data = False):

    if not isinstance(clusters, list): # If only processing one cluster at a time
        kmeans = KMeans(n_clusters=clusters, random_state=0) # Call KMeans with random_state=0 to allow replication
        model = kmeans.fit(data_frame) # Fitting the kmeans model
        labels = kmeans.labels_ # Storing the predicted labels
        centroids = model.cluster_centers_ # Storing the cluster centers
        wc_value = model.inertia_ # Storing the within cluster value
        bc_distance = bc_distance_calc(centroids) # Storing the between cluster distance values
        if plotting == True: # If plotting is true
            scatter_plotting(data_frame, labels, centroids, clusters) # The above defined scatter_plotting function is called and the plots made
        if cluster_data == True: # If cluster data is true
            display_cluster_data(wc_value, bc_distance, clusters) # The above mentioned display_cluster_data is called and the WC and BC related data is printed
    else: # If there are multiple number of clusters passed in the code below achieves the same as above, iteratively executing for each value.
        wc_list = []
        bc_store = []
        for k_number in clusters:
            kmeans = KMeans(n_clusters=k_number, random_state=0)
            model = kmeans.fit(data_frame)
            labels = model.labels_
            centroids = model.cluster_centers_
            wc_values = model.inertia_
            bc_distance = bc_distance_calc(centroids)
            wc_list.append(wc_values)
            bc_store.append(bc_distance)
            if plotting == True:
                scatter_plotting(data_frame, labels, centroids, k_number)
        if cluster_data == True:
            display_cluster_data(wc_list, bc_store, clusters)

# # # # # # # # # # # # # 
# CALLING THE FUNCTIONS #
# # # # # # # # # # # # #

# Reading in the required csv file and creating the dataframes used for the rest of the solutions.
data_cols_1 = ['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicassen']
wholsesale_df = read_data("data/wholesale_customers.csv",data_cols_1)

# Question 2.1: 
# Create a table in the report with the mean and range (min and max) for each attribute.
mean_and_ranges_df(wholsesale_df)

# Question 2.2:
# Run k-means with k = 3 and construct a scatterplot for each pair of attributes using Pyplot. 
# Therefore, 15 scatter plots should be constructed in total. Diﬀerent clusters should appear with diﬀerent colors in the scatter plot. 
k_means_pairs_run(wholsesale_df, 3, True, False)

# Question 2.3:
# Run k-means for each possible value of k in the set {3,5,10}. 
# Create a table with the between cluster distance BC, within cluster distance WC and ratio BC/WC
# For each K value.
k_list = [3,5,10]
k_means_pairs_run(wholsesale_df, k_list, False, True)
