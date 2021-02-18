#####################################################################################
####################### FILE NAME: adult_data.py ####################################
#####################################################################################
1. read_data(data_path, data_cols, excluded_columns = '')
	Inputs: datapath, list of strings, string
	Outputs: Either one dataframe or two dataframes
	
	Function: Reads in csv and outputs a dataframe. 
		  Additionally a df with a specific column dropped can also be created.

2. data_information_gathering(data_frame)
	Inputs: dataframe
	Outputs: Prints table

	Function: Returns information about the data within the dataframe
		  THE INFORMATION BEING:
 			- Number of instances
 			- Number of missing values
 			- Fraction of missing values over all attribute values
 			- Number of instances with missing values
 			- Fraction of instances with missing values over all instances

3. cols_encoding(data_frame, to_print = True)
	Inputs: dataframe, boolean 
	Outputs: encoded column values

	Function: Encodes for each attribute and returns the set of encoded values

4. x_y_encoding(data_frame, target_column_index)
	Inputs: dataframe, integer value 
	Outputs: two lists "encoded_rows" and "target_encoded_row"  

	Function: A function called from create_decision_tree() function and used to encode the data used for training and testing.

5. create_decision_tree(data_frame, target_column, test_data_frame = pd.DataFrame(), ignore_missing = True)
	Inputs: dataframe, string, dataframe, boolean 
	Outputs: printed line, showing the error rate of a decision tree 

	Function: Function which creates trains and tests a decision tree classifier

6. missing_val_df_split(data_frame)
	Inputs: dataframe 
	Outputs: dataframe

	Function: Return df with only missing values

7. add_random_df(data_frame1, data_frame2, amount = 0)
	Inputs: dataframe, dataframe, integer
	Outputs: dataframe

	Function: Randomly gets values from one df and adds to another used for creating D'

8. populate_df_na_val(data_frame, fill_value = "", mode = False)
	Inputs: dataframe, string, boolean 
	Outputs: printed line, showing the error rate of a decision tree 

	Function: Filling missing values withing a dataframe with a specified value or with the mode value

#####################################################################################
####################### FILE NAME: wholesale_data.py #####################################
#####################################################################################
1. read_data(data_path, data_cols, excluded_columns = '')
	Inputs: datapath, list of strings, string
	Outputs: Either one dataframe or two dataframes
	
	Function: Reads in csv and outputs a dataframe. 
		  Additionally a df with a specific column dropped can also be created.

2. mean_and_ranges_df(data_frame)
	Inputs: datapath
	Outputs: prints a table
	
	Function: function is used to retrieve the means and ranges for every column of a specified dataframe

3. bc_distance_calc(centroids)
	Inputs: np.array()
	Outputs: float
	
	Function: Used to calculate the between cluster score..

4. display_cluster_data(wc, bc, clusters)
	Inputs: list, list, list or integer
	Outputs: Prints a line, K number WC value, BC value and BC/WC ratio
	
	Function: Prints the within cluster and between cluster distances and scores

5. scatter_plotting(data_frame, labels, centroids, clusters)
	Inputs: dataframe, np.array, np.array, list
	Outputs: saves a png image
	
	Function: Called by the k_means_pair_run function and used to plot scatterplots when triggered