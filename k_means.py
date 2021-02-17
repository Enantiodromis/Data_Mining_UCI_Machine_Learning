import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

def k_means_pairs_run(data_frame, clusters):
    nrows = 5
    ncols = 3
    
    cols = data_frame.columns
    num_columns = len(cols)
    
    colors = ["blue","orange","purple","yellow","pink","magenta","beige","brown","gray","cyan","black","red","green",]

    if not isinstance(clusters, list):
        kmeans = KMeans(n_clusters=clusters, random_state=0)
        kmeans.fit(data_frame)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        # Scaling so that the figure is not too small
        scaler = clusters * 2
        # Creating the figure
        fig = plt.figure(figsize=(scaler*ncols, scaler*nrows))
        fig.subplots_adjust(wspace=0.4)
        fig.subplots_adjust(hspace=0.4)
        # Initialising the index of the subplots
        sub_idx = 1
        # Looking through the index for the first element of the pair
        for col_1 in range(num_columns):
            # Looking through the index for the second element of the pair
            for col_2 in range(col_1+1, num_columns):
                # Iterate over the number of clusters
                for cluster_num in range(clusters):
                    fig.add_subplot(nrows, ncols, sub_idx)
                    
                    Label_col_1 = data_frame[labels == cluster_num].iloc[:,col_1]
                    Label_col_2 = data_frame[labels == cluster_num].iloc[:,col_2]  

                    plt.scatter(x=Label_col_1, y=Label_col_2, c=colors[cluster_num],s=30,linewidths=0,alpha=0.5, label="Cluster " + str(cluster_num))
                    plt.scatter(centroids[:,col_1],centroids[:,col_2], marker='s', s=30, color='k')
                    plt.legend(fontsize='8',)

                    plt.xlabel('This is the X axis label')
                    plt.ylabel('This is the Y axis label')
                    plt.title(str(cols[col_1]) + " & " + str(str(cols[col_2])) + " pair", fontsize = 10)
                    plt.xlabel(str(cols[col_1]), fontsize = 10)
                    plt.ylabel(str(cols[col_2]), fontsize = 10)

                sub_idx += 1    
        plt.show()