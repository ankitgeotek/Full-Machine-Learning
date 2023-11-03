import random
import numpy as np


class KMeans:

    def __init__(self, n_clusters = 2, max_itteration = 100):
        self.n_clusters = n_clusters    #deciding Cluster
        self.max_iter = max_itteration
        self.centroids = None

    def fit_predict(self, x):
        random_index = random.sample(range(0, x.shape[0]), self.n_clusters)
        self.centroids = x[random_index]    # Selecting randon points for centroids
        
        for i in range(self.max_iter):
            # assign cluster
            cluster_group = self.assign_clusters(x)

           
            # move centroids
            old_centroids = self.centroids
            self.centroids = self.move_centroid(x, cluster_group)


            # check finish
            if (old_centroids == self.centroids).all:
                break

        return cluster_group

    def assign_clusters(self, x):
        cluster_group = []
        distances= []

        for row in x:
            for centroid in self.centroids:
                distances.append(np.sqrt(np.dot(row-centroid,row-centroid)))
            
            min_distances = min(distances)  #finding minimum euclidian distance
            index_position = distances.index(min_distances)    #index postion of minimum eucludian distance
            cluster_group.append(index_position)
            distances.clear()

        return np.array(cluster_group)


    def move_centroid(self, x, cluster_group):
        new_centroids = []
        cluster_type = np.unique(cluster_group)

        for type in cluster_type:
            new_centroids = x[cluster_group == type].mean(axis = 0)

        return np.array(new_centroids)



