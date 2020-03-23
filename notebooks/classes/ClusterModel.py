import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from category_encoders import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, estimate_bandwidth

class ClusterModel():
    
    def __init__(self, df, pca_percent=.95, nb_centroids=[4, 8, 16]):
        self.df = df
        self.pca_percent = pca_percent
        self.nb_centroids = nb_centroids
        
        self.do_pca()
            
    
    def do_pca(self):
        pipe = Pipeline([
            ('ohe',OneHotEncoder()),
            ('Scaler',StandardScaler()),
            ('clf',PCA(self.pca_percent))
        ])
        
        self.pipeline = pipe.fit(self.df)
        self.pca =  self.pipeline[2]
        self.components = pipe.transform(self.df)
            
            
            
    def get_pca_ratio(self):
        return self.pca.explained_variance_ratio_
    
    def get_pca_values(self):
        return self.components
    
    def kmeans_cluster(self):
        result = []
        for nb in self.nb_centroids:
            df_cluster = self.df.copy()
            km = KMeans(n_clusters=nb)
            km.fit(self.components)
            pred = km.predict(self.components)
            
            df_cluster['cluster']= pred 
            result.append(df_cluster)
        
        return result, km
    
    def affinity_propagation(self):
        af = AffinityPropagation(verbose=True, max_iter=3000, damping=0.8).fit(self.components)
        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_
        n_clusters = len(cluster_centers_indices)
        df_cluster = self.df.copy()
        df_cluster['cluster'] = labels
        
        return df_cluster, af, cluster_centers_indices, labels, n_clusters
    
    def mean_shift(self):
        # The following bandwidth can be automatically detected using
        bandwidth = estimate_bandwidth(self.components, quantile=0.2, n_samples=self.df.shape[0])
        model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        model.fit(self.components)
        labels = model.labels_
        df_cluster = self.df.copy()
        df_cluster['cluster'] = labels
        
        return df_cluster, model
        

    