import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from category_encoders import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, estimate_bandwidth, DBSCAN

class ClusterModel():
    
    def __init__(self, df, pca_percent=.95, nb_centroids=[4, 8, 16], do_pca=True):
        self.df = df
        self.pca_percent = pca_percent
        self.nb_centroids = nb_centroids
        self.do_pca = do_pca
        
        if self.do_pca:
            self.pca()
            
    
    def pca(self):
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
        
            if self.do_pca:
                km = KMeans(n_clusters=nb)
                km.fit(self.components)
                pred = km.predict(self.components)
                
            else:
                pipe = Pipeline([
                    ('ohe',OneHotEncoder()),
                    ('Scaler',StandardScaler()),
                    ('clf',KMeans(n_clusters=nb))
                ])
                kmeans_pipe = pipe.fit(self.df)
                km =  kmeans_pipe[2]
                pred = kmeans_pipe.predict(self.df)
            
            df_cluster['cluster']= pred 
            result.append(df_cluster)
        
        return result, km
    
    def affinity_propagation(self):
        
        if self.do_pca:
            af = AffinityPropagation(verbose=True, max_iter=200, damping=0.6).fit(self.components)
            df_cluster = self.df.copy()
            df_cluster['cluster'] = af.labels_
        
        else:
            pipe = Pipeline([
                ('ohe',OneHotEncoder()),
                ('Scaler',StandardScaler()),
                ('clf',AffinityPropagation( max_iter=500, damping=0.7, verbose=True))
            ])
            af_pipe = pipe.fit(self.df)
            af =  pipe[2]
            df_cluster['cluster'] = af_pipe.predict(self.df)
        
        return df_cluster, af
    
    def mean_shift(self):
        bandwidth = estimate_bandwidth(self.components, quantile=0.2, n_samples=self.df.shape[0])
        model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        model.fit(self.components)
        labels = model.labels_
        df_cluster = self.df.copy()
        df_cluster['cluster'] = labels
        
        return df_cluster, model
    
    def dbscan(self):
        pipe = Pipeline([
            ('ohe',OneHotEncoder()),
            ('Scaler',StandardScaler()),
            ('clf', DBSCAN(eps=0.3, min_samples=15))
        ])
        
        db_pipe = pipe.fit(self.df)
        model = db_pipe[2]
        labels = model.labels_
        df_cluster = self.df.copy()
        df_cluster['cluster'] = labels
        
        return df_cluster, model
        
        

    