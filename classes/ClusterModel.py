import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from category_encoders import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

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
        y = []
        for nb in self.nb_centroids:
            df_cluster = self.df.copy()
            km = KMeans(n_clusters=nb)
            km.fit(self.components)
            pred = km.predict(self.components)
            
            df_cluster['cluster']= pred 
            y.append(df_cluster)
        
        return y, km
    