from time import time
from scipy.cluster import hierarchy
from scipy.spatial import distance
import numpy as np
import pylab as plt
from sklearn.manifold import MDS

from sklearn.base import BaseEstimator,TransformerMixin

class TPE(BaseEstimator,TransformerMixin):

    def __init__(self,linkage='single',profile=False):
        self.linkage = linkage
        self.profile = profile

    def fit_transform(self,X):
        ''' (1) compute linkage matrix,
        (2) compute pairwise distances using linkage matrix,
        (3) compute embedding using distances'''

        t = time()
        n = len(X)
        c = hierarchy.linkage(X,self.linkage)
        d = distance.squareform(hierarchy.cophenet(c))
        t1 = time()
        if self.profile:
          print('Computed cophenetic distances in {0:.2f}s'.format(t1-t))

        mds = MDS(dissimilarity='precomputed')
        X_ = mds.fit_transform(d[:n,:n])
        t2 = time()
        if self.profile:
          print('Computed MDS embedding in {0:.2f}s'.format(t2-t1))

        self.cluster_tree = c

        return X_
