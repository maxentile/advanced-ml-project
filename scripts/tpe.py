from time import time
from scipy.cluster import hierarchy
from scipy.spatial import distance
from scipy.spatial.distance import squareform,pdist
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

class HybridTPE(BaseEstimator,TransformerMixin):

    def __init__(self,linkages=['single','complete'],
                 combination_method='add',profile=False):
        ''' combination methods: 'add', 'geometric-mean',
        'normalize-add' '''
        self.linkages = linkages
        self.profile = profile
        self.combination_method=combination_method

    def fit_transform(self,X):
        ''' (1) compute linkage matrix,
        (2) compute pairwise distances using linkage matrix,
        (3) compute embedding using distances'''

        t = time()
        n = len(X)

        # how to combine several cophenetic distance matrices into one?
        # 1. mean,median,mode
        # 2. geometric mean of rank
        # 3. ???
        cs = [hierarchy.linkage(X,l) for l in self.linkages]
        cophenets = [hierarchy.cophenet(c) for c in cs]
        cophenet_array = np.array(cophenets).T
        if self.combination_method == 'geometric-mean':
            coph_orders = [sorted(np.arange(len(d)),key=lambda i:d[i]) for d in cophenets]
            coph_orders = np.array(coph_orders).T
            coph_prod = np.array((coph_orders+1).prod(1),dtype=float)
            dim = coph_orders.shape[1]
            geom_mean = np.array([(float(c)**(1.0/dim)).real for c in coph_prod])
            d = squareform(geom_mean)
        if self.combination_method == 'add':
            d = squareform(cophenet_array.sum(1))
        if self.combination_method == 'normalize-add':
            normalized = (cophenet_array-cophenet_array.min(0))
            normalized /= normalized.max(0)
            d = squareform(normalized.sum(1))

        t1 = time()
        if self.profile:
          print('Computed cophenetic distances in {0:.2f}s'.format(t1-t))

        mds = MDS(dissimilarity='precomputed')
        X_ = mds.fit_transform(d[:n,:n])
        t2 = time()
        if self.profile:
          print('Computed MDS embedding in {0:.2f}s'.format(t2-t1))

        return X_
