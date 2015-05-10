from sklearn import neighbors
import numpy as np
import numpy.random as npr

def local_density_k(X,k=10,metric=None,rows_sorted=False):
    if metric=='precomputed':
        # not tested yet, disagrees with ground-truth
        raise NotImplementedError()
        if rows_sorted:
            s = X
        else:
            # sort each row
            s = np.sort(X,1)

        # slice k-nearest (not including self), compute row means
        distances = s[:,1:k+1].mean(1)
    else:
        if metric != None:
            bt = neighbors.BallTree(X,200,metric=metric)
            neighbor_graph = neighbors.kneighbors_graph(X,k,'distance')
        else:
            neighbor_graph = neighbors.kneighbors_graph(X,k,'distance')
        distances = np.array(neighbor_graph.mean(1))[:,0]
    #return np.exp(distances) * np.exp(distances.max())
    #return distances
    return 1-((distances - distances.min())/(distances.max() - distances.min()))

def local_density_k_transformed(X, k, metric=None, rows_sorted=False):
  d = X.shape[1]
  result = local_density_k(X, k, metric, rows_sorted)
  return np.exp(result**d)

def determine_r(X,alpha=5,num_to_check=2000):
    X_ = np.array(X)
    npr.shuffle(X_)
    sampled = X_[:num_to_check]

    neighbor_graph = neighbors.kneighbors_graph(X,2,'distance')

    dist_to_1nn = neighbor_graph.max(1)

    med_min_dist = np.median(dist_to_1nn.toarray()[:,0])

    return alpha*med_min_dist

def local_density_r(X,r=0.1,metric=None):
    if metric=='precomputed':
        counts = (X<r).sum(0)
        counts = np.array(counts,dtype=float) # convert to float
    else:
        if metric != None:
            bt = neighbors.BallTree(X,200,metric=metric)
            neighbor_graph = neighbors.radius_neighbors_graph(bt,r)
        else:
            neighbor_graph = neighbors.radius_neighbors_graph(X,r)
        counts = np.array(neighbor_graph.sum(1))[:,0]
    return ((counts - counts.min())/(1+ counts.max() - counts.min()))
