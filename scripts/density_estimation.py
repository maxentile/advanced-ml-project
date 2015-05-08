from sklearn import neighbors
import numpy as np

def local_density_k(X,k=10,metric=None):
    if metric != None:
        bt = neighbors.BallTree(X,200,metric=metric)
        neighbor_graph = neighbors.kneighbors_graph(X,k,'distance')
    else:
        neighbor_graph = neighbors.kneighbors_graph(X,k,'distance')
    distances = np.array(neighbor_graph.mean(1))[:,0]
    #return np.exp(distances) * np.exp(distances.max())
    return distances
    #return np.exp(1-((distances - distances.min())/(distances.max() - distances.min())))

def local_density_r(X,r=0.1,metric=None):
    if metric != None:
        bt = neighbors.BallTree(X,200,metric=metric)
        neighbor_graph = neighbors.radius_neighbors_graph(bt,r)
    else:
        neighbor_graph = neighbors.radius_neighbors_graph(X,r)
    counts = np.array(neighbor_graph.sum(1))[:,0]
    return ((counts - counts.min())/(counts.max() - counts.min()))
