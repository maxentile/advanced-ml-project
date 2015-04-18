import numpy as np
import pylab as pl
pl.rcParams['font.family']='Serif'
import networkx as nx
from sklearn import neighbors
from sklearn.neighbors import KernelDensity
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance
from sklearn.base import BaseEstimator,TransformerMixin

class FlowMap(BaseEstimator,TransformerMixin):
    def __init__(self,target_clust_num=200,
                min_edges=2,
                max_edges=20,
                r_percentile=1.0,
                density_estimator='k',
                k=10,r=0.1):

        if density_estimator=='k':
            self.density_estimator=lambda X: self.local_density_k(X,k)
        else:
            if density_estimator not in 'rk':
                print('density_estimator not understood: defaulting to radial')
            self.density_estimator=lambda X: self.local_density_r(X,r)

        self.k=k
        self.r=r
        self.target_clust_num=target_clust_num
        self.min_edges=min_edges
        self.max_edges=max_edges
        self.r_percentile=r_percentile
        self.accept_prob_func=self.compute_accept_prob

    def local_density_k(self,X,k=10,metric=None):
        if metric != None:
            bt = neighbors.BallTree(X,200,metric=metric)
            neighbor_graph = neighbors.kneighbors_graph(X,k,'distance')
        else:
            neighbor_graph = neighbors.kneighbors_graph(X,k,'distance')
        distances = np.array(neighbor_graph.mean(1))[:,0]
        return 1-((distances - distances.min())/(distances.max() - distances.min()))

    def local_density_r(self,X,r=0.1,metric=None):
        if metric != None:
            bt = neighbors.BallTree(X,200,metric=metric)
            neighbor_graph = neighbors.radius_neighbors_graph(bt,r)
        else:
            neighbor_graph = neighbors.radius_neighbors_graph(X,r)
        counts = np.array(neighbor_graph.sum(1))[:,0]
        return ((counts - counts.min())/(counts.max() - counts.min()))

    def compute_accept_prob(self,densities,
                            outlier_density_percentile=1.0,
                            target_density_percentile=3.0):
        ''' densities is a vector of densities '''
        OD = np.percentile(densities,outlier_density_percentile)
        TD = np.percentile(densities,target_density_percentile)

        accept_prob = np.zeros(len(densities))

        for i,LD in enumerate(densities):
            if LD < OD:
                accept_prob[i] = 0
            elif LD > OD and LD <= TD:
                accept_prob[i] = 1
            elif LD > TD:
                accept_prob[i] = TD/LD
        return accept_prob

    def accept_according_to_probs(self,accept_prob):
        ''' just output indices'''
        return accept_prob > np.random.rand(len(accept_prob))

    # compute cluster centers given cluster assignments
    def compute_cluster_centers(self,X,C):
        centers = np.zeros((len(set(C)),len(X.T)))
        for i in set(C):
            points = X[C==i]
            centers[i] = np.mean(points,0)
        return centers

    def num_edges(self,densities,min_edges=2,max_edges=20):
        ''' pass in an array of densities'''
        assert(len(densities)>1)
        min_density = np.min(densities)
        max_density = np.max(densities)
        lambdas = densities / (max_density - min_density)
        return np.array(min_edges + lambdas*(max_edges - min_edges),dtype=int)

    def construct_graph(self,centers,num_edges_array):
        max_edges = np.max(num_edges_array)
        G = nx.Graph()
        nn = neighbors.NearestNeighbors(max_edges+1)
        nn.fit(centers)
        for i in range(len(centers)):
            dist,neigh = nn.kneighbors(centers[i],num_edges_array[i]+1)
            dist = dist[0]
            neigh = neigh[0]
            for j in range(1,len(dist)):
                G.add_edge(i,neigh[j],weight=dist[j])
        return G

    def fit_transform(self,X):

        # to-do: need to implement edge pruning using time-data, as in paper
        # namely, edges can only be drawn between points in the same timepoint or
        # adjacent timepoints

        # Density-dependent downsampling
        est_density = self.density_estimator(X)
        accept_prob = self.accept_prob_func(est_density)
        accept_ind = self.accept_according_to_probs(accept_prob)
        downsampled = X[accept_ind]

        # Clustering
        cluster_model = AgglomerativeClustering(self.target_clust_num)
        C = cluster_model.fit_predict(downsampled)

        # Graph construction over cluster centers
        centers = self.compute_cluster_centers(downsampled,C)
        pdist = distance.pdist(centers)
        #distmat = distance.squareform(pdist)
        r = np.percentile(pdist,self.r_percentile)
        #adj = (distmat < r)

        num_neighbors = self.local_density_r(centers,self.r)
        #pl.hist(num_neighbors,bins=len(set(num_neighbors)));
        #sorted_clust_id = sorted(range(len(num_neighbors)),key=lambda i:num_neighbors[i])

        num_edges_array = self.num_edges(num_neighbors,self.min_edges,self.max_edges)

        G = self.construct_graph(centers,num_edges_array)
        #w = 1/(distmat)
        #w[w==np.inf]=0
        #weighted_adj_mat = w*adj

        # Rendering
        pos = nx.graphviz_layout(G)
        positions = np.array(pos.values())
        pl.scatter(positions[:,0],positions[:,1])
        nx.draw_networkx_edges(G,pos=positions)
        pl.show()
        return positions

def main():

    def generate_blobs(num_samples=5000,separation=8):
        centers = np.array([[0,0],[1,0],[0,1],[1,1]],dtype=float)
        centers -= 0.5
        centers = np.vstack((centers,#centers*2,centers*3,
                             #centers*4,centers*5,centers*6,
                             #centers*7,centers*8,centers*9,
                             #centers*10,centers*11,centers*12,
                             #centers*13,centers*14,
                             [0,0]))
        centers *= separation
        kde = KernelDensity()
        kde.fit(centers)
        samples = kde.sample(num_samples)
        density = kde.score_samples(samples)
        return samples,density

    samples,density = generate_blobs(1000,10)
    f = FlowMap()
    pos = f.fit_transform(samples)

if __name__ == '__main__':
    main()
