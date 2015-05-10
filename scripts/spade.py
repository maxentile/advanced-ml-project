import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from sklearn import neighbors
from density_estimation import local_density_k_transformed,determine_r,local_density_r
from sklearn.neighbors import KernelDensity
from sklearn.cluster import AgglomerativeClustering
import networkx as nx
from sklearn.base import BaseEstimator,TransformerMixin

class SPADE(BaseEstimator,TransformerMixin):
    def __init__(self,
                  num_clusters=50,
                  density_estimator='r', # or 'k'
                  k=20,
                  r=None):

        self.num_clusters = num_clusters

        self.k=k
        if r != None:
            self.r=r
        else:
            self.r=None

        if density_estimator=='k':
            self.density_estimator=lambda X: local_density_k_transformed(X,k)
        else:
            if density_estimator not in 'rk':
                print('density_estimator not understood: defaulting to radial')
            self.density_estimator=lambda X: local_density_r(X,determine_r(X))

        self.accept_prob_func=self.compute_accept_prob

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

    def downsample(self,X):
        if self.r == None:
            self.r = determine_r(X)
        scores = self.density_estimator(X)
        accept_prob = self.compute_accept_prob(scores)
        down_sampled_X = X[self.accept_according_to_probs(accept_prob)]
        return down_sampled_X

    # compute cluster centers given cluster assignments
    def compute_cluster_centers(self,X,C):
        centers = np.zeros((len(set(C)),len(X.T)))
        for i in set(C):
            points = X[C==i]
            centers[i] = np.mean(points,0)
        return centers

    def fit_transform(self,X,render=False):

        print('density-dependent downsampling...')
        # density-dependent downsampling
        down_sampled_X = self.downsample(X)
        print(len(X),len(down_sampled_X))

        print('clustering...')
        cluster_model = AgglomerativeClustering(self.num_clusters)
        cluster_model.fit(down_sampled_X)
        cluster_pred = cluster_model.labels_
        cluster_centers = self.compute_cluster_centers(X,cluster_pred)


        print('upsampling...')
        knn = neighbors.KNeighborsClassifier(1,metric='l1')
        knn.fit(down_sampled_X,cluster_pred)
        upsampled_cluster_pred = knn.predict(X)
        occupancy = np.array([sum(upsampled_cluster_pred==i) for i in range(self.num_clusters)])
        norm_occupancy = (occupancy - occupancy.min()) / (occupancy.max() - occupancy.min())

        print('constructing graph...')

        G = self.distance_graph(cluster_centers)
        mst = nx.minimum_spanning_tree(G)

        if render:
            print('rendering...')
            self.render(cluster_centers,mst,norm_occupancy)

        return cluster_centers,mst,norm_occupancy

    def render(self,cluster_centers,mst,norm_occupancy,
              marker_scale=1.0,savefig=True,fname=''):
        pos = nx.graphviz_layout(mst)
        positions = np.zeros((len(pos),2))
        for p in pos:
            positions[p] = pos[p]

        for e in mst.edges():
            cpts = positions[np.array(e)]
            plt.plot(cpts[:,0],cpts[:,1],c='0.7',linewidth=1,zorder=1)
        plt.scatter(positions[:,0],positions[:,1],linewidth=0,zorder=10,
                   c=cluster_centers[:,0],s=marker_scale*(100+(200*norm_occupancy)));

        plt.axis('off')
        if savefig:
            fname = "spade.pdf"
            plt.savefig(fname)

    def multiview_fit_and_render(self,X,fname=''):
        results = []
        for i in range(9):
            result = self.fit_transform(X)
            plt.subplot(3,3,i+1)
            self.render(*result,savefig=False,marker_scale=0.1)
            results.append(result)

        if fname=='':
            import os.path
            fname_pre = "spade_multiview"
            fname_post = ".pdf"
            fname = fname_pre + fname_post
            if os.path.isfile(fname):
                x = 1
                fname_pre_x = fname_pre + '({0})'.format(x)
                fname = fname_pre_x + fname_post
                while os.path.isfile(fname):
                    x +=1
                    fname_pre_x = fname_pre + '({0})'.format(x)
                    fname = fname_pre_x + fname_post
        plt.savefig(fname)

        return results

class SPADE_alt(BaseEstimator,TransformerMixin):
    def __init__(self,num_clusters=50,
                 density_estimator='k-nearest',# or 'kde' or 'eps-neighbors'
                 k=20,
                 r=1.0,
                 verbose=True,
                 naive_accept_prob=False):
        self.num_clusters=num_clusters
        self.density_estimator = density_estimator
        self.k=k
        self.r=r
        self.verbose=verbose
        self.naive_accept_prob=naive_accept_prob

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

    def distance(self,x,y):
        return np.sqrt(np.sum((x-y)**2))

    def distance_graph(self,centers):
        G = nx.Graph()
        for i in range(len(centers)-1):
            for j in range(i+1,len(centers)):
                G.add_edge(i,j,weight=self.distance(centers[i],centers[j]))
        return G

    def fit_transform(self,X):
        num_points = len(X)

        from time import time
        t0 = time()
        if self.verbose:
            print("Estimating density...")

        if self.density_estimator not in ['KDE','k-nearest','eps-neighbors']:
            print('invalid density_estimator, defaulting to k-nearest')
            density_estimator = 'k-nearest'

        if self.density_estimator == 'k-nearest':
            scores = self.local_density_k(X,self.k)
            #scores = local_density_k_transformed(X,self.k)
        if self.density_estimator == 'eps-neighbors':
            scores = self.local_density_r(X,self.r)
        if self.density_estimator == 'kde':

            from sklearn.grid_search import GridSearchCV

            # this snippet from documentation:
            # http://scikit-learn.org/stable/auto_examples/neighbors/plot_digits_kde_sampling.html
            params = {'bandwidth': np.logspace(-1, 1, 20)}
            grid = GridSearchCV(KernelDensity(), params)
            grid.fit(samples)

            print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

            # use the best estimator to compute the kernel density estimate
            kde = grid.best_estimator_
            # estimate density
            kde.fit(X)
            scores = kde.score_samples(X)

        if self.verbose:
            print("Done! Elapsed time: {0:.2f}s".format(time() - t0))

        # "down-sample"
        if self.verbose:
            print("Down-sampling...")

        def compute_accept_prob_spade(densities,
                                outlier_density_percentile=1.0,
                                target_density_percentile=3.0):
            ''' densities is a vector of densities '''
            OD = np.percentile(densities,outlier_density_percentile)
            TD = np.percentile(densities,target_density_percentile)

            accept_prob = np.zeros(len(densities))

            for i,LD in enumerate(densities):
                if LD < OD:
                    accept_prob[i] = 0.0
                elif LD > OD and LD <= TD:
                    accept_prob[i] = 1.0
                elif LD > TD:
                    accept_prob[i] = TD/LD
            return accept_prob

        def compute_accept_prob_naive(densities):
            ''' densities is a vector of densities '''
            accept_prob = 1.0 - (scores - scores.min()) / (scores.max() - scores.min())
            return accept_prob

        if self.naive_accept_prob:
            accept_prob = compute_accept_prob_naive(scores)
        else:
            accept_prob = compute_accept_prob_spade(scores)

        down_sampled_X = []
        down_sampled_i = []
        for i in range(num_points):
            if npr.rand() < accept_prob[i]:
                down_sampled_i.append(i)
                down_sampled_X.append(X[i])
        down_sampled_X = np.array(down_sampled_X)
        if self.verbose:
            print("Done! Reduced data from {1} to {2} points. Elapsed time: {0:.2f}s".format(time() - t0,len(X),len(down_sampled_X)))

            print("Clustering...")
        # cluster down-sampled data
        cluster_model = AgglomerativeClustering(self.num_clusters)
        cluster_model.fit(down_sampled_X)

        # compute cluster centers
        cluster_pred = cluster_model.labels_
        cluster_centers = []
        for i in range(self.num_clusters):
            cluster_centers.append(np.mean(down_sampled_X[cluster_pred==i],0))
        cluster_centers = np.array(cluster_centers)

        if self.verbose:
            print("Done! Elapsed time: {0:.2f}s".format(time() - t0))

            print("Up-sampling...")

        knn = neighbors.KNeighborsClassifier(1,metric='l1')
        knn.fit(down_sampled_X,cluster_pred)
        upsampled_cluster_pred = knn.predict(X)
        # "up-sample" (assign all points to nearest cluster center)
        #upsampled_cluster_pred = assign_to_nearest_landmark(X,cluster_centers)

        # compute normalized cluster occupancies
        occupancy = np.array([sum(upsampled_cluster_pred==i) for i in range(self.num_clusters)])
        norm_occupancy = (occupancy - occupancy.min()) / (occupancy.max() - occupancy.min())

        if self.verbose:
            print("Done! Elapsed time: {0:.2f}s".format(time() - t0))

            print("Computing MST...")
        # compute MST
        G = self.distance_graph(cluster_centers)
        mst = nx.minimum_spanning_tree(G)
        if self.verbose:
            print("Done! Elapsed time: {0:.2f}s".format(time() - t0))


            print("Plotting SPADE tree...")
        # plot SPADE tree
        if X.shape[1] == 2:
            print("Plotting overlay on data")
            plt.scatter(X[:,0],X[:,1],c=upsampled_cluster_pred,linewidths=0,alpha=0.5)
            plt.scatter(cluster_centers[:,0],cluster_centers[:,1]);
            for e in mst.edges():
                cpts = cluster_centers[np.array(e)]
                plt.plot(cpts[:,0],cpts[:,1],c='black',linewidth=2)
            plt.axis('off')

            plt.figure()

        # set positions by force-directed layout
        pos = nx.graphviz_layout(mst)
        positions = np.zeros((len(pos),2))
        for p in pos:
            positions[p] = pos[p]

        # if the data is 2D, we can optionally use cluster centers
        #if X.shape[1] == 2:
        #    positions = cluster_centers

        for e in mst.edges():
            cpts = positions[np.array(e)]
            plt.plot(cpts[:,0],cpts[:,1],c='black',linewidth=2)
        #plt.scatter(down_sampled_X[:,0],down_sampled_X[:,1],c=cluster_pred,linewidths=0,alpha=0.5)
        plt.scatter(positions[:,0],positions[:,1],
                   c=cluster_centers[:,0],s=100+(200*norm_occupancy));

        plt.axis('off')

        print("Done! Total elapsed time: {0:.2f}s".format(time() - t0))

        return cluster_centers,mst

if __name__=='__main__':
    # generate fake data
    npr.seed(0)
    from synthetic_data import generate_blobs,generate_branches
    #samples,density = generate_blobs(10000,10)
    samples,density = generate_branches(10000)

    # plot fake data, colored by actual
    plt.rcParams['font.family']='Serif'
    cmap='rainbow'
    s=2
    plt.scatter(samples[:,0],samples[:,1],c=density,linewidths=0,s=s*4,cmap=cmap)
    plt.title('Synthetic data: colored by actual density')
    plt.savefig('synthetic_data.pdf')


    #sp = SPADE_alt(density_estimator='eps-neighbors')

    # using r-based estimator
    #sp = SPADE(density_estimator='r')

    def make_plots(sp,suffix='_r'):

        # plot data colored by estimated_density
        est_density = sp.density_estimator(samples)
        plt.scatter(samples[:,0],samples[:,1],c=est_density,linewidths=0,s=s*4,cmap=cmap)
        plt.title('Synthetic data: colored by estimated density')
        plt.savefig('synthetic_data_est_density'+suffix+'.pdf')

        # plot data colored by acceptance probability
        accept_prob = sp.compute_accept_prob(est_density)
        plt.scatter(samples[:,0],samples[:,1],c=accept_prob,linewidths=0,s=s*4,cmap=cmap)
        plt.colorbar()
        plt.title('Synthetic data: colored by accept_prob')
        plt.savefig('synthetic_data_accept_prob'+suffix+'.pdf')

        # plot a few downsamplings of the data
        plt.figure()
        plt.title('Synthetic data: downsampled')
        for i in range(4):
            downsampled = sp.downsample(samples)
            est_density = sp.density_estimator(downsampled)
            plt.subplot(2,2,i+1)
            plt.scatter(downsampled[:,0],downsampled[:,1],c=est_density,linewidths=0,s=s,cmap=cmap)

        plt.savefig('synthetic_data_downsampled'+suffix+'.pdf')

    make_plots(SPADE(density_estimator='r'),'_r')
    plt.close()
    make_plots(SPADE(density_estimator='k'),'_k')

    # plot results
    #sp.fit_transform(samples)
    #plt.show()
    #_ = sp.fit_transform(samples,render=True)
    #sp.multiview_fit_and_render(samples)
