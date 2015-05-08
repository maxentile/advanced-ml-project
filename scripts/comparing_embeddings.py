import pandas as pd
import numpy as np
import pylab as plt
import seaborn as sns
from sklearn import neighbors

def one_nn_class_baseline(X,labels):
    ''' given a pointcloud X and labels, compute the classification accuracy of
    1NN-classifier
    '''
    one_nn = neighbors.kneighbors_graph(X,2)
    inds = np.zeros(len(X),dtype=int)
    for i in range(len(X)):
        inds[i] = [ind for ind in one_nn[i].indices if ind != i][0]
    preds = Y[inds]
    return 1.0*sum(preds==Y) / len(Y)

def one_nn_baseline(X,Y):
    ''' given two clouds of corresponding points, X and Y, report the fraction
    of nearest-neighbors preserved.

    Algorithm:
    - each pair of corresponding points is labeled by an index i
    - for each point in X, find its nearest neighbor, labeled j
    - for the corresponding point in Y, find its nearest neighbor, labeled j'
    - if j==j', then the nearest neighbors of i are preserved
    - return number of preserved neighbors / the total number possible'''

    # 2, since self is counted as a neighbor by the neighbors module
    one_nn_X = neighbors.kneighbors_graph(X,2)
    one_nn_Y = neighbors.kneighbors_graph(Y,2)
    sames = 0
    for i in range(len(X)):
        neighbor_X = one_nn_X[i].indices[one_nn_X[i].indices!=i][0]
        neighbor_Y = one_nn_Y[i].indices[one_nn_Y[i].indices!=i][0]
        if neighbor_X == neighbor_Y:
            sames+=1

    same_frac = 1.0*sames / len(X)
    return same_frac


def knn_baseline(X,Y,k=5):
    ''' generalization of the one_nn_baseline algorithm...
    given two clouds of corresponding points, X and Y, and a parameter k,
    compute the fraction of each k-nearest-neighborhood conserved


    return the overall fraction of neighborhoods preserved, as well as the
    fraction of each local neighborhood preserved

    '''

    k = k+1 # since self is counted as a neighbor in the kneighbors graph
    knn_X = neighbors.kneighbors_graph(X,k)
    knn_Y = neighbors.kneighbors_graph(Y,k)
    sames = np.zeros(len(X))
    for i in range(len(X)):
        neighbors_X = set(knn_X[i].indices[knn_X[i].indices!=i])
        neighbors_Y = set(knn_Y[i].indices[knn_Y[i].indices!=i])
        sames[i] = len(neighbors_X.intersection(neighbors_Y))
    same_frac = 1.0*sum(sames) / (len(X)*(k-1))
    return same_frac, sames

def knn_baseline_curve(X,Y,ks=range(1,50)):
    ''' slightly less wasteful way to sweep over a range of ks
    when computing knn_baseline, if computing the neighbors graph is expensive'''
    max_k = max(ks)+1 # since self is counted as a neighbor in the kneighbors graph
    knn_X = neighbors.kneighbors_graph(X,max_k)
    knn_Y = neighbors.kneighbors_graph(Y,max_k)
    sames = np.zeros(len(ks))
    for ind,k in enumerate(ks):
        for i in range(len(X)):
            neighbors_X = set(knn_X[i].indices[knn_X[i].indices!=i][:k])
            neighbors_Y = set(knn_Y[i].indices[knn_Y[i].indices!=i][:k])
            sames[ind] += len(neighbors_X.intersection(neighbors_Y))
        sames[ind] /= (len(X)*(k))
    return sames

def plot_1nn_classification_comparison():
    fig, ax = pl.subplots()
    barlist = ax.bar(range(len(vec)),vec)
    pl.hlines(one_nn_class_baseline(X,Y),0,len(vec),linestyles='--')
    pl.xlabel('Algorithm')
    pl.ylabel('1NN Classification Accuracy')
    pl.title('1NN Classification in Low-Dimensional Embeddings')
    baseline_names = ['PCA','Isomap','LLE']
    pl.xticks(range(len(vec)), baseline_names + method_names,rotation=30)
    #pl.ylim(0.25,1.0)

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., height-0.075, '{0:.2f}'.format(height),
                    ha='center', va='bottom',color='white')
    autolabel(barlist)

    for i in range(len(baseline_names)):
        barlist[i].set_color('gray')

    for i in range(len(baseline_names),len(vec)):
        barlist[i].set_color('blue')

    pl.savefig('../figures/embedding-comparison.pdf')

#def plot_neighborhood_preservation()
