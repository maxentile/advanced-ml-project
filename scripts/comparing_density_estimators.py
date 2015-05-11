import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.neighbors import KernelDensity
from sklearn.metrics import mutual_info_score
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_blobs
from synthetic_data import generate_blobs, generate_n_blobs
from density_estimation import *
from scipy.spatial.distance import squareform,pdist

def plot_estimator_pred(samples,density,estimator,params,
                         param_name='r'):
    for i in range(6):
        plt.subplot(3,2,i+1)
        est_density = estimator(samples,params[i])
        pred_density = est_density
        #lr = LinearRegression()
        #X = np.reshape(est_density,(len(density),1))
        #lr.fit(X,density)
        #pred_density = lr.predict(X)
        plt.scatter(pred_density,density,
                   c='blue',
                   #c=abs(density-pred_density),
                   #linewidth=0,
                   #cmap='spectral',
                   alpha=0.5,
                   )
        padding=0.05
        plt.ylim(density.min()-(density.min()*padding),density.max()*(1+padding))
        plt.title(param_name+'={0}'.format(params[i]))
        plt.axis('off')



# during sweeps over parameters r and k, compute R^2, mutual information of predicted vs.
# actual density
def sweep_estimator_accuracy(samples, density, estimator, sweep, render=False):
    # to-do: produce plot
    # return the curves
    from scipy.stats import spearmanr

    result_r2 = []
    result_spearman = []
    result_mutual_info = []
    d = np.vstack(density)
    for s in sweep:
      pred_density = estimator(samples, s)
      pd = np.vstack(pred_density)
      linearModel = LinearRegression()
      linearModel.fit(pd,d)
      r2 = linearModel.score(pd,d)
      spearman = spearmanr(pd,d)
      #mutual_info = mutual_info_score(density, pred_density)
      result_r2.append(r2)
      result_spearman.append(spearman[0])
      #result_mutual_info.append(mutual_info)

    if render:
      plt.figure()
      plt.plot(sweep, result_r2, '-o', lw = 2, markersize=6)
      plt.ylabel(r'$R^2$ Score', fontsize=14)
      plt.ylim((0,1))
      plt.grid()
    return result_r2,result_spearman
    #plt.figure()
    #plt.plot(sweep, result_mutual_info)


def plot_scatter(samples,density,output_path=''):
    # produces scatterplot of first 2 dimensions of samples, colored by density
    plt.scatter(samples[:,0],samples[:,1], c=density, linewidth=0)
    if output_path != '':
      plt.savefig(output_path, format='pdf')
    return


def main():
    plt.rcParams['font.family']='Serif'
    npr.seed(0)
    #samples,density = generate_blobs(5000,10)
    #samples,density = generate_n_blobs(5000,10,ndim=10)
    samples,density = generate_n_blobs(5000,10,ndim=50)

    #r = [0.01,0.1,0.25,0.5,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0] #np.arange(0.01, 3, 0.1) #[0.01,0.1,0.25,0.5,1.0,2.0]
    #r = np.hstack((np.arange(0,5,5),np.arange(5,10,0.5),np.arange(10,25,5),np.arange(25,56,2)))
    #r = np.arange(0,20,0.1) # for 2D or 10D
    r = np.arange(0,80,0.1) # for 50D

    k = list(range(1,11)) + [15,20] + list(range(30,500)[::50])
    #k = [1,5,10,50,100,200] #np.arange(1,500,15) #[1,5,10,50,100,200]

    #r_spade = np.array([determine_r(samples) for _ in range(100)])
    #print(r_spade.mean(),r_spade.std())
    r_spade = determine_r(samples)

    l1_distmat = squareform(pdist(samples,'minkowski',1))
    l2_distmat = squareform(pdist(samples))

    '''
    #plot_estimator_pred(samples,density,local_density_r,r,'r')
    plt.figure()
    plot_scatter(samples, density)
    plt.title('Synthetic data')

    # scatter plots
    plt.figure()
    plot_estimator_pred(samples, density, lambda x, kk: local_density_k_transformed(x,kk,'l1'), k, 'k')
    plt.title('K-nearest, L1')
    plt.figure()
    plot_estimator_pred(samples, density, local_density_k_transformed, k, 'k')
    plt.title('K-nearest, L2')
    plt.figure()
    plot_estimator_pred(samples, density, lambda x, rr: local_density_r(x,rr,'l1'), r, 'r')
    plt.title('r-sphere, L1')
    plt.figure()
    plot_estimator_pred(samples, density, local_density_r, r, 'r')
    plt.title('r-sphere, L2')'''

    _,l1_k = sweep_estimator_accuracy(samples, density, lambda x, kk: local_density_k_transformed(x,kk,'l1'), k)
    _,l2_k = sweep_estimator_accuracy(samples, density, local_density_k_transformed, k)
    #l1_k = sweep_estimator_accuracy(np.sort(l1_distmat,1), density, lambda x, kk: local_density_k_transformed(x,kk,'precomputed',True), k)
    #l2_k = sweep_estimator_accuracy(np.sort(l2_distmat,1), density, lambda x, kk: local_density_k_transformed(x,kk,'precomputed',True), k)
    plt.figure()
    plt.plot(k,l1_k,label=r'$\ell_1$',linewidth=2)
    plt.plot(k,l2_k,label=r'$\ell_2$',linewidth=2)
    plt.title(r'$k$-nearest-based density-estimator accuracy')
    plt.legend(loc='best')
    plt.xlabel(r'$k$')
    plt.ylabel(r"Spearman's $\rho$")
    #plt.ylabel(r'$R^2$')
    plt.ylim(0,1)
    #plt.show()
    plt.savefig('../figures/paper/density-estimation/k-nearest.pdf')

    plt.figure()
    #l1_r = sweep_estimator_accuracy(samples, density, lambda x, rr: local_density_r(x,rr,'l1'), r)
    #l2_r = sweep_estimator_accuracy(samples, density, local_density_r, r)

    _,l1_r = sweep_estimator_accuracy(l1_distmat, density, lambda x, rr: local_density_r(x,rr,'precomputed'), r)
    _,l2_r = sweep_estimator_accuracy(l2_distmat, density, lambda x, rr: local_density_r(x,rr,'precomputed'), r)

    plt.plot(r,l1_r,label=r'$\ell_1$',linewidth=2)
    plt.plot(r,l2_r,label=r'$\ell_2$',linewidth=2)
    plt.title(r'$r$-sphere-based density-estimator accuracy')
    plt.vlines(r_spade,0,1,linestyle='--',label=r'$r$ selected by SPADE')
    plt.legend(loc='best')
    plt.xlabel(r'$r$')
    plt.ylabel(r"Spearman's $\rho$")
    #plt.ylabel(r'$R^2$')

    plt.ylim(0,1)
    #plt.show()
    plt.savefig('../figures/paper/density-estimation/r-sphere.pdf')


    """
    transforms = [(lambda x:x, 'id'), (np.log, 'log'), (np.exp, 'exp'), (np.sqrt, 'sqrt'), (lambda x: np.exp(x**2), 'exp(x^d)')]
    for t in transforms:
      plt.figure()
      density_est = lambda samples,param: t[0](local_density_k(samples,param))
      plot_estimator_pred(samples,density,density_est,k,'k')
      plt.savefig('../figures' + t[1] + '.png')
      plt.title(t[1])
    """


if __name__=='__main__':
    main()
