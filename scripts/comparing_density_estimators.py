import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.neighbors import KernelDensity
from sklearn.metrics import mutual_info_score
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from synthetic_data import generate_blobs
from density_estimation import *

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
        plt.title(param_name+'={0}'.format(params[i]))
        plt.axis('off')



# during sweeps over parameters r and k, compute R^2, mutual information of predicted vs.
# actual density
def sweep_estimator_accuracy(samples, density, estimator, sweep):
    # to-do: produce plot
    # return the curves
    result_r2 = []
    result_mutual_info = []
    d = np.vstack(density)
    for s in sweep:
      pred_density = estimator(samples, s)
      pd = np.vstack(pred_density)
      linearModel = LinearRegression()
      linearModel.fit(d, pd)
      r2 = linearModel.score(d, pd)
      #mutual_info = mutual_info_score(density, pred_density)
      result_r2.append(r2)
      #result_mutual_info.append(mutual_info)
    plt.figure()
    plt.plot(sweep, result_r2, '-o', lw = 2, markersize=6)
    plt.ylabel(r'$R^2$ Score', fontsize=14)
    plt.ylim((0,1))
    plt.grid()
    #plt.figure()
    #plt.plot(sweep, result_mutual_info)


def plot_scatter(samples,density,output_path=''):
    # produces scatterplot of first 2 dimensions of samples, colored by density
    plt.scatter(samples[:,0],samples[:,1], c=density, linewidth=0)
    plt.show()
    if output_path != '':
      plt.savefig(output_path, format='pdf')
    return


def main():
    samples,density = generate_blobs(5000,10)
    sampleshd, densityhd = generate_blobs(5000,10,10)
    r = np.arange(0.01, 3, 0.1) #[0.01,0.1,0.25,0.5,1.0,2.0]
    #plot_estimator_pred(samples,density,local_density_r,r,'r')
    plt.figure()
    k = np.arange(1,300,15) #[1,5,10,50,100,200]
    #plot_scatter(sampleshd, densityhd)
    #plot_estimator_pred(samples,density,local_density_k_transformed,k,'k')
    #sweep_estimator_accuracy(samples, density, local_density_k_transformed, k)
    #sweep_estimator_accuracy(samples, density, local_density_r, r)
    sweep_estimator_accuracy(sampleshd, densityhd, local_density_k_transformed, k)
    sweep_estimator_accuracy(sampleshd, densityhd, local_density_r, r)
    plt.show()

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
