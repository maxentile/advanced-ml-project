import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.neighbors import KernelDensity
from sklearn.metrics import mutual_info_score
from sklearn.metrics import r2_score
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
def sweep_estimator_accuracy(samples, density, estimator, sweep, plot=False):
    # to-do: produce plot
    # return the curves
    result_r2 = []
    result_mutual_info = []
    for s in sweep:
      pred_density = estimator(samples, s)
      r2 = r2_score(density, pred_density)
      mutual_info = mutual_info_score(density, pred_density)
      result_r2.append(r2) 
      result_mutual_info.append(mutual_info)
    plt.figure()
    plt.plot(sweep, result_r2)
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
    r = [0.01,0.1,0.25,0.5,1.0,2.0]
    #plot_estimator_pred(samples,density,local_density_r,r,'r')
    plt.figure()
    k = [1,5,10,50,100,200]
    
    """
    transforms = [(lambda x:x, 'id'), (np.log, 'log'), (np.exp, 'exp'), (np.sqrt, 'sqrt'), (lambda x: np.exp(x**2), 'exp(x^d)')]
    for t in transforms:
      plt.figure()
      density_est = lambda samples,param: t[0](local_density_k(samples,param))
      plot_estimator_pred(samples,density,density_est,k,'k')
      plt.savefig('../figures' + t[1] + '.png')
      plt.title(t[1])
    """
    plot_scatter(samples, density)
    plot_estimator_pred(samples,density,local_density_k_transformed,k,'k')
    sweep_estimator_accuracy(samples, density, local_density_k_transformed, k)
    #sweep_estimator_accuracy(samples, density, local_density_r, r)
    plt.show()
    

if __name__=='__main__':
    main()
