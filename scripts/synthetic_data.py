from sklearn.neighbors import KernelDensity
import numpy as np
import numpy.random as npr
import itertools

def hypercube(ndim=2):
    corners = [-1,1]
    corner_list = [corners for _ in xrange(ndim)]
    return np.array([i for i in itertools.product(*corner_list)])

def generate_n_blobs(num_samples=5000, nblobs=10,separation=8,ndim=2):
    centers = np.random.rand(nblobs, ndim)
    centers *= separation
    kde = KernelDensity()
    kde.fit(centers)
    samples = kde.sample(num_samples)
    density = np.exp(kde.score_samples(samples))
    return samples,density

def generate_blobs(num_samples=5000,separation=8,ndim=2):
#    centers = np.array([[0,0],[1,0],[0,1],[1,1]],dtype=float)
#    centers -= 0.5
#    centers = np.vstack((centers,#centers*2,centers*3,
                         #centers*4,centers*5,centers*6,
                         #centers*7,centers*8,centers*9,
                         #centers*10,centers*11,centers*12,
                         #centers*13,centers*14,
#                         [0,0]))
    centers = hypercube(ndim)
    centers *= separation
    kde = KernelDensity()
    kde.fit(centers)
    samples = kde.sample(num_samples)
    density = np.exp(kde.score_samples(samples))
    return samples,density

def generate_branches(num_samples=5000,
                      branch_width=0.5,
                      branch_length=5,
                      ndim=2):

    h_corners = np.array(hypercube(ndim))*branch_length
    branches = np.array(h_corners)
    for i in range(20):
        branches = np.vstack((branches, h_corners*(i+2)))
    branches = np.vstack((branches,np.zeros(branches.shape[1])))

    kde = KernelDensity(bandwidth=branch_width)
    kde.fit(branches)
    samples = kde.sample(num_samples)
    density = np.exp(kde.score_samples(samples))

    return samples,density
