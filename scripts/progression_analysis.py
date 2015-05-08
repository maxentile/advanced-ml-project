import pandas as pd
import numpy as np
import pylab as plt
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
#plt.rcParams['font.family']='Serif'

def import_data():

    path = '../data/stem-cell/'
    file_names=['NG_norm_concat.csv','NN_norm_concat.csv','4FI_norm_concat.csv']

    NN_filenames = ['NN_00.export.csv',
    'NN_02.export.csv',
    'NN_04.export.csv',
    'NN_06.export.csv',
    'NN_08.export.csv',
    'NN_10.export.csv',
    'NN_12.export.csv',
    'NN_14.export.csv',
    'NN_16.export.csv',
    'NN_18.export.csv',
    'NN_20.export.csv',
    'NN_22.export.csv',
    'NN_24.export.csv',
    'NN_30.export.csv']

    NG_filenames = ['NG_00.export.csv',
    'NG_02.export.csv',
    'NG_04.export.csv',
    'NG_06.export.csv',
    'NG_08.export.csv',
    'NG_10.export.csv',
    'NG_12.export.csv',
    'NG_14.export.csv',
    'NG_16.export.csv',
    'NG_18.export.csv',
    'NG_20.export.csv',
    'NG_22.export.csv',
    'NG_24.export.csv',
    'NG_30.export.csv']

    Oct4_filenames = [
    '4FI_00.export.csv',
    '4FI_01.export.csv',
    '4FI_02.export.csv',
    '4FI_04.export.csv',
    '4FI_06.export.csv',
    '4FI_08.export.csv',
    '4FI_10.export.csv',
    '4FI_12.export.csv',
    '4FI_14.export.csv',
    '4FI_16.export.csv',
    '4FI_17.export.csv',
    '4FI_18.export.csv',
    '4FI_20.export.csv']

    NG = [pd.read_csv(path+'NG/'+fname) for fname in NG_filenames]
    NN = [pd.read_csv(path+'NN/'+fname) for fname in NN_filenames]
    Oct4 = [pd.read_csv(path+'4FI/'+fname) for fname in Oct4_filenames]

    # short, nonredundant lists of the days covered
    days_ng = [int(s[3:5]) for s in NG_filenames]
    days_nn = [int(s[3:5]) for s in NN_filenames]
    days_oct = [int(s[4:6]) for s in Oct4_filenames]

    # full-length arrays assigning each observation to a day
    times_ng = np.hstack(np.ones(len(n))*days_ng[i] for i,n in enumerate(NG))
    times_nn = np.hstack(np.ones(len(n))*days_nn[i] for i,n in enumerate(NN))
    times_oct = np.hstack(np.ones(len(n))*days_oct[i] for i,n in enumerate(Oct4))


    #if dataframes:
    #    return NG,NN,Oct4,times_ng,times_nn,times_oct
    #else:
    data_NG = np.vstack(n for n in NG)[:,2:-3]
    data_NN = np.vstack(n for n in NG)[:,2:-3]
    data_Oct4 = np.vstack(n for n in NG)[:,2:-3]

    return data_NG,data_NN,data_Oct4,times_ng,times_nn,times_oct

def plot_cells_per_day():
    plt.plot(times,[len(n) for n in NG],label='Nanog-GFP')
    plt.plot(times,[len(n) for n in NN],label='Nanog-Neo')
    plt.plot(times_oct,[len(n) for n in Oct4],label='Oct4-GFP')
    plt.legend(loc='best')
    plt.ylabel('Number of cells')
    plt.xlabel('Day')
    plt.title('Cells per day')
    plt.savefig('../figures/cells_per_day.pdf')

def process_data(X,cofactor=15):
    return np.arcsinh(X/cofactor)

def compare_2d_embeddings(X,Y,algos):
    raise NotImplementedError('[WIP]')

    X_ = dict()
    for i,algo in enumerate(algos):
        X_[i] = algo.fit_transform(X)

def cluster_prog_analysis(X,times,num_clust=20):
    algorithm=MiniBatchKMeans(num_clust)
    y = algorithm.fit_predict(X)

    def num_in_clust(Y,num_clusters=50):
        assert(num_clusters >= max(Y))
        occupancy = np.zeros(num_clusters)
        for y in Y:
            occupancy[y] += 1
        return occupancy

    clust_ind = sorted(range(num_clust),key=lambda i:(y==i).dot(np.arange(len(y))))
    clust_mat = np.array([num_in_clust(y[times==t],num_clust)[clust_ind] for t in sorted(set(times))]).T


    aspect = float(clust_mat.shape[1]) / clust_mat.shape[0]
    plt.imshow(clust_mat,aspect=aspect,
              interpolation='none',cmap='Blues')
    plt.xlabel('Timestep')
    plt.ylabel('Cluster ID')
    plt.title('Cluster occupancy per time step')
    plt.colorbar()
    plt.savefig('../figures/clust_prog_analysis.jpg')

    return clust_mat, y

def main():
    NG_raw,NN_raw,Oct4_raw,t_NG,t_NN,t_Oct4 = import_data()
    NG,NN,Oct4 = (process_data(dataset) for dataset in (NG_raw,NN_raw,Oct4_raw))

    X = np.vstack((NG,NN,Oct4))
    #Y = np.vstack((t_NG,t_NN+2*t_NG.max(),t_Oct4+2*(t_NG.max()+t_NN.max())))

    #compare_2d_embeddings(X,Y)
    clust_mat,y = cluster_prog_analysis(NG,t_NG)

if __name__ == '__main__':
    main()
