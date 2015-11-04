#!/usr/bin/python

import numpy as np

class KMeans(object):
    def __init__(self):
        super(KMeans, self).__init__()

    def load_data(self, file_in):
        data = np.genfromtxt(file_in, delimiter=',')
        return data

    def clustering(self, data, k, eps):
        x = data[:, :-1]
        n, d = x.shape
        k_means = np.random.rand(k, d)
        converged = False
        while not converged:
            pre_k_means = k_means.copy()
            cluster = np.zeros((n, k))
            for i in range(n):
                clus_id = np.argmin(np.linalg.norm(x[i] - k_means, axis=1))
                cluster[i, clus_id] = 1
            # re-compute means for each cluster
            for i in range(k):
                mask = cluster[:, i].reshape((n, 1))
                if np.sum(mask) != 0:
                    k_means[i] = (mask*x).sum(0)/np.sum(mask)
            if np.linalg.norm(k_means - pre_k_means) < eps:
                converged = True
        return k_means, cluster

if __name__ == '__main__':
    import sys
    KMeans = KMeans()
    try:
        in_file = sys.argv[1]
        k = int(sys.argv[2])
        eps = float(sys.argv[3])
    except:
        print "ERROR: missing or invalid arguments"
        exit()
    data = KMeans.load_data(in_file)
    k_means, cluster = KMeans.clustering(data, k, eps)
    print 'kmeans'
    print k_means
    print 'cluster'
    print cluster
