#/usr/bin/python

import numpy as np

class AttrStat(object):
    """docstring for """
    def __init__(self):
        super(AttrStat, self).__init__()

    def load_data(self, file_in): # 'OnlineNewsPopularity'
        data = np.genfromtxt(file_in, delimiter=',')[1:, 2:] # the first line in the csv file is attr names
        return data

    def load_data2(self, file_in): # 'iris'
        data = np.genfromtxt(file_in, delimiter=',')[:, :2] # the first line in the csv file is attr names
        return data

    def calc_covariance(self, data):
        centeredData = data - data.mean(0) # centering
        covariance = centeredData.transpose().dot(centeredData)/data.shape[0]
        return covariance

    def calc_correlation(self, data):
        covariance = self.calc_covariance(data)
        variance = covariance.diagonal()
        variance = np.reshape(variance, (1, len(variance))) # convert 1-d array to 2-d array
        correlation = covariance/((variance.transpose().dot(variance))**0.5)
        return correlation

    def calc_dominant_eigens(self, covariance, eps):
        d = covariance.shape[0]
        init_vec = np.ones((d, 1))
        cur_vec = init_vec
        cur_max_elem = 1.0
        diff = np.inf
        while diff >= eps:
            pre_vec = cur_vec
            pre_max_elem = cur_max_elem
            cur_vec = covariance.dot(pre_vec)
            cur_max_elem = cur_vec[np.argmax(np.absolute(cur_vec))]
            # cur_max_elem = np.absolute(cur_vec).max()
            cur_vec = cur_vec/cur_max_elem # re-scale
            diff = np.linalg.norm(cur_vec - pre_vec)
        eigen_vec = cur_vec/np.linalg.norm(cur_vec)
        # eigen_val = cur_max_elem/pre_max_elem
        eigen_val = cur_max_elem
        return [eigen_val, eigen_vec]

    def calc_eigens(self, covariance, eps):
        d = covariance.shape[0]
        init_X = np.random.randn(d, d) # random d*d matrix
        init_X = init_X/np.linalg.norm(init_X, axis=0) # each coloum is unit vector
        cur_X = init_X
        diff = np.inf
        while diff >= eps:
            pre_X = cur_X
            cur_X = covariance.dot(pre_X)
            for i in range(0, d):
                for j in range(0, i):
                    cur_X[:, i] = cur_X[:, i] - cur_X[:, i].dot(cur_X[:, j])*cur_X[:, j]
                cur_X[:, i] = cur_X[:, i]/np.linalg.norm(cur_X[:, i])
            diff = np.linalg.norm(cur_X - pre_X)
        return cur_X

    def proj_data(self, data, U):
        projectedData = data.dot(U)
        return projectedData

