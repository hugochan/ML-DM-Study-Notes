#!/usr/bin/python

import numpy as np
import random

class EMClus(object):
    """
    EM clustering algorithm
    """
    def __init__(self):
        super(EMClus, self).__init__()
        old_settings = np.seterr(over='ignore')


    def load_data(self, file_in, sep=','):
        data = []
        try:
            with open(file_in, 'r') as f:
                for i in f:
                    tmp = i.rstrip('\r\n').split(sep)
                    data.append(tmp)
        except Exception, e:
            print e
            exit()
        f.close()
        data = np.array(data)
        return data

    def gaussian(self, x, mean, cov_inv, cov_det):
        d = cov_inv.shape[0]
        centered_x = (x - mean).reshape((d, 1))
        exponent = -centered_x.transpose().dot(cov_inv).dot(centered_x) / 2
        y = np.exp(exponent) / np.power(2 * np.pi, d / 2.0) / np.power(cov_det, 0.5)
        return y

    def calc_post_prob(self, x, means, covariances, prior_prob, _lambda=0.0001):
        n = x.shape[0]
        k = means.shape[0]
        w = np.zeros((k, n))
        for i in range(k):
            cov_det = np.linalg.det(covariances[i])
            if cov_det == 0:
                covariances[i] += np.diag([_lambda]*(covariances[i].shape[0]))
                cov_det = np.linalg.det(covariances[i])
            cov_inv = np.linalg.inv(covariances[i])
            for j in range(n):
                w[i, j] = prior_prob[i, 0] * self.gaussian(x[j, :], means[i, :], cov_inv, cov_det)
        w = w/w.sum(0)
        return w

    def EM(self, data, k, eps):
        x = data[:, :-1].astype(np.float64)
        n, d = x.shape
        # initilization
        means = np.zeros((k, d))
        covariances = [np.zeros((d, d)) for i in range(k)]
        prior_prob = np.zeros((k, 1))
        cluster = np.zeros((n, k))
        rand_pool = range(k)
        for i in range(n):
            cluster[i, random.choice(rand_pool)] = 1 # random assignment of points to clusters
        # compute initial means, covariances and prior probabilities for each cluster
        for i in range(k):
            mask = cluster[:, i].reshape((n, 1))
            valid_count = np.sum(mask)

            if valid_count != 0:
                means[i, :] = (mask * x).sum(0) / valid_count
                centered_x = mask * (x - means[i, :])
                covariances[i] = centered_x.transpose().dot(centered_x) / valid_count
                prior_prob[i, 0] = float(valid_count) / n

        # iteration
        itn = 0
        while True:
            itn += 1
            pre_means = means.copy()
            w = self.calc_post_prob(x, means, covariances, prior_prob)

            for i in range(k):
                total_weigths = np.sum(w[i, :])
                means[i, :] = w[i, :].dot(x) / total_weigths
                centered_x = x - means[i, :]
                covariances[i] = (w[i, :] * (centered_x.transpose())).dot(centered_x) / total_weigths
                prior_prob[i, 0] = total_weigths / n

            error = 0
            for i in range(k):
                diff = means[i, :] - pre_means[i, :]
                error += diff.dot(diff)
            if error <= eps:
                break
        return means, covariances, prior_prob, itn

    def cluster(self, data, means, covariances, prior_prob):
        x = data[:, :-1].astype(np.float64)
        w = self.calc_post_prob(x, means, covariances, prior_prob)
        cluster = w.argmax(0).tolist()
        clus_size = {}
        clus_pts = {}
        for j in range(x.shape[0]):
            try:
                clus_size[cluster[j]] += 1
                clus_pts[cluster[j]].append(j)
            except:
                clus_size[cluster[j]] = 1
                clus_pts[cluster[j]] = [j]
        return cluster, clus_size, clus_pts

    def calc_purity(self, data, clus_pts):
        n = data.shape[0]
        y = data[:, -1]
        truth_clus = {}
        for j in range(n):
            try:
                truth_clus[y[j]].append(j)
            except:
                truth_clus[y[j]] = [j]
        k = len(clus_pts)
        purity = 0.0
        for i in range(k):
            _max = -1
            for key, val in truth_clus.items():
                count_intersection = len(set(clus_pts[i]) & set(val))
                _max = count_intersection if _max < count_intersection else _max
            purity += _max
        purity /= n
        return purity


if __name__ == '__main__':
    import sys
    try:
        in_file = sys.argv[1]
        K = int(sys.argv[2])
        eps = float(sys.argv[3])
    except:
        # in_file = 'dancing_truth.txt'
        # K = 5
        # eps = 0.00001
        print "ERROR: missing or invalid arguments"
        exit()
    emc = EMClus()
    data = emc.load_data(in_file)
    means, covariances, prior_prob, itn = emc.EM(data, K, eps)
    print "finial means for each cluster:"
    print means
    print "finial covariance matrix for each cluster:"
    print covariances
    print "number of iterations:"
    print itn
    cluster, clus_size, clus_pts = emc.cluster(data, means, covariances, prior_prob)
    print "finial cluster assignment of all the points:"
    print clus_pts
    print "finial size of each cluster:"
    print clus_size
    purity = emc.calc_purity(data, clus_pts)
    print "purity:"
    print purity
