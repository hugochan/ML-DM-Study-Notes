#!/usr/bin/python

import numpy as np

class PCA(object):
    def __init__(self):
        super(PCA, self).__init__()

    def load_data(self, file_in):
        data = np.genfromtxt(file_in, delimiter=',')
        return data

    def train(self, data, fr):
        D = data[:, :-1]
        n, d = D.shape
        u = D.mean(0)
        D = D - u # centered data
        covarance = D.transpose().dot(D)/n
        w, v = np.linalg.eigh(covarance)
        ordered_eigh = sorted(zip(w, v.transpose()), key=lambda d:d[0], reverse=True)
        total_variance = np.trace(covarance)
        threshold = total_variance*fr
        sumf = 0.0
        r = 0
        for each in ordered_eigh:
            sumf += each[0]
            r += 1
            if sumf >= threshold:
                break
        return np.array([ordered_eigh[i][1] for i in range(r)]).transpose()

    def project(self, w, point):
        return w.transpose().dot(point)

if __name__ == '__main__':
    import sys
    pca = PCA()
    try:
        in_file = sys.argv[1]
        fr = float(sys.argv[2])
    except:
        print "ERROR: missing or invalid arguments"
        exit()
    data = pca.load_data(in_file)
    w = pca.train(data, fr)
    print "reduced basis: %s"%w
    print "projection of %s: %s"%(data[0, :-1], pca.project(w, data[0, :-1]))
