#!/usr/bin/python

import math
import numpy as np

class NaiveBayes(object):
    def __init__(self):
        super(NaiveBayes, self).__init__()

    def load_data(self, file_in):
        data = np.genfromtxt(file_in, delimiter=',')
        return data

    def train(self, data):
        n = data.shape[0]
        self.d = data.shape[1] - 1
        self.class_set = list(set(data[:, -1].tolist()))
        self.class_num = len(self.class_set)
        data_list = [[] for i in range(self.class_num)]
        self.prior_prob = [[] for i in range(self.class_num)]
        for i in range(data.shape[0]):
            for j in range(self.class_num):
                if data[i, -1] == self.class_set[j]:
                    data_list[j].append(data[i, :-1])
                    break
        self.u_list = []
        Z_list = []
        self.S = []
        for j in range(self.class_num):
            data_list[j] = np.array(data_list[j])
            self.prior_prob[j] = data_list[j].shape[0]/float(n)
            self.u_list.append(data_list[j].mean(0))
            Z_list.append(data_list[j] - self.u_list[j])
            self.S.append(Z_list[j].transpose().dot(Z_list[j])/n)

    def predict(self, point):
        posterior_prob = []
        for i in range(self.class_num): # each class
            likelihood = 1.0
            for j in range(self.d): # each dimensionality
                likelihood = likelihood/((2*np.pi*self.S[i][j, j])**0.5)*math.exp(-(point[j]-self.u_list[i][j])**2/2/self.S[i][j, j])
            posterior_prob.append(likelihood*self.prior_prob[i])
        class_index = np.argmax(posterior_prob)
        return self.class_set[class_index]

    def calc_accuracy(self, data):
        n = data.shape[0]
        x = data[:, :-1]
        y = data[:, -1]
        count = 0.0
        for i in range(0, n):
            if self.predict(x[i]) == y[i]:
                count += 1
        accuracy = count/n
        return accuracy

if __name__ == '__main__':
    import sys
    nbayes = NaiveBayes()
    try:
        in_file = sys.argv[1]
    except:
        print "ERROR: missing or invalid arguments"
        exit()
    data = nbayes.load_data(in_file)
    nbayes.train(data)
    accuracy = nbayes.calc_accuracy(data)
    print "accuracy on training data: %s"%accuracy
