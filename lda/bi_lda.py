#!/usr/bin/python

import numpy as np

class BiLDA(object):
    def __init__(self):
        super(BiLDA, self).__init__()

    def load_data(self, file_in):
        data = np.genfromtxt(file_in, delimiter=',')
        return data

    def train(self, data):
        data_list = [[], []]
        self.class_set = list(set(data[:, -1].tolist()))
        for i in range(data.shape[0]):
            if data[i, -1] == self.class_set[0]:
                data_list[0].append(data[i, :-1])
            else:
                data_list[1].append(data[i, :-1])
        data_list[0] = np.array(data_list[0])
        data_list[1] = np.array(data_list[1])
        u_list = [0, 0]
        u_list[0] = data_list[0].mean(0)
        u_list[1] = data_list[1].mean(0)
        Z_list = [0, 0]
        Z_list[0] = data_list[0] - u_list[0]
        Z_list[1] = data_list[1] - u_list[1]
        S = [0, 0]
        S[0] = Z_list[0].transpose().dot(Z_list[0])
        S[1] = Z_list[1].transpose().dot(Z_list[1])
        Total_S = S[0] + S[1]
        w = np.linalg.inv(Total_S).dot(u_list[0] - u_list[1])
        self.m0 = w.transpose().dot(u_list[0])
        self.m1 = w.transpose().dot(u_list[1])
        return w

    def project(self, w, point):
        return w.transpose().dot(point)

    def predict(self, w, point):
        projection = self.project(w, point)
        if np.linalg.norm(projection - self.m0) < np.linalg.norm(projection - self.m1):
            return self.class_set[0]
        else:
            return self.class_set[1]

    def calc_accuracy(self, data):
        n = data.shape[0]
        x = data[:, :-1]
        y = data[:, -1]
        count = 0.0
        for i in range(0, n):
            if self.predict(w, x[i]) == y[i]:
                count += 1
        accuracy = count/n
        return accuracy

if __name__ == '__main__':
    import sys
    blda = BiLDA()
    try:
        in_file = sys.argv[1]
    except:
        print "ERROR: missing or invalid arguments"
        exit()
    data = blda.load_data(in_file)
    w = blda.train(data)
    print "direction: %s"%w
    accuracy = blda.calc_accuracy(data)
    print "accuracy on training data: %s"%accuracy
