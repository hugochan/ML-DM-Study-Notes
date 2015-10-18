#!/usr/bin/python

import math
import numpy as np

class KNN(object):
    def __init__(self):
        super(KNN, self).__init__()

    def load_data(self, file_in):
        data = np.genfromtxt(file_in, delimiter=',')
        return data

    def predict(self, data, point, k):
        x = data[:, :-1]
        y = data[:, -1]
        dst = ((x - point)**2).sum(1)
        nn_index = []
        for i in range(k):
            index = np.argmin(dst)
            nn_index.append(index)
            dst[index] = np.inf
        class_bag = y[nn_index].tolist()
        votes = {}
        for each in set(class_bag):
            votes[each] = class_bag.count(each)
        dominant_class = sorted(votes.items(), key=lambda d:d[1], reverse=True)[0][0]
        return dominant_class

    def calc_accuracy(self, data, k):
        n = data.shape[0]
        x = data[:, :-1]
        y = data[:, -1]
        count = 0.0
        for i in range(0, n):
            if self.predict(data, x[i], k) == y[i]:
                count += 1
        accuracy = count/n
        return accuracy

if __name__ == '__main__':
    import sys
    knn = KNN()
    try:
        in_file = sys.argv[1]
        k = int(sys.argv[2])
    except:
        print "ERROR: missing or invalid arguments"
        exit()
    data = knn.load_data(in_file)
    accuracy = knn.calc_accuracy(data, k)
    print "accuracy on training data: %s"%accuracy
