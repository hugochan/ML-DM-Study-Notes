#!/usr/bin/python

import numpy as np
import random

class SMO(object):
    def __init__(self):
        super(SMO, self).__init__()

    def load_data(self, file_in):
        data = np.genfromtxt(file_in, delimiter=',')
        return data

    def train(self, data, C, kernel_type, eps):
        n = data.shape[0]
        x = data[:, :-1]
        y = data[:, -1]
        a = np.zeros((n, ))
        diff = np.inf
        while diff > eps:
            pre_a = a.copy()
            for j in range(0, n):
                i = j
                while i == j:
                    i = random.choice(range(0, n))
                kxii = self.kernel(x[i], x[i], kernel_type)
                kxjj = self.kernel(x[j], x[j], kernel_type)
                kxij = self.kernel(x[i], x[j], kernel_type)
                kij = kxii + kxjj - 2*kxij
                if kij == 0:
                    continue
                old_aj, old_ai = a[j], a[i]
                if not y[i] == y[j]:
                    L = max(0, old_aj - old_ai)
                    H = min(C, C - old_ai + old_aj)
                else:
                    # import pdb;pdb.set_trace()
                    L = max(0, old_ai + old_aj - C)
                    H = min(C, old_ai + old_aj)
                Ei = self.predict(a, x, y, x[i], 0, kernel_type) - y[i]
                Ej = self.predict(a, x, y, x[j], 0, kernel_type) - y[j]
                a[j] = old_aj + y[j]*(Ei - Ej)/float(kij)
                a[j] = max(a[j], L)
                a[j] = min(a[j], H)
                a[i] = old_ai + y[i]*y[j]*(old_aj - a[j])
            diff = np.linalg.norm(a - pre_a)
            # print "diff %s"%diff
        # after convergence
        # compute bias b
        b = 0.0
        count = 0
        for i in range(0, n):
            if a[i] > 0 and a[i] < C:
                b += (y[i] - self.predict(a, x, y, x[i], 0, kernel_type))
                count += 1
        b = b/count
        w = None
        if kernel_type == 'linear':
            w = (a*y).dot(x)
        return a, b, w

    def calc_accuracy(self, data, a, b, kernel_type):
        # compute accuracy on training set
        n = data.shape[0]
        x = data[:, :-1]
        y = data[:, -1]
        count = 0.0
        for i in range(0, n):
            y_hat = self.predict(a, x, y, x[i], b, kernel_type)
            if y_hat*y[i] > 0:
                count += 1
        accuracy = count/n
        return accuracy


    def predict(self, a, x, y, xk, b, kernel_type):
        ay = a*y
        if kernel_type == 'linear':
            return ay.dot(x.dot(xk)) + b
        elif kernel_type == 'quadratic':
            return ay.dot((x.dot(xk))**2) + b
        else:
            raise ValueError('Invalid arguments: %s'%kernel_type)

    def kernel(self, x, y, kernel_type):
        if kernel_type == 'linear':
            return x.dot(y)
        elif kernel_type == 'quadratic':
            # k(x, y) = (x1y1 + x2y2)**2
            return (x.dot(y))**2
        else:
            raise ValueError('Invalid argments: %s'%kernel_type)

if __name__ == '__main__':
    smo = SMO()
    try: # use the comandline parameters
        in_file = sys.argv[1]
        C = sys.argv[2]
        kernel_type = sys.argv[3]
        eps = sys.argv[4]
    except: # use the default parameters
        in_file = 'iris-slwc.txt'
        C = 1
        kernel_type = 'linear'
        eps = 0.001
    data = smo.load_data(in_file)
    a, b, w = smo.train(data, C, kernel_type, eps)
    accuracy = smo.calc_accuracy(data, a, b, kernel_type)
    # print support-vectors, i.e., the pairs i, ai>0
    print 'The support vectors are:'
    count = 0
    for i in range(0, data.shape[0]):
        if a[i] > 0:
            print '%s, %s'%(i, a[i])
            count += 1
    print 'number of support vectors: %s'%count
    print 'bias: %s'%b
    if kernel_type == 'linear':
        print 'w: %s'%w
    print 'accuracy: %s'%accuracy
