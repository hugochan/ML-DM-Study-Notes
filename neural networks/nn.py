#!/usr/bin/python

import numpy as np

class NN(object):
    """backpropagation neural networks: one hidden layer"""
    def __init__(self):
        super(NN, self).__init__()

    def load_data(self, file_in, sep):
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

    def init_settings(self, data, Nh):
        self.X = data[:, :-1].astype(np.float64)
        self.Y = data[:, -1]
        self.N, self.Ni = self.X.shape # num of records & attrbutes
        self.Nh = Nh
        self.class_set = list(set(self.Y.tolist()))
        self.No = len(self.class_set) # num of classes
        self.class_coding = dict(zip(self.class_set, np.diag([1 for i in range(self.No)])))
        self.Wih = np.random.rand(self.Ni + 1, self.Nh) - 0.5 # input - hidden layer
        self.Who = np.random.rand(self.Nh + 1, self.No) - 0.5 # hidden - output layer

    def train(self, data, Nh, eps, eta, epochs):
        self.init_settings(data, Nh)
        for e in range(epochs):
            count = 0
            patterns = range(self.N)
            np.random.shuffle(patterns)
            for i in patterns:
                x = self.X[i].reshape(self.Ni, 1)
                y = self.class_coding[self.Y[i]].reshape(self.No, 1)
                converged = False
                while not converged:
                    y_hat, h = self.feedforward(x)
                    mse = 0.5*np.linalg.norm(y_hat - y)**2
                    if mse <= eps:
                        converged = True
                    else:
                        self.backpropagation(x, y, y_hat, h, eta)
                    count += 1
            # print '%s interations'%count
            # print 'epoch: %s'%e

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def feedforward(self, x):
        x = np.vstack((x, np.ones((1, 1))))
        net = self.Wih.transpose().dot(x) # nh * 1 matrix
        h = self.sigmoid(net)
        h = np.vstack((h, np.ones((1, 1))))
        net = self.Who.transpose().dot(h) # no * 1 matrix
        o = self.sigmoid(net)
        return o, h

    def backpropagation(self, x, y, y_hat, h, eta):
        # update Who: weight connecting hidden and output layers
        Dt = (y - y_hat)*y_hat*(1 - y_hat) # no * 1 matrix
        delta_Who = (eta*Dt.dot(h.transpose())).transpose() # (nh + 1) * no matrix
        self.Who += delta_Who

        # update Wih: wight connecting input and hidden layers
        x = np.vstack((x, np.ones((1, 1))))
        Dt = h[:-1, :]*(1 - h[:-1, :])*(self.Who[:-1, :].dot(Dt)) # nh * 1 matrix
        delta_Wih = (eta*Dt.dot(x.transpose())).transpose() # (ni + 1) * nh matrix
        self.Wih += delta_Wih

    def calc_accuracy(self, data):
        # compute accuracy on training set
        n = data.shape[0]
        X = data[:, :-1].astype(np.float64)
        Y = data[:, -1]
        count = 0.0
        for i in range(0, n):
            x = X[i].reshape(self.Ni, 1)
            y = self.class_coding[Y[i]].reshape(self.No, 1)
            y_hat = self.feedforward(x)[0]
            if np.argmax(y_hat) == np.argmax(y):
                count += 1
        accuracy = count/n
        return accuracy


if __name__ == '__main__':
    import sys
    try:
        in_file = sys.argv[1]
        Nh = int(sys.argv[2])
        eps = float(sys.argv[3])
        eta = float(sys.argv[4])
        epochs = int(sys.argv[5])
    except:
        # in_file = 'iris.data.txt'
        # Nh = 4
        # eps = 0.01
        # eta = 0.5
        # epochs = 100
        print "ERROR: missing or invalid arguments"
        exit()
    nn = NN()
    data = nn.load_data(in_file, ',')
    import time
    t0 = time.time()
    nn.train(data, Nh, eps, eta, epochs)
    print 'runtime of training: %ss'%int(time.time()-t0)
    print 'Wih matrix'
    print nn.Wih
    print 'Who matrix'
    print nn.Who
    accuracy = nn.calc_accuracy(data)
    print 'accuracy: %s'%accuracy
