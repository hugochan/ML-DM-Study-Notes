#!/usr/bin/python

import numpy as np

class MultiLDA(object):
    def __init__(self):
        super(MultiLDA, self).__init__()

    def load_data(self, file_in):
        data = np.genfromtxt(file_in, delimiter=',')
        return data

    def train(self, data):
        d = data.shape[1] - 1
        self.class_set = list(set(data[:, -1].tolist()))
        self.class_num = len(self.class_set)
        data_list = [[] for i in range(self.class_num)]
        for i in range(data.shape[0]):
            for j in range(self.class_num):
                if data[i, -1] == self.class_set[j]:
                    data_list[j].append(data[i, :-1])
                    break
        self.u_list = []
        Z_list = []
        S = []
        Total_S = np.zeros((d, d))
        mean_u = np.zeros((d,))
        for j in range(self.class_num):
            data_list[j] = np.array(data_list[j])
            self.u_list.append(data_list[j].mean(0))
            Z_list.append(data_list[j] - self.u_list[j])
            S.append(Z_list[j].transpose().dot(Z_list[j]))
            Total_S += S[j]
            mean_u += self.u_list[j]
        mean_u /= self.class_num
        B = np.zeros((d, d))
        for j in range(self.class_num):
            tmp = (self.u_list[j] - mean_u).reshape((d, 1))
            B += (tmp).dot(tmp.transpose())
        B /= self.class_num
        S_I_B = np.linalg.inv(Total_S).dot(B)
        w, v = np.linalg.eigh(S_I_B)
        dominant_direction = v[np.argmax(w)]
        self.m_list = []
        for j in range(self.class_num):
            self.m_list.append(dominant_direction.transpose().dot(self.u_list[j]))
        return dominant_direction

    def project(self, dominant_direction, point):
        return dominant_direction.transpose().dot(point)

    def predict(self, dominant_direction, point):
        projection = self.project(dominant_direction, point)
        min_dst = np.inf
        min_index = -1
        for j in range(self.class_num):
            tmp = np.linalg.norm(projection - self.project(dominant_direction, self.u_list[j]))
            if tmp < min_dst:
                min_dst = tmp
                min_index = j
        return self.class_set[min_index]

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
    blda = MultiLDA()
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
