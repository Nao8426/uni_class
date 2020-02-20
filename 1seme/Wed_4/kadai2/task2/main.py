# SVMでMNISTの数字画像を識別するプログラム
import cv2
import numpy as np
import os
import pandas as pd
from collections import Counter
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def dataset():
    data = []
    label = []
    print('Loading data...', end='')
    for num in range(0, 10):
        for root, dirs, files in os.walk('mini_dataset/{}'.format(num)):
            for f in files:
                data.append(cv2.imread('{}/{}'.format(root, f), 0).reshape(-1))     # 1次元配列に変換してdataに貯めていく
                label.append(num)
    print('Finish!')

    return data, label


def f(x, w, b):
    return np.dot(w, x) + b


def distance(x, w, b):
    return abs(np.dot(w, x) + b) / np.sqrt(np.dot(w, w))


if __name__ == '__main__':
    while 1:
        svm_type = int(input('\"one_versus_the_rest(0)\" or \"one_versus_one(1)\"? : '))
        if svm_type == 0 or svm_type == 1:
            break
        else:
            print('Error. Please, input \"0\" or \"1\"')

    data, label = dataset()

    if svm_type == 0:
        par = pd.read_csv(filepath_or_buffer="classifier/one_versus_the_rest/SVM0_1.csv", sep=",")
        c = 0
        _sum = 0
        for img in data:
            judge = []
            for num in range(0, 10):
                w = par.iloc[num, 1:785].values
                b = par.iloc[num, 785]
                if f(img, w, b) >= 0:
                    judge.append((num, distance(img, w, b)))
                elif f(img, w, b) < 0:
                    judge.append((-1, distance(img, w, b)))

            pred = (-1, 1.7976931348623157e+308)
            for i in judge:
                if i[0] != -1 and i[1] < pred[1]:
                    pred = i

            if pred[0] != -1:
                if pred[0] == label[_sum]:
                    c += 1
            elif pred[0] == -1:
                _min = 1.7976931348623157e+308
                for i in judge:
                    if i[1] < _min:
                        n = i[0]
                        _min = i[1]
                if n == label[_sum]:
                    c += 1

            _sum += 1

            print('Finish No.{} image'.format(_sum))

    elif svm_type == 1:
        par = pd.read_csv(filepath_or_buffer="classifier/one_versus_one/SVM0_2.csv", sep=",")
        c = 0
        _sum = 0
        c_num = [0]*10
        s_num = [0]*10
        for img in data:
            judge = []
            index = 0
            for i in range(0, 10):
                for j in range(i+1, 10):
                    w = par.iloc[index, 1:785].values
                    b = par.iloc[index, 785]
                    if f(img, w, b) >= 0:
                        judge.append(i)
                    elif f(img, w, b) < 0:
                        judge.append(j)
                    index += 1

            count = Counter(judge)
            if count.most_common()[0][0] == label[_sum]:
                c += 1
                c_num[label[_sum]] += 1

            s_num[label[_sum]] += 1
            _sum += 1

            print('Finish No.{} image'.format(_sum))

    for i in range(0, 10):
        accuracy = c_num[i] / s_num[i]
        print('Accuracy for {} : {}'.format(i, accuracy))
        
    accuracy = c / _sum
    print('Accuracy : {}'.format(accuracy))