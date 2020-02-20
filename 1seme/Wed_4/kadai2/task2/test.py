# SVMでMNISTの数字画像を識別する学習プログラム
import cv2
import numpy as np
import os
import pandas as pd
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def f(x, w, b):
    return np.dot(w, x) + b


if __name__ == '__main__':
    img = cv2.imread('input/image1.png', 0).reshape(-1)
    img = img / 255.0
    #par = pd.read_csv(filepath_or_buffer="classifier/one_versus_the_rest/SVM0_1.csv", sep=",")
    par = pd.read_csv(filepath_or_buffer="classifier/one_versus_one/SVM0_2.csv", sep=",")

    judge = []
    '''
    for num in range(0, 10):
        w = par.iloc[num, 1:785].values
        b = par.iloc[num, 785]
        value.append(f(img, w, b))
    '''
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

    print(mode(judge))