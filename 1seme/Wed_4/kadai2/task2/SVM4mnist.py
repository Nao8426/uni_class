# SVMでMNISTの数字画像を識別するために多数決用の分類器を作成
import cv2
import cvxopt
import numpy as np
import os
import pandas as pd
import time
from sklearn.model_selection import train_test_split
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# データセットの作成
class DATASET:
    def __init__(self, num1, num2, num3):
        self.cls = num1
        self.num1 = num2
        self.num2 = num3


    def one_vs_the_rest(self):
        data = []
        label = []
        print('Loading data (positive:{})...'.format(self.cls), end='')
        for num in range(0, 10):
            for root, dirs, files in os.walk('train_img/{}'.format(num)):
                for f in files:
                    '''
                    # リサイズによる次元削減
                    img = cv2.imread('{}/{}'.format(root, f), 0)
                    dst = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
                    dst = dst.reshape(-1)
                    data.append(dst)
                    '''
                    data.append(cv2.imread('{}/{}'.format(root, f), 0).reshape(-1))     # 1次元配列に変換してdataに貯めていく
                    if num == self.cls:
                        label.append(1.0)
                    else:
                        label.append(-1.0)
        print('Finish!')
        
        data = np.array(data)
        label = np.array(label)

        data = data / 255.0

        train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.9)
        
        return train_data, test_data, train_label, test_label


    def one_vs_one(self):
        data = []
        label = []
        print('Loading data (\"pos\"_\"neg\":{}_{})...'.format(self.num1, self.num2), end='')
        for root, dirs, files in os.walk('train_img/{}'.format(self.num1)):
            for f in files:
                '''
                # リサイズによる次元削減
                img = cv2.imread('{}/{}'.format(root, f), 0)
                dst = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
                dst = dst.reshape(-1)
                data.append(dst)
                '''
                data.append(cv2.imread('{}/{}'.format(root, f), 0).reshape(-1))     # 1次元配列に変換してdataに貯めていく
                label.append(1.0)
        for root, dirs, files in os.walk('train_img/{}'.format(self.num2)):
            for f in files:
                '''
                # リサイズによる次元削減
                img = cv2.imread('{}/{}'.format(root, f), 0)
                dst = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
                dst = dst.reshape(-1)
                data.append(dst)
                '''
                data.append(cv2.imread('{}/{}'.format(root, f), 0).reshape(-1))     # 1次元配列に変換してdataに貯めていく
                label.append(-1.0)
        print('Finish!')
        
        data = np.array(data)
        label = np.array(label)

        data = data / 255.0

        train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.8)
        
        return train_data, test_data, train_label, test_label


# SVM
class SVM:
    def __init__(self, data, label):
        self.data = data
        self.label = label
    

    # 線形カーネル
    def kernel(self, x, y):
        return np.dot(x, y)


    # ラグランジュ乗数を二次計画法で求める
    def Lagrange(self, n):
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self.label[i] * self.label[j] * np.dot(self.data[i], self.data[j])
        Q = cvxopt.matrix(K)
        p = cvxopt.matrix(-np.ones(n))      # -1がn個の列ベクトル
        G = cvxopt.matrix(np.diag([-1.0]*n))        # 対角成分が-1の(n × n)行列
        h = cvxopt.matrix(np.zeros(n))      # 0がn個の列ベクトル
        A = cvxopt.matrix(self.label, (1,n))     # N個の教師信号が要素の行ベクトル（1 × n）
        b = cvxopt.matrix(0.0)      # 定数0.0
        solution = cvxopt.solvers.qp(Q, p, G, h, A, b)       # 二次計画法でラグランジュ乗数alphaを求める

        alpha = np.array(solution['x']).flatten()

        return alpha


    # サポートベクトルを抽出
    def support_vector(self, alpha):
        S = []
        for i in range(len(alpha)):
            if alpha[i] >= 0.00001:
                S.append(i)
        
        return S


    # wを計算
    def w_cal(self, S, alpha):
        w = np.zeros(784)
        for n in S:
            w += alpha[n] * self.label[n] * self.data[n]
        
        return w

    
    # bを計算
    def b_cal(self, S, alpha):
        _sum = 0
        for n in S:
            tmp = 0
            for m in S:
                tmp += alpha[m] * self.label[m] * self.kernel(self.data[n], self.data[m])
            _sum += (self.label[n] - tmp)
        b = _sum / len(S)

        return b


    def main(self):
        alpha = self.Lagrange(len(self.data))        # ラグランジュ乗数
        S = self.support_vector(alpha)       # サポートベクトル
        w = self.w_cal(S, alpha)     # w
        b = self.b_cal(S, alpha)        # b

        return w, b, S


# ラベルごとにデータを分割
def data_split(data, label):
    cls1 = []
    cls2 = []
    for i in range(len(data)):
        if label[i] == 1:
            cls1.append(data[i])
        elif label[i] == -1:
            cls2.append(data[i])
    
    return cls1, cls2


def f(x, w, b):
    return np.dot(w, x) + b


# 精度を計算
def accuracy(cls1, cls2, w, b):
    num = len(cls1) + len(cls2)
    c1 = 0
    c2 = 0
    for i in cls1:
        if f(i, w, b) >= 0:
            c1 += 1
        elif f(i, w, b) < 0:
            c2 += 1
    for i in cls2:
        if f(i, w, b) < 0:
            c1 += 1
        elif f(i, w, b) >= 0:
            c2 += 1
    if c1 > c2:
        acc = c1 / num
    elif c1 < c2:
        acc = c2 / num

    return acc


# 保存
class SAVE:
    def __init__(self, output_w, output_b, output_acc_train, output_acc_test, processing_time):
        self.output_w = output_w
        self.output_b = output_b
        self.output_acc_train = output_acc_train
        self.output_acc_test = output_acc_test
        self.processing_time = processing_time

    
    def one_vs_the_rest(self):
        # 学習結果のパラメータを保存
        df = pd.DataFrame(self.output_w, index=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        df['b'] = self.output_b

        dirname = 'classifier'
        if not os.path.exists('{}'.format(dirname)):
            os.mkdir('{}'.format(dirname))

        file_num = 1
        while 1:
            if not os.path.exists('{}/one_versus_the_rest/SVM{}.csv'.format(dirname, file_num)):
                df.to_csv('{}/one_versus_the_rest/SVM{}.csv'.format(dirname, file_num))
                print('Save classifier as \"SVM{}.csv\"'.format(file_num))
                break
            else:
                file_num += 1

        # 各学習の精度と学習時間を保存
        df = pd.DataFrame({'acc_for_train':self.output_acc_train, 'acc_for_test':self.output_acc_test, 'processing_time':self.processing_time}, index=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        
        dirname = 'result'
        if not os.path.exists('{}'.format(dirname)):
            os.mkdir('{}'.format(dirname))

        file_num = 1
        while 1:
            if not os.path.exists('{}/one_versus_the_rest/result{}.csv'.format(dirname, file_num)):
                df.to_csv('{}/one_versus_the_rest/result{}.csv'.format(dirname, file_num))
                print('Save accuracy and processing time as \"result{}.csv\"'.format(file_num))
                break
            else:
                file_num += 1


    def one_vs_one(self):
        # 学習結果のパラメータを保存
        id = []
        for i in range(0, 10):
            for j in range(i+1, 10):
                id.append('{}_{}'.format(i, j))

        df = pd.DataFrame(self.output_w, index=id)
        df['b'] = self.output_b

        dirname = 'classifier'
        if not os.path.exists('{}'.format(dirname)):
            os.mkdir('{}'.format(dirname))

        file_num = 1
        while 1:
            if not os.path.exists('{}/one_versus_one/SVM{}.csv'.format(dirname, file_num)):
                df.to_csv('{}/one_versus_one/SVM{}.csv'.format(dirname, file_num))
                print('Save classifier as \"SVM{}.csv\"'.format(file_num))
                break
            else:
                file_num += 1

        # 各学習の精度と学習時間を保存
        df = pd.DataFrame({'acc_for_train':self.output_acc_train, 'acc_for_test':self.output_acc_test, 'processing_time':self.processing_time}, index=id)
        
        dirname = 'result'
        if not os.path.exists('{}'.format(dirname)):
            os.mkdir('{}'.format(dirname))

        file_num = 1
        while 1:
            if not os.path.exists('{}/one_versus_one/result{}.csv'.format(dirname, file_num)):
                df.to_csv('{}/one_versus_one/result{}.csv'.format(dirname, file_num))
                print('Save accuracy and processing time as \"result{}.csv\"'.format(file_num))
                break
            else:
                file_num += 1


if __name__ == '__main__':
    while 1:
        svm_type = int(input('\"one_versus_the_rest(0)\" or \"one_versus_one(1)\"? : '))
        if svm_type == 0 or svm_type == 1:
            break
        else:
            print('Error. Please, input \"0\" or \"1\"')
    output_w = []
    output_b = []
    output_acc_train = []
    output_acc_test = []
    processing_time = []

    if svm_type == 0:
        for num in range(0, 10):
            print('＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝')

            dataset_make = DATASET(num, None, None)
            train_data, test_data, train_label, test_label = dataset_make.one_vs_the_rest()

            # SVM
            print('Start learning')
            svm = SVM(train_data, train_label)
            start = time.time()
            w, b, S = svm.main()
            elapsed_time = time.time() - start
            processing_time.append(elapsed_time)
            print('Finish!')

            output_w.append(w)
            output_b.append(b)

            # トレーニングデータ、テストデータをラベル別に分割
            cls1_train, cls2_train = data_split(train_data, train_label)
            cls1_test, cls2_test = data_split(test_data, test_label)

            # トレーニングデータ、テストデータのそれぞれに対して精度を計算
            acc_train = accuracy(cls1_train, cls2_train, w, b)
            output_acc_train.append(acc_train)
            acc_test = accuracy(cls1_test, cls2_test, w, b)
            output_acc_test.append(acc_test)
            print('Accuracy for training data (classify \"{}\") : {}'.format(num, acc_train))
            print('Accuracy for test data (classify \"{}\") : {}'.format(num, acc_test))

        print('＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝')

    elif svm_type == 1:
        for num1 in range(0, 10):
            for num2 in range(num1+1, 10):
                print('＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝')

                dataset_make = DATASET(None, num1, num2)
                train_data, test_data, train_label, test_label = dataset_make.one_vs_one()

                # SVM
                print('Start learning')
                svm = SVM(train_data, train_label)
                start = time.time()
                w, b, S = svm.main()
                elapsed_time = time.time() - start
                processing_time.append(elapsed_time)
                print('Finish!')

                output_w.append(w)
                output_b.append(b)

                # トレーニングデータ、テストデータをラベル別に分割
                cls1_train, cls2_train = data_split(train_data, train_label)
                cls1_test, cls2_test = data_split(test_data, test_label)

                # トレーニングデータ、テストデータのそれぞれに対して精度を計算
                acc_train = accuracy(cls1_train, cls2_train, w, b)
                output_acc_train.append(acc_train)
                acc_test = accuracy(cls1_test, cls2_test, w, b)
                output_acc_test.append(acc_test)
                print('Accuracy for training data (classify \"{}_{}\") : {}'.format(num1, num2, acc_train))
                print('Accuracy for test data (classify \"{}_{}\") : {}'.format(num1, num2, acc_test))

        print('＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝')

    # 学習結果のパラメータ、各学習の精度と学習時間を保存
    save = SAVE(output_w, output_b, output_acc_train, output_acc_test, processing_time)
    if svm_type == 0:
        save.one_vs_the_rest()
    elif svm_type == 1:
        save.one_vs_one()