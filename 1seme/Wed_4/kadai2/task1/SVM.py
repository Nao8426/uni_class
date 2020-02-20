# SVM（内部まで実装）
import cvxopt
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
import MakeData
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# データを学習できる形に変換
def data_exchange(data1, data2):
    data = data1 + data2        # データを１つにまとめる
    label = label1 + label2     # ラベルを１つにまとめる
    data = np.array(data)       # データ配列をnumpy型に変換
    label = np.array(label)     # ラベル配列をnumpy型に変換
    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.3)       # トレーニング用とテスト用に分ける

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
        w = np.zeros(2)
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


def f_plot(x1, w, b):
    return -(w[0] / w[1]) * x1 - (b / w[1])


# 精度を計算
def accuracy(cls1, cls2, w, b):
    num = len(cls1) + len(cls2)
    c = 0
    for i in cls1:
        if f(i, w, b) >= 0:
            c += 1
    for i in cls2:
        if f(i, w, b) < 0:
            c += 1

    acc = c / num

    return acc


# 結果を描画
class Draw:
    def __init__(self, data, cls1, cls2, x_min, x_max, y_min, y_max, w, b, S, acc, check):
        self.data = data
        self.cls1 = cls1
        self.cls2 = cls2
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.w = w
        self.b = b
        self.S = S
        self.acc = acc
        self.check = check


    # 結果を描画
    def main(self):
        # 訓練データを描画
        x1, x2 = np.array(self.cls1).transpose()
        plt.plot(x1, x2, 'rx')    
        x1, x2 = np.array(self.cls2).transpose()
        plt.plot(x1, x2, 'bx')

        # サポートベクトルを描画
        if self.check == 'train':
            for n in self.S:
                plt.scatter(self.data[n,0], self.data[n,1], s=80, c='c', marker='o')
        
        # 識別境界を描画
        x1 = np.linspace(self.x_min, self.x_max, 1000)
        x2 = [f_plot(x, self.w, self.b) for x in x1]
        plt.plot(x1, x2, 'g-')

        plt.xlim(self.x_min, self.x_max)
        plt.ylim(self.y_min, self.y_max)
        plt.text(-25, -25, 'Accuracy : {}%'.format(round(self.acc*100, 2)))
        plt.show()


if __name__ == '__main__':
    N1 = 100     # クラス１のデータ数
    N2 = 100     # クラス２のデータ数

    # データを作成
    data1, label1 = MakeData.gauss(5, 2.5, 5, 2.5, 1.0, N1)       # クラス１のデータとラベル
    data2, label2 = MakeData.gauss(-5, 2.5, -5, 2.5, -1.0, N2)        # クラス２のデータとラベル
    train_data, test_data, train_label, test_label = data_exchange(data1, data2)

    # SVM
    svm = SVM(train_data, train_label)
    w, b, S = svm.main()

    # トレーニングデータ、テストデータをラベル別に分割
    cls1_train, cls2_train = data_split(train_data, train_label)
    cls1_test, cls2_test = data_split(test_data, test_label)

    # トレーニングデータ、テストデータのそれぞれに対して精度を計算
    acc_train = accuracy(cls1_train, cls2_train, w, b)
    acc_test = accuracy(cls1_test, cls2_test, w, b)
    print('Accuracy for training data : {}'.format(acc_train))
    print('Accuracy for test data : {}'.format(acc_test))
    
    x_min = -30      # xの最小値（描画範囲）
    x_max = 30      # xの最大値（描画範囲）
    y_min = -30      # yの最小値（描画範囲）
    y_max = 30      # yの最大値（描画範囲）

    # トレーニングデータとテストデータに対する結果を描画
    draw_train = Draw(train_data, cls1_train, cls2_train, x_min, x_max, y_min, y_max, w, b, S, acc_train, check='train')
    draw_test = Draw(train_data, cls1_test, cls2_test, x_min, x_max, y_min, y_max, w, b, S, acc_test, check='test')
    draw_train.main()
    draw_test.main()