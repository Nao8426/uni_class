# データ作成用プログラム
import random


def gauss(mu_x, sigma_x, mu_y, sigma_y, class_label, size):
    data = []
    label = []

    for i in range(size):
        elem = []
        elem.append(random.gauss(mu_x, sigma_x))
        elem.append(random.gauss(mu_y, sigma_y))
        data.append(elem)
        label.append(class_label)

    return data, label