import gzip
import numpy as np
import os
import pickle
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def load_img(file_name):
    with gzip.open('{}'.format(file_name), 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 784)

    return data


def load_label(file_name):
    with gzip.open('{}'.format(file_name), 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    
    return labels


if __name__ == '__main__':
    dataset = {}
    dataset['train_img'] = load_img('mnist_origin/train-images-idx3-ubyte.gz')
    dataset['train_label'] = load_label('mnist_origin/train-labels-idx1-ubyte.gz')
    dataset['test_img'] = load_img('mnist_origin/t10k-images-idx3-ubyte.gz')
    dataset['test_label'] = load_label('mnist_origin/t10k-labels-idx1-ubyte.gz')

    # dirnameで指定した名前のファイル（出力先のファイル）がなければ作る
    dirname = 'dataset'
    if not os.path.exists('{}'.format(dirname)):
        os.mkdir('{}'.format(dirname))
    
    with open('{}/mnist.pkl'.format(dirname), 'wb') as d:
        pickle.dump(dataset, d, -1)