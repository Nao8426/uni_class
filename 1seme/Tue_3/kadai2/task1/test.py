# 学習済みの識別機を用いて識別するプログラム
import os
import pickle
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    with open('dataset/mnist.pkl', 'rb') as f:
        dataset = pickle.load(f)
    
    # 輝度値を０～１に変換
    dataset['test_img'] = dataset['test_img'] / 255.0
    
    model = joblib.load('classifier/model.joblib')

    # テストデータに対する精度
    pred_test = model.predict(dataset['test_img'])
    accuracy_test = accuracy_score(dataset['test_label'], pred_test)
    print('accuracy for test data： %.2f' % accuracy_test)