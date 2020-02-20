# 識別機を学習するプログラム
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
os.chdir(os.path.dirname(os.path.abspath(__file__)))


if __name__ == '__main__':
    with open('dataset/mnist.pkl', 'rb') as f:
        dataset = pickle.load(f)

    # 輝度値を０～１に変換
    dataset['train_img'] = dataset['train_img'] / 255.0
    dataset['test_img'] = dataset['test_img'] / 255.0

    # SVMのインスタンスを作成
    model = SVC(kernel='rbf', gamma='scale', random_state=111)

    print('Learning start...')
    model.fit(dataset['train_img'], dataset['train_label'])
    print('Finish!')
    
    # トレーニングデータに対する精度
    pred_train = model.predict(dataset['train_img'])
    accuracy_train = accuracy_score(dataset['train_label'], pred_train)
    print('accuracy for training data：{:.4f}'.format(accuracy_train))

    # テストデータに対する精度
    pred_test = model.predict(dataset['test_img'])
    accuracy_test = accuracy_score(dataset['test_label'], pred_test)
    print('accuracy for test data：{:.4f}'.format(accuracy_test))

    while 1:
        save_check = input('save model?(y/n) : ')
        if save_check == 'y' or save_check == 'n':
            break
        else:
            print('Error. Please, input again.')
    
    if save_check == 'y':
        # dirnameで指定した名前のファイル（出力先のファイル）がなければ作る
        dirname = 'classifier'
        if not os.path.exists('{}'.format(dirname)):
            os.mkdir('{}'.format(dirname))
        
        # モデルの保存
        joblib.dump(model, '{}/model.joblib'.format(dirname), compress=True)
        print('\"{}\"にモデルを保存'.format(dirname))