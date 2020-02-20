# サンプルプログラム　＋　ガウスノイズ　by 永井
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:55:20 2019
 
@author: nasub
"""
 
import cv2
import math
import time
import numpy as np
 
 
def mosaicGetMin(image3, image4, xmin, xmax, ymin, ymax, tmin, tmax, tstep, width1, height1, width2, height2):
    min = 1.7976931348623157e+308
 
    t = tmin
    while t < tmax:
        s1 = math.sin(t * math.pi / 180.0)
        c1 = math.cos(t * math.pi / 180.0)
        y = ymin
        while y <= ymax:
            x = xmin
            while x <= xmax:
                print('x:{} y:{} t:{}'.format(x, y, t))
                s = 0
                count = 0
                i = 0
                while i < height2:
                    j = 0
                    while j < width2:
                        # 画像を回転したときのピクセルの位置
                        u = int(math.floor((s1 * i + c1 * j) + x + 0.5))
                        v = int(math.floor((c1 * i - s1 * j) + y + 0.5))
                        # 画像が重ならない部分は誤差を計算しない
                        if u < 0 or u >= width1 or v < 0 or v >= height1:
                            j += 1
                            continue
                        pixelValue1 = image3[v, u]
                        pixelValue2 = image4[i, j]
                        # 二乗誤差の計算
                        tmp = int(pixelValue1[0]) - int(pixelValue2[0])
                        s += tmp * tmp
                        tmp = int(pixelValue1[1]) - int(pixelValue2[1])
                        s += tmp * tmp
                        tmp = int(pixelValue1[2]) - int(pixelValue2[2])
                        s += tmp * tmp
                        count += 1
 
                        j += 1
                    i += 1
 
                # 平均二乗誤差の計算
                ave = s / count
                # 平均二乗誤差が最小値より小さい場合はパラメータの更新
                if min > ave:
                    min = ave
                    i_min = y
                    j_min = x
                    t_min = t
 
                x += 1
            y += 1
        t += tstep
     
    return i_min, j_min, t_min
 
 
def mosaic(image1, image2):
    # 1/5に縮小した画像を作成
    image3 = cv2.resize(image1, dsize=None, fx=0.2, fy=0.2)
    image4 = cv2.resize(image2, dsize=None, fx=0.2, fy=0.2)
 
    # 元画像と縮小画像それぞれの幅と高さを抽出
    h1, w1, c1 = image1.shape
    h2, w2, c2 = image2.shape
    h3, w3, c3 = image3.shape
    h4, w4, c4 = image4.shape
 
    # 縮小した画像で、画像が最も重なるときのパラメータを概算
    a, b, c = mosaicGetMin(image3, image4, 2, w3-3, 2, h3-3, 0, 2.0, 2.0, w3, h3, w4, h4)
    # 元の画像でパラメータを計算
    a, b, c = mosaicGetMin(image1, image2, (b-1)*5, (b+1)*5-1, (a-1)*5, (a+1)*5-1, c-1.0, c+1.0, 1.0, w1, h1, w2, h2)
 
    return b, a, c

 
def addGaussianNoize(image):
    # 元画像の幅と高さを抽出
    h, w, c = image.shape
     
    #　元画像にガウシアンノイズ付加
    mean = 0
    var = 0.1
    sigma = 15
    gauss = np.random.normal(mean, sigma, (h, w, c))
    gauss = gauss.reshape(h, w ,c)
    image = image + gauss
     
    return image


if __name__ == '__main__':
    filename1 = './Level1/1-001-1.jpg'
    filename2 = './Level1/1-001-2.jpg'
 
    image1 = cv2.imread(filename1)
    image2 = cv2.imread(filename2)
     
    image1 = addGaussianNoize(image1)
    image2 = addGaussianNoize(image2)
     
    start = time.time()
     
    dx, dy, dt = mosaic(image1, image2)
 
    elapsed_time = time.time() - start
 
    print('dx={}'.format(dx))
    print('dy={}'.format(dy))
    print('dt={}'.format(dt))
 
    print('処理時間：{}'.format(elapsed_time))