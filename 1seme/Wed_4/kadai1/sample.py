# サンプルプログラム（python用）

import cv2
import math
import numpy as np
import os
import time
os.chdir(os.path.dirname(os.path.abspath(__file__)))


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


def mosaicResizeImage(image, re_w, re_h):
    h, w, c = image.shape

    re_image = np.arange(re_w * re_h * 3).reshape(re_h, re_w, 3)

    # 元の画像と新しい画像の大きさの比を計算
    stepx = int(w / re_w)
    stepy = int(h / re_h)

    i = 0
    while i < re_h:
        j = 0
        while j < re_w:
            b = g = r = 0
            count = 0
            k = stepy * i
            while k <= stepy * (i + 1) - 1:
                l = stepx * j
                while l <= stepx * (j + 1) - 1:
                    if k >= h or l >= w:
                        l += 1
                        continue
                    pixelValue = image[k, l]
                    b += pixelValue[0]
                    g += pixelValue[1]
                    r += pixelValue[2]
                    count += 1
                    l += 1
                k += 1
            re_image[i, j][0] = b / count
            re_image[i, j][1] = g / count
            re_image[i, j][2] = r / count
            j += 1
        i += 1

    return re_image


def mosaic(image1, image2):
    # 入力画像の幅と高さを抽出
    h1, w1, c1 = image1.shape
    h2, w2, c2 = image2.shape

    # 1/5に縮小した画像を作成
    image3 = mosaicResizeImage(image1, int(w1/5), int(h1/5))
    image4 = mosaicResizeImage(image2, int(w2/5), int(h2/5))

    # 縮小画像の幅と高さを抽出
    h3, w3, c3 = image3.shape
    h4, w4, c4 = image4.shape

    # 縮小した画像で、画像が最も重なるときのパラメータを概算
    tempy, tempx, tempt = mosaicGetMin(image3, image4, 2, w3-3, 2, h3-3, -15, 15, 2.0, w3, h3, w4, h4)
    # 元の画像でパラメータを計算
    tempy, tempx, tempt = mosaicGetMin(image1, image2, (tempx-1)*5, (tempx+1)*5-1, (tempy-1)*5, (tempy+1)*5-1, tempt-1.0, tempt+1.0, 1.0, w1, h1, w2, h2)

    return tempx, tempy, tempt


if __name__ == '__main__':
    filename1 = 'Level1/1-001-1.jpg'
    filename2 = 'Level1/1-001-2.jpg'

    image1 = cv2.imread(filename1)
    image2 = cv2.imread(filename2)

    start = time.time()

    dx, dy, dt = mosaic(image1, image2)

    elapsed_time = time.time() - start

    print('dx={}'.format(dx))
    print('dy={}'.format(dy))
    print('dt={}'.format(dt))

    print('処理時間：{}'.format(elapsed_time))