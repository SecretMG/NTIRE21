import numpy as np
import cv2 as cv
import skimage
import matplotlib.pyplot as plt
import math
import time
from tqdm import tqdm

def cal_psnr(src, noised):

    # 三通道
    if len(src.shape) == 3:
        for k in range(0, 2):
            sse = 0.0
            for i in range(src.shape[0]):
                for j in range(src.shape[1]):
                    sse += ((int)(src[i][j][k]) - (int)(noised[i][j][k]))**2
        sse /= 3
        mse = sse / (src.shape[0] * src.shape[1])
        psnr = 10 * math.log10(256 * 256 / mse)

    # 单通道
    else:
        sse = 0.0
        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                sse += ((int)(src[i][j]) - (int)(noised[i][j]))**2
        mse = sse / (src.shape[0] * src.shape[1])
        psnr = 10 * math.log10(256 * 256 / mse)
    return psnr
    

def distance(img1, img2):
    dis = np.mean((img1 - img2) ** 2)
    return dis


def NL_MeansBlur(noised, templete, search, h, sigma):
    rows = noised.shape[0]
    cols = noised.shape[1]
    templete_center = templete >> 1
    search_center = search >> 1
    boarder = templete_center + search_center
    dest = np.ones(noised.shape)


    noised = cv.copyMakeBorder(noised, boarder, boarder, boarder, boarder, cv.BORDER_REFLECT)

    # 三通道
    if len(noised.shape) == 3:
       coeff = -(1.0/ 3.0) * (1.0 / (h * h))
       weight = np.zeros(256 * 256 * 3 * templete * templete)
       for i in range(len(weight)):
           weight[i] = math.exp(max(i - 2.0 * sigma * sigma, 0.0) * coeff)
           if weight[i] < 0.001:
               weight[i] = 0
               break
       for i in range(boarder, boarder + rows):
           for j in range(boarder, boarder + cols):
               seg1 = noised[i - templete_center : i + templete_center,
                      j - templete_center : j + templete_center]
               p = 0
               sum = 0
               for m in range(-search_center, search_center):
                   for n in range(-search_center, search_center):
                       seg2 = noised[i + m - templete_center : i + m +templete_center,
                              j + n - templete_center : j + n + templete_center]
                       dis = distance(seg1, seg2)# Todo需要引入三通道的距离计算
                       w = weight[dis]
                       p += noised[i + m][j + n] * w
                       sum += w
               dest[i - boarder, j - boarder] = p / sum
       return dest
    # 单通道
    else:
        weight = np.ones(256 * 256 * templete * templete)
        for i in range(len(weight)):
            weight[i] = math.exp(- max(i - 2.0 * sigma * sigma, 0.0) / (h * h))
            if weight[i] < 0.001:
                weight[i] = 0.0
                break
        for i in tqdm(range(boarder, boarder + rows)):
            for j in range(boarder, boarder + cols):
                seg1 = noised[i - templete_center: i + templete_center,
                       j - templete_center: j + templete_center]
                p = 0
                sum = 0
                for m in range(-search_center, search_center):
                    for n in range(-search_center, search_center):
                        seg2 = noised[i + m - templete_center: i + m + templete_center,
                               j + n - templete_center: j + n + templete_center]
                        dis = distance(seg1, seg2)
                        w = weight[(int)(dis)]
                        p += noised[i + m][j + n] * w
                        sum += w
                re = p / sum
                if re < 0:
                    re = 0
                elif re > 255:
                    re = 255
                dest[i - boarder, j - boarder] = re
        return dest

if __name__ == '__main__':
    src = cv.imread("lenna.jpg", 0)#[:,:,::-1]
    noised = skimage.util.random_noise(src, mode='gaussian', mean=0, var=0.05)  # 添加高斯噪声
    # noised = skimage.util.random_noise(noised, mode='s&p')                      # 添加椒盐噪声
    noised = cv.normalize(noised, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)       # 标准化
    print(src.data)
    psnr_noised = cal_psnr(src, noised)
    print("Noised PSNR is:", cal_psnr(src, noised))

    start = time.perf_counter()                                                 # 高斯滤波并计时
    gaussion = cv.GaussianBlur(noised, (7, 7), -1)
    end = time.perf_counter()
    psnr_gaussion = cal_psnr(src, gaussion)
    print("\n*********************Gaussion Blur*********************")
    print("\ntime:", 1000 * (end - start), "ms")
    print("\nBlured PSNR is:", cal_psnr(src, gaussion))

    start = time.perf_counter()                                                 # 中值滤波并计时
    median = cv.medianBlur(noised, 3)
    end = time.perf_counter()
    psnr_median = cal_psnr(src, median)
    print("\n**********************Median Blur**********************")
    print("\ntime:", 1000 * (end - start), "ms")
    print("\nBlured PSNR is:", cal_psnr(src, median))

    start = time.perf_counter()                                                 # 双边滤波并计时
    bilater = cv.bilateralFilter(noised, 15, 100, 50)
    end = time.perf_counter()
    psnr_bilater = cal_psnr(src, bilater)
    print("\n*********************Bilateral Blur********************")
    print("\ntime:", 1000 * (end - start), "ms")
    print("\nBlured PSNR is:", cal_psnr(src, bilater))

    start = time.perf_counter()                                                 # OpenCV非局部均值滤波并计时
    nlmeans = cv.fastNlMeansDenoising(noised, None, 40, 7, 21)
    end = time.perf_counter()
    psnr_nlm = cal_psnr(src, nlmeans)
    print("\n**********************NL-Means Blur********************")
    print("\ntime:", 1000 * (end - start), "ms")
    print("\nBlured PSNR is:", psnr_nlm)

    # start = time.perf_counter()                                                 # 自己编写的非局部均值滤波并计时
    # my_nlmeans = NL_MeansBlur(noised, 3, 9, 10, 10)
    # end = time.perf_counter()
    # psnr_mynlm = cal_psnr(src, my_nlmeans)
    # print("\n********************My NL-Means Blur*******************")
    # print("\ntime:", 1000 * (end - start), "ms")
    # print("\nBlured PSNR is:", cal_psnr(src, my_nlmeans))

    plt.subplot(2, 3, 1), plt.imshow(src)        , plt.title("src")             # 图片显示
    plt.subplot(2, 3, 2), plt.imshow(noised)     , plt.title("noised-{:.2f}dB".format(psnr_noised))
    plt.subplot(2, 3, 3), plt.imshow(gaussion)   , plt.title("Gaussion Blur-{:.2f}dB".format(psnr_gaussion))
    plt.subplot(2, 3, 4), plt.imshow(median)     , plt.title("Median Blur-{:.2f}dB".format(psnr_median))
    plt.subplot(2, 3, 5), plt.imshow(bilater)    , plt.title("Bilater Blur-{:.2f}dB".format(psnr_bilater))
    plt.subplot(2, 3, 6), plt.imshow(nlmeans)    , plt.title("CV NL-Means Blur-{:.2f}dB".format(psnr_nlm))
    # plt.subplot(2, 3, 1), plt.imshow(my_nlmeans) , plt.title("My NL-Means Blur-{:.2f}dB".format(psnr_mynlm))
    plt.subplots_adjust(hspace=0.5)
    plt.show()