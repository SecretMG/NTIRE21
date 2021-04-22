import cv2 as cv
import numpy as np
import os
import math


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def load_img(dir):                                                                                                      ## 载入图片

    img_name = os.listdir(dir)
    img_num = len(img_name)
    imgs = []
    for i in range(img_num):
        img = cv.imread(dir+'/'+img_name[i], 1)
        img = cv.normalize(img, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
        imgs.append(img)
    return imgs


def cal_saturation(src):                                                                                                ## 计算饱和度

    return np.std(src, axis=2)


def cal_contrast(src):                                                                                                  ## 计算对比度

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ker = np.float32([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    lap = abs(cv.filter2D(gray, -1, kernel=ker))
    return lap


def cal_wellexposedness(src, sigma=0.2):                                                                                ## 计算曝光度

    w = np.exp(-0.5 * (src - 0.5)**2 / sigma**2)
    w = np.prod(w, axis=2)
    return w


def gaussian_pyramid(src):                                                                                              ## 生成高斯金字塔

    rows = np.array(imgs).shape[1]
    cols = np.array(imgs).shape[2]
    pyr_len = math.floor(math.log(min(rows, cols), 2))# 层数
    pyr = []
    pyr.append(src)
    for i in range(1, pyr_len):
        src = cv.pyrDown(src)
        pyr.append(src)
    return pyr


def laplacian_pyramid(src):                                                                                             ## 生成拉普拉斯金字塔

    pyr = gaussian_pyramid(src)
    pyr_len = len(pyr)
    lap_pyr = []
    for i in range(pyr_len-1, 0, -1):
        tem1 = cv.pyrUp(pyr[i])
        tem2 = pyr[i-1] - tem1
        lap_pyr.append(tem2)
    return lap_pyr


def padding(src):                                                                                                       ## 扩充图片

    img_num = np.array(imgs).shape[0]
    rows = np.array(imgs).shape[1]
    cols = np.array(imgs).shape[2]
    for i in range(img_num):
        src[i] = cv.copyMakeBorder(src[i], 0, 2**(math.ceil(math.log(rows, 2)))-rows, 0, 2**(math.ceil(math.log(cols, 2)))-cols, borderType=cv.BORDER_REFLECT)
    return src


def naive_ef(imgs, wc=1, ws=1, we=1):

    img_num = np.array(imgs).shape[0]
    rows = np.array(imgs).shape[1]
    cols = np.array(imgs).shape[2]

    img_contrast = np.empty([img_num, rows, cols], dtype='float32')                                                     ## 初始化权重矩阵
    img_sat = np.empty([img_num, rows, cols], dtype='float32')
    img_well = np.empty([img_num, rows, cols], dtype='float32')
    weight = np.ones([img_num, rows, cols], dtype='float32')

    for k in range(img_num):                                                                                            ## 计算权重
        img_contrast[k] = cal_contrast(imgs[k])
        img_sat[k] = cal_saturation(imgs[k])
        img_well[k] = cal_wellexposedness(imgs[k])
        weight[k] = img_contrast[k]**wc * img_sat[k]**ws * img_well[k]**we + 1e-12

    weight = softmax(weight)                                                                                            # 归一化权重
    # weight_total = np.sum(weight, 0)
    # weight /= weight_total

    # for k in range(img_num):                                                                                          # 保存权重图像
    #     weight[k] *= 255
    #     cv.imwrite("./outputs/naive_softmax_weight"+str(k)+".jpg", weight[k])

    output = np.einsum('kij, kijc -> ijc', weight, imgs)                                                                ## 按权重合成图像

    for i in range(img_num):
        imgs[i] = cv.normalize(imgs[i], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    output = cv.normalize(output, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    return output, weight


def pyramid_ef(imgs, wc=1, ws=1, we=1):

    img_num = np.array(imgs).shape[0]
    rows = np.array(imgs).shape[1]
    cols = np.array(imgs).shape[2]
    _, weight = naive_ef(imgs, wc, ws, we)
    pyr = gaussian_pyramid(np.zeros([rows, cols, 3], dtype='float32'))
    pyr_len = len(pyr)

    for k in range(img_num):                                                                                            # 按权重合成拉普拉斯金字塔
        imgs[k] = cv.normalize(imgs[k], None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
        weight_pyr = gaussian_pyramid(weight[k])
        img_pyr  = laplacian_pyramid(imgs[k])
        for l in range(pyr_len-1):
            w = cv.cvtColor(weight_pyr[l], cv.COLOR_GRAY2BGR)
            pyr[l] += w*img_pyr[pyr_len-2-l]

    # for l in range(pyr_len):                                                                                          # 保存加权的拉普拉斯金字塔
    #     pyr[l] = cv.normalize(pyr[l], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    #     cv.imwrite("./outputs/w_lap/"+str(l)+".jpg", pyr[l])

    output = pyr[pyr_len - 2]                                                                                           # 依据拉普拉斯金字塔重建图像
    for l in range(pyr_len-3, -1, -1):
        output = pyr[l] + cv.pyrUp(output)
    output = cv.normalize(output, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    return output


if __name__ == '__main__':

    dir = "./figs"
    imgs = load_img(dir)
    rows = np.array(imgs).shape[1]
    cols = np.array(imgs).shape[2]
    imgs = padding(imgs)

    naive_output,_ = naive_ef(imgs)
    naive_output_cut = naive_output[0:rows, 0:cols]

    pyramid_output = pyramid_ef(imgs)
    pyramid_output_cut = pyramid_output[0:rows, 0:cols]

    cv.imwrite("./outputs/Naive_output_softmax.jpg", naive_output_cut)
    cv.imwrite("./outputs/Pyramid_output.jpg", pyramid_output_cut)