from utils import *
import cv2 as cv
import numpy as np
from math import *
import pdb




def naive_EF(locations, w_exps, sigma, INF):
    def calc_C(imgs):
        # 越边缘处，权重越大
        weights = np.empty(imgs.shape[:-1])  # 取消channel维，因为只对灰度图做计算
        for i, img in enumerate(imgs):
            img = cv.cvtColor(img, code=cv.COLOR_BGR2GRAY)  # 默认空间为BGR，将它转换为灰度图
            img = cv.Laplacian(img, ddepth=cv.CV_32F)  # 边缘检测, float32
            img = np.absolute(img)  # 对所有元素取绝对值
            weights[i] = img
        return weights
    def calc_S(imgs):
        'todo：探究全是'
        # std越大，权重越大
        return np.std(imgs, axis=3)  # 取消第三维（通道维），以通道方向为一组，计算每个点的三个通道的值的std
    def calc_W(imgs, sigma=0.2):
        # 值越接近0.5，权重越大
        imgs = np.exp(- (imgs - 0.5) ** 2 / (2 * sigma * sigma))
        return np.prod(imgs, axis=3)  # 在通道方向为一组，计算3个值的乘积

    imgs = readImageSeq(locations)  # 已经是np数组
    imgs = np.float32(imgs) / INF   # 将uint8转换成float32，0-1区间
    # print(imgs.shape, imgs.dtype)   # (N, W, H, C), dtype
    C = calc_C(imgs)**w_exps[0]
    S = calc_S(imgs)**w_exps[1]
    W = calc_W(imgs)**w_exps[2]
    W0 = C * S * W  # section 3.1 done
    W0 += 1e-9   # for stability
    # print(W.shape)  # (3, 512, 1024)

    '''divsum'''
    W1 = np.einsum('ij, kij -> kij', (1 / W0.sum(axis=0)), W0)  # normalized，对应原文中的W_hat
    '''softmax'''
    W2 = np.exp(W0)
    W2 = np.einsum('ij, kij -> kij', (1 / W2.sum(axis=0)), W2)  # 不可行，权重太接近

    R = np.einsum('kij, kijc -> ijc', W1, imgs)
    return W1, R # weight map以及生成的img，注意W是单通道的，btw.[0, 1]，R是三通道的，btw.[0, 1]


def pyramid_EF(locations, w_exps, sigma, INF, depth):
    def PyrBuild(img, depth=5):
        # D * [Hd * Wd * C]
        Gauss = [None] * depth
        Lapla = [None] * (depth - 1)
        Gauss[0] = img
        for i in range(depth - 1):
            Gauss[i+1] = cv.pyrDown(Gauss[i])
            Lapla[i] = Gauss[i] - cv.pyrUp(Gauss[i+1])
        return Gauss, Lapla
    def PyrRecovery(img, Lapla):
        depth = len(Lapla)
        Recov = [None] * (depth + 1)
        Recov[depth] = img
        for d in range(depth-1, -1, -1):
            Recov[d] = Lapla[d] + cv.pyrUp(Recov[d+1])
        return Recov


    imgs = readImageSeq(locations)  # 已经是np数组
    imgs = np.float32(imgs) / INF   # 将uint8转换成float32，0-1区间
    W, _ = naive_EF(locations, w_exps, sigma, INF)

    '--- 计算合成图片的Laplace金字塔'
    # 根据公式，需要计算W的Gaussian金字塔，以及I的Laplacian金字塔
    Gauss_W = [None] * imgs.shape[0]
    Gauss_I = [None] * imgs.shape[0]
    Lapla_I = [None] * imgs.shape[0]
    for k in range(imgs.shape[0]):
        Gauss_W[k], _ = PyrBuild(W[k], depth)
        Gauss_I[k], Lapla_I[k] = PyrBuild(imgs[k], depth)   # Gauss_I需要保存，在collapse时使用
    Lapla_final = []
    for d in range(depth - 1):
        L = np.zeros(Lapla_I[0][d].shape)
        for k in range(imgs.shape[0]):
            L += np.expand_dims(Gauss_W[k][d], axis=-1).repeat(imgs.shape[-1], axis=-1) * Lapla_I[k][d]
        Lapla_final.append(L)

    '--- 计算合成图片的底片'
    base = np.zeros(Gauss_I[0][-1].shape)
    for k in range(imgs.shape[0]):
        base += np.expand_dims(Gauss_W[k][-1], axis=-1).repeat(imgs.shape[-1], axis=-1) * Gauss_I[k][-1]

    '--- collapse'
    img_pyr = PyrRecovery(base, Lapla_final)
    return img_pyr[0]