from utils import *
import cv2 as cv
import numpy as np
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
    I = imgs

    # print(imgs.shape, imgs.dtype)   # (N, W, H, C), dtype
    imgs = np.float32(imgs) / INF   # 将uint8转换成float64，0-1区间
    W = calc_C(imgs)**w_exps[0] * calc_S(imgs)**w_exps[1] * calc_W(imgs, sigma)**w_exps[2]  # section 3.1 done
    W += 1e-9   # for stability
    W = np.einsum('ij, kij -> kij', (1./W.sum(axis=0)), W)  # normalized，对应原文中的W_hat
    # print(W[:, 0, 0].sum())

    R = np.einsum('kij, kijc -> ijc', W, I)
    return W, R # weight map以及生成的img，注意W是单通道的而且是0-1，R是三通道的而且是0-255


def full_EF(locations, w_exps, sigma, INF, depth):
    def Gaussian_Pyramid(imgs, depth=5):
        res = [[] for _ in range(len(imgs))]    # 生成一个含有图片个数个列表的空列表
        for i, img in enumerate(imgs):
            for d in range(depth):
                if d:
                    img = cv.pyrDown(img)
                # 从原图开始
                res[i].append(img)
                # cv.imwrite(f'outputs/Gaussian_Pyramids/weights/{i, d}.jpg', img*INF)
        return res
    def Laplacian_Pyramid(imgs, depth=5):
        res = [[] for _ in range(len(imgs))]    # 生成一个含有图片个数个列表的空列表
        Gauss_pyr = Gaussian_Pyramid(imgs, depth + 1)   # 计算最后一层高斯金字塔对应的拉普拉斯金字塔时，需要更下一层的高斯金字塔
        for i, Gauss_pyr in enumerate(Gauss_pyr):
            # 第i张图片的高斯金字塔
            for d, Gauss_img in enumerate(Gauss_pyr[ : -1]):
                # 计算第j张高斯图对应的拉普拉斯图
                down_up = cv.pyrUp(Gauss_pyr[d+1])
                res[i].append(Gauss_img - down_up)
                cv.imwrite(f'outputs/Laplacian_Pyramids/imgs/{i, d}.jpg', res[i][d])
        return res




    imgs = readImageSeq(locations)  # 已经是np数组
    W, _ = naive_EF(locations, w_exps, sigma, INF)

    '---根据公式，需要计算I的Laplacian金字塔，以及W的Gaussian金字塔'
    Laplace_I = Laplacian_Pyramid(imgs, depth)
    Gauss_W = Gaussian_Pyramid(W, depth)
    for d in range(depth-1, -1, -1):
        L_final = np.empty(Laplace_I[0][d].shape)
        for i in range(len(imgs)):
            G = Gauss_W[i][d]   # shape=(i, j)
            L = Laplace_I[i][d] # shape=(i, j, c)
            G, L = np.asarray(G), np.asarray(L)
            L_final += np.einsum('ij, ijc -> ijc', G, L)
        # cv.imwrite(f'outputs/Laplacian_Pyramids/final/{d}.jpg', L_final)
        # todo: how to collapse?










