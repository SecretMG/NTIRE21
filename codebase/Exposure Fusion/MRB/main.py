import cv2 as cv
import numpy as np

from utils import *
from algorithms import *

locations = ['figs/venice_canal_exp_0.jpg', 'figs/venice_canal_exp_1.jpg', 'figs/venice_canal_exp_2.jpg']
sigma = 0.2
INF = 255
depth = 5


_, R = naive_EF(locations, [1, 1, 1], sigma, INF)
cv.imwrite('outputs/naive_fused.jpg', R)  # 计算并保存naive版本的EF fusion

pyramid_fused_img = full_EF(locations, [1, 1, 1], sigma, INF, depth)



cv.waitKey()