import cv2 as cv
import numpy as np

from utils import *
from algorithms import *

locations = ['figs/venice_canal_exp_0.jpg', 'figs/venice_canal_exp_1.jpg', 'figs/venice_canal_exp_2.jpg']
sigma = 0.2
INF = 255
depth = 5


_, naive_fused_img = naive_EF(locations, [1, 1, 1], sigma, INF)
cv.imwrite('outputs/naive_fused.jpg', naive_fused_img*INF)  # 计算并保存naive版本的EF fusion

pyramid_fused_img = pyramid_EF(locations, [1, 1, 1], sigma, INF, depth)
cv.imwrite('outputs/pyramid_fused.jpg', pyramid_fused_img*INF)



cv.waitKey()