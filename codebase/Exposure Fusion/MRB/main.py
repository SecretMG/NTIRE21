from algorithms import *
import random

locations = ['figs/venice_canal_exp_0.jpg', 'figs/venice_canal_exp_1.jpg', 'figs/venice_canal_exp_2.jpg']
sigma = 0.2
INF = 255
depth = 6   # Gauss depth, not Laplacian depth
w_exps = [4, 3, 1]


'''naive-EF'''
weights, naive_fused_img = naive_EF(locations, w_exps, sigma, INF)
cv.imwrite('outputs/naive_fused.jpg', naive_fused_img*INF)  # 计算并保存naive版本的EF fusion
for i, weight in enumerate(weights):
    cv.imwrite(f'outputs/naive/weights/{i}.jpg', weight*INF)
for i in range(30):
    row = random.randint(0, weights.shape[1]-1)
    col = random.randint(0, weights.shape[2]-1)
    print(weights[:, row, col])

'''pyramid-EF'''
pyramid_fused_img = pyramid_EF(locations, w_exps, sigma, INF, depth)
cv.imwrite('outputs/pyramid_fused.jpg', pyramid_fused_img*INF)

'''opencv实现的ExposureFusion'''
imgs = readImageSeq(locations)
img = cv.createMergeMertens().process(imgs)
cv.imwrite('outputs/opencv.jpg', img*INF)  # 计算并保存naive版本的EF fusion

cv.waitKey()