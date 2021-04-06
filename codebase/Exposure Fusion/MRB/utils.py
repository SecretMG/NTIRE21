import cv2 as cv
import numpy as np

def readImageSeq(locations):
    assert len(locations) > 1  # 至少要读到多于1张图片

    imgs = []
    for loc in locations:
        img = cv.imread(loc, 1)
        # img = cv.resize(img, (1024, 512))
        # cv.imwrite(loc, img)
        imgs.append(img)
    imgs = np.asarray(imgs)  # 从列表转换成numpy数组

    return imgs
