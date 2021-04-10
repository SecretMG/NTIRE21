import numpy as np
from matplotlib import pyplot as plt
import cv2
import os

def Expand(img):
    return np.stack([img]*3, axis=-1)

def ImgShow(img, name='img', wait=True):
    cv2.imshow(name, img)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def PyrShow(imgs, name='img', wait=True):
    for i in range(len(imgs)):
        cv2.imshow('{}_{}'.format(name, i), imgs[i])
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def PyrBuild(img, depth=4):
    GauPyr = [None]*depth
    LapPyr = [None]*(depth-1)
    GauPyr[0] = img
    for i in range(depth-1):
        GauPyr[i+1] = cv2.pyrDown(GauPyr[i])
        LapPyr[i] = GauPyr[i] - cv2.pyrUp(GauPyr[i+1])
    return GauPyr, LapPyr

def PyrRecovery(LapPyr, base_img):
    depth = len(LapPyr)
    RecPyr = [None]*(depth+1)
    RecPyr[depth] = base_img
    for i in range(depth):
        RecPyr[depth-i-1] = cv2.pyrUp(RecPyr[depth-i]) + LapPyr[depth-i-1]
    return RecPyr

class ExposureFusion():
    def __init__(self, root_dir='./figs'):
        self.root_dir = root_dir
        self.read_data()
        self.length = len(self.data)
    
    def __len__(self):
        return self.length
    
    def norm(self, img, eps=1e-6):
        img = (img - img.min()) / (img.max()-img.min()) + eps
        return img

    def read_data(self):
        self.data = []
        for name in os.listdir(self.root_dir):
            img = cv2.imread(os.path.join(self.root_dir, name)).astype(np.float32) / 255.
            self.data.append(img)
    
    def cal_C(self, img, norm=False):
        img_gray = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)
        img_lap = cv2.Laplacian(img_gray, cv2.CV_32F, ksize=3)
        img_C = np.abs(img_lap)
        # img_C[img_C==0] = 1
        if norm: 
            img_C = self.norm(img_C)
        # PyrShow([img_gray, img_lap, img_C], name='calC', wait=False)
        return img_C

    def cal_S(self, img, norm=False):
        img_S = np.std(img, axis=-1)
        # img_S[img_S==0] = 1
        if norm: 
            img_S = self.norm(img_S)
        # ImgShow(img_S, name='calS', wait=False)
        return img_S

    def cal_E(self, img, sigma=0.2, norm=False):
        img_well = np.exp(- (img - 0.5) ** 2 / (2 * sigma * sigma))
        img_E =  np.prod(img_well, axis=-1)
        if norm: 
            img_E = self.norm(img_E)
        # ImgShow(img_E, name='calE', wait=False)
        return img_E
    
    def cal_WeightMap(self, img):
        # WeightMap = self.cal_C(img) * self.cal_S(img)
        WeightMap = self.cal_C(img) * self.cal_S(img) * self.cal_E(img)
        return WeightMap
    
    def normalize(self, imgs):
        img_sum = np.ones_like(imgs[0]) * 0
        for i in range(len(imgs)): 
            img_sum += np.exp(imgs[i])
        for i in range(len(imgs)):
            imgs[i] = np.exp(imgs[i]) / img_sum
        return imgs
    
    def LapMerge(self, LapPyrs, GauPyrs):
        LapPyr_new = []
        for i in range(self.depth-1):
            Lap_new = np.zeros_like(LapPyrs[0][i])
            for k in range(self.length):
                Lap_new += LapPyrs[k][i] * Expand(GauPyrs[k][i])
            LapPyr_new.append(Lap_new)
        return LapPyr_new

    def Fusion(self, imgs, WeightMaps):
        result = np.zeros_like(imgs[0])
        for i in range(self.length):
            result += Expand(WeightMaps[i]) * imgs[i]
        return result

    def NaiveFusion(self):
        WeightMaps = [None] * self.length
        for i in range(self.length):
            WeightMaps[i] = self.cal_WeightMap(self.data[i])
        WeightMaps = self.normalize(WeightMaps)
        result = self.Fusion(self.data, WeightMaps)
        # PyrShow(WeightMaps, name='WeightMap', wait=False)
        # ImgShow(result, name='Result', wait=True)
        return result

    def MultiScaleFusion(self, depth=6):
        self.depth = depth
        WeightMaps = [None] * self.length
        GauPyrs = [None] * self.length
        LapPyrs = [None] * self.length
        ImgPyrs = [None] * self.length

        for k in range(self.length):
            WeightMaps[k] = self.cal_WeightMap(self.data[k])
        WeightMaps = self.normalize(WeightMaps)
        # PyrShow(WeightMaps, name='WeightMap', wait=True)

        for k in range(self.length):
            ImgPyrs[k], LapPyrs[k] = PyrBuild(self.data[k], depth=depth)
            GauPyrs[k], _ = PyrBuild(WeightMaps[k], depth=depth)
            # PyrShow(GauPyrs[k], name='GauPyr', wait=False)
            # PyrShow(LapPyrs[k], name='LapPyr', wait=False)
            # PyrShow(ImgPyrs[k], name='ImgPyr', wait=True)
        LapPyrs_final = self.LapMerge(LapPyrs, GauPyrs)
        GauBase = np.stack([ImgPyrs[k][-1] for k in range(self.length)], axis=0)
        WeightBase = np.stack([GauPyrs[k][-1] for k in range(self.length)], axis=0)
        ImgBase = self.Fusion(GauBase, WeightBase)
        RecPyr = PyrRecovery(LapPyrs_final, ImgBase)
        # PyrShow(RecPyr, name='Result', wait=True)
        return RecPyr[0]
    
if __name__=='__main__':
    ef = ExposureFusion('./figs')
    naive_result = ef.NaiveFusion()
    MS_result = ef.MultiScaleFusion(depth=4)
    mergeMertens = cv2.createMergeMertens()
    opencv_result = mergeMertens.process([np.uint8(img*255) for img in ef.data])
    ImgShow(naive_result, name='Naive Result', wait=False)
    ImgShow(opencv_result, name='OpenCV Result', wait=False)
    ImgShow(MS_result, name='Multi Scale Result', wait=True)
    
    # 验证金字塔是否有错
    # GauPyr, LapPyr = PyrBuild(ef.data[1], depth=2)
    # RecPyr = PyrRecovery(LapPyr, GauPyr[-1])
    # PyrShow(GauPyr, name='GauPyr', wait=False)
    # PyrShow(LapPyr, name='LapPyr', wait=False)
    # PyrShow(RecPyr, name='RecPyr', wait=True)
    # diff = RecPyr[0] - ef.data[1]
    # PyrShow([diff], name='diff', wait=True)