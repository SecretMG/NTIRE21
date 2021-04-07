import cv2
import numpy as np
def psnr(A, B):
    return 10*np.log(255*255.0/(((A.astype(np.float)-B)**2).mean()))/np.log(10)
 
def double2uint8(I, ratio=1.0):
    #I*ratio 的矩阵中各个值将被限制在0到255之间
    return np.clip(np.round(I*ratio), 0, 255).astype(np.uint8)
 
def make_kernel(f):
    kernel = np.zeros((int(2*f+1), int(2*f+1)))
    for d in range(1, f+1):
        kernel[f-d:f+d+1, f-d:f+d+1] += (1.0/((2*d+1)**2))
    return kernel/kernel.sum()
 
def NLmeansfilter(I, h_=10, templateWindowSize=5,  searchWindowSize=11):
    f = templateWindowSize//2
    t = searchWindowSize//2
    height, width = I.shape[:2]
    print("height:{},width:{}".format(height,width))
    padLength = t+f
    #对称填充，每个轴填充t+f
    #0:7,520:527为填充的空白像素
    I2 = np.pad(I, ((padLength,padLength),(padLength,padLength),(0,0)), 'symmetric')
    #print(I2.shape[:3])
    kernel = make_kernel(f)
    #print(kernel)
    #滤波系数h
    h = (h_**2)
    #5:521
    I_ = I2[padLength-f:padLength+f+height, padLength-f:padLength+f+width]
    #print(I_.shape[:3])
    #print(I_) 
    average = np.zeros(I.shape)
    nx = np.zeros(I.shape)
    #-5:6
    #窗口滑动，采用整个图片滑动，
    for i in range(-t, t+1):
        for j in range(-t, t+1):
            if i==0 and j==0:
                continue
            #5+i:521+i 516*516,已经做好填充的图片，卷积核5*5
            I2_ = I2[padLength+i-f:padLength+i+f+height, padLength+j-f:padLength+j+f+width]
            w=np.exp((I2_-I_)**2/h)[f:f+height, f:f+width]
            #边缘各有四行四列未处理，516*516转为512*512，高斯加权欧氏距离
            w = np.exp(-cv2.filter2D ((I2_ - I_)**2, -1, kernel)/h)[f:f+height, f:f+width]
            #归一化因子
            nx += w
            #w(x,y)*v(y)
            average += (w*I2_[f:f+height, f:f+width])
    return average/nx
 
if __name__ == '__main__':
    I = cv2.imread(r'E:\2020-2021-2\CV\NL_Means\lena.png')
    #cv2.imshow(r'E:\2020-2021-2\CV\NL_Means\lena.png',I)
    #cv2.imwrite(r'E:\2020-2021-2\CV\NL_Means\gray.png',I)
    #原图每个像素点增加一个sigma乘与原图相同大小的随机矩阵，产生噪声图
    #*I.shape返回一个list
    sigma = 20.0
    I1 = double2uint8(I + np.random.randn(*I.shape) *sigma)
    print (u'噪声图像PSNR{}'.format(psnr(I, I1)))
    cv2.imwrite(r'E:\2020-2021-2\CV\NL_Means\Noise.png',I1)
    R1  = cv2.medianBlur(I1, 5)
    print (u'中值滤波PSNR',psnr(I, R1))
    R2 = cv2.fastNlMeansDenoising(I1, None, sigma, 5, 11)
    print (u'opencv的NLM算法',psnr(I, R2))
    cv2.imwrite(r'E:\2020-2021-2\CV\NL_Means\NLM1.png',R2)
    R3 = double2uint8(NLmeansfilter(I1.astype(np.float), sigma, 5, 11))
    print (u'NLM PSNR',psnr(I, R3))
    cv2.imwrite(r'E:\2020-2021-2\CV\NL_Means\NLM2.png',R3)
