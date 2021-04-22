import cv2
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
Pic=[]
PicGray=[]
C=[]
S=[]
E=[]
W=[]
for i in range(3):
    Pic.append(cv2.imread('data/venice_canal_exp_{}.jpg'.format(i)))
    PicGray.append(cv2.imread('data/venice_canal_exp_{}.jpg'.format(i),0))
    GrayLap=cv2.Laplacian(PicGray[i],cv2.CV_16S,ksize=3)
    C.append(cv2.convertScaleAbs(GrayLap))
    B=Pic[i][:,:,0]
    R=Pic[i][:,:,1]
    G=Pic[i][:,:,2]
    Mean=(R+G+B)/3
    S.append(np.sqrt(((R-Mean)**2+(G-Mean)**2+(B-Mean)**2)/3))
    HSV=cv2.cvtColor(Pic[i],cv2.COLOR_BGR2HSV)
    E.append(np.exp(-((HSV[:,:,2]/256-0.5)**2)/0.08))
    W.append((C[i]**0.5)*(S[i]**0.5)*E[i])
WSum=W[0]+W[1]+W[2]
R=np.zeros([512,1024,3])
for i in range(3):
    for j in range(3):
        R[:,:,j]=R[:,:,j]+((W[i]/WSum)*Pic[i][:,:,j])
cv2.imwrite('out/NainveEF_Output.jpg',R)

