import cv2
import numpy as np
from numpy.core.fromnumeric import shape
#计算拉普拉斯金字塔和高斯金字塔
def LapPyr(image):
    img =image.copy()
    dowm=cv2.pyrDown(img)
    up=cv2.pyrUp(dowm)
    diff = img-up
    return dowm,diff
#三张图片的权重，单个通道
def CalW(Picture):
    W=[]
    for i in range(3):
        PicGray=cv2.cvtColor(Picture[i],cv2.COLOR_RGB2GRAY)
        GrayLap=cv2.Laplacian(PicGray,cv2.CV_16S,ksize=3)
        Ct=cv2.convertScaleAbs(GrayLap)
        B=Picture[i][:,:,0]
        R=Picture[i][:,:,1]
        G=Picture[i][:,:,2]
        Mean=(R+G+B)/3
        St=np.sqrt(((R-Mean)**2+(G-Mean)**2+(B-Mean)**2)/3)
        HSV=cv2.cvtColor(Picture[i],cv2.COLOR_BGR2HSV)
        Et=np.exp(-((HSV[:,:,2]/255-0.5)**2)/0.08)
        W.append((Ct+2)*(St+2)*(Et+2))
    WW=[]
    WSum=W[0]+W[1]+W[2]
    for i in range(3):
        WW.append(np.exp(W[i]/WSum))

    WSum2=WW[0]+WW[1]+WW[2]
    
    for i in range(3):
        W[i]=WW[i]/WSum2
    return W

np.seterr(divide='ignore',invalid='ignore')
Pic=[]
#L存放原图片的拉普拉斯金字塔，G_Original存放原图片的高斯金字塔，G存放权重的高斯金字塔
#L，G_Original为3*m*n*3，G为3*m*n
L=[]
G_Original=[]
GO0=[]

G=[]
L0=[]
R_L=[]
ENum=5
for i in range(3):
    Pic.append(cv2.imread('data/venice_canal_exp_{}.jpg'.format(i)))
    tp1,tp2=LapPyr(Pic[i])
    L0.append(tp2)
    GO0.append(tp1)
L.append(L0)
G_Original.append(Pic)
G_Original.append(GO0)

G1=[]
print(shape(CalW(Pic)))
for i in range(3):
    Gt,df=LapPyr(CalW(Pic)[i])
    G1.append(Gt)
print(shape(GO0))
G.append(CalW(Pic))
G.append(G1)
#test=[]
#ENum为金字塔高度
for i in range(ENum):
    L_t=[]
    G_t=[]
    GO_t=[]
    W_Now=G[i]
    #test.append(L[i])
    L[i]=np.array(L[i])
    for k in range(3):
        L[i][:,:,:,k]=L[i][:,:,:,k]*np.array(W_Now)
    R_L.append(L[i])
    #计算下一层
    for j in range(3):
        tp1,tp2=LapPyr(G_Original[i+1][j])
        L_t.append(tp2)
        GO_t.append(tp1)
    
    for j in range(3):
        Gt,df=LapPyr(G[i+1][j])
        G_t.append(Gt)

    G_Original.append(GO_t)
    L.append(L_t)
    G.append(G_t)

# test.append(G_Original[ENum])
# for i in range(ENum):
#     up=cv2.pyrUp(test[ENum-i][1])
#     #up=cv2.GaussianBlur(up,(3,3),0)
#     test[ENum-1-i][1]=test[ENum-1-i][1]+up
# cv2.imwrite('out/Pyr.jpg',test[0][1])

R_LF=[]
#计算最终的拉普拉斯金字塔
for i in range(ENum):
    R_LF.append(R_L[i][0]+R_L[i][1]+R_L[i][2])
#计算基图
for i in range(3):
    for k in range(3):
        G_Original[ENum][i][:,:,k] = np.array(G_Original[ENum][i][:,:,k])*np.array(G[ENum][i])  
base=G_Original[ENum][0]+G_Original[ENum][1]+G_Original[ENum][2]

R_LF.append(base)
#还原金字塔
for i in range(ENum):
    up=cv2.pyrUp(R_LF[ENum-i])
    R_LF[ENum-1-i]=R_LF[ENum-1-i]+up
cv2.imwrite('out/PyramidEF_Output.jpg',R_LF[0])

