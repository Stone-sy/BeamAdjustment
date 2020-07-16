# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:38:31 2020

数字摄影测量实习
光束法平差
    1 后方交会
    2 前方交会
    3 光束法平差
    4 精度评定
    5 核线

@author: shi'ying
"""
import numpy as np
import math as m 
import pandas as pd
import cv2

path='./Data'
result_path='./Result'

#读取像点坐标
def read_xy(file):
    file=path+'/'+file
    point=[[] for i in range(3)]
    with open(file,'r')as f:
        next(f)
        lines=f.readlines()
        for line in lines:
            a,b,c=[i for i in line.split()]
            point[0].append(a)      #点号
            point[1].append(b)      #x坐标
            point[2].append(c)      #y坐标
        f.close()
    return point

#读取物方点坐标
def read_XY(file):
    file=path+'/'+file
    point=[[] for i in range(4)]
    with open(file,'r')as f:
        next(f)
        lines=f.readlines()
        for line in lines:
            a,b,c,d=[i for i in line.split()]
            point[0].append(a)      #点号
            point[1].append(b)      #x坐标
            point[2].append(c)      #y坐标
            point[3].append(d)      #z坐标
        f.close()
    return point

#将像素坐标转换为像平面坐标，单位从像素变成mm
def xyplane(p):
    x=p[1]
    y=p[2]
    #平移到像主点为原点的坐标系（此时还是像素坐标）
    x=[float(i)-5344.0/2 for i in x]
    y=[4008.0/2-float(i) for i in y]
    K1 = -5.00793e-009
    K2 = 1.83462e-016
    P1 = -2.24419e-008
    P2 = 1.76820e-008
    x0=47.48571
    y0=12.02756 
    #畸变纠正(像素坐标)
    for i in range(0,len(x)):
        r2=(x[i]-x0)**2+(y[i]-y0)**2
        delx=(x[i]-x0)*(K1*r2+K2*r2**2)+P1*(r2+2*(x[i]-x0)**2)+2*P2*(x[i]-x0)*(y[i]-y0)
        dely=(y[i]-y0)*(K1*r2+K2*r2**2)+P2*(r2+2*(y[i]-y0)**2)+2*P1*(x[i]-x0)*(y[i]-y0)
        x[i]=x[i]-delx
        y[i]=y[i]-dely
        #转成mm单位
        x[i]=x[i]*25.4/300
        y[i]=y[i]*25.4/300
    p[1]=x
    p[2]=y
    return p
    
#计算旋转矩阵R
def calR(angle):        #angle=[fi,w,k] 外方位元素
    fi=angle[0]
    w=angle[1]
    k=angle[2]
    a1=m.cos(fi)*m.cos(k)-m.sin(fi)*m.sin(w)*m.sin(k)
    a2=-m.cos(fi)*m.sin(k)-m.sin(fi)*m.sin(w)*m.cos(k)
    a3=-m.sin(fi)*m.cos(w)
    b1=m.cos(w)*m.sin(k)
    b2=m.cos(w)*m.cos(k)
    b3=-m.sin(w)
    c1=m.sin(fi)*m.cos(k)+m.cos(fi)*m.sin(w)*m.sin(k)
    c2=-m.sin(fi)*m.sin(k)+m.cos(fi)*m.sin(w)*m.cos(k)
    c3=m.cos(fi)*m.cos(w)
    #R=[[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]]
    return a1,a2,a3,b1,b2,b3,c1,c2,c3

#后方交会
def BundleAdjustment(point1,pointXY):
    point1=xyplane(point1) #像素坐标转像平面坐标
    kzd=[2201,2209,4201,4211,4601,4509,2601,2609]
    kzdnum=[]
    
    for i in kzd:
        for k in range(0,len(point1[0])):
            if int(point1[0][k])==i:
                kzdnum.append(k)
    X=[float(pointXY[1][i]) for i in kzdnum]
    Y=[float(pointXY[2][i]) for i in kzdnum]
    Z=[float(pointXY[3][i]) for i in kzdnum]#提取出对应的物方点坐标
    x=[float(point1[1][i]) for i in kzdnum]
    y=[float(point1[2][i]) for i in kzdnum]#提取对应的像点坐标
    
    f=4547.93519*25.4/300
    x0=47.48571*25.4/300
    y0=12.02756*25.4/300
   
    #初值
    Xs=np.mean(np.array(X))
    Ys=np.mean(np.array(Y))
    Zs=-f
    fi=0
    w=0
    k=0
    
    angle=[fi,w,k]
    a1,a2,a3,b1,b2,b3,c1,c2,c3=calR(angle)
    count=0
    while True:
        angle=[fi,w,k]
        a1,a2,a3,b1,b2,b3,c1,c2,c3=calR(angle)
        A=[]
        #B=[]
        L=[]
        
        count+=1
        
        for i in range(0,len(kzdnum)):
            X0=a1*(X[i]-Xs)+b1*(Y[i]-Ys)+c1*(Z[i]-Zs)
            Y0=a2*(X[i]-Xs)+b2*(Y[i]-Ys)+c2*(Z[i]-Zs)
            Z0=a3*(X[i]-Xs)+b3*(Y[i]-Ys)+c3*(Z[i]-Zs)
            
            #计算像点坐标近似值
            x1=x0-f*X0/Z0
            y1=y0-f*Y0/Z0
            
            #计算误差矩阵
            a11=(a1*f+a3*(x[i]-x0))/Z0
            a12=(b1*f+b3*(x[i]-x0))/Z0
            a13=(c1*f+c3*(x[i]-x0))/Z0
            a14=(y[i]-y0)*m.sin(w)-((x[i]-x0)/f*((x[i]-x0)*m.cos(k)-(y[i]-y0)*m.sin(k))+f*m.cos(k))*m.cos(w)
            a15=-f*m.sin(k)-(x[i]-x0)/f*((x[i]-x0)*m.sin(k)+(y[i]-y0)*m.cos(k))
            a16=y[i]-y0
            a21=(a2*f+a3*(y[i]-y0))/Z0
            a22=(b2*f+b3*(y[i]-y0))/Z0
            a23=(c2*f+c3*(y[i]-y0))/Z0
            a24=-(x[i]-x0)*m.sin(w)-((y[i]-y0)/f*((x[i]-x0)*m.cos(k)-(y[i]-y0)*m.sin(k))-f*m.sin(k))*m.cos(w)
            a25=-f*m.cos(k)-(y[i]-y0)/f*((x[i]-x0)*m.sin(k)+(y[i]-y0)*m.cos(k))
            a26=-(x[i]-x0)
            
            A.append([a11,a12,a13,a14,a15,a16])
            #B.append([-a11,-a12,-a13])
            A.append([a21,a22,a23,a24,a25,a26])
            #B.append([-a21,-a22,-a23])
            
            l1=x[i]-x1
            l2=y[i]-y1
            L.append([l1])
            L.append([l2])
            
        #将A\B矩阵和L矩阵转换成numpy数组进行运算
        A=np.array(A)
        #B=np.array(B)
        L=np.array(L)
        
        #解法方程
        #t=(A.T@A-A.T@B@np.linalg.inv(B.T@B)@(B.T@A))@(A.T@L-A.T@B@np.linalg.inv(B.T@B)@(B.T@L))
        t=np.matmul(np.matmul(np.linalg.inv(A.T@A),A.T),L)
        Xs+=float(t[0])
        Ys+=float(t[1])
        Zs+=float(t[2])
        fi+=float(t[3])
        w+=float(t[4])
        k+=float(t[5])
        if abs(t[3])<0.00001 and abs(t[4])<0.00001 and abs(t[5])<0.00001 or count>20:
            break
    
    return [Xs,Ys,Zs,fi,w,k],count

#前方交会
def FrontAdjustment(xy,allXs):
    f=4547.93519*25.4/300
    x0=47.48571*25.4/300
    y0=12.02756*25.4/300
    
    wf=[]
    for i in range(0,len(xy[0][0])):
        B=[]
        L=[]
        for n in range(0,len(allXs)):
            a1,a2,a3,b1,b2,b3,c1,c2,c3=calR(allXs[n][3:])
            x1=xy[n][0][i]-x0
            y1=xy[n][1][i]-y0
            l1=f*a1+x1*a3
            l2=f*b1+x1*b3
            l3=f*c1+x1*c3
            lx=l1*allXs[n][0]+l2*allXs[n][1]+l3*allXs[n][2]
            l4=f*a2+y1*a3
            l5=f*b2+y1*b3
            l6=f*c2+y1*c3
            ly=l4*allXs[n][0]+l5*allXs[n][1]+l6*allXs[n][2]
            B.append([l1,l2,l3])
            B.append([l4,l5,l6])
            L.append([lx])
            L.append([ly])
        B=np.array(B)
        L=np.array(L)
        wf.append([i[0] for i in (np.linalg.inv(B.T@B)@B.T@L).tolist()])
    WuFang=[]
    kzd=[2201,2209,4201,4211,4601,4509,2601,2609]
    for i in range(0,len(point1[0])):
        if (int(point1[0][i]) in kzd):
            WuFang.append([pointXY[0][i],float(pointXY[1][i]),float(pointXY[2][i]),float(pointXY[3][i])])
        else:
            WuFang.append([point1[0][i],wf[i][0],wf[i][1],wf[i][2]])
    return WuFang

#光束法平差
def BeamAdjustment(WuFang, nxy, allXs):
    count=0
    while True:
        A=np.zeros((2*6*132,6*6))
        B=np.zeros((2*6*132,124*3))
        L=np.zeros((2*6*132,1))
        for n in range(0,len(allXs)):
            Xs,Ys,Zs=allXs[n][0:3]
            fi,w,k=allXs[n][3:]
            a1,a2,a3,b1,b2,b3,c1,c2,c3=calR([fi,w,k])
            A1=[]
            B2=np.zeros((2*132,124*3))
            L1=[]
            k=0
            
            for i in range(0,len(xy[0][0])):
                B1=[]
                #提取第一张相片的第一个点
                x=nxy[n][1][i]
                y=nxy[n][2][i]
                X=WuFang[i][1]
                Y=WuFang[i][2]
                Z=WuFang[i][3]
                X0=a1*(X-Xs)+b1*(Y-Ys)+c1*(Z-Zs)
                Y0=a2*(X-Xs)+b2*(Y-Ys)+c2*(Z-Zs)
                Z0=a3*(X-Xs)+b3*(Y-Ys)+c3*(Z-Zs)
                
                #计算像点坐标近似值
                x1=x0-f*X0/Z0
                y1=y0-f*Y0/Z0
                
                #计算误差矩阵
                a11=(a1*f+a3*(x-x0))/Z0
                a12=(b1*f+b3*(x-x0))/Z0
                a13=(c1*f+c3*(x-x0))/Z0
                a14=(y-y0)*m.sin(w)-((x-x0)/f*((x-x0)*m.cos(k)-(y-y0)*m.sin(k))+f*m.cos(k))*m.cos(w)
                a15=-f*m.sin(k)-(x-x0)/f*((x-x0)*m.sin(k)+(y-y0)*m.cos(k))
                a16=y-y0
                a21=(a2*f+a3*(y-y0))/Z0
                a22=(b2*f+b3*(y-y0))/Z0
                a23=(c2*f+c3*(y-y0))/Z0
                a24=-(x-x0)*m.sin(w)-((y-y0)/f*((x-x0)*m.cos(k)-(y-y0)*m.sin(k))-f*m.sin(k))*m.cos(w)
                a25=-f*m.cos(k)-(y-y0)/f*((x-x0)*m.sin(k)+(y-y0)*m.cos(k))
                a26=-(x-x0)
                
                A1.append([a11,a12,a13,a14,a15,a16])
                A1.append([a21,a22,a23,a24,a25,a26])
                B1.append([-a11,-a12,-a13])
                B1.append([-a21,-a22,-a23])
                
                l1=x-x1
                l2=y-y1
                L1.append([l1])
                L1.append([l2])
                #循环一张相片后，得到一张相片132个点的264个方程的系数，都是
                if (int(nxy[0][0][i]) not in kzd):
                    B2[2*i:2*i+2,3*k:3*k+3]=np.array(B1).reshape(2,3)
                    k+=1
            A[2*132*n:2*132*(n+1),6*n:6*(n+1)]=A1
            B[2*132*n:2*132*(n+1),0:124*3]=B2
            L[2*132*n:2*132*(n+1),0:1]=L1
        
        N11=A.T@A
        N12=A.T@B
        N22=B.T@B
        u1=A.T@L
        u2=B.T@L
        a=N11-N12@np.linalg.inv(N22)@N12.T
        b=u1-N12@np.linalg.inv(N22)@u2
        t_=np.linalg.inv(a)@b
        c=N22-N12.T@np.linalg.inv(N11)@N12
        d=u2-N12.T@np.linalg.inv(N11)@u1
        X_=np.linalg.inv(c)@d
        
        #处理解
        for i in range(0,6):
            allXs[i][0]+=t_[6*i]
            allXs[i][1]+=t_[6*i+1]
            allXs[i][2]+=t_[6*i+2]
            allXs[i][3]+=t_[6*i+3]
            allXs[i][4]+=t_[6*i+4]
            allXs[i][5]+=t_[6*i+5]
            
        j=0
        for i in range(0,len(xy[0][0])):
            if (int(nxy[0][0][i]) not in kzd):
                WuFang[i][1]+=float(X_[j*3])
                WuFang[i][2]+=float(X_[j*3+1])
                WuFang[i][3]+=float(X_[j*3+2])
                j+=1
        
        sum1=0.0
        for i in range(0,int(len(t_)/6)):
            sum1+=sum(t_[6*i+3:6*i+6])       #线元素之和
        #print('第{}次迭代'.format(count))
        count+=1
        
        if sum1<0.00001:
            break
    
    print("光束法平差 迭代了 {}次，其结果已写入Result文件夹 ：".format(count))
    
    #坐标差
    ZuoBiaoCha(WuFang, pointXY,allXs)
    
    #理论精度
    accuracy_eval(np.hstack((A,B)),np.vstack((t_,X_)),L)

    #实际精度     
    autual_eval(WuFang,pointXY)
    
    return WuFang,allXs

#坐标差
def ZuoBiaoCha(WuFang,pointXY,allXs):
    p=np.array(pointXY,dtype=np.float).T
    WuFang=np.array(WuFang,dtype=np.float)
    temp=np.zeros((WuFang.shape[0],WuFang.shape[1]+4),dtype=np.float)
    temp[:,4:7]=WuFang[:,1:]-p[:,1:]
    temp[:,0:4]=WuFang[:,:]
    temp[:,7]=np.sqrt(temp[:,4]**2+temp[:,5]**2+temp[:,6]**2)
    data=pd.DataFrame(temp)
    data.to_csv(result_path+'/光束法平差结果.csv',header=['Point','X(mm)','Y(mm)','Z(mm)','dX(mm)','dY(mm)','dZ(mm)','dP(mm)'],index=False)
    data4=pd.DataFrame(np.array(np.array(allXs).reshape(6,6),dtype=np.float))
    data4.to_csv(result_path+'/光束法平差结果.csv',mode='a',header=['Xs(mm)','Ys(mm)','Zs(mm)','Phi(rad)','Omega(rad)','Kappa(rad)'])
    print("【光束法平差结果】：加密点坐标（含计算坐标与已知坐标的差）和外方位元素")

#理论精度
def accuracy_eval(B,X,L):
    V=B@X-L
    Qii=np.linalg.inv(B.T@B)
    #中误差
    m0=m.sqrt(V.T@V/(np.shape(B)[0]-np.shape(X)[0]))
    theory_m=m0*np.array([m.sqrt(i) for i in Qii.diagonal()])
    temp1=[]
    for i in range(0,6):
        temp1.append(theory_m[6*i:6*i+6])
    temp2=[]
    for i in range(0,124):
        temp2.append(theory_m[36+3*i:36+3*i+3])
    for i in range(0,len(point1[0])):
        if(int(point1[0][i]) in kzd):
            temp2=np.insert(temp2,i,np.zeros((1,3)),axis=0)
    delta=np.zeros((len(WuFang),4))
    delta[:,0:3]=np.array(temp2)
    delta[:,3]=[m.sqrt(i[0]**2+i[1]**2+i[2]**2) for i in delta[:,0:3]]
    m0=pd.DataFrame(np.array([m0]))
    m0.to_csv(result_path+'/光束法平差理论精度.csv',header=['Mean square error'],index=False)
    data1=np.hstack((WuFang,delta))
    data1=pd.DataFrame(data1)
    data1.to_csv(result_path+'/光束法平差理论精度.csv',mode='a',header=['Point','X(mm)','Y(mm)','Z(mm)','dX(mm)','dY(mm)','dZ(mm)','dP(mm)'],index=False)
    data2=pd.DataFrame(temp1)
    data2.to_csv(result_path+'/光束法平差理论精度.csv',mode='a',header=['dXs(mm)','dYs(mm)','dZs(mm)','dPhi(rad)','dOmega(rad)','dKappa(rad)'])    
    print("【光束法平差理论精度】：中误差、所有点和外方位元素的理论精度")

#实际精度
def autual_eval(WuFang,pointXY):
    x1=np.array(WuFang).astype(float)[:,1:]
    x2=np.array(pointXY).astype(float).T[:,1:]
    x3=(x1-x2)**2
    ux=sum(x3[:,0])/np.shape(x3)[0]
    uy=sum(x3[:,1])/np.shape(x3)[0]
    uz=sum(x3[:,2])/np.shape(x3)[0]
    data3=pd.DataFrame(np.array([ux,uy,uz]).reshape(1,3))
    data3.to_csv(result_path+'/光束法平差实际精度.csv',header=['ux(mm)','uy(mm)','uz(mm)'],index=False)
    print("【光束法平差实际精度】：所有点的实际精度")
    
def get_AB(R,x,y,f):
    v=R[1,0]*x+R[1,1]*y-R[1,2]*f
    w=R[2,0]*x+R[2,1]*y-R[2,2]*f
    A=v*R[2,0]-w*R[1,0]
    B=w*R[1,1]-v*R[2,1]
    C=w*R[1,2]-v*R[2,2]
    A_=A/B
    B_=C/B
    x0=47.48571
    y0=12.02756 
    f=4547.93519
    #从以x0,y0为原点的坐标系，移到以相片中心为原点的坐标系
    B_=B_*f+y0-A_*x0
    return A_,B_

#获得直线方程
def img_line(A,B,x,style=0):
    if style==0:
        return A*x+B   #以x为自变量
    else:
        return (x-B)/A #以y为自变量
    
#画线（不包括点和标注）
def draw_line(A,B,linefunction,img):
    height=img.shape[0]
    width=img.shape[1]
    point=[]
    #画第一个点
    x1=width/2
    y1=img_line(A,B,x1,0)
    if abs(y1)<=height/2:
        point.append([x1,y1])
    
    x2=-(width/2)
    y2=img_line(A,B,x2,0)
    if abs(y2)<=height/2:
        point.append([x2,y2])
    #画第二个点
    y3=height/2
    x3=img_line(A,B,y3,1)
    if abs(x3)<=width/2:
        point.append([x3,y3])
    
    y4=-(height/2)
    x4=img_line(A,B,y4,1)
    if abs(x4)<=width/2:
        point.append([x4,y4])
    
    point=np.array(point,dtype=np.float)
    point[:,1]=-point[:,1]
    point=point+np.array([width/2,height/2])
    #从像主点坐标系转换到以左上角为原点的坐标系
    px1=int(round(point[0,0]))
    py1=int(round(point[0,1]))
    px2=int(round(point[1,0]))
    py2=int(round(point[1,1]))
    
    new_img=cv2.line(img,(px1,py1),(px2,py2),(0,0,255),4,4)
    return new_img
 

#核线绘制（含点和标注）     
def Nuclear_line(img_file,points,Xs):
    img=cv2.imread(img_file)
    #读取002.jpg
    #读取以像主点为原点的坐标002和004
    for i in range(0,len(points[0])):
        if int(points[0][i])==4507:
            #获得左相片p点 以像主点为原点的坐标（mm）
            xp=points[1][i]
            yp=points[2][i]
    
    #旋转矩阵        
    a1,a2,a3,b1,b2,b3,c1,c2,c3=calR(Xs[3:])
    R1=np.array([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]])
    #获得A、B矩阵
    L_A,L_B=get_AB(R1, xp, yp, f)
    #画经过4507点的核线
    new_img=draw_line(L_A, L_B, img_line, img)
    xp_pixel=int(round(xp*300/25.4+5344.0/2))
    yp_pixel=int(round(4008.0/2-yp*300/25.4))
    #标注4507号点
    new_img=cv2.circle(new_img,(xp_pixel,yp_pixel),16,(0,255,0),-1)
    new_img=cv2.putText(new_img, '4507', (xp_pixel+20,yp_pixel+20),cv2.FONT_HERSHEY_COMPLEX, 5, (255, 255, 255), 2)
    cv2.waitKey(30)
    cv2.imwrite(result_path+"/{}_result.JPG".format(img_file[-7:-4]), new_img)
    print("已绘制相片{}的核线".format(img_file[-7:-4]))
    
    
# 控制点：2201、2209、4201、4211、4601、4509、2601、2609
if __name__=='__main__':
    #已知条件
    f=4547.93519*25.4/300
    x0=47.48571*25.4/300
    y0=12.02756*25.4/300
    kzd=[2201,2209,4201,4211,4601,4509,2601,2609]
    
    #读取像点坐标和已知点物方坐标
    point1=read_xy('001.txt')
    point2=read_xy('002.txt')
    point3=read_xy('003.txt')
    point4=read_xy('004.txt')
    point5=read_xy('005.txt')
    point6=read_xy('006.txt')
    pointXY=read_XY('物点坐标.TXT')
    
    #后方交会
    Xs1,count=BundleAdjustment(point1, pointXY)
    print("相片001 后方交会 迭代了 {}次".format(count))
    Xs2,count=BundleAdjustment(point2, pointXY)
    print("相片002 后方交会 迭代了 {}次".format(count))
    Xs3,count=BundleAdjustment(point3, pointXY)
    print("相片003 后方交会 迭代了 {}次".format(count))
    Xs4,count=BundleAdjustment(point4, pointXY)
    print("相片004 后方交会 迭代了 {}次".format(count))
    Xs5,count=BundleAdjustment(point5, pointXY)
    print("相片005 后方交会 迭代了 {}次".format(count))
    Xs6,count=BundleAdjustment(point6, pointXY)
    print("相片006 后方交会 迭代了 {}次".format(count))
    
    nxy=[]
    #这里point1-6已经进行了像平面坐标的转化
    nxy.append(point1[:])
    nxy.append(point2[:])
    nxy.append(point3[:])
    nxy.append(point4[:])
    nxy.append(point5[:])
    nxy.append(point6[:])
    # nxy 表示点号、x、y  
    # xy 表示坐标x、y
    xy=[i[1:] for i in nxy]
    allXs=[]
    allXs=[Xs1,Xs2,Xs3,Xs4,Xs5,Xs6]
    
    #前方交会
    WuFang=FrontAdjustment(xy,allXs)
    
    #光束法平差
    WuFang,allXs=BeamAdjustment(WuFang, nxy, allXs)
    
    #绘制002.JPG的核线
    img_file=path+'/002.JPG'
    points=point2
    Xs=Xs2
    Nuclear_line(img_file, points, Xs)
    
    #绘制004.JPG的核线
    img_file=path+'/004.JPG'
    points=point4
    Xs=Xs4
    Nuclear_line(img_file, points, Xs)
    
    
    
    

