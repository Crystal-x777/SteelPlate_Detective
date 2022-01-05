# encoding:utf-8
''''''
'''
1.同时打开5张图
2.如果第1-3张做高斯模糊
3.小区域平移计算找出9个像素块均值大于周边9个像素块某阈值的区域，标记为白色，其他区域标记为黑色
4.白色区域计算连通域
5.计算白色区域面积
6.计算白色区域x轴，y轴中心点
如果是前1-3张：
    为表面斑痕缺陷
如果是后4-5张为结构性缺陷：
    如果面积小于某阈值为棍印
        否则
            如果位置在边部为边列开
            如果连通域大于某长条值为划痕
            
孔洞规则、棍印、斑迹、油斑、压痕、划伤、翘皮、孔洞、边裂
'''



import numpy as np
import cv2
import imutils
from collections import Counter
import pandas as pd
from skimage import measure,draw,data


#中文乱码
def zh_ch(string):
    return string.encode("gbk").decode(errors="ignore")



#二值化
def Image_Binarization(url):
    img = cv2.imread("org\\{}".format(url), 0)
    #取众值，做蒙版
    mask_img = img.flatten()
    counts = np.bincount(mask_img)
    num = np.argmax(counts)
    x = img.shape[0]
    y = img.shape[1]
    zeros_img = np.zeros(shape=(x, y))
    mask_img = num + zeros_img
    #进行像素级差值计算
    qx_array=abs(img-mask_img)
    qx_array[qx_array>50]=255
    qx_array[qx_array<=50]=0

    return qx_array,zeros_img#返回的是二值化原图

#连通域
def Connected_Component(qx_array,zeros_img):
    list_white=[]
    list_bbox=[]
    labels =measure.label(qx_array,connectivity=2)
    properties = measure.regionprops(labels)
    for prop in properties:
        if prop.area >8:#16
            print('区域编号：',prop.label)#区域标记
            print('面积mm² ：',prop.area*0.03)#面积mm
            print('宽wPx：',prop.bbox[2]-prop.bbox[0])
            print('高hPx：',prop.bbox[3]-prop.bbox[1])
            print('中心点坐标(x,y)：',prop.centroid)#x,y坐标
            print('周长Px：',prop.perimeter)#周长
            print('像素总数：',prop.area)#区域内像素点总数
            print('边框坐标：',prop.bbox)#边框起始位
            print('-'*30)
            list_bbox.append(prop.bbox)
            list_white.append(prop.coords)
    for white1 in list_white:
        for white2 in white1:
            zeros_img[white2[0]][white2[1]]=255
    return zeros_img,list_bbox

#检测框&展示图
def show_img(url,zeros_img,list_bbox):
    #原图
    o_img = cv2.imread("org\\{}".format(url), 0)
    cv2.imshow(zh_ch('原图'), imutils.resize(o_img, 600))
    #结果图
    x,y=zeros_img.shape[0],zeros_img.shape[1]
    zeros_img.resize(x,y,1)
    cv2.imwrite("result\\{}".format(url), zeros_img)
    r_img = cv2.imread("result\\{}".format(url), 0)
    r_img = cv2.cvtColor(r_img, cv2.COLOR_GRAY2RGB)
    for bbox in list_bbox:
        Y = np.array([bbox[0]-20, bbox[0]-20, bbox[2]+20, bbox[2]+20])
        X = np.array([bbox[1]-20, bbox[3]+20, bbox[3]+20, bbox[1]-20])
        rr, cc = draw.polygon_perimeter(Y, X)
        draw.set_color(r_img, [rr, cc], [0, 0, 255])
    cv2.imwrite("result\\{}".format(url), r_img)
    r_img = cv2.imread("result\\{}".format(url), 1)
    cv2.imshow(zh_ch('>0.5mm缺陷图'), imutils.resize(r_img, 600))
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
    # cv2.waitKey(4500)
    # cv2.destroyAllWindows()



import time

if __name__=="__main__":
    for i in range(1,21):#url = '20.png'
        url = '{}.jpg'.format(i)
        # print(url)
        img,zeros_img=Image_Binarization(url)#生成二值图
        zeros_img,list_bbox=Connected_Component(img,zeros_img)
        show_img(url,zeros_img,list_bbox)




