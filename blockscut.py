import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch

def read_blockscut(image,num):
    # 图像读取和预裁剪
    img_origin = cv2.imread(image, 0)           # 读取灰度图像
    img_cut = img_origin[350:700, 100:800]      # 对九宫格进行大致的裁剪
    imgcolor = cv2.imread(image)                # 读取彩色图像
    imgcolor_cut = imgcolor[350:700,100:800]    # 同样进行大致裁剪

    # 分割出包括九个框的最大框
    # 滤波降噪
    img_filter = cv2.GaussianBlur(img_cut, (3, 3), 0)    # 高斯滤波
    img_filter = cv2.medianBlur(img_filter,3)            # 中值滤波

    # 形态学处理及二值化
    # cv2.threshold()方法参数：1.原图像，2.分割阈值，3.高于阈值时赋予的值，4.方法选择
    retval, img_BW = cv2.threshold(img_filter, 50, 255, cv2.THRESH_BINARY)  # 二值化
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # cv2.morphologyEx()方法参数：1.原图像，2.形态学运算方法，3.结构元素，4.运算次数
    imgbw_open = cv2.morphologyEx(img_BW, cv2.MORPH_OPEN, kernel, iterations=2)

    # 提取九宫格每一个的轮廓
    # cv2.findContours()参数：1.原图像，2.轮廓检索模式，RETR_EXTERNAL只检测最外围轮廓，3.轮廓近似方法，这里选择仅保留轮廓拐点信息
    # 输出变量中，contours表示每个轮廓的点集的集合，hierarchy中包含每个轮廓的索引编号
    contours, hierarchy = cv2.findContours(imgbw_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # sorted()参数：1.待排序变量，2.排序方法，这里选择面积大小，3.是否倒序
    hierarchy = sorted(contours,key=cv2.contourArea,reverse=True)   # 按面积排序，最大的九个就是九个矩形的轮廓
    blockindex = []     # 得到每个轮廓的索引编号
    box = []            # 存储每个矩形框的四个顶点
    for i in range(9):
        blockindex.append(cv2.minAreaRect(hierarchy[i]))
        box.append(np.array(cv2.boxPoints(blockindex[i]), dtype=int))
    # cv2.drawContours(imgcolor_cut, box, -1, (0, 255, 0), 3)
    # cv2.imshow("box", imgcolor_cut)
    # cv2.waitKey()

    # 求出每行/列起始格子的顶点坐标
    for i in range(9):
        axisx = []
        axisy = []
        for j in range(4):
            x = box[i][j][0]
            y = box[i][j][1]
            axisx.append(x)
            axisy.append(y)
        axisx = sorted(axisx, reverse=False)    # 按大小升序排列
        axisy = sorted(axisy, reverse=False)    # 按大小升序排列

        # 可视化及保存文件
        singleblock = img_cut[axisy[1]+5:axisy[2]-5,axisx[1]+5:axisx[2]-5]
        # cv2.imwrite("./singblock/block{}_{}.jpg".format(num,3*i+j), singleblock)
        # cv2.imshow("singleblock", singleblock)
        # cv2.waitKey()


if __name__ == '__main__':
    # for num in range(10):
    read_blockscut('./small/{}.bmp'.format(1),1)




