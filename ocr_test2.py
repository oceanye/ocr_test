# 导入所需模块
import cv2
from matplotlib import pyplot as plt
import os
import numpy as np


# plt显示彩色图片
def plt_show0(img):
    # cv2与plt的图像通道不同：cv2为[b,g,r];plt为[r, g, b]
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.show()


# plt显示灰度图片
def plt_show(img):
    plt.imshow(img, cmap='gray')
    plt.show()


# 图像去噪灰度处理
def gray_guss(image):
    image = cv2.GaussianBlur(image, (3, 3), 0)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_image


# 读取待检测图片
origin_image = cv2.imread('./test.jpg')
# 复制一张图片，在复制图上进行图像操作，保留原图
image = origin_image.copy()
# 图像去噪灰度处理
gray_image = gray_guss(image)
# x方向上的边缘检测（增强边缘信息）
Sobel_x = cv2.Sobel(gray_image, cv2.CV_16S, 1, 0)
absX = cv2.convertScaleAbs(Sobel_x)
image = absX

# 图像阈值化操作——获得二值化图
ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
# 显示灰度图像
plt_show(image)
# 形态学（从图像中提取对表达和描绘区域形状有意义的图像分量）——闭操作
kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX,iterations = 1)
# 显示灰度图像
plt_show(image)
