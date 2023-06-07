import cv2
import numpy as np
from sklearn.cluster import KMeans

def process_image(image, n_colors):
    # 将图片数据转化为一维数组
    pixels = image.reshape(-1, 3)

    # 定义和训练k-means模型
    kmeans = KMeans(n_clusters=n_colors,n_init=10)
    kmeans.fit(pixels)

    # 用模型的聚类中心替换原来的像素点，得到分割后的图片
    segmented_image = kmeans.cluster_centers_[kmeans.labels_]
    segmented_image = segmented_image.reshape(image.shape).astype(np.uint8)

    return segmented_image

def draw_contours(image):
    # 转换到灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测边缘
    edges = cv2.Canny(gray, 50, 150)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 画出轮廓
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

    return image

# 读取图片
img = cv2.imread('test.jpg')

# 图像分割
segmented_img = process_image(img, 5)

# 画出轮廓
final_img = draw_contours(segmented_img)

# 显示结果
cv2.imshow('Result', final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
