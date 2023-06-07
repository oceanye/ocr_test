import cv2
import numpy as np
from sklearn.cluster import KMeans

# 颜色聚类和图像分割
def process_image(image, n_colors):
    # 将图片数据转化为一维数组
    pixels = image.reshape(-1, 3)

    # 定义和训练k-means模型
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    kmeans.fit(pixels)

    # 用模型的聚类中心替换原来的像素点，得到分割后的图片
    segmented_image = kmeans.cluster_centers_[kmeans.labels_]
    segmented_image = segmented_image.reshape(image.shape).astype(np.uint8)

    return segmented_image

# 寻找四边形并绘制边界
def draw_bounding_box(image):
    # 转换到灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化图像
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 检查每个轮廓是否为四边形，并绘制边界框
    rectangle_num = 1
    for contour in contours:
        # 近似多边形
        epsilon = 0.02*cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 如果是四边形
        if len(approx) == 4:
            # 绘制边界框
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)

            # 在框的中心绘制矩形编号
            center_x = x + w//2
            center_y = y + h//2
            cv2.putText(image, str(rectangle_num), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            rectangle_num += 1

    return image

# 读取图片
img = cv2.imread('test.jpg')

# 图像分割
segmented_img = process_image(img, 5)

# 绘制四边形轮廓和编号
final_img = draw_bounding_box(segmented_img)

# 显示结果
cv2.imshow('Result', final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
