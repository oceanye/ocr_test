import cv2
import numpy as np

def color_and_brightness_based_segmentation(image, lower_color, upper_color, brightness_threshold):
    # 将图像转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 根据颜色范围创建颜色掩码
    color_mask = cv2.inRange(hsv, lower_color, upper_color)

    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 根据亮度阈值创建亮度掩码
    _, brightness_mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)

    # 组合颜色掩码和亮度掩码
    combined_mask = cv2.bitwise_and(color_mask, brightness_mask)

    # 对掩码进行形态学操作以去除噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # 找到轮廓并筛选四边形区域
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quadrilateral_contours = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(approx) == 4:  # 四边形判断条件可以根据实际情况调整
            quadrilateral_contours.append(approx)

    return combined_mask, quadrilateral_contours

# 加载图像
image = cv2.imread('test.jpg')

# 颜色范围（示例：橙色、灰色和绿色）
lower_color_orange = np.array([5, 50, 50], dtype=np.uint8)
upper_color_orange = np.array([15, 255, 255], dtype=np.uint8)

lower_color_gray = np.array([200, 200, 200], dtype=np.uint8)
upper_color_gray = np.array([255, 255, 254], dtype=np.uint8)

lower_color_green = np.array([35, 50, 50], dtype=np.uint8)
upper_color_green = np.array([85, 255, 255], dtype=np.uint8)

# 亮度阈值
brightness_threshold = 10

# 根据颜色和亮度进行区域划分并保留四边形区域
segmented_mask_orange, quadrilateral_contours_orange = color_and_brightness_based_segmentation(image, lower_color_orange, upper_color_orange, brightness_threshold)
segmented_mask_gray, quadrilateral_contours_gray = color_and_brightness_based_segmentation(image, lower_color_gray, upper_color_gray, brightness_threshold)
segmented_mask_green, quadrilateral_contours_green = color_and_brightness_based_segmentation(image, lower_color_green, upper_color_green, brightness_threshold)

# 绘制四边形区域和输出尺寸
for contour in quadrilateral_contours_orange:
    cv2.drawContours(image, [contour], 0, (255,0,0), 2)
    # 计算四边形区域的尺寸
    rect = cv2.minAreaRect(contour)
    (x, y), (w, h), angle = rect
    print("橙色四边形尺寸：")
    print("宽度:", w)
    print("高度:", h)
    print("角度:", angle)

for contour in quadrilateral_contours_gray:
    cv2.drawContours(image, [contour], 0, (50, 50, 50), 2)
    # 计算四边形区域的尺寸
    rect = cv2.minAreaRect(contour)
    (x, y), (w, h), angle = rect
    print("灰色四边形尺寸：")
    print("宽度:", w)
    print("高度:", h)
    print("角度:", angle)

for contour in quadrilateral_contours_green:
    cv2.drawContours(image, [contour], 0, (0, 0, 255), 2)
    # 计算四边形区域的尺寸
    rect = cv2.minAreaRect(contour)
    (x, y), (w, h), angle = rect
    print("绿色四边形尺寸：")
    print("宽度:", w)
    print("高度:", h)
    print("角度:", angle)

    # 显示结果图像
cv2.imshow("Segmented Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
