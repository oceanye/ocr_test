import cv2
import numpy as np
from shapely.geometry import Polygon
import os
import matplotlib.pyplot as plt

points = []
scores = []

def click_event(event, x, y, flags, params):
    global image
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        if len(points) == 1:
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

def get_center(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    center_x = int(sum(x_coords) / len(points))
    center_y = int(sum(y_coords) / len(points))
    return (center_x, center_y)

def similarity_score(polygon_points, contour):
    polygon = Polygon(polygon_points)
    polygon_area = polygon.area
    polygon_center = get_center(polygon_points)

    contour_area = cv2.contourArea(contour)
    moments = cv2.moments(contour)
    contour_center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))

    area_difference = abs(polygon_area - contour_area) / max(polygon_area, contour_area)
    center_distance = np.sqrt((polygon_center[0] - contour_center[0]) ** 2 + (polygon_center[1] - contour_center[1]) ** 2)

    score = 0.5 * area_difference + 0.5 * center_distance
    return score

def evaluate_similarity(points, contours):
    best_score = float('inf')
    best_contour = None

    for contour in contours:
        # 使用凸包来补全轮廓
        hull = cv2.convexHull(contour)

        # 多边形逼近
        approx = cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True)

        # 检查逼近后的轮廓是否有四个顶点并且面积大于400
        if len(approx) == 4 and cv2.contourArea(approx) > 400:
            score = similarity_score(points, approx)
            if score < best_score:
                best_score = score
                best_contour = approx

    return best_score, best_contour

image = cv2.imread('test.jpg')
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_event)

while True:
    cv2.imshow("image", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if len(points) == 4:
        blur_param = 1
        thresh_params = range(0, 256, 2)

        # 检查是否存在名为 cv_rev 的文件夹，如果不存在，就新建一个
        if not os.path.exists('cv_rev'):
            os.makedirs('cv_rev')

        while True:
            blur = cv2.GaussianBlur(image, (blur_param, blur_param), 0)
            gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

            best_score = float('inf')
            best_contour = None

            for thresh_param in thresh_params:
                _, threshold = cv2.threshold(gray, thresh_param, 255, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                score, contour = evaluate_similarity(points, contours)
                if score < best_score:
                    best_score = score
                    best_contour = contour

                # 保存轮廓图片
                temp_image = gray.copy()
                temp_image = cv2.cvtColor(temp_image, cv2.COLOR_GRAY2BGR)
                for point in points:
                    cv2.circle(temp_image, point, 5, (0, 255, 0), -1)
                if best_contour is not None:
                    cv2.drawContours(temp_image, [best_contour], -1, (0, 255, 0), 2)
                param_text = f"blur_param: {blur_param}, thresh_param: {thresh_param}, score: {best_score:.2f}"
                cv2.putText(temp_image, param_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imwrite(f"cv_rev/image_{blur_param}_{thresh_param}_score_{best_score:.2f}.jpg", temp_image)

                scores.append(best_score)

            if blur_param < 9:
                blur_param += 2  # 确保 blur_param 总是奇数
            else:
                print('满足相似度要求，终止循环')
                print('最终的参数设置是：', 'blur_param:', blur_param, ', thresh_param:', thresh_param)
                break

# 绘制相似度分数曲线图
plt.plot(list(thresh_params), scores)
plt.xlabel('Threshold')
plt.ylabel('Similarity Score')
plt.title('Similarity Score vs. Threshold')
plt.show()

cv2.destroyAllWindows()
