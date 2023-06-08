import cv2
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import os

points = []
scores = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

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

    score = area_difference + center_distance
    return score

image = cv2.imread('test.jpg')
cv2.namedWindow("image")
cv2.namedWindow("parameters")
cv2.setMouseCallback("image", click_event)

blur_param = 1
thresh_param = 127

# 检查是否存在名为 cv_rev 的文件夹，如果不存在，就新建一个
if not os.path.exists('cv_rev'):
    os.makedirs('cv_rev')

while True:
    if len(points) == 4:
        blur = cv2.GaussianBlur(image, (blur_param, blur_param), 0)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        _, threshold = cv2.threshold(gray, thresh_param, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_score = float('inf')
        best_contour = None

        for contour in contours:
            # 使用凸包来补全轮廓
            hull = cv2.convexHull(contour)

            # 检查凸包是否有四个顶点并且面积大于400
            if len(hull) == 4: #and cv2.contourArea(hull) > 400:
                score = similarity_score(points, hull)
                if score < best_score:
                    best_score = score
                    best_contour = hull

        # 在新窗口显示参数和相似度评分
        param_text = f"blur_param: {blur_param}, thresh_param: {thresh_param}, score: {best_score}"
        param_image = np.zeros((200, 500))
        cv2.putText(param_image, param_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("parameters", param_image)

        # 在每次迭代中，在图片上画出最相似的色块
        if best_contour is not None:
            cv2.drawContours(image, [best_contour], -1, (0, 255, 0), 3)
            cv2.putText(image, param_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("contour",image)
            # 保存轮廓图片，文件名包含关键参数
            cv2.imwrite(f"cv_rev/image_{blur_param}_{thresh_param}.jpg", image)

        scores.append(best_score)

        if blur_param < 17:
            blur_param += 2  # 确保 blur_param 总是奇数
        elif thresh_param < 255:
            thresh_param += 2
        else:
            print('满足相似度要求，终止循环')
            print('最终的参数设置是：', 'blur_param:', blur_param, ', thresh_param:', thresh_param)
            break

    cv2.imshow("image", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 绘制 score 的曲线图，水平轴为 thresh_param
plt.plot(range(0, 256, 2), range(0, 100, 2))
plt.title('Score over threshold')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.savefig('cv_rev/score_curve.png')
plt.show()



cv2.destroyAllWindows()
