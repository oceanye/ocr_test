import cv2
import numpy as np
from shapely.geometry import Polygon
import os

points = []
scores = []

def click_event(event, x, y, flags, params):
    global image
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
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

    score = area_difference + center_distance
    return score

image = cv2.imread('test.jpg')
cv2.namedWindow("image")
cv2.namedWindow("parameters")
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

                # 保存轮廓图片，文件名包含关键参数
                temp_image = image.copy()


                for contour in contours:
                    # 使用凸包来补全轮廓
                    hull = cv2.convexHull(contour)

                    cv2.drawContours(temp_image, contour, -1, (0, 255, 0), 2)
                    # 检查凸包是否有四个顶点并且面积大于400
                    if len(hull) == 4 and cv2.contourArea(hull) > 400:
                        score = similarity_score(points, hull)
                        if score < best_score:
                            best_score = score
                            best_contour = hull


                for point in points:
                    cv2.circle(temp_image, point, 5, (0, 255, 0), -1)
                if best_contour is not None:
                    cv2.drawContours(temp_image, [best_contour], -1, (0, 255, 0), 2)
                param_text = f"blur_param: {blur_param}, thresh_param: {thresh_param}"
                cv2.putText(temp_image, param_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imwrite(f"cv_rev/image_{blur_param}_{thresh_param}.jpg", temp_image)

                scores.append(best_score)

            if blur_param < 9:
                blur_param += 2  # 确保 blur_param 总是奇数
            else:
                print('满足相似度要求，终止循环')
                print('最终的参数设置是：', 'blur_param:', blur_param, ', thresh_param:', thresh_param)
                exit()

cv2.destroyAllWindows()
