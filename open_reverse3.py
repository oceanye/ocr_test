import cv2
import numpy as np
from shapely.geometry import Polygon
import os
import matplotlib.pyplot as plt

points = []
selected_color = None
mode = "point"  # This is to manage the different modes of operation

def mouse_event(event, x, y, flags, params):
    global image, points, mode
    # Select points in both modes
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    # Select color only in "color" mode
    if event == cv2.EVENT_RBUTTONDOWN and mode == "color":
        color = image[y, x]
        hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]
        lower_bound = np.array([hsv_color[0] - 10, 100, 100])
        upper_bound = np.array([hsv_color[0] + 10, 255, 255])
        selected_color = (lower_bound, upper_bound)

def get_center(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    center_x, center_y = int(sum(y_coords) / len(points))
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
        hull = cv2.convexHull(contour)
        approx = cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True)
        if len(approx) == 4 and cv2.contourArea(approx) > 400:
            score = similarity_score(points, approx)
            if score < best_score:
                best_score = score
                best_contour = approx

    return best_score, best_contour

image = cv2.imread('test.jpg')
cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_event)

while True:
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('p'):
        mode = "point"

    elif key == ord('c'):
        mode = "color"

    if len(points) == 4:
        blur_params = [1, 3, 5, 7, 9]
        thresh_params = range(0, 256, 2)

        # check if 'cv_rev' directory exists, if not, create one
        if not os.path.exists('cv_rev'):
            os.makedirs('cv_rev')

        scores = []
        for blur_param in blur_params:
            all_scores = []
            for thresh_param in thresh_params:
                blur = cv2.GaussianBlur(image, (blur_param, blur_param), 0)
                gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

                _, threshold = cv2.threshold(gray, thresh_param, 255, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                score, _ = evaluate_similarity(points, contours)
                all_scores.append(score)

            scores.append(all_scores)

            # save the contour image
            temp_image = gray.copy()
            temp_image = cv2.cvtColor(temp_image, cv2.COLOR_GRAY2BGR)
            for point in points:
                cv2.circle(temp_image, point, 5, (0, 255, 0), -1)
            param_text = f"blur_param: {blur_param}"
            cv2.putText(temp_image, param_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imwrite(f"cv_rev/image_{blur_param}.jpg", temp_image)

        # Plot similarity scores
        plt.figure()
        for i in range(len(blur_params)):
            plt.plot(list(thresh_params), scores[i], label=f"blur_param: {blur_params[i]}")
        plt.xlabel('Threshold')
        plt.ylabel('Similarity Score')
        plt.title('Similarity Score vs. Threshold')
        plt.legend()
        plt.savefig("cv_rev/similarity_scores.jpg")
        plt.show()

        break

cv2.destroyAllWindows()

