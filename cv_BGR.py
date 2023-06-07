import cv2
import numpy as np

selected_pixel = None
brightness_threshold = 100

def on_trackbar_change(value):
    lower = [cv2.getTrackbarPos(f"Lower {channel}", "Filter") for channel in ("B", "G", "R")]
    upper = [cv2.getTrackbarPos(f"Upper {channel}", "Filter") for channel in ("B", "G", "R")]
    lower_color = np.array(lower)
    upper_color = np.array(upper)
    mask = cv2.inRange(image, lower_color, upper_color)
    filtered = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow("Filtered Image", filtered)

def on_mouse_click(event, x, y, flags, param):
    global selected_pixel
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_pixel = image[y, x]
        update_trackbars(selected_pixel)
        print_lower_upper(selected_pixel)

def update_trackbars(pixel):
    lower = [max(0, int(pixel[i] - pixel[i] * 0.1)) for i in range(3)]
    upper = [min(255, int(pixel[i] + pixel[i] * 0.1)) for i in range(3)]
    for i, channel in enumerate(("B", "G", "R")):
        cv2.setTrackbarPos(f"Lower {channel}", "Filter", lower[i])
        cv2.setTrackbarPos(f"Upper {channel}", "Filter", upper[i])

def print_lower_upper(pixel):
    lower = [max(0, int(pixel[i] - pixel[i] * 0.1)) for i in range(3)]
    upper = [min(255, int(pixel[i] + pixel[i] * 0.1)) for i in range(3)]
    print(f"Lower: {lower}")
    print(f"Upper: {upper}")

# 读取图像
image = cv2.imread('test.jpg')

# 调整图像亮度
image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)

# 创建窗口并绑定事件
cv2.namedWindow("Filter")
cv2.setMouseCallback("Filter", on_mouse_click)

# 创建滑块
for channel in ("B", "G", "R"):
    cv2.createTrackbar(f"Lower {channel}", "Filter", 0, 255, on_trackbar_change)
    cv2.createTrackbar(f"Upper {channel}", "Filter", 255, 255, on_trackbar_change)

# 创建亮度阈值滑块
cv2.createTrackbar("Brightness Threshold", "Filter", brightness_threshold, 255, on_trackbar_change)

# 显示原始图像
cv2.imshow("Filter", image)

while True:
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
