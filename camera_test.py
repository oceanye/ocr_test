import cv2
import numpy as np
import datetime
import os

# 打开摄像头
cap = cv2.VideoCapture()
cap.open(0, cv2.CAP_DSHOW)

# 设置视频流的分辨率和帧率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 10)

# 创建保存图片的文件夹
pic_folder = "pic_folder"
if not os.path.exists(pic_folder):
    os.makedirs(pic_folder)

# 创建保存截图的文件夹
screenshot_folder = os.path.join(pic_folder, "screenshot")
if not os.path.exists(screenshot_folder):
    os.makedirs(screenshot_folder)

# 创建保存裁剪区域的文件夹
clip_folder = os.path.join(pic_folder, "clip")
if not os.path.exists(clip_folder):
    os.makedirs(clip_folder)

while True:
    # 读取视频流的一帧
    ret, frame = cap.read()

    if not ret:
        break

    # 转换为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 图像平滑处理
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # 边缘检测
    edges = cv2.Canny(blurred_frame, 50, 150)

    # 寻找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0  # 计数器

    for contour in contours:
        # 进行多边形逼近，将轮廓近似为较少的顶点
        approx = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)

        # 判断是否为非凹四边形
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # 计算面积
            area = cv2.contourArea(approx)

            # 判断面积是否大于等于20x20像素
            if area >= 400:
                # 绘制轮廓
                cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)

                # 获取中心点坐标
                M = cv2.moments(approx)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # 切割四边形区域并保存为图像文件
                now = datetime.datetime.now()
                filename = "{}-{}-{}-{}-{}-{}-{}-{}.png".format(now.year, now.month, now.day, now.hour, now.minute,
                                                                now.second, now.microsecond, count)
                filepath = os.path.join(clip_folder, filename)

                # 裁剪四边形区域并保存为图像文件
                x, y, w, h = cv2.boundingRect(approx)
                clip = frame[y:y + h, x:x + w]
                cv2.imwrite(filepath, clip)

                count += 1

    # 保存当前帧到screenshot_folder
    screenshot_filename = "screenshot_{:06d}.png".format(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))

    screenshot_filepath = os.path.join(screenshot_folder, screenshot_filename)
    cv2.imwrite(screenshot_filepath, frame)

    # 显示处理后的图像
    cv2.imshow('Frame', frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 清理资源
cap.release()
cv2.destroyAllWindows()
