import math

import cv2
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from pytesseract import pytesseract

camera_on = False


if camera_on:
    # 打开摄像头
    cap = cv2.VideoCapture()
    cap.open(0, cv2.CAP_DSHOW)

    # 设置视频流的分辨率和帧率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 10)

else:
    cap = cv2.VideoCapture('test.mp4')



def get_angle(pt1, pt2):
    # 计算两个点之间的角度
    x_diff = pt2[0] - pt1[0]
    y_diff = pt2[1] - pt1[1]
    #print("x_diff:",x_diff)
    #print("y_diff:",y_diff)
    return np.degrees(np.arctan2(y_diff, x_diff))

def is_parallel(angle1, angle2, tolerance):
    # 判断两个角度是否平行
    angle_diff = np.abs(angle1 - angle2)
    #print("angle_diff:",angle_diff)
    if angle_diff <= tolerance or angle_diff >= 180 - tolerance:
        return True
    else:
        return False

def parallel_quar(contour, tolerance=5):
    #rect = cv2.minAreaRect(contour)
    #box = cv2.boxPoints(contour)
    #获得变量contour的点坐标，存成list
    points=contour.reshape(-1,2)


    #print("contour:",points)

    angle1 = get_angle(points[0], points[1])
    angle2 = get_angle(points[3], points[2])

    angle3 = get_angle(points[1], points[2])
    angle4 = get_angle(points[0], points[3])

    #print("angle1:",angle1)
    #print("angle2:",angle2)
    #print("angle3:",angle3)
    #print("angle4:",angle4)

    if is_parallel(angle1, angle2, tolerance) and is_parallel(angle3, angle4, tolerance):
        return contour
    else:
        return None

def sort_coordinates(src_bound):
    src_bound = np.array(src_bound, dtype=np.float32)
    # 找到左下角的坐标
    leftmost = np.min(src_bound, axis=0)
    bottommost = np.min(src_bound, axis=0)
    start_point = [leftmost[0], bottommost[1]]

    # 按顺时针方向排序
    sorted_indices = np.argsort(np.arctan2(src_bound[:, 0] - start_point[0], src_bound[:, 1] - start_point[1]))
    sorted_coordinates = src_bound[sorted_indices]

    # 转换为相对左下角的坐标
    relative_coordinates = sorted_coordinates - start_point

    fig_on = False

    if fig_on :
        # 绘制排序结果
        fig, ax = plt.subplots()
        ax.plot(relative_coordinates[:, 0], relative_coordinates[:, 1], '-o')
        ax.set_aspect('equal')

        # 标记最终顺序
        for i, (x, y) in enumerate(relative_coordinates):
            ax.text(x, y, str(i+1), va='center', ha='center')

        plt.show()

    return relative_coordinates



import cv2
import numpy as np
import pytesseract

def ocr(image):
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算背景颜色范围
    pixel_count = image.shape[0] * image.shape[1] // 5
    min_color = np.percentile(image.reshape(-1, 3), 2.5, axis=0, interpolation='lower')
    max_color = np.percentile(image.reshape(-1, 3), 97.5, axis=0, interpolation='higher')

    # 提取纯色背景
    mask = np.all((image >= min_color) & (image <= max_color), axis=2)

    # 显示mask结果
    cv2.imshow('Mask Image', mask.astype(np.uint8) * 255)

    # 等待用户按下空格键
    while cv2.waitKey(0) != ord(' '):
        pass

    # 对图像进行二值化处理
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 使用轮廓检测分割字符区域
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制矩形框并识别字符
    for contour in contours:
        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)

        # 绘制矩形框
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 提取字符区域
        roi = binary[y:y+h, x:x+w]

        # 使用Tesseract进行文字识别
        result = pytesseract.image_to_string(roi, lang='chi_sim+eng')

        # 显示识别结果
        cv2.putText(image, result, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示二值化结果和识别结果
    cv2.imshow('Binary Image', binary)
    cv2.imshow('OCR Result', image)

    # 等待用户按下空格键
    while cv2.waitKey(0) != ord(' '):
        pass

    cv2.destroyAllWindows()

    return result
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

    frame = cv2.rotate(frame, cv2.ROTATE_180)

    if not ret:
        break

    # 转换为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 图像平滑处理
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # 边缘检测
    edges = cv2.Canny(blurred_frame, 100, 250)

    # 寻找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0  # 计数器

    for contour in contours:
        # 进行多边形逼近，将轮廓近似为较少的顶点

        hull = cv2.convexHull(contour)
        approx = cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True)

        # 判断是否为非凹四边形
        if len(approx) == 4 and cv2.isContourConvex(approx):


            print("---------"+str(count)+"-----------")
            parallel_approx=parallel_quar(approx,5)

            if(parallel_approx is None):
                continue

            area = cv2.contourArea(parallel_approx)

            frame_shown = blurred_frame
            #cv2.drawContours(frame, [hull], 0, (255, 0, 0), 2)
            # 判断面积是否大于等于20x20像素
            if area >= 2000:
                # 绘制轮廓

                cv2.drawContours(frame_shown, [parallel_approx], 0, (0, 255, 0), 2)

                # 获取中心点坐标
                M = cv2.moments(parallel_approx)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                cv2.putText(frame_shown, str(count), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # 切割四边形区域并保存为图像文件
                now = datetime.datetime.now()
                filename = "{}-{}-{}-{}-{}-{}-{}-{}".format(now.year, now.month, now.day, now.hour, now.minute,
                                                                now.second, now.microsecond, count)
                filepath = os.path.join(clip_folder, filename)

                # 裁剪四边形区域并保存为图像文件

                x, y, w, h = cv2.boundingRect(parallel_approx)
                clip = frame[y:y + h, x:x + w]
                #将x,y,w,h 生成dst_bound
                dst_bound = np.float32([[0,0], [0,  h], [w, h], [w, 0]])

                src_bound = parallel_approx.reshape(-1, 2)
                src_bound = sort_coordinates(src_bound)
                #src_bound是[x,y]的4个点坐标，按顺时针排序，左上角开始

                #print("src_bound:",src_bound)
                #print("dst_bound:",dst_bound)
                transform_matrix = cv2.getPerspectiveTransform(src_bound,dst_bound )
                clip_correct = cv2.warpPerspective(clip, transform_matrix, (w, h))
                ocr_text = "none"#ocr(clip_correct)
                #print(ocr_text)
                cv2.imwrite(filepath+".png", clip)
                cv2.imwrite(filepath+"-correct"+ocr_text+".png", clip_correct)

                count += 1


    # 保存当前帧到screenshot_folder
    screenshot_filename = "screenshot_{:06d}.png".format(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))

    screenshot_filepath = os.path.join(screenshot_folder, screenshot_filename)
    cv2.imwrite(screenshot_filepath, frame)

    # 显示处理后的图像
    cv2.imshow('Frame', frame_shown)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 清理资源
cap.release()
cv2.destroyAllWindows()
