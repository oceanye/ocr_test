import copy
import math
import threading
import time

import cv2
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt

import ocr_clip
from tkinter import Tk, filedialog

import logging

# 配置日志输出格式
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')

# 设置日志级别为info
logging.getLogger().setLevel(logging.CRITICAL)



from pytesseract import pytesseract

camera_on = False


if camera_on:
    # 打开摄像头
    cap = cv2.VideoCapture()
    cap.open(0, cv2.CAP_DSHOW)

    # 设置视频流的分辨率和帧率
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    #cap.set(cv2.CAP_PROP_FPS, 10)
    print("camera open")

else:

    # 打开文件选择框
    file_path = filedialog.askopenfilename()
    cap = cv2.VideoCapture(file_path)
    print("video open")


def get_angle(pt1, pt2):
    # 计算两个点之间的角度
    x_diff = pt2[0] - pt1[0]
    y_diff = pt2[1] - pt1[1]
    logging.info("x_diff:"+str(x_diff))
    logging.info("y_diff:"+str(y_diff))
    return np.degrees(np.arctan2(y_diff, x_diff))

def is_parallel(angle1, angle2, tolerance):
    # 判断两个角度是否平行
    angle_diff = np.abs(angle1 - angle2)
    logging.info("angle_diff:"+str(angle_diff))
    if angle_diff <= tolerance or angle_diff >= 180 - tolerance:
        return True
    else:
        return False

def parallel_quar(contour, tolerance=5):
    #rect = cv2.minAreaRect(contour)
    #box = cv2.boxPoints(contour)
    #获得变量contour的点坐标，存成list
    points=contour.reshape(-1,2)




    angle1 = get_angle(points[0], points[1])
    angle2 = get_angle(points[3], points[2])

    angle3 = get_angle(points[1], points[2])
    angle4 = get_angle(points[0], points[3])

    #logging.info("angle1:"+str(angle1))
    #logging.info("angle2:"+str(angle2))
    #logging.info("angle3:"+str(angle3))
    #logging.info("angle4:"+str(angle4))

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

    mean_point = np.mean(src_bound, axis=0)

    # 按顺时针方向排序
    sorted_indices = np.argsort(np.arctan2(src_bound[:, 0] - mean_point[0], src_bound[:, 1] - mean_point[1]))
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
            ax.text(x+3, y+3, str(i+1), va='center', ha='center')

        plt.show()






    return relative_coordinates





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
    # cv2.imshow('Mask Image', mask.astype(np.uint8) * 255)

    # 等待用户按下空格键
    # while cv2.waitKey(0) != ord(' '):
    #     pass

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
        result = pytesseract.image_to_string(roi, lang='eng')

        # 显示识别结果
        cv2.putText(image, result, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示二值化结果和识别结果
    #cv2.imshow('Binary Image', binary)
    #cv2.imshow('OCR Result', image)

    # 等待用户按下空格键
    while cv2.waitKey(0) != ord(' '):
        pass

    cv2.destroyAllWindows()

    return result
# 创建保存图片的文件夹

def mouse_callback(event, x, y, flags, param):
    global pmx, pmy
    if event == cv2.EVENT_LBUTTONDOWN:  # 检测到鼠标左键点击事件
        print(f"鼠标点击坐标：({x}, {y})")
        global is_clicked
        is_clicked = True
        pmx = x
        pmy = y


def ocr_percent(image):
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用Tesseract进行文字识别
    result = pytesseract.image_to_string(gray, lang='eng')

    # 显示识别结果
    return result


# 判定点是否在 contour 内部
def is_point_inside_contour(contour, point):
    # 将点转换为tuple格式
    point_tuple = (point[0], point[1])

    # 判定点与轮廓的关系
    result = cv2.pointPolygonTest(contour, point_tuple, False)

    # 返回结果
    return result >= 0


def remove_similar_contours(contours, max_distance):
    filtered_contours = []

    for contour in contours:
        is_similar = False

        for other_contour in filtered_contours:
            center1, _ = cv2.minEnclosingCircle(contour)
            #center1 = contour.mean(axis=0)
            center2, _ = cv2.minEnclosingCircle(other_contour)
            #center2 = other_contour.mean(axis=0)

            distance = np.linalg.norm(np.array(center1) - np.array(center2))

            if distance < max_distance:
                is_similar = True
                break

        if not is_similar:
            filtered_contours.append(contour)

    return filtered_contours

# 计时器类
class FPSCounter:
    def __init__(self):
        self.start_time = None
        self.true_count = 0

    def start(self):
        self.start_time = time.time()
        self.true_count = 0

    def update(self, value):
        if value:
            self.true_count += 1

    def get_fps(self):
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        fps = self.true_count / elapsed_time if elapsed_time > 0 else 0
        return fps


def filter_contours(contours):
    filtered_contours = []
    global frame_shown
    for contour in contours:

        hull = cv2.convexHull(contour)
        # 进行多边形逼近，将轮廓近似为较少的顶点
        approx = cv2.approxPolyDP(hull, 0.1 * cv2.arcLength(hull, True), True)

        #调试边缘识别结果
        #cv2.drawContours(frame_shown, [approx], -1, (0, 255, 0), 2)

        if len(approx) == 4 :
            area = cv2.contourArea(approx)
            if area >= 5000 and area < 50000 and cv2.isContourConvex(approx):


                #判断对边平行与否
                # print("---------"+str(count)+"-----------")
                parallel_approx = parallel_quar(approx, 10)

                if (parallel_approx is None):
                    continue

                filtered_contours.append(parallel_approx)

        #是否打开 相近多边形的合并
        #filtered_contours = remove_similar_contours(filtered_contours, 10)
    return filtered_contours
#-------------------------------
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



ret, frame = cap.read()

f_height, f_width, _ = frame.shape


# 创建窗口并显示第一帧图像
cv2.namedWindow('Video')
frame = cv2.rotate(frame, cv2.ROTATE_180)
cv2.imshow('Video', frame)


pmx,pmy = 964, 501
mouse_mode = True
if mouse_mode == True :
    # 设置鼠标回调函数
    cv2.setMouseCallback('Video', mouse_callback)


    # 等待鼠标点击或20秒等待期结束
    is_clicked = False
    timer = threading.Timer(20.0, lambda: None)
    timer.start()
    while not is_clicked and timer.is_alive():
        if cv2.waitKey(1) == 27:  # 按下ESC键退出程序
            break

    # 取消计时器
    timer.cancel()


# 创建帧率统计器
fps_counter = FPSCounter()

# 循环执行 evaluate() 并统计帧率
fps_counter.start()
fps = 0

while True:
    # 读取视频流的一帧
    ret, frame = cap.read()

    frame = cv2.rotate(frame, cv2.ROTATE_180)

    if not ret:
        break
    #保留原图
    frame_copy = frame.copy()
    # 转换为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 图像平滑处理
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # 边缘检测
    edges = cv2.Canny(blurred_frame, 00, 50)

    # 寻找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_shown = frame


    count = 0  # 计数器

    contours_f = filter_contours(contours)


    for contour in contours_f:



        parallel_approx = contour


        area = cv2.contourArea(parallel_approx)



        # 绘制轮廓

        cv2.drawContours(frame_shown, [parallel_approx], 0, (0, 240, 0), 2)

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
        clip = frame_copy[y:y + h, x:x + w]
        #将x,y,w,h 生成dst_bound
        dst_bound = np.float32([[0,0], [0,  h], [w, h], [w, 0]])

        src_bound = parallel_approx.reshape(-1, 2)
        src_bound = sort_coordinates(src_bound)
        #src_bound是[x,y]的4个点坐标，按顺时针排序，左上角开始

        #print("src_bound:",src_bound)
        #print("dst_bound:",dst_bound)
        transform_matrix = cv2.getPerspectiveTransform(src_bound,dst_bound )
        clip_correct = cv2.warpPerspective(clip, transform_matrix, (w, h))

        #cv2.imwrite(filepath + ".png", clip)
        #cv2.imwrite(filepath + "-correct" + ".png", clip_correct)
        #print("parallel_approx:",parallel_approx)
        #print("pmx,pmy:"+pmx+","+pmy)
        in_contour = is_point_inside_contour(parallel_approx, (pmx, pmy))

        if (pmx+pmy) == 0 or in_contour:
            #ocr_text = ocr_percent(clip_correct)
            ocr_text = ""
            print(ocr_text)
            fn_correct = filepath+"-correct "+ocr_text+".png"
            cv2.imwrite(fn_correct, clip_correct)

            ocr_clip.ocr_clip(fn_correct)

            cv2.drawContours(frame_shown, [parallel_approx], 0, ( 0,0, 255), 2)
            fps_counter.update(True)
            # 打印当前帧率
            fps = round(fps_counter.get_fps(),2)
            logging.info("帧率:"+str(fps))



        cv2.putText(frame_shown, "FPS:"+str(fps), (round(f_width*0.75/10)*10,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        count += 1


    # 保存当前帧到screenshot_folder
    screenshot_filename = "screenshot_{:06d}.png".format(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))

    screenshot_filepath = os.path.join(screenshot_folder, screenshot_filename)
    cv2.imwrite(screenshot_filepath, frame_shown)

    # 显示处理后的图像
    cv2.imshow('Frame', frame_shown)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 清理资源
cap.release()
cv2.destroyAllWindows()
