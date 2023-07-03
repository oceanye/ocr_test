import copy
import math
import re
import threading
import time
import random

import cv2
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt

#from ocr_clip import ocr_clip_img
from tkinter import Tk, filedialog

import logging
from pytesseract import pytesseract, image_to_string



# 配置日志输出格式
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')

# 设置日志级别为info
logging.getLogger().setLevel(logging.CRITICAL)

lang_model = 'final'#''eng'





camera_on = False

#file_path 为当前文件夹路径
file_path = os.path.dirname(os.path.abspath(__file__))



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
    #将file_path中的\替换为//
    file_path = file_path.replace('/', '\\')
    cap = cv2.VideoCapture(file_path)
    print("video open")


#---------------------------------------------------
def get_angle(pt1, pt2):
    # 计算两个点之间的角度
    x_diff = pt2[0] - pt1[0]
    y_diff = pt2[1] - pt1[1]
    logging.info("x_diff:"+str(x_diff))
    logging.info("y_diff:"+str(y_diff))
    return np.degrees((np.arctan2(y_diff, x_diff)))

def is_parallel(angle1, angle2, tolerance):
    # 判断两个角度是否平行
    angle_diff = np.abs(angle1-angle2)
    logging.info("angle_diff:"+str(angle_diff))
    if angle_diff <= tolerance or angle_diff >= 360 - tolerance:
        return True
    else:
        return False

def parallel_quar(contour, tolerance=1):
    #rect = cv2.minAreaRect(contour)
    #box = cv2.boxPoints(contour)
    #获得变量contour的点坐标，存成list
    points=contour.reshape(-1,2)




    angle1 = get_angle(points[0], points[1])
    angle2 = get_angle(points[3], points[2])

    angle3 = get_angle(points[1], points[2])
    angle4 = get_angle(points[0], points[3])

    logging.info("angle1:"+str(angle1))
    logging.info("angle2:"+str(angle2))
    logging.info("angle3:"+str(angle3))
    logging.info("angle4:"+str(angle4))

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

def mouse_callback(event, x, y, flags, param):
    global pmx, pmy
    if event == cv2.EVENT_LBUTTONDOWN:  # 检测到鼠标左键点击事件
        print(f"鼠标点击坐标：({x}, {y})")
        global is_clicked
        is_clicked = True
        pmx = x
        pmy = y


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
        approx = cv2.approxPolyDP(hull, 0.05 * cv2.arcLength(hull, True), True)

        #调试边缘识别结果
        #cv2.drawContours(frame_shown, [approx], -1, (0, 255, 0), 2)

        if len(approx) == 4 :
            # 调试四边形识别结果
            #cv2.drawContours(frame_shown, [approx], -1, (0, 255, 0), 2)
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





def proj_split(binary, threshold, direction):
    b_height , b_width = binary.shape

    #建立morphologyex结构元素
    if direction == "vertical":
        structure_element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
    else:
        structure_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    #闭运算
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, structure_element)

    cv2.imshow("textimg", closed)

    # 沿垂直方向统计
    if direction =="vertical":
        projection = np.sum(closed, axis=1)
    else:
        projection = np.sum(closed, axis=0)

    projection_edge= 0 #round(len(projection)*0.1)
    projection_start = projection_edge
    projection_end = len(projection) - projection_edge
    min_projection = np.min(projection[projection_start:projection_end])
    threshold = min_projection*1.5+1
    # 找到小于阈值的区域
    split_regions = []
    start = 0
    for i in range(projection_start, projection_end):
        if projection[i] < threshold and projection[i-1] >= threshold:
            start = i
        elif projection[i] >= threshold and projection[i-1] < threshold:
            end = i
            split_regions.append((start, end))

    # 获取划分区域的中点坐标
    split_points_mid = [int((start + end) / 2) for start, end in split_regions]
    split_points_mid.insert(0, projection_start)
    split_points_mid.append(projection_end)

     #split_points_mid.append(b_height)
    i = len(split_points_mid) - 1
    while i > 0:
        if split_points_mid[i] - split_points_mid[i - 1] < 10:
            split_points_mid.pop(i)
        i -= 1

    #split_points_mid升序排列
    #split_points_mid.sort()

    # 绘制统计图像
    plt_on = False
    if plt_on:
        if direction =="vertical":
            plt.plot(projection, range(len(projection)))
            # 绘制一条直线，X坐标在threshold处
            plt.plot([threshold] * len(projection), range(len(projection)))
            plt.xlabel('Pixel Sum')
            plt.ylabel('Y-coordinate')
        else:
            plt.plot(range(len(projection)),projection)
            # 绘制一条直线，Y坐标在threshold处
            plt.plot(range(len(projection)), [threshold] * len(projection))
            plt.xlabel('X-coordinate')
            plt.ylabel('Pixel Sum')


        plt.show()
        filename = str(random.randint(1000000, 9999999)) + '.png'
        plt.savefig(filename)


    return split_points_mid


def split_image(img, split_y, split_x):
    regions = []
    for i in range(len(split_y)-1):
        for j in range(len(split_x)-1):
            region = img[split_y[i]:split_y[i+1], split_x[j]:split_x[j+1]]
            regions.append(region)
    return regions

def compute_binary_sum(regions):
    binary_sums = []
    for region in regions:
        _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary_sum = np.sum(binary)
        binary_sums.append(binary_sum)
    return binary_sums

def save_regions(file_path,regions, binary_sums):
    global lang_model

    rst = []
    binary_sums_sort = sorted(binary_sums, reverse=True)
    max_binary = binary_sums_sort[:3]


    # 根据binary_sums 对regions进行排序，取region前三个
    regions_sort = []
    for i in range(len(max_binary)):
        index = binary_sums.index(max_binary[i])
        regions_sort.append(index)


    file_name = os.path.basename(file_path)
    fn_parent = os.path.dirname(os.path.dirname(file_path))

    for i in range(len(regions)):#"#regions_sort:

        region = regions[i]
        #binary_sum = binary_sums[i]

        # 将region turn black to white
        region = cv2.bitwise_not(region)
        #rst = "v="+pytesseract.image_to_string(region, lang='eng')
        #if np.sum(region)>1000:
        if i == 1 or i==2:
            rst1 = image_to_string(region, lang=lang_model,config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
            #rst1中替换所有换行符\n
            rst1 = rst1.replace("\n", "")
            rst1 = rst1.replace("%", "")
            #提取rst1中的数字，形成整数
            rst1 = re.findall(r"\d+\.?\d*", rst1)
            rst1 = "".join(rst1)

            #如果rst1是数字
            if rst1.isdigit():
                fn = fn_parent + "//ocr//"+str(rst1)+"//" + file_name.split('.')[0] + f"-region{i + 1}_NUM{rst1}.png"
                str(rst1).replace("%", "")
                print("v=" + rst1)
                logging.info("v=" + rst1)
                rst.append(rst1)
            else:
                # rst1等于100-200之间的随机数
                rst1 = "X" + str(random.randint(100, 200))
                fn = fn_parent + "//ocr//" + file_name.split('.')[0] + f"-region{i + 1}_NUM{rst1}.png"


        else:
            #rst1等于100-200之间的随机数
            rst1 = "X"+str(random.randint(100, 200))
            fn = fn_parent + "//ocr//" + file_name.split('.')[0] + f"-region{i + 1}_NUM{rst1}.png"

        if True:

            print("OCR_split_image: " + fn)
            cv2.imwrite(fn, region)


    #value转换为1个数值
    value = "".join(rst)
    #value = value.replace("\n","")
    #如果value非空，转换为数值
    value = str2int(value)

    return value
def fill_disconnected_regions(binary_image):
    # 进行连通组件标记
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)

    # 获取标记图像中像素值为1的区域（即不连续的部分）
    disconnected_regions = (labels == 1).astype('uint8')

    # 使用膨胀操作补足不连续的部分
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 可根据需要调整结构元素的大小
    reconnected_regions = cv2.dilate(disconnected_regions, kernel, iterations=1)

    # 将补足后的区域与原始图像进行融合
    result = cv2.bitwise_or(binary_image, reconnected_regions)

    return result

def str2int(rst):
    try:
        if rst != "":
            #判断rst是否为int
            value = int(rst.replace("\n",""))

        if 10< value < 50:
            return value
    except:
        return ""
def ocr_clip_img(image,fn_path):
    global lang_model
    #fn, _ = os.path.splitext(file_path2)

    #统计当前函数运行时间
    start = time.time()


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    # 先通过单字段识别两位整数，如果不在有效区域内，则进行拆分识别

    rst = image_to_string(binary,lang=lang_model,config='--psm 7 --oem 1 -c tessedit_char_whitelist=0123456789%'  )    #rst只保留前两个char，判断是否小于40-c tessedit_char_whitelist=0123456789%

    #rst中只保留数字
    rst = re.findall(r"\d+\.?\d*", rst)
    #rst = rst[:2]
    value = str2int(rst)

    #筛选出rst中除了数字之外的字符


    if value != "" and value != None and value < 40 and value > 10:
        end = time.time()

        print("ocr_clip_img string time:", end - start)
        print ("字段-检测成功 数值 = ", value)

        return value
    else:
        print("字段-检测结果：",rst)
        print("字段-检测失败，拆分后重新检测")
        #拆分后识别

        #split_y = proj_split(binary, binary.shape[1]*5, "vertical")
        #print("split_y:",split_y)
        split_x = proj_split(binary, binary.shape[0]*5, "horizontal")
        print("split_x:",split_x)

        #for y in split_y:
        #    cv2.line(gray, (0, y), (binary.shape[1], y), (0, 0, 255), 1)  # 线条颜色为红色

        # 在image上绘制水平线
        for x in split_x:
            cv2.line(gray, (x, 0), (x, binary.shape[0]), (0, 0, 255), 1)  # 线条颜色为红色

        split_image_on = False

        if split_image_on:
            cv2.imshow("image",gray)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        cv2.imwrite(fn_path+'-split.png', image)

        regions=split_image(binary, [0,300], split_x)
        binary_sum = compute_binary_sum(regions)
        value = save_regions(fn_path,regions, binary_sum)

        #统计当前函数运行时间

        end = time.time()
        print("ocr_clip_img char time:",end-start)
        #如果value 不为None
        if value != None:
            print("单字符-检测成功 数值 = ",value)
        else:
            print("单字符-检测失败")

    return value
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


ocr_folder = os.path.join(pic_folder, "ocr")
if not os.path.exists(ocr_folder):
    os.makedirs(ocr_folder)

# 检查在ocr_folder下是否存在以数字1-9命名的文件夹，如果不存在则创建
for i in range(1, 10):
    folder = os.path.join(ocr_folder, str(i))
    if not os.path.exists(folder):
        os.makedirs(folder)

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

num=0

# 通过调用tinker，建立一个窗口，有一个滑条，设置2个滑块，分别是最小值和最大值，控制canny的阈值
# 通过滑块调节阈值，实时显示canny的效果
# 通过滑块调节阈值，实时显示canny的效果


while True:
    num=num+1
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

    frame_shown = frame

    # 边缘检测
    edges = cv2.Canny(blurred_frame, 0, 20) # 调整canny阈值




    # 寻找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #删除contours中面积小于400的轮廓
    contours = [contour for contour in contours if cv2.contourArea(contour) < 400]

    #检测到的轮廓数量
    #cv2.drawContours(frame_shown, contours, 0, (255, 0, 0), 2)




    count = 0  # 计数器
    value = 0

    contours_f = filter_contours(contours)

    for contour in contours_f:
        in_contour = is_point_inside_contour(contour, (pmx, pmy))
        if in_contour:
            break

    if in_contour == True:


        for contour in contours_f:


            global parallel_approx

            parallel_approx = contour


            area = cv2.contourArea(parallel_approx)



            # 绘制轮廓

            #cv2.drawContours(frame_shown, [parallel_approx], 0, (0, 240, 0), 2)

            # 获取中心点坐标
            M = cv2.moments(parallel_approx)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # 标记当前轮廓的编号
            # cv2.putText(frame_shown, str(count), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # 切割四边形区域并保存为图像文件
            now = datetime.datetime.now()

            #filename为选中文件的文件名，不包含后缀，另加 count
            if camera_on == True:
                # 实时读取摄像头时，以时间命名
                filename = "{}-{}-{}-{}-{}-{}".format(now.month, now.day, now.hour, now.minute,now.second, count)
            else:
                filename = file_path.split("\\")[-1].split(".")[0] + "-" + str(count)

            filepath = os.path.join(clip_folder, filename)


            filepath_ocr = os.path.join(ocr_folder, filename)




            #cv2.imwrite(filepath + ".png", clip)
            #cv2.imwrite(filepath + "-correct" + ".png", clip_correct)
            #print("parallel_approx:",parallel_approx)
            #print("pmx,pmy:"+pmx+","+pmy)
            in_contour = is_point_inside_contour(parallel_approx, (pmx, pmy))



            if (pmx+pmy) == 0 or in_contour:

                # 裁剪四边形区域并保存为图像文件
                global x,y,w,h  # 裁剪区域的坐标和宽高
                x, y, w, h = cv2.boundingRect(parallel_approx)
                print("锁定当前帧 x,y,w,h:",x,y,w,h)

                clip = frame_copy[y:y + h, x:x + w]

                # 将x,y,w,h 生成dst_bound
                dst_bound = np.float32([[0, 0], [0, h], [w, h], [w, 0]])

                src_bound = parallel_approx.reshape(-1, 2)
                src_bound = sort_coordinates(src_bound)
                # src_bound是[x,y]的4个点坐标，按顺时针排序，左上角开始

                # print("src_bound:",src_bound)
                # print("dst_bound:",dst_bound)
                global transform_matrix




                transform_matrix = cv2.getPerspectiveTransform(src_bound, dst_bound)
                clip_correct = cv2.warpPerspective(clip, transform_matrix, (w, h))



                # 统一裁剪区域大小
                clip_resize = cv2.resize(clip_correct, (350, 120), interpolation=cv2.INTER_CUBIC)

                # 去除边框
                clip_final = clip_resize[10:110, 35:335]

                print("clip.shape:", clip.shape)
                print("transform_matrix:", transform_matrix.shape)


                #ocr_text = ocr_percent(clip_correct)
                #ocr_text = ""
                #print(ocr_text)
                fn_final = filepath+"-correct"+str(num)
                cv2.imwrite(fn_final+".png", clip_final)

                value= ocr_clip_img(clip_final,fn_final)



                cv2.drawContours(frame_shown, [parallel_approx], 0, ( 255,0, 0), 2)

                # drawcontour 用虚线

                #cv2 绘制矩形填充区域，颜色红色，透明度50%
                #cv2.fillPoly(frame_shown, [parallel_approx], (0, 0, 255), 0.5)

                fps_counter.update(True)
                # 打印当前帧率
                fps = round(fps_counter.get_fps(),2)
                logging.info("帧率:"+str(fps))
                logging.info("当前帧："+str(num))





                #结束当前for循环
                break
            count += 1


    else:

        try:


            now = datetime.datetime.now()

            #filename为选中文件的文件名，不包含后缀，另加 count
            if camera_on == True:
                # 实时读取摄像头时，以时间命名
                filename = "{}-{}-{}-{}-{}-{}".format(now.month, now.day, now.hour, now.minute,now.second, count)
            else:
                filename = file_path.split("\\")[-1].split(".")[0] + "-" + str(count)

            filepath = os.path.join(clip_folder, filename)


            filepath_ocr = os.path.join(ocr_folder, filename)

            frame_copy = frame.copy()

            # 裁剪四边形区域并保存为图像文件

            #logging.info("frame_copy:",frame_copy)

            #logging.info("采用上一帧数据 x,y,w,h:",str(x),str(y),str(w),str(h))

            clip = frame_copy[y:y + h, x:x + w]


            clip_correct = cv2.warpPerspective(clip, transform_matrix, (w, h))

            # 统一裁剪区域大小
            clip_resize = cv2.resize(clip_correct, (350, 120), interpolation=cv2.INTER_CUBIC)

            # 去除边框
            clip_final = clip_resize[10:110, 35:335]

            # ocr_text = ocr_percent(clip_correct)
            # ocr_text = ""
            # print(ocr_text)
            fn_final = filepath + "-simi"+str(num) # 未识别得到矩形轮廓帧，则沿用前一帧的结果，并命名 simi
            cv2.imwrite(fn_final + ".png", clip_final)

            value = ocr_clip_img(clip_final,fn_final)



            cv2.drawContours(frame_shown, [parallel_approx], 0, (0, 0, 255), 2)


            fps_counter.update(False)
            # 打印当前帧率
            fps = round(fps_counter.get_fps(), 2)
            logging.info("帧率:" + str(fps))
            logging.info("当前帧："+str(num))
        except Exception as e:

            print('Reason:', e)
            pass


    cv2.putText(frame_shown, "FPS:" + str(fps), (round(f_width * 0.75 / 10) * 10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    cv2.putText(frame_shown, "VALUE:" + str(value), (round(f_width * 0.75 / 10) * 10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)


    print("当前帧：", num)
    # 保存当前帧到screenshot_folder
    screenshot_filename = "screenshot_{:06d}.png".format(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))

    screenshot_filepath = os.path.join(screenshot_folder, screenshot_filename)
    cv2.imwrite(screenshot_filepath, frame_shown)

    # 显示处理后的图像
    cv2.imshow('Frame', frame_shown)

    #if num>95:
    #    parallel_quar(parallel_approx, 2)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




# 清理资源
cap.release()
cv2.destroyAllWindows()
