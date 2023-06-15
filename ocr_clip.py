import os
from tkinter import filedialog

import cv2
import numpy as np
import matplotlib.pyplot as plt

def proj_split(binary, threshold, direction):
    b_height , b_width = binary.shape
    # 沿垂直方向统计
    if direction =="vertical":
        projection = np.sum(binary, axis=1)
    else:
        projection = np.sum(binary, axis=0)

    # 找到小于阈值的区域
    split_regions = []
    start = 0
    for i in range(1, len(projection)):
        if projection[i] < threshold and projection[i-1] >= threshold:
            start = i
        elif projection[i] >= threshold and projection[i-1] < threshold:
            end = i
            split_regions.append((start, end))

    # 获取划分区域的中点坐标
    split_points_mid = [int((start + end) / 2) for start, end in split_regions]


    if direction == "vertical":
        split_points_mid.append(b_height)
        i = len(split_points_mid) - 1
        while i > 0:
            if split_points_mid[i] - split_points_mid[i - 1] < 10:
                split_points_mid.pop(i)
            i -= 1

    # 绘制统计图像
    plt_on = False
    if plt_on:
        if direction =="vertical":
            plt.plot(projection, range(len(projection)))
            plt.xlabel('Pixel Sum')
            plt.ylabel('Y-coordinate')
        else:
            plt.plot(range(len(projection)),projection)
            plt.xlabel('X-coordinate')
            plt.ylabel('Pixel Sum')


    plt.show()



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

def save_regions(regions, binary_sums):
    binary_sums_sort = sorted(binary_sums, reverse=True)
    max_binary = binary_sums_sort[:2]
    for i in range(len(max_binary)):
        region = regions[i]
        binary_sum = binary_sums[i]
        cv2.imwrite(fn+f"-region_{i+1}.png", region)





def ocr_clip(file_path2):
    fn, _ = os.path.splitext(file_path2)
    image = cv2.imread(file_path2,0)
    _,binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    split_y = proj_split(binary, binary.shape[1]*5, "vertical")
    print("split_y:",split_y)
    split_x = proj_split(binary, binary.shape[0]*5, "horizontal")
    print("split_x:",split_x)

    for y in split_y:
        cv2.line(binary, (0, y), (binary.shape[1], y), (0, 0, 255), 1)  # 线条颜色为红色

    # 在image上绘制水平线
    for x in split_x:
        cv2.line(binary, (x, 0), (x, binary.shape[0]), (0, 0, 255), 1)  # 线条颜色为红色

    split_image_on = False
    if split_image_on:
        cv2.imshow("image",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.imwrite(fn+'-split.png', image)

    regions=split_image(binary, split_y, split_x)
    binary_sum = compute_binary_sum(regions)
    save_regions(regions, binary_sum)


file_path = filedialog.askopenfilename()

ocr_clip(file_path)