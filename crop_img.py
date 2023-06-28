import os
from PIL import Image
import numpy as np

# 读取路径下所有图片，包括子文件夹
def get_images(root_path):
    img_list = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(('jpg', 'png', 'jpeg')):
                img_list.append(os.path.join(root, file))

    return img_list


# crop所有白边
def crop_image(image_path):
    img = Image.open(image_path)
    img_data = np.asarray(img)

    if len(img_data.shape) == 2:  # 灰度图像
        non_white_pixels = np.where(img_data < 250)
    else:  # RGB图像
        non_white_pixels = np.where((img_data < [250, 250, 250]).all(axis=2))

    x_min, x_max, y_min, y_max = np.min(non_white_pixels[0]), np.max(non_white_pixels[0]), np.min(
        non_white_pixels[1]), np.max(non_white_pixels[1])
    cropped_img = img.crop((y_min, x_min, y_max, x_max))

    return cropped_img


# 文件名加上-crop，存在原路径下
def save_cropped_images(img_list):
    for img_path in img_list:
        img = crop_image(img_path)
        base, ext = os.path.splitext(img_path)
        new_img_path = f"{base}{ext}"
        img.save(new_img_path)
        print('ok')

# 使用方法
root_path = "C://Users//ZJLZ1026.LZSJY//PycharmProjects//ocr_test//ocr_train2"
# 填写你的图片路径
img_list = get_images(root_path)
save_cropped_images(img_list)
