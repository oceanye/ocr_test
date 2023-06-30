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
    # 图像img向内crop 5个像素





    if len(img_data.shape) == 2:  # 灰度图像
        non_white_pixels = np.where(img_data < 240)
    else:  # RGB图像
        non_white_pixels = np.where((img_data < [240, 240, 240]).all(axis=2))

    x_min, x_max, y_min, y_max = np.min(non_white_pixels[0]), np.max(non_white_pixels[0]), np.min(
        non_white_pixels[1]), np.max(non_white_pixels[1])
    cropped_img = img.crop((y_min-1, x_min-1, y_max+1, x_max+1))

    return cropped_img


# 文件名加上-crop，存在原路径下
def save_images(img_list):
    for img_path in img_list:
        img = crop_image(img_path)
        img_exp = expand_image(img)
        base, ext = os.path.splitext(img_path)


        new_img_path = f"{base}{ext}"
        img_exp.save(new_img_path)
        print('ok')

def expand_image(img):

    #img=Image.open(image_path)

    target_width,target_height = img.size
    new_width = max(round(target_height/1.5),target_width)
    # Create a new blank canvas with the desired size and fill it with white
    expanded_image = Image.new('RGB', (new_width, target_height), 'white')

    # Calculate the position to paste the original image onto the canvas
    x = (new_width - target_width) // 2


    # Paste the original image onto the canvas
    expanded_image.paste(img, (x, 0))

    return expanded_image

# 使用方法
root_path = "C://Users//ZJLZ1026.LZSJY//PycharmProjects//ocr_test//ocr_train2"
# 填写你的图片路径
img_list = get_images(root_path)
save_images(img_list)
