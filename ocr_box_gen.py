import os
from PIL import Image

import subprocess

batch_file = 'lstmf_batch.bat'

if os.path.isfile(batch_file):
    os.remove(batch_file)

# 初始化空字典

images = {}

# 遍历ocr_train2文件夹下的所有子文件夹
for label in os.listdir('ocr_train2'):
    subfolder = os.path.join('ocr_train2', label)
    if os.path.isdir(subfolder):
        # 如果是文件夹，遍历其中的所有文件
        for image_file in os.listdir(subfolder):
            # 只处理png文件
            if image_file.endswith('.png'):
                image_path = os.path.join(subfolder, image_file)
                # 将图片路径和标签添加到字典中
                images[image_path] = label

# 显示images字典




# 创建一个空列表来保存所有的.lstmf文件路径
lstmf_files = []

for image_file, label in images.items():
    # 载入图像并保存为.tiff格式
    image = Image.open(image_file)
    tif_file = f'{image_file[:-4]}.tif'
    image.save(tif_file,dpi=(300,300))
    image.close()

    image_tif = Image.open(tif_file)



    # 创建对应的.box文件
    box_file = f'{tif_file[:-4]}.box'
    with open(box_file, 'w') as f:
        width, height = image_tif.size
        f.write(f'{label} 1 1 {width-1} {height-1} 0\n')


    print(image_tif.info.get('dpi'))

    # 使用Tesseract命令生成.lstmf文件
    abs_tif_file = os.path.abspath(tif_file)
    base_name = os.path.splitext(abs_tif_file)[0]
    print(abs_tif_file)
    print(base_name)
    #command = ['tesseract', abs_tif_file, base_name, 'lstmbox', 'lstm.train','c','tessedit_char_whitelist=0123456789']#'lstmbox'
    #print('error_check_point-1')
    #subprocess.run(command, check=True)
    # 获取abs_tif_file的文件名
    abs_tif_file_name = abs_tif_file.split('\\')[-1].split('.')[0]
    print (abs_tif_file_name)


    abs_tif_file_name.replace('%','%%')
    base_name.replace('%','%%')

    command = ['tesseract', abs_tif_file, base_name,'--psm 10',' lstm.train'] #'lstmbox'
    #print(command)
    #subprocess.run(command, check=True)

    # 将命令写入到批处理文件中
    with open(batch_file, 'a') as file:
        file.write(' '.join(command) + '"\n')

    # 将.lstmf文件的路径添加到列表中
    lstmf_file = tif_file.replace('.tif', '.lstmf')
    lstmf_files.append(lstmf_file)

# 将所有的.lstmf文件路径写入到一个文本文件中
with open('listfile.txt', 'w') as f:
    for lstmf_file in lstmf_files:
        f.write(f'{lstmf_file}\n')
    f.write("<<<EOF\n")



subprocess.call(batch_file,shell=True)