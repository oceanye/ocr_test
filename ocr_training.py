import os
import shutil


# Step 1: Create training directory
if os.path.exists('training'):
    shutil.rmtree('training')
os.mkdir('training')

# Step 2: Move character images to the training directory
ocr_train_folder = 'ocr_train'
training_folder = 'training'
for folder_name in os.listdir(ocr_train_folder):
    label = folder_name
    folder_path = os.path.join(ocr_train_folder, folder_name)
    dest_folder_path = os.path.join(training_folder, label)
    os.mkdir(dest_folder_path)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        shutil.copy(file_path, dest_folder_path)

# Step 3: Generate training data file
with open('training/training_data.txt', 'w') as file:
    for root, _, files in os.walk(training_folder):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            label = os.path.basename(root)
            file.write(f'{file_path} {label}\n')

# Step 4: Generate box files
for root, _, files in os.walk(training_folder):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        #os.system(f'tesseract  --psm 10 {file_path} {file_path[:-4]} batch.nochop makebox')

        image = Image.open(file_path)
        box_file = f'{file_name}.box'
        with open(box_file, 'w') as f:
            width, height = image.size
            f.write(f'{label} 1 1 {width - 1} {height - 1} 0\n')

        os.system(f'tesseract  lstm.train {file_path} {file_path[:-4]} batch.nochop makebox')

# Step 5: 生成训练数据文件
os.system('tesseract training/training_data.txt output_base batch.box')

