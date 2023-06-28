#!/bin/bash

# Step 1: 创建训练文件夹
mkdir training

# Step 2: 移动字符图片到训练文件夹
for folder in ocr_train/*/; do
    label=$(basename "$folder")
    mv "$folder"/*.png training/"$label"/
done

# Step 3: 生成训练数据文件
cd training
for folder in */; do
    label=$(basename "$folder")
    for file in "$folder"*.png; do
        echo "$file" "$label"
    done
done > training_data.txt

# Step 4: 生成box文件
tesseract training_data.txt output_base batch.nochop makebox

# Step 5: 生成训练数据文件
tesseract training_data.txt output_base batch.boxa

# Step 6: 生成字体特征文件
tesseract training_data.txt output_base box.train

# Step 7: 生成识别模型
unicharset_extractor training_data.txt
mftraining -F font_properties -U unicharset -O output_base.unicharset output_base.tr
cntraining output_base.tr
mv inttemp output_base.inttemp
mv normproto output_base.normproto
mv pffmtable output_base.pffmtable
mv shapetable output_base.shapetable
combine_tessdata output_base.

# 完成训练，生成的模型文件为"output_base.traineddata"
echo "训练完成！"
