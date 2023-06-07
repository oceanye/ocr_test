import cv2
import pytesseract

# 设置tesseract路径
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # 修改为你的 tesseract 路径

# 读取图片
img = cv2.imread('test.jpg')

# 转为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用pytesseract提取文字
text = pytesseract.image_to_string(gray)  # 使用中文简体语言模型

# 寻找“锯条贷款”和“%”之间的数字
import re
match = re.search('(.*)%', text)
if match:
    number = match.group(1)
    print(number.strip())  # 打印数字
