import cv2
import pytesseract


def recognize_percentage(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 进行阈值处理，将图像转换为二值图像
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 使用OCR识别文本
    result = pytesseract.image_to_string(threshold)

    # 查找百分号字符
    percentage_indices = [i for i, char in enumerate(result) if char == '%']

    # 输出结果
    if percentage_indices:
        print("在图像中找到了百分号！")
        for index in percentage_indices:
            print("位置：", index)
    else:
        print("图像中未找到百分号。")


# 调用函数并传入图像路径
image_path = "test.jpg"
recognize_percentage(image_path)
