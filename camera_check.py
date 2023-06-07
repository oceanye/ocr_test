import cv2


def check_available_cameras():
    index = 0
    while True:
        # 尝试打开摄像头设备
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break

        # 获取摄像头的名称
        camera_name = f"Camera {index}"

        # 打印摄像头索引和名称
        print(f"Camera Index: {index} - Camera Name: {camera_name}")

        # 释放摄像头资源
        cap.release()

        index += 1


# 检查可用摄像头
check_available_cameras()
