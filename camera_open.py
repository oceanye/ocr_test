import cv2
import os
import datetime

print(cv2.getBuildInformation())
# 创建存储视频的文件夹
record_folder = "record"
if not os.path.exists(record_folder):
    os.makedirs(record_folder)



# 打开摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if cap.isOpened():
    # 设置视频流的分辨率和帧率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 10)



else:
    #通过cv2打开2023-06-07-11-46-14.mp4

    cap = cv2.videocapture('2023-06-07-11-46-14.mp4', cv2.CAP_DSHOW)


# 视频编码器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 初始化计时器
timer = 0
interval = 2 * 60  # 2分钟

out = None
out_filename = ""

while True:
    # 读取视频流的一帧
    ret, frame = cap.read()

    if not ret:
        break

    # 在帧上进行一些处理，如果需要的话

    # 显示处理后的帧
    cv2.imshow('Frame', frame)

    # 计时器增加帧间隔
    timer += 1

    # 如果计时器达到保存间隔
    if timer >= interval:
        # 释放之前的VideoWriter对象
        if out is not None:
            out.release()

        # 获取当前时间作为文件名
        current_time = datetime.datetime.now()
        filename = current_time.strftime("%Y-%m-%d-%H-%M-%S") + ".mp4"

        # 创建新的VideoWriter对象，指定输出文件路径、编码器、帧率和分辨率
        output_file = os.path.join(record_folder, filename)
        out = cv2.VideoWriter(output_file, fourcc, 30.0, (640, 480))

        # 更新当前的输出文件名
        out_filename = filename

        # 重置计时器
        timer = 0

    # 如果当前有VideoWriter对象，则将帧写入其中
    if out is not None:
        out.write(frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()

# 释放最后一个VideoWriter对象
if out is not None:
    out.release()

cv2.destroyAllWindows()
