import cv2
import cvui
import numpy as np

WINDOW_NAME = 'Resizable Window'

# 初始化窗口大小
window_width = 600
window_height = 400

# 创建一个空白的图像作为窗口的背景
window = np.zeros((window_height, window_width, 3), dtype=np.uint8)

while True:
    # 清空窗口
    window.fill(49)

    # 使用cvui的resizeWindow函数创建可调整大小的窗口
    cvui.resizeWindow(WINDOW_NAME, window_width, window_height)

    # 在窗口中绘制一些内容
    cvui.text(window, 50, 50, 'Resizable Window')

    # 显示窗口
    cvui.imshow(WINDOW_NAME, window)

    # 检查是否按下了ESC键
    if cv2.waitKey(1) == 27:
        break

    # 处理窗口的事件
    if cvui.WINDOW_EVENT in cvui.mouse(cvui.EVENT_MOUSEWHEEL):
        # 获取鼠标滚轮的滚动量
        _, dy = cvui.mouse(cvui.EVENT_MOUSEWHEEL)
        window_height += dy

    # 拖动调整窗口大小
    cvui.update()

# 销毁窗口
cv2.destroyAllWindows()
