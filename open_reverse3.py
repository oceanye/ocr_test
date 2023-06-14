import cv2
import numpy as np
from shapely.geometry import Polygon
import os
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageProcessor:
    def __init__(self, master):
        self.master = master
        self.frame = Frame(self.master)
        self.frame.pack()
        self.points = []
        self.color_mode = False
        self.color_lower = None
        self.color_upper = None

        self.image_label = Label(self.frame)
        self.image_label.pack()

        self.button = Button(self.frame, text='Select Image', command=self.load_image)
        self.button.pack()

        self.color_button = Button(self.frame, text='Color Selection', command=self.color_selection)
        self.color_button.pack()

        self.canvas = None
        self.image = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        self.image = cv2.imread(file_path)

        if self.image is not None:
            self.display_image()

    def color_selection(self):
        self.color_mode = not self.color_mode
        if self.color_mode:
            self.color_button.config(text='Exit Color Selection')

            # Initialize color boundaries to white
            self.color_lower = np.array([255, 255, 255])
            self.color_upper = np.array([255, 255, 255])
        else:
            self.color_button.config(text='Color Selection')

    def click_event(self, event):
        if len(self.points) < 4:
            x, y = event.x, event.y
            self.points.append((x, y))
            if len(self.points) == 1:
                cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)
                self.display_image()

        if self.color_mode and self.image is not None:
            h, w, _ = self.image.shape
            x, y = int(event.x * w / self.canvas.winfo_width()), int(event.y * h / self.canvas.winfo_height())
            color = self.image[y, x, :]

            # Update color boundaries
            self.color_lower = np.minimum(self.color_lower, color)
            self.color_upper = np.maximum(self.color_upper, color)

            # Apply color mask
            mask = cv2.inRange(self.image, self.color_lower, self.color_upper)
            self.image = cv2.bitwise_and(self.image, self.image, mask=mask)
            self.display_image()

    def display_image(self):
        b,g,r = cv2.split(self.image)
        img = cv2.merge((r,g,b))
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)
        self.image_label.imgtk = imgtk
        self.image_label.configure(image=imgtk)

    def get_center(self, points):
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        center_x = int(sum(x_coords) / len(points))
        center_y = int(sum(y_coords) / len(points))
        return (center_x, center_y)

    def similarity_score(self, polygon_points, contour):
        polygon = Polygon(polygon_points)
        polygon_area = polygon.area
        polygon_center = self.get_center(polygon_points)

        contour_area = cv2.contourArea(contour)
        moments = cv2.moments(contour)
        contour_center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))

        area_difference = abs(polygon_area - contour_area) / max(polygon_area, contour_area)
        center_distance = np.sqrt((polygon_center[0] - contour_center[0]) ** 2 + (polygon_center[1] - contour_center[1]) ** 2)

        score = 0.5 * area_difference + 0.5 * center_distance
        return score

    def evaluate_similarity(self, points, contours):
        best_score = float('inf')
        best_contour = None

        for contour in contours:
            # 使用凸包来补全轮廓
            hull = cv2.convexHull(contour)

            # 多边形逼近
            approx = cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True)

            # 检查逼近后的轮廓是否有四个顶点并且面积大于400
            if len(approx) == 4 and cv2.contourArea(approx) > 400:
                score = self.similarity_score(points, approx)
                if score < best_score:
                    best_score = score
                    best_contour = approx

        return best_score, best_contour

if __name__ == '__main__':
    root = Tk()
    ImageProcessor(root)
    root.mainloop()
