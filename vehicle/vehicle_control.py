import random
from vehicle.Vehicle import Vehicle
import asyncio
from connection.SocketConnection import SocketConnection
import json
import struct
import time
import cv2
import numpy as np
import math
import threading

old_center = []


def remove_small_particles(img, min_size=150):
    # Проверяем, цветное ли изображение (3 канала)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Конвертируем в grayscale
    else:
        gray = img  # Если уже grayscale, оставляем как есть

    # Бинаризация (если изображение не бинарное)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Находим контуры
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Создаем маску для крупных объектов
    mask = np.zeros_like(binary)

    for contour in contours:
        # Если площадь контура больше min_size, добавляем его в маску
        if cv2.contourArea(contour) > min_size:
            cv2.drawContours(mask, [contour], -1, 255, -1)

    # Применяем маску к оригинальному изображению
    if len(img.shape) == 3:
        # Если исходное изображение цветное, применяем маску к каждому каналу
        result = cv2.bitwise_and(img, img, mask=mask)
    else:
        # Если исходное изображение grayscale, просто применяем маску
        result = cv2.bitwise_and(img, img, mask=mask)

    return result


class Camera:
    def __init__(self):
        self.image = None
        self.color_data = []
        self.blended = None
        self.lower = np.array([0, 0, 0])
        self.upper = np.array([179, 30, 187])
        # upper = np.array([179, 48, 187])
        # Set minimum and maximum HSV values to display
        self.lower_line = np.array([0, 54, 157])
        self.upper_line = np.array([60, 255, 210])

        self.angle = 90

    def update_image(self, img):
        self.image = img

    def update_angle(self, ang):
        self.angle = ang

    def get_color_data(self):
        return self.color_data

    def get_blended(self):
        return self.blended

    def thread(self):
        while True:
            try:
                if self.image is not None:
                    # Convert to HSV format and color threshold
                    hsv_line = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
                    mask_line = cv2.inRange(hsv_line, self.lower_line, self.upper_line)
                    result_line = cv2.bitwise_and(self.image, self.image, mask=mask_line)
                    # Convert to HSV format and color threshold
                    hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, self.lower, self.upper)
                    image = cv2.bitwise_and(self.image, self.image, mask=mask)
                    # ret, frame = video.read()
                    image = remove_small_particles(image)
                    gauss_result_line = cv2.GaussianBlur(result_line, (3, 3), 0)

                    alpha = 5.0  # например, 2.0 — сильный контраст
                    beta = 0  # можно добавить яркость

                    result_line = cv2.convertScaleAbs(gauss_result_line, alpha=alpha, beta=beta)

                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    alpha = 0.5  # Прозрачность (0-1)
                    self.blended = cv2.addWeighted(image, alpha, result_line, 1 - alpha, 0)
                    for cnt in contours:
                        # Пропускаем маленькие контуры
                        if cv2.contourArea(cnt) < 100:
                            continue

                        # Преобразуем контур в прямоугольник
                        rect, _, center = contour_to_rotated_rectangle(cnt)

                        # Рисуем результат
                        self.color_data = draw_rotated_rectangle(self.blended, rect, self.angle, center)
                    # cv2.imshow('Rotated Rectangles', self.blended)
            except Exception as ex:
                print(ex)
            # if self.image is not None:
            #     if cv2.waitKey(1) == ord('x'):
            #         break

    def start(self):
        t1 = threading.Thread(target=self.thread, args=())
        t1.start()
        # t1.join()



class Robot:
    def __init__(self, vehicle: Vehicle):
        self.v = vehicle

    def forword(self):
        self.v.setMotorPower(50, 50)
        time.sleep(0.01)
        self.v.setMotorPower(0, 0)
        time.sleep(0.0001)

    def forword_t(self):
        t = threading.Thread(self.forword)
        t.start()


def contour_to_rotated_rectangle(contour):
    # Получаем rotated rectangle (повернутый прямоугольник)
    rect = cv2.minAreaRect(contour)

    # Извлекаем параметры прямоугольника
    (center, (width, height), angle) = rect

    # Корректируем угол в зависимости от соотношения сторон
    if width < height:
        angle += 90
        width, height = height, width

    # Получаем 4 вершины прямоугольника
    box = cv2.boxPoints((center, (width, height), angle))
    box = np.int32(box)

    # Сортируем точки по часовой стрелке, начиная с левой верхней
    box = sorted(box, key=lambda x: (x[0], x[1]))
    box = np.array(box)

    # Нормализуем угол в диапазон [-45, 45]
    angle = angle % 90
    if angle > 45:
        angle -= 90

    return box, angle, center


def draw_rotated_rectangle(image, box, angle, center):
    """
    Рисует прямоугольник и информацию об угле на изображении

    Параметры:
        image - исходное изображение
        box - точки прямоугольника
        angle - угол наклона
        center - центр прямоугольника
    """
    # Рисуем прямоугольник
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

    # Рисуем центр
    cv2.circle(image, (int(center[0]), int(center[1])), 5, (255, 0, 0), -1)

    # Добавляем текст с углом
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # text = f"Angle: {angle:.1f} deg"
    # cv2.putText(image, text, (int(center[0]) - 50, int(center[1]) - 30),
    #             font, 0.7, (0, 0, 255), 2)

    # Рисуем линию, показывающую ориентацию
    # line_length = max(image.shape[0], image.shape[1]) // 4
    # t_angle = 25
    line_length = 40
    pixels_color = []
    for t_angle in range(-90, 90, 10):
        end_x = int(center[0] + line_length * math.cos(math.radians(angle + t_angle)))
        end_y = int(center[1] + line_length * math.sin(math.radians(angle + t_angle)))
        val = image[end_y][end_x].copy()
        pixels_color.append(val)
        cv2.line(image, (int(center[0]), int(center[1])), (end_x, end_y), (255, 0, 0), 2)
    # print(len(pixels_color))
    # end_x_2 = int(center[0] + line_length * math.cos(math.radians(angle+t_angle)))
    # end_y_2 = int(center[1] + line_length * math.sin(math.radians(angle+t_angle)))
    # val_2 = image[end_y_2][end_x_2].copy()
    # cv2.line(image, (int(center[0]), int(center[1])), (end_x_2, end_y_2), (0, 255, 0), 2)
    # end_x_3 = int(center[0] + line_length * math.cos(math.radians(angle - t_angle)))
    # end_y_3 = int(center[1] + line_length * math.sin(math.radians(angle - t_angle)))
    # val_3 = image[end_y_3][end_x_3].copy()
    # cv2.line(image, (int(center[0]), int(center[1])), (end_x_3, end_y_3), (0, 0, 255), 2)
    return pixels_color


class BinaryDataHandler:
    def __init__(self, vehicle: Vehicle, connection: SocketConnection):
        self.connection = connection
        self.vehicle = vehicle

    async def start_driving(self):
        await self.example_1()

    def save_image(self, image_data, index_camera: str):
        filename = f"{index_camera}.webp"

        with open(filename, 'wb') as f:
            f.write(image_data)

    def forword(self, t_w):
        self.vehicle.setMotorPower(50, 50)
        time.sleep(t_w)
        self.vehicle.setMotorPower(0, 0)
        time.sleep(0.0001)

    def rotay_to_one(self):
        self.vehicle.rotate(1)
        time.sleep(0.4)

    def rotay_to_twice(self):
        self.vehicle.rotate(-1)
        time.sleep(0.4)

    def forword_t(self, t_w):
        t = threading.Thread(target=self.forword, args=(t_w,), )
        t.start()

    def rotay_to_twice_t(self):
        t = threading.Thread(target=self.rotay_to_twice, args=(), )
        t.start()

    def rotay_to_one_t(self):
        t = threading.Thread(target=self.rotay_to_one, args=(), )
        t.start()

    # пример получения скриншотов с камер
    async def example_1(self):
        global old_center
        lower = np.array([0, 0, 0])
        upper = np.array([179, 30, 187])
        # upper = np.array([179, 48, 187])
        # Set minimum and maximum HSV values to display
        lower_line = np.array([0, 54, 157])
        upper_line = np.array([60, 255, 210])
        angle = 90
        t_work_m = 0.02
        t_stop_m = 0.02
        i_iter = 0
        c = Camera()
        c.start()

        while True:
            s = time.time()
            # скриншот с камеры 1
            # посылаем на клиент сообщение, с какой камеры сделать скриншот (возможные варианты:
            # camera1, camera2, camera3, camera4, camera5, camera6)
            self.connection.send_data("camera1")
            # получаем скриншот с камеры 1. Код ниже не будет выполняться, пока не придет скриншот с камеры 1
            image_data = self.connection.receive_data()
            # print("get_image_data")
            nparr = np.frombuffer(image_data, np.uint8)
            images = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            c.update_image(images)
            blended = c.get_blended()
            # print(blended)
            if blended is not None:
                cv2.imshow('Rotated Rectangles', blended)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            color_data = c.get_color_data()
            for id, i in enumerate(color_data):
                if i[0] != 0:
                    if id < 8:
                        angle -= 2.4
                        c.update_angle(angle)
                        self.rotay_to_one()
                        # t_work_m = 0.1
                    elif id > 10:
                        angle += 2.4
                        c.update_angle(angle)
                        self.rotay_to_twice()
                        # t_work_m = 0.1
                # else:
                #     pass_count_data += 1
                #     self.vehicle.setMotorPower(50, 50)
                #     time.sleep(0.007)
                #     self.vehicle.setMotorPower(0, 0)
                #     time.sleep(0.007)
                # print(id, i)
            # if abs(old_center[0] - center[0]) > 0.5 or abs(old_center[1] - center[1]) > 0.5:
            #     t_work_m = 0.02
            #     t_stop_m = 0.02
            for i in color_data[7:11]:
                if i[0] != 0:
                    self.forword(t_work_m)
                    # self.vehicle.setMotorPower(45, 45)
                    # time.sleep(t_work_m)
                    # if t_work_m > 1:
                    #     self.vehicle.setMotorPower(-50, -50)
                    #     time.sleep(0.001)
                    # self.vehicle.setMotorPower(0, 0)
                    # time.sleep(t_stop_m)
                    # print('t_work_m', t_work_m)
                    # print(abs(old_center[0] - center[0]), abs(old_center[1] - center[1]))
                    # print()
                    # if 0 < abs(old_center[0] - center[0]) < 0.3 or 0 < abs(old_center[1] - center[1]) < 0.3:
                    #     t_work_m += 1.5
                        # t_stop_m -= 0.01
                    break
            print("time: ", time.time()-s)
            # hsv = cv2.cvtColor(images, cv2.COLOR_BGR2HSV)

            # try:
            #     # Convert to HSV format and color threshold
            #     hsv_line = cv2.cvtColor(images, cv2.COLOR_BGR2HSV)
            #     mask_line = cv2.inRange(hsv_line, lower_line, upper_line)
            #     result_line = cv2.bitwise_and(images, images, mask=mask_line)
            #     # Convert to HSV format and color threshold
            #     hsv = cv2.cvtColor(images, cv2.COLOR_BGR2HSV)
            #     mask = cv2.inRange(hsv, lower, upper)
            #     images = cv2.bitwise_and(images, images, mask=mask)
            #     # ret, frame = video.read()
            #     image = remove_small_particles(images)
            #
            #     gauss_result_line = cv2.GaussianBlur(result_line, (3, 3), 0)
            #     # median_result_line = cv2.medianBlur(gauss_result_line, 3)
            #
            #     alpha = 5.0  # например, 2.0 — сильный контраст
            #     beta = 0  # можно добавить яркость
            #
            #     # (опционально) Убираем шум
            #     # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            #     # result_noise = cv2.morphologyEx(median_result_line, cv2.MORPH_OPEN, kernel)
            #
            #     result_line = cv2.convertScaleAbs(gauss_result_line, alpha=alpha, beta=beta)
            #
            #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #     _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            #     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #     alpha = 0.5  # Прозрачность (0-1)
            #     blended = cv2.addWeighted(image, alpha, result_line, 1 - alpha, 0)
            #     for cnt in contours:
            #         # Пропускаем маленькие контуры
            #         if cv2.contourArea(cnt) < 100:
            #             continue
            #
            #         # Преобразуем контур в прямоугольник
            #         rect, _, center = contour_to_rotated_rectangle(cnt)
            #
            #         # Рисуем результат
            #         color_data = draw_rotated_rectangle(blended, rect, angle, center)
            #         if i_iter == 0:
            #             old_center = center
            #         count_data = len(color_data)
            #         pass_count_data = 0
            #         # print(old_center[0] - center[0])
            #         # if old_center[0] - center[0] < 3 and old_center[1] - center[1] < 3:
            #         #     t_work_m += 0.1
            #         # else:
            #         #     t_work_m -= 0.1
            #         if pass_count_data == count_data:
            #             print("eeeeee")
            #             self.vehicle.setMotorPower(50, 50)
            #             time.sleep(0.01)
            #             self.vehicle.setMotorPower(0, 0)
            #             time.sleep(0.0001)
            #         old_center = center
            #         # print()
            #         # print("color_data", color_data)
            #         # if len(color_data) > 0:
            #         #     print('color_data[0]', color_data[0])
            #         #     if color_data[0] == 0:
            #         #         print("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            #         #         if 0 <= start_angle <= 360:
            #         #             start_angle = 1
            #         #             angle -= 2.4
            #         #         else:
            #         #             start_angle = 0
            #         #             angle = 90
            #         #         # self.vehicle.setMotorPower(0, 0)
            #         #         print(start_angle, angle)
            #         #         self.vehicle.rotate(start_angle)
            #         #         time.sleep(0.3)
            #         #     else:
            #         #         # pass
            #         #         self.vehicle.setMotorPower(100, 100)
            #         #         time.sleep(0.05)
            #         # self.vehicle.setMotorPower(0, 0)
            #     # Показываем результат
            #     # alpha = 0.5  # Прозрачность (0-1)
            #     # blended = cv2.addWeighted(image, alpha, result_line, 1 - alpha, 0)
            #     # alpha = 0.5  # Прозрачность (0-1)
            #     # blended = cv2.addWeighted(image, alpha, result_line, 1 - alpha, 0)
            #     cv2.imshow('Rotated Rectangles', blended)
            # except Exception as e:
            #     print(e)
            # i_iter = 1

    # # пример использования setMotorPower и rotate
    # def example_4(self):
    #     self.vehicle.rotate(90)
    #     time.sleep(1)

    #     self.vehicle.rotate(-90)
    #     time.sleep(1)

    #     self.vehicle.setMotorPower(100, 100)
    #     time.sleep(1)

    #     self.vehicle.rotate(-90)
    #     time.sleep(1)

    #     self.vehicle.setMotorPower(0, 0)

    #     self.vehicle.rotate(180)
    #     time.sleep(1)


async def control_vehicle(vehicle: Vehicle, connection: SocketConnection):
    data_handler = BinaryDataHandler(vehicle, connection)
    await data_handler.start_driving()
