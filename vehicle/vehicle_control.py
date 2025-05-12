import random
from vehicle.Vehicle import Vehicle
import asyncio
from connection.SocketConnection import SocketConnection
from collections import deque
import json
import struct
import time
import cv2
import numpy as np
import math


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


def draw_rotated_rectangle(image, box, angle, center, speed):
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
    cv2.circle(image, (int(center[0]), int(center[1])), 5, (255, 0, 0), -1)

    # Угол и скорость
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_angle = f"Angle: {angle:.1f} deg"
    text_speed = f"Speed: {speed:.2f} px/s"
    
    # Рисуем угол поворота
    cv2.putText(image, text_angle, (int(center[0]) - 80, int(center[1]) - 70),
                font, 0.6, (0, 0, 255), 2)
    # Рисуем скорость
    cv2.putText(image, text_speed, (int(center[0]) - 80, int(center[1]) - 45),
                font, 0.6, (0, 0, 255), 2)

    # Линия направления робота
    line_length = 27
    end_x = int(center[0] + line_length * math.cos(math.radians(angle)))
    end_y = int(center[1] + line_length * math.sin(math.radians(angle)))
    val = image[end_y][end_x].copy()
    cv2.line(image, (int(center[0]), int(center[1])), (end_x, end_y), (255, 0, 0), 2)
    
    return val


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


class BinaryDataHandler:
    def __init__(self, vehicle: Vehicle, connection: SocketConnection):
        self.connection = connection
        self.vehicle = vehicle
        
        self.movement = False
        self.prev_center = None
        self.prev_time = None
        self.speed = 0
        self.speed_buffer = deque(maxlen=3)


    async def start_driving(self):
        self.example_1()

    def save_image(self, image_data, index_camera: str):
        filename = f"{index_camera}.webp"

        with open(filename, 'wb') as f:
            f.write(image_data)

    # пример получения скриншотов с камер
    def example_1(self):
        lower = np.array([0, 0, 0])
        # upper = np.array([179, 30, 187])
        upper = np.array([179, 48, 187])
        # Set minimum and maximum HSV values to display
        lower_line = np.array([0, 54, 157])
        upper_line = np.array([60, 255, 210])
        angle = 90
        start_angle = 0
        while True:
            # скриншот с камеры 1
            # посылаем на клиент сообщение, с какой камеры сделать скриншот (возможные варианты:
            # camera1, camera2, camera3, camera4, camera5, camera6)
            self.connection.send_data("camera1")
            # получаем скриншот с камеры 1. Код ниже не будет выполняться, пока не придет скриншот с камеры 1
            image_data = self.connection.receive_data()
            print("get_image_data")
            nparr = np.frombuffer(image_data, np.uint8)
            images = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            hsv = cv2.cvtColor(images, cv2.COLOR_BGR2HSV)

            try:
                # Convert to HSV format and color threshold
                hsv_line = cv2.cvtColor(images, cv2.COLOR_BGR2HSV)
                mask_line = cv2.inRange(hsv_line, lower_line, upper_line)
                result_line = cv2.bitwise_and(images, images, mask=mask_line)
                # Convert to HSV format and color threshold
                hsv = cv2.cvtColor(images, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower, upper)
                images = cv2.bitwise_and(images, images, mask=mask)
                # ret, frame = video.read()
                image = remove_small_particles(images)

                # gauss_result_line = cv2.GaussianBlur(result_line, (3, 3), 0)
                # median_result_line = cv2.medianBlur(gauss_result_line, 3)
                #
                # alpha = 5.0  # например, 2.0 — сильный контраст
                # beta = 0     # можно добавить яркость
                #
                # # (опционально) Убираем шум
                # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                # result_noise = cv2.morphologyEx(median_result_line, cv2.MORPH_OPEN, kernel)
                #
                # result_line = cv2.convertScaleAbs(result_noise, alpha=alpha, beta=beta)

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                alpha = 0.5  # Прозрачность (0-1)
                blended = cv2.addWeighted(image, alpha, result_line, 1 - alpha, 0)
                
                current_time = time.time()

                for cnt in contours:
                    if cv2.contourArea(cnt) < 100:
                        continue

                    rect, _, center = contour_to_rotated_rectangle(cnt)

                    # Вычисляем мгновенную скорость
                    if self.prev_center is not None and self.prev_time is not None:
                        dx = center[0] - self.prev_center[0]
                        dy = center[1] - self.prev_center[1]
                        distance = math.sqrt(dx**2 + dy**2)
                        dt = current_time - self.prev_time
                        if dt > 0:
                            instant_speed = distance / dt
                            self.speed_buffer.append(instant_speed)
                            
                            # Сглаживаем скорость
                            self.speed = sum(self.speed_buffer) / len(self.speed_buffer)
                    else:
                        self.speed = 0
                        self.speed_buffer.append(self.speed)

                    # Запоминаем для следующего кадра
                    self.prev_center = center
                    self.prev_time = current_time

                    # Рисуем прямоугольник с информацией о скорости
                    alpha = 0.5
                    blended = cv2.addWeighted(image, alpha, result_line, 1 - alpha, 0)
                    color_data = draw_rotated_rectangle(blended, rect, angle, center, self.speed)

                    if self.movement:
                        if len(color_data) > 0:
                            if color_data[0] == 0:
                                if 0 <= start_angle <= 360:
                                    start_angle += 1
                                    angle -= 5
                                else:
                                    start_angle = 0
                                    angle = 90
                                print(start_angle, angle)
                                self.vehicle.rotate(start_angle)
                            else:
                                self.vehicle.setMotorPower(100, 100)
                                time.sleep(0.05)
                                self.vehicle.setMotorPower(0, 0)
                cv2.imshow('Rotated Rectangles', blended)

            except Exception as e:
                print(e)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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
    
