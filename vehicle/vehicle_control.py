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
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Angle: {angle:.1f} deg"
    cv2.putText(image, text, (int(center[0]) - 50, int(center[1]) - 30),
                font, 0.7, (0, 0, 255), 2)

    # Рисуем линию, показывающую ориентацию
    # line_length = max(image.shape[0], image.shape[1]) // 4
    line_length = 23
    end_x = int(center[0] + line_length * math.cos(math.radians(angle)))
    end_y = int(center[1] + line_length * math.sin(math.radians(angle)))
    print(image[end_y][end_x])
    cv2.line(image, (int(center[0]), int(center[1])), (end_x, end_y), (255, 0, 0), 2)    

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

    async def start_driving(self):
        self.example_1()

    def save_image(self, image_data, index_camera: str):
        filename = f"{index_camera}.webp"

        with open(filename, 'wb') as f:
            f.write(image_data)

    # пример получения скриншотов с камер
    def example_1(self):
        lower = np.array([0, 0, 0])
        upper = np.array([179, 30, 187])
        # Set minimum and maximum HSV values to display
        lower_line = np.array([0, 54, 157])
        upper_line = np.array([60, 255, 210])
        while True:
            # скриншот с камеры 1
            # посылаем на клиент сообщение, с какой камеры сделать скриншот (возможные варианты: camera1, camera2, camera3, camera4, camera5, camera6)
            self.connection.send_data("camera1")
            # получаем скриншот с камеры 1. Код ниже не будет выполняться, пока не придет скриншот с камеры 1
            image_data = self.connection.receive_data()
            nparr = np.frombuffer(image_data, np.uint8)
            images = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
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
                
                result_line = cv2.blur(result_line,(6,6))

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    # Пропускаем маленькие контуры
                    if cv2.contourArea(cnt) < 100:
                        continue

                    # Преобразуем контур в прямоугольник
                    rect, angle, center = contour_to_rotated_rectangle(cnt)

                    # Рисуем результат
                    alpha = 0.5  # Прозрачность (0-1)
                    blended = cv2.addWeighted(image, alpha, result_line, 1 - alpha, 0)
                    draw_rotated_rectangle(blended, rect, angle, center)

                # Показываем результат
                # alpha = 0.5  # Прозрачность (0-1)
                # blended = cv2.addWeighted(image, alpha, result_line, 1 - alpha, 0)
                # alpha = 0.5  # Прозрачность (0-1)
                # blended = cv2.addWeighted(image, alpha, result_line, 1 - alpha, 0)
                cv2.imshow('Rotated Rectangles', blended)
            except Exception as e:
                print(e)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # self.save_image(image_data, "camera1")

    #     # # скриншот с камеры 2
    #     # self.connection.send_data("camera2")
    #     #
    #     # image_data = self.connection.receive_data()
    #     #
    #     # self.save_image(image_data, "camera2")
    #     #
    #     # # скриншот с камеры 3
    #     # self.connection.send_data("camera3")
    #     #
    #     # image_data = self.connection.receive_data()
    #     #
    #     # self.save_image(image_data, "camera3")
    #     #
    #     # # скриншот с камеры 4
    #     # self.connection.send_data("camera4")
    #     #
    #     # image_data = self.connection.receive_data()
    #     #
    #     # self.save_image(image_data, "camera4")
    #     #
    #     # # скриншот с камеры 5
    #     # self.connection.send_data("camera5")
    #     #
    #     # image_data = self.connection.receive_data()
    #     #
    #     # self.save_image(image_data, "camera5")
    #     #
    #     # # скриншот с камеры 6
    #     # self.connection.send_data("camera6")
    #     #
    #     # image_data = self.connection.receive_data()
    #     #
    #     # self.save_image(image_data, "camera6")

    # # пример управления роботом последовательным запуском команд управления
    # def example_2(self):
    #     # первый параметр - мощность вращения левых колес в процентах, второй параметр - мощность вращения правых колес.
    #     # если значение положительное, то колесо вращается по часовой стрелке, если отрицательно, то против часовой стрелки.
    #     self.vehicle.setMotorPower(100, 100)
    #     time.sleep(1)

    #     self.vehicle.setMotorPower(100, -100)
    #     time.sleep(1)

    #     self.vehicle.setMotorPower(60, 60)
    #     time.sleep(3)

    #     self.vehicle.setMotorPower(100, -100)
    #     time.sleep(3)

    #     self.vehicle.setMotorPower(50, 50)
    #     time.sleep(1)

    #     self.vehicle.setMotorPower(100, -80)
    #     time.sleep(1.5)

    #     self.vehicle.setMotorPower(70, 70)
    #     time.sleep(4)

    # # пример получения скриншотов и управления роботом последовательным запуском команд управления
    # def example_3(self):
    #     # скриншот с камеры 1
    #     self.connection.send_data("camera1")

    #     image_data = self.connection.receive_data()

    #     self.save_image(image_data)

    #     self.vehicle.setMotorPower(100, 100)
    #     time.sleep(1)

    #     # скриншот с камеры 2
    #     self.connection.send_data("camera2")

    #     image_data = self.connection.receive_data()

    #     self.save_image(image_data)

    #     self.vehicle.setMotorPower(-100, 100)
    #     time.sleep(1)

    #     # скриншот с камеры 3
    #     self.connection.send_data("camera3")

    #     image_data = self.connection.receive_data()

    #     self.save_image(image_data)

    #     self.vehicle.setMotorPower(60, 60)
    #     time.sleep(3)

    #     # скриншот с камеры 4
    #     self.connection.send_data("camera4")

    #     image_data = self.connection.receive_data()

    #     self.save_image(image_data)

    #     self.vehicle.setMotorPower(-100, 100)
    #     time.sleep(3)

    #     # скриншот с камеры 5
    #     self.connection.send_data("camera5")

    #     image_data = self.connection.receive_data()

    #     self.save_image(image_data)

    #     self.vehicle.setMotorPower(50, 50)
    #     time.sleep(1)

    #     # скриншот с камеры 6
    #     self.connection.send_data("camera6")

    #     image_data = self.connection.receive_data()

    #     self.save_image(image_data)

    #     self.vehicle.setMotorPower(-100, 80)
    #     time.sleep(1.5)

    #     self.vehicle.setMotorPower(70, 70)
    #     time.sleep(4)

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
    
