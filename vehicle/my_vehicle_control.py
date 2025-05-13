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

def angle_between_points(pt1, pt2):
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    return angle_deg % 360  # Приводим к [0°, 360°)

class Camera:
    def __init__(self, connection: SocketConnection):
        self.connection = connection

        self.min_size_particles = 150

        self.lower_line = np.array([0, 54, 157])
        self.upper_line = np.array([60, 255, 210])

        self.lower_robot = np.array([0, 0, 0])
        self.upper_robot = np.array([179, 30, 187])

        self.lower_first_part_robot = np.array([0, 0, 0])
        self.upper_first_part_robot = np.array([179, 255, 155])

        self.lower_second_part_robot = np.array([0, 0, 130])
        self.upper_second_part_robot = np.array([179, 255, 255])
        
        self.squareSize = 32
        self.halfSquare = squareSize // 2

    def remove_small_particles(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(binary)

        for contour in contours:
            if cv2.contourArea(contour) > self.min_size_particles:
                cv2.drawContours(mask, [contour], -1, 255, -1)
        result = cv2.bitwise_and(img, img, mask=mask)
        return result

    def processing_mask_line(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_line = cv2.inRange(hsv, self.lower_line, self.upper_line)
        img_line = cv2.bitwise_and(img, img, mask=mask_line)
        return img_line

    def processing_mask_robot(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_robot = cv2.inRange(hsv, self.lower_robot, self.upper_robot)
        img_robot = cv2.bitwise_and(img, img, mask=mask_robot)
        img_robot_with_out_small_particles = self.remove_small_particles(img_robot)
        return img_robot_with_out_small_particles

    def processing_mask_robot_part_one(self, img_robot):
        hsv = cv2.cvtColor(img_robot, cv2.COLOR_BGR2HSV)
        mask_line = cv2.inRange(hsv, self.lower_first_part_robot, self.upper_first_part_robot)
        img_robot_part_one = cv2.bitwise_and(img_robot, img_robot, mask=mask_line)
        return img_robot_part_one

    def processing_mask_robot_part_two(self, img_robot):
        hsv = cv2.cvtColor(img_robot, cv2.COLOR_BGR2HSV)
        mask_line = cv2.inRange(hsv, self.lower_second_part_robot, self.upper_second_part_robot)
        img_robot_part_two = cv2.bitwise_and(img_robot, img_robot, mask=mask_line)
        return img_robot_part_two

    @staticmethod
    def find_center(contour):
        rect = cv2.minAreaRect(contour)
        (center, (_, _), _) = rect
        return center

    def find_centers_on_parts_robot(self, img_robot_part_one, img_robot_part_two):
        center_robot_part_one = (0, 0)
        center_robot_part_two = (0, 0)

        gray_robot_part_one = cv2.cvtColor(img_robot_part_one, cv2.COLOR_BGR2GRAY)
        gray_robot_part_two = cv2.cvtColor(img_robot_part_two, cv2.COLOR_BGR2GRAY)

        _, thresh_robot_part_one = cv2.threshold(gray_robot_part_one, 5, 255, cv2.THRESH_BINARY)
        _, thresh_robot_part_two = cv2.threshold(gray_robot_part_two, 5, 255, cv2.THRESH_BINARY)

        contours_robot_part_one, _ = cv2.findContours(thresh_robot_part_one, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_robot_part_two, _ = cv2.findContours(thresh_robot_part_two, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours_robot_part_one:
            # Пропускаем маленькие контуры
            if cv2.contourArea(cnt) < 100:
                continue

            center_robot_part_one = self.find_center(cnt)

        for cnt in contours_robot_part_two:
            # Пропускаем маленькие контуры
            if cv2.contourArea(cnt) < 100:
                continue

            center_robot_part_two = self.find_center(cnt)

        # center_robot_part_one = self.find_center(contours_robot_part_one[0])
        # center_robot_part_two = self.find_center(contours_robot_part_two[0])
        return center_robot_part_one, center_robot_part_two

    def processing(self):
        while True:
            self.connection.send_data("camera1")
            image_data = self.connection.receive_data()
            new_format_image_data = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(new_format_image_data, cv2.IMREAD_COLOR)

            img_line = self.processing_mask_line(image)

            img_robot = self.processing_mask_robot(image)
            img_robot_part_one = self.processing_mask_robot_part_one(img_robot)
            img_robot_part_two = self.processing_mask_robot_part_two(img_robot)

            center_robot_part_one, center_robot_part_two = self.find_centers_on_parts_robot(img_robot_part_one, img_robot_part_two)
            # print(center_robot_part_one, center_robot_part_two)

            cv2.circle(img_robot, (int(center_robot_part_one[0]), int(center_robot_part_one[1])), 2, (255, 0, 0), -1)
            cv2.circle(img_robot, (int(center_robot_part_two[0]), int(center_robot_part_two[1])), 2, (0, 255, 0), -1)

            print(angle_between_points(center_robot_part_one, center_robot_part_two))
            # cv2.line(img_robot, center_robot_part_one, center_robot_part_two, (0, 255, 255), 2)

            cv2.imwrite("Image.png", img_robot)
            # cv2.imshow("Image", image)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    def run(self):
        thread_processing = threading.Thread(target=self.processing, args=())
        thread_processing.start()


class BinaryDataHandler:
    def __init__(self, vehicle: Vehicle, cam: Camera):
        self.cam = cam
        self.vehicle = vehicle
        self.cam.run()

    async def start_driving(self):
        self.go()

    def go(self):
        pass


async def control_vehicle(vehicle: Vehicle, connection: SocketConnection):
    cam = Camera(connection)
    data_handler = BinaryDataHandler(vehicle, cam)
    await data_handler.start_driving()