import random
from vehicle.Vehicle import Vehicle
import asyncio
from connection.SocketConnection import SocketConnection
import json
import struct
import time

class BinaryDataHandler:
    def __init__(self, vehicle: Vehicle, connection: SocketConnection):
        self.connection = connection
        self.vehicle = vehicle

    async def start_driving(self):

        # self.example_1()
        self.example_2()
        # self.example_3()
        # self.example_4()

    def save_image(self, image_data):
        filename = f"{str(int(time.time()))}_{str(random.random())}.webp"

        with open(filename, 'wb') as f:
            f.write(image_data)

    # пример получения скриншотов с камер
    # def example_1(self):
    #     # скриншот с камеры 1
    #     # посылаем на клиент сообщение, с какой камеры сделать скриншот (возможные варианты: camera1, camera2, camera3, camera4, camera5, camera6)
    #     self.connection.send_data("camera1")

    #     # получаем скриншот с камеры 1. Код ниже не будет выполняться, пока не придет скриншот с камеры 1
    #     image_data = self.connection.receive_data()

    #     self.save_image(image_data)

    #     # скриншот с камеры 2
    #     self.connection.send_data("camera2")

    #     image_data = self.connection.receive_data()

    #     self.save_image(image_data)

    #     # скриншот с камеры 3
    #     self.connection.send_data("camera3")

    #     image_data = self.connection.receive_data()

    #     self.save_image(image_data)

    #     # скриншот с камеры 4
    #     self.connection.send_data("camera4")

    #     image_data = self.connection.receive_data()

    #     self.save_image(image_data)

    #     # скриншот с камеры 5
    #     self.connection.send_data("camera5")

    #     image_data = self.connection.receive_data()

    #     self.save_image(image_data)

    #     # скриншот с камеры 6
    #     self.connection.send_data("camera6")

    #     image_data = self.connection.receive_data()

    #     self.save_image(image_data)

    # пример управления роботом последовательным запуском команд управления
    def example_2(self):
        # первый параметр - мощность вращения левых колес в процентах, второй параметр - мощность вращения правых колес.
        # если значение положительное, то колесо вращается по часовой стрелке, если отрицательно, то против часовой стрелки.
        self.vehicle.setMotorPower(-50, 50)
        time.sleep(1)

        self.vehicle.setMotorPower(50, 50)
        time.sleep(1)

        self.vehicle.setMotorPower(-40, 90)
        time.sleep(1)

        self.vehicle.setMotorPower(10, 30)
        time.sleep(1)

        self.vehicle.setMotorPower(-40, 80)
        time.sleep(1)

        self.vehicle.setMotorPower(-50, 50)
        time.sleep(1)
        # заехал на бугорок (стою)

        self.vehicle.setMotorPower(-70, 70)
        time.sleep(1)

        self.vehicle.setMotorPower(-20, 20)
        time.sleep(1)

        self.vehicle.rotate(6)
        time.sleep(1)

        self.vehicle.rotate(3)
        time.sleep(1)

        self.vehicle.setMotorPower(-30, 30)
        time.sleep(1)

        self.vehicle.setMotorPower(6, 6)
        time.sleep(1)

        self.vehicle.setMotorPower(-70, 70)
        time.sleep(1)

        self.vehicle.setMotorPower(10, 10)
        time.sleep(1)

        self.vehicle.rotate(-15)
        time.sleep(1)

        self.vehicle.setMotorPower(5, 5)
        time.sleep(1)

        self.vehicle.rotate(-6)
        time.sleep(1)

        # self.vehicle.setMotorPower(2, 2)
        # time.sleep(1)

        self.vehicle.setMotorPower(4.5, 4.5)
        time.sleep(1)

        self.vehicle.setMotorPower(0, 0)
        time.sleep(1)

        self.vehicle.setMotorPower(6, 6)
        time.sleep(1)

        #тут мне плохо стало

        self.vehicle.setMotorPower(-7, -7)
        time.sleep(1)

        self.vehicle.rotate(-30)
        time.sleep(1)

        self.vehicle.setMotorPower(10, 10)
        time.sleep(1)

        self.vehicle.setMotorPower(10, 10)
        time.sleep(1)

        self.vehicle.rotate(10)
        time.sleep(1)

        self.vehicle.rotate(10)
        time.sleep(1)

        self.vehicle.setMotorPower(10, 10)
        time.sleep(1)

        self.vehicle.rotate(9)
        time.sleep(1)

        self.vehicle.setMotorPower(10, 10)
        time.sleep(1)

        self.vehicle.rotate(-25)
        time.sleep(1)

        self.vehicle.rotate(-7)
        time.sleep(1)

        self.vehicle.setMotorPower(0, 0)
        time.sleep(1)

        self.vehicle.setMotorPower(30, 30)
        time.sleep(3)

        self.vehicle.setMotorPower(-5, -5)
        time.sleep(1)

        self.vehicle.rotate(-83)
        time.sleep(1)

        # на таран 2-го чек-поинта
        self.vehicle.setMotorPower(100, 100)
        time.sleep(1)

        # self.vehicle.rotate(-7)
        # time.sleep(1)

        self.vehicle.setMotorPower(20, 20)
        time.sleep(1)

        self.vehicle.rotate(30)
        time.sleep(1)

        self.vehicle.setMotorPower(7, 7)
        time.sleep(1)

        self.vehicle.rotate(20)
        time.sleep(1)

        self.vehicle.setMotorPower(5, 5)
        time.sleep(1)

        self.vehicle.rotate(23)
        time.sleep(1)

        # self.vehicle.setMotorPower(4, 4)
        # time.sleep(1)

        # путь до 3 чек-поинта (уходим со 2-го)
        self.vehicle.setMotorPower(5, 5)
        time.sleep(1)

        self.vehicle.rotate(12)

        self.vehicle.setMotorPower(15, 15)
        time.sleep(1)

        self.vehicle.setMotorPower(0, 0)
        time.sleep(0)


    # пример получения скриншотов и управления роботом последовательным запуском команд управления
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

    # пример использования setMotorPower и rotate
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
